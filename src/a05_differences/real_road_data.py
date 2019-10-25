
import numpy as np
import torch
import logging
log = logging.getLogger('exp.road_data_discrepancy')
from ..paths import DIR_EXP
from ..pipeline.config import add_experiment
from ..pipeline.pipeline import Pipeline
from ..pipeline.evaluations import Evaluation, TrChannelLoad, TrChannelSave
from ..pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix, tr_print
from ..pipeline.transforms_imgproc import TrZeroCenterImgs, TrShow
from ..pipeline.transforms_pytorch import tr_torch_images, TrCUDA, TrNP, torch_onehot
from ..datasets.generic_sem_seg import TrSemSegLabelTranslation
from ..datasets.cityscapes import CityscapesLabelInfo

from ..a01_sem_seg.transforms import TrColorimg
from ..a01_sem_seg.exp0130_half_precision import ExpSemSegPSP_Apex
from ..a04_reconstruction.experiments import TrPix2pixHD_Generator, load_pix2pixHD_default

from .E0_article_evaluation import get_anomaly_net
from .experiments import Exp0517_Diff_SwapFgd_ImgVsLabels_semGT, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT
from ..datasets.dataset import imwrite, ImageBackgroundService, ChannelLoaderImage




class Exp_PSP_BDD(ExpSemSegPSP_Apex):
	"""
	Load saved weights for PSP BDD from the enseble
	"""
	cfg = add_experiment(ExpSemSegPSP_Apex.cfg,
		name = '0121_PSPEns_BDD_00',
		net = dict(
			use_aux = True, # sadly there are some weights in the checkpoint associated with this
			apex_mode = False,
		)
	)

	def training_run(self):
		raise NotImplementedError()


class EvaluationDiscrepancyPipeline(Evaluation):

	def __init__(self):
		self.construct_persistence()

		self.loader_args = dict(
			shuffle = False,
			batch_size = 4,
			num_workers = 4,
			drop_last = False,
		)

	def init_semseg(self):
		self.exp_sem_seg = Exp_PSP_BDD()
		self.exp_sem_seg.init_net('eval')

	def tr_apply_semseg(self, image, **_):
		
		logits = self.exp_sem_seg.net_mod(image)

		_, class_id = torch.max(logits, 1)

		return dict(
			pred_labels_trainIds = class_id,
		)

	def init_gan(self):
		# TODO context?
		self.mod_pix2pix = load_pix2pixHD_default('0405_nostyle_crop_ctc')
		self.mod_pix2pix.cuda()

		self.table_trainId_to_fullId = CityscapesLabelInfo.table_trainId_to_label
		self.table_trainId_to_fullId_cuda = torch.from_numpy(self.table_trainId_to_fullId).byte().cuda()

		self.tr_trainId_to_fullId = TrSemSegLabelTranslation(
			self.table_trainId_to_fullId,
			fields=[('pred_labels_trainIds', 'labels_source')],
		),

	def tr_apply_pix2pix(self, pred_labels_trainIds, **_):

		labels = self.table_trainId_to_fullId_cuda[pred_labels_trainIds.reshape(-1)].reshape(pred_labels_trainIds.shape)

		labels_onehot = torch_onehot(
			labels,
			num_channels=self.mod_pix2pix.opt.label_nc, 
			dtype=torch.float32,
		)

		# labels = tr_trainId_to_fullId.forward(None, pred_labels_trainIds)
		inst = None
		img = None
		log.debug(f'labels {labels_onehot.shape} {labels_onehot.dtype}')

		log.debug(f'labels_onehot {labels_onehot.shape} {labels_onehot.dtype}')

		gen_out = self.mod_pix2pix.inference(labels_onehot, inst, img)

		log.debug(f'gen out shape {gen_out.shape} from labels {labels.shape}')

		desired_shape = labels_onehot.shape[2:]
		if gen_out.shape[2:] != desired_shape:
			gen_out = gen_out[:, :, :desired_shape[0], :desired_shape[1]]

		gen_image = (gen_out + 1) * 128
		gen_image = torch.clamp(gen_image, min=0, max=255)
		gen_image = gen_image.type(torch.uint8)
		# gen_image = gen_image.clamp(12, 254).byte()

		return dict(
			gen_image_raw = gen_out,
			gen_image = gen_image,
		)

	# def tr_postprocess_gen_image(self, gen_image, **_):
	# 	# gen_image = (gen_image_raw + 1) * 128
	# 	# gen_image = gen_image.clamp(12, 254).byte()

	# 	# return dict(
	# 	# 	gen_image = gen_image_raw,
	# 	# )

	# 	# lambda gen_image, **_: dict(gen_image = gen_image.transpose([1, 2, 0])),

	# 	gen_image = gen_image.transpose([1, 2, 0])
	# 	# gen_image = np.clip(gen_image, 0, 255)
	# 	gen_image = gen_image.astype(np.uint8)

	# 	return dict(
	# 		gen_image = gen_image,
	# 	)

	def init_discrepancy(self):

		self.exp_discrepancy = Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT()
		self.exp_discrepancy.init_net('eval')

	def tr_apply_discrepancy(self, image, pred_labels_trainIds, gen_image_raw, **_):
		
		pred_anomaly_logits = self.exp_discrepancy.net_mod(
			image = image, 
			labels = pred_labels_trainIds,
			gen_image = gen_image_raw,
		)

		pred_anomaly_prob = torch.nn.functional.softmax(pred_anomaly_logits, dim=1)
		
		# extract channel 1 - probability for the "1" result
		pred_anomaly_prob = pred_anomaly_prob[:, 1]

		return dict(
			pred_anomaly_prob = pred_anomaly_prob,
		)

	def tr_make_demo(self, image, pred_labels_trainIds_colorimg, pred_anomaly_prob, gen_image = None, **_):

		EMPTY_IMG = np.zeros(image.shape, dtype=np.uint8)

		# labels_colorimg = self.tr_colorimg.forward(None, labels) if labels is not None else EMPTY_IMG

		# prob_based_img = (pred_class_trainId_colorimg.astype(np.float32) * (0.25 + 0.75*pred_class_prob[:, :, None])).astype(np.uint8)

		gen_image = EMPTY_IMG if gen_image is None else gen_image
		
		out = self.img_grid_2x2([image, pred_labels_trainIds_colorimg, gen_image, pred_anomaly_prob])
	
		return dict(
			demo = out,
		)


	def construct_persistence(self):
		super().construct_persistence()

		# self.persistence_base_dir = '{channel.ctx.exp.workdir}/pred/{dset.name}_{dset.split}/'
		self.workdir = DIR_EXP / '0550_real_road_data'
		self.persistence_base_dir = '{channel.ctx.workdir}/pred/{dset.name}_{dset.split}/'

		self.chan_demo = ChannelLoaderImage(self.persistence_base_dir+'demo/{fid_no_slash}_demo.webp')
		self.chan_demo.ctx = self



	def construct_pipeline(self):
		self.tr_pre_batch = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image')
		)

		self.tr_colorimg = TrColorimg('pred_labels_trainIds')
		self.tr_colorimg.set_override(19, [0, 0, 0])

		# self.tr_onehot = TrOnehotTorch(pred_labels_trainIds)

		return Pipeline(
			tr_input = TrsChain(),
			tr_batch_pre_merge = self.tr_pre_batch,
			tr_batch = TrsChain(
				TrCUDA(),
				self.tr_apply_semseg,
				self.tr_apply_pix2pix,
				self.tr_apply_discrepancy,
				# self.tr_postprocess_gen_image,
				tr_print,
				TrKeepFields('gen_image', 'pred_anomaly_prob', 'pred_labels_trainIds'),
			),
			tr_output = TrsChain(
				TrNP(),
				self.tr_colorimg,
				# self.tr_postprocess_gen_image,
				lambda gen_image, **_: dict(gen_image = gen_image.transpose([1, 2, 0])),
				# TrShow('gen_image'),
				# TrShow('pred_labels_trainIds_colorimg'),
				# TrShow('pred_anomaly_prob'),
				self.tr_make_demo,
				TrChannelSave(self.chan_demo, 'demo'),
			),
			loader_args =self.loader_args,
		)



# def gen_run(gen_variant, dset, b_show=True):

# 	tr_gen = TrsChain(
# 		TrSemSegLabelTranslation(
# 			fields=dict(pred_labels_trainIds='labels_source'),
# 			table=CityscapesLabelInfo.table_trainId_to_label
# 		),
# 		TrPix2pixHD_Generator(gen_variant, b_postprocess=True),
# 	)

# 	tr_gen_and_show = TrsChain(
# 		tr_gen,
# 		TrColorimg('pred_labels_trainIds'),
# 		TrShow(['image', 'gen_image', 'pred_labels_trainIds_colorimg'])
# 	)

# 	tr_gen_and_save = TrsChain(
# 		tr_gen,
# 		TrSaveChannelsAutoDset(['gen_image']),
# 	)

# 	dset.set_channels_enabled('pred_labels_trainIds', 'image')
# 	dset.discover()

# 	if b_show:
# 		dset[1].apply(tr_gen_and_show)
# 	else:
# 		Frame.frame_list_apply(tr_gen_and_save, dset, ret_frames=False)




# def anomaly_run(exp, dset, b_show=True):
# 	pipe = exp.construct_default_pipeline('test')

# 	trout = exp.tr_out_for_eval_show if b_show else exp.tr_out_for_eval

# 	pipe.tr_output += trout

# 	dset.set_channels_enabled('image', 'pred_labels_trainIds', 'gen_image')
# 	dset.discover()


# 	if b_show:
# 		pipe.execute(dset, b_one_batch=True)
# 	else:
# 		pipe.execute(dset, b_accumulate=False)
# 		dset.flush_hdf5_files()
