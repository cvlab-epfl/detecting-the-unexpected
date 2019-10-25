
import numpy as np
import torch
from pathlib import Path
import logging, gc
log = logging.getLogger('exp.eval')

from ..paths import DIR_EXP
from ..pipeline.config import add_experiment
from ..pipeline.frame import Frame
from ..pipeline.pipeline import Pipeline
from ..pipeline.evaluations import Evaluation, TrChannelLoad, TrChannelSave
from ..pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix, tr_print, TrRenameKw
from ..pipeline.transforms_imgproc import TrZeroCenterImgs, TrShow
from ..pipeline.transforms_pytorch import tr_torch_images, TrCUDA, TrNP, torch_onehot
from ..datasets.dataset import imwrite, ImageBackgroundService, ChannelLoaderImage, ChannelLoaderHDF5
from ..datasets.generic_sem_seg import TrSemSegLabelTranslation
from ..datasets.cityscapes import CityscapesLabelInfo

from ..a01_sem_seg.transforms import SemSegLabelsToColorImg, TrColorimg
from ..a01_sem_seg.experiments import ExpSemSegPSP_Ensemble_BDD, ExpSemSegBayes_BDD, ExpSemSegPSP
from ..a04_reconstruction.experiments import Pix2PixHD_Generator
from .experiments import Exp0516_Diff_SwapFgd_ImgVsGen_semGT, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT
from ..a05_road_rec_baseline.experiments import Exp0525_RoadReconstructionRBM_LowLR

from .E0_article_evaluation import get_anomaly_net
from .E1_plot_utils import TrImgGrid, TrBlend, tr_draw_anomaly_contour

class EvaluationSemSeg(Evaluation):
	def __init__(self, exp):
		super().__init__(exp)

		self.tr_colorimg = TrColorimg('pred_labels')
		self.tr_colorimg_gt = TrColorimg('labels')

		self.tr_make_demo = TrImgGrid(
			['image', None, 'labels_colorimg', 'pred_labels_colorimg'],
			num_cols = 2,
			out_name = 'demo_semseg',
		)

	def construct_persistence(self):
		self.persistence_base_dir = '{channel.ctx.workdir}/{dset.name}_{dset.split}/'

		self.chan_pred_trainId = ChannelLoaderImage(self.persistence_base_dir+'semantic/{fid_no_slash}_predTrainId.png')
		self.chan_pred_trainId.ctx = self
		self.chan_pred_colorimg = ChannelLoaderImage(self.persistence_base_dir+'semantic/{fid_no_slash}_colorimg.png')
		self.chan_pred_colorimg.ctx = self
		self.chan_demo = ChannelLoaderImage(self.persistence_base_dir+'semantic/demo/{fid_no_slash}_demo.webp')
		self.chan_demo.ctx = self

	def construct_transforms(self, dset):
		self.tr_output = TrsChain(
			self.tr_colorimg,
			self.tr_colorimg_gt,
			self.tr_make_demo,
			TrChannelSave(self.chan_pred_trainId, 'pred_labels'),
			TrChannelSave(self.chan_pred_colorimg, 'pred_labels_colorimg'),
			TrChannelSave(self.chan_demo, 'demo'),
		)


class EvaluationSemSegWithUncertainty(EvaluationSemSeg):
	def __init__(self, exp):
		super().__init__(exp)

		self.tr_make_demo = TrImgGrid(
			['image', self.exp.uncertainty_field_name, 'labels_colorimg', 'pred_labels_colorimg'],
			num_cols = 2,
			out_name = 'demo',
		)

	def construct_persistence(self):
		super().construct_persistence()

		self.chan_demo = ChannelLoaderImage(self.persistence_base_dir+'anomaly/demo/{fid_no_slash}_anomaly_demo.webp')
		self.chan_demo.ctx = self

		self.chan_uncertainty = ChannelLoaderHDF5(
			self.persistence_base_dir+'anomaly/anomaly_score.hdf5',
			'{fid}',
			compression=6,
		)
		self.chan_uncertainty.ctx = self

	def construct_transforms(self, dset):
		super().construct_transforms(dset)

		self.tr_output.append(
			TrChannelSave(self.chan_uncertainty, self.exp.uncertainty_field_name),
		)




# 	def tr_make_demo(self, image, pred_labels_colorimg, labels = None, **_):
# 	#def out_demo(fid, dset, image, pred_class_trainId_colorimg, pred_class_by_bitstring_trainId_colorimg, pred_class_prob, labels_colorimg=None, **_):
# 	#fid_no_slash = fid.replace('/', '__')
	
# # 	pred_class_prob_img = adapt_img_data(pred_class_prob)
# # 	print(np.min(pred_class_prob), np.max(pred_class_prob))

# 		EMPTY_IMG = np.zeros(image.shape, dtype=np.uint8)

# 		labels_colorimg = self.tr_colorimg.forward(None, labels) if labels is not None else EMPTY_IMG

# 		# prob_based_img = (pred_class_trainId_colorimg.astype(np.float32) * (0.25 + 0.75*pred_class_prob[:, :, None])).astype(np.uint8)
		
# 		out = self.img_grid_2x2([image, labels_colorimg, pred_labels_colorimg, EMPTY_IMG])
	
# 		return dict(
# 			demo = out,
# 		)



class EvaluationDetectingUnexpected:
	SEM_SEG_UNCERTAINTY_NAMES = {
		'BaySegBdd': 'dropout',
		'PSPEnsBdd': 'ensemble',
	}

	ANOMALY_DETECTORS_CLASSES = {
		'discrepancy_gen_only': Exp0516_Diff_SwapFgd_ImgVsGen_semGT,
		'discrepancy_label_only': Exp0517_Diff_SwapFgd_ImgVsLabels_semGT,
		'discrepancy_label_and_gen': Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT,
		'rbm': Exp0525_RoadReconstructionRBM_LowLR,
	}

	def __init__(self, sem_seg_variant):

		self.sem_seg_variant = sem_seg_variant
		self.uncertainty_variant = self.SEM_SEG_UNCERTAINTY_NAMES[self.sem_seg_variant]
		self.anomaly_detector_variants = (
			list(self.ANOMALY_DETECTORS_CLASSES.keys())
		 	+ 
		 	[self.uncertainty_variant]
		)
		self.anomaly_detector_variants.sort()

		self.init_persistence()

	def init_persistence(self, out_dir_override=None):
		"""
		Storage locations for results and intermediate data
		"""

		out_dir_default = DIR_EXP / '05_Eval' / '{dset.name}_{dset.split}' / f'sem_{self.sem_seg_variant}'
		out_dir = Path(out_dir_override or out_dir_default)

		self.storage = dict(
			pred_labels_trainIds = ChannelLoaderImage(out_dir / 'sem_labels/{fid}_predTrainIds.png'),
			pred_labels_colorimg = ChannelLoaderImage(out_dir / 'sem_labels/{fid}_predColorImg.png'),
			gen_image = ChannelLoaderImage(out_dir / 'gen_image/{fid}_gen_image.webp'),

			demo_with_labels = ChannelLoaderImage(out_dir / 'demo/{fid_no_slash}_pipeline.webp'),
			demo_with_baselines = ChannelLoaderImage(out_dir / 'demo/{fid_no_slash}_scores.webp'),
		)

		self.storage.update({
			f'anomaly_{name}': ChannelLoaderHDF5(
				out_dir / f'anomaly_score/anomaly_{name}.hdf5',
				'{fid}',
				compression=6,
			)
			for name in self.anomaly_detector_variants
		})


	def init_semseg(self):
		if self.sem_seg_variant == 'BaySegBdd':
			exp = ExpSemSegBayes_BDD()
			exp.init_net('eval')

			anomaly_field_name = 'anomaly_dropout'

			tr_renames = TrRenameKw(
				pred_labels = 'pred_labels_trainIds', 
				pred_var_dropout = anomaly_field_name,
			)

		elif self.sem_seg_variant == 'PSPEnsBdd':
			exp = ExpSemSegPSP_Ensemble_BDD()
			exp.load_subexps()
			exp.init_net('master_eval')

			anomaly_field_name = 'anomaly_ensemble'

			tr_renames = TrRenameKw(
				pred_labels = 'pred_labels_trainIds', 
				pred_var_ensemble = anomaly_field_name,
			)
			
		else:
			raise NotImplementedError(self.sem_seg_variant)
		
		tr_write_results = TrsChain(
			TrChannelSave(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
			TrChannelSave(self.storage['pred_labels_colorimg'], 'pred_labels_colorimg'),
			TrChannelSave(self.storage[anomaly_field_name], anomaly_field_name)
		)

		exp.tr_out_for_eval = TrsChain(
			tr_renames,
			tr_write_results
		)

		tr_show_results = TrsChain(
			# TrImgGrid(
			# 	['image', anomaly_field_name, 'labels_colorimg', 'pred_labels_colorimg'],
			# 	num_cols = 2,
			# 	out_name = 'demo',
			# ),
			TrShow(['image', anomaly_field_name], ['labels_colorimg', 'pred_labels_colorimg']),
		)

		exp.tr_out_for_eval_show = TrsChain(
			tr_renames,
			tr_show_results
		)

		self.exp_semseg = exp

	def run_semseg(self, dset, b_show=False):
		pipe = self.exp_semseg.construct_default_pipeline('test')

		tr_out = self.exp_semseg.tr_out_for_eval_show if b_show else self.exp_semseg.tr_out_for_eval

		pipe.tr_output += tr_out

		dset.set_channels_enabled('image')

		if b_show:
			pipe.execute(dset, b_one_batch=True)
		else:
			pipe.execute(dset, b_accumulate=False)
			dset.flush_hdf5_files()

	def init_gen_image(self):
		self.pix2pix = Pix2PixHD_Generator()

	def run_gen_image(self, dset, b_show=False):
		pipe = self.pix2pix.construct_pipeline()
		
		pipe.tr_input += [
			TrChannelLoad(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
		]

		if b_show:
			pipe.tr_output.append(
				TrShow('gen_image'),
			)

			pipe.execute(dset, b_one_batch=True)
		else:
			pipe.tr_output.append(
				TrChannelSave(self.storage['gen_image'], 'gen_image'),
			)

			pipe.execute(dset, b_accumulate=False)
			dset.flush_hdf5_files()


	def init_detector(self, detector_name):
		# load network
		exp_class = self.ANOMALY_DETECTORS_CLASSES[detector_name]
		exp = exp_class()
		exp.init_net('eval')

		score_field = f'anomaly_{detector_name}'

		exp.tr_out_for_eval = TrChannelSave(self.storage[score_field], 'anomaly_p')
		exp.tr_out_for_eval_show = TrsChain(
			TrColorimg('pred_labels_trainIds'),
			TrShow(['image', 'gen_image', 'pred_labels_trainIds_colorimg', 'anomaly_p']),
		)
		
		return exp

	def run_detector(self, detector_exp_or_name, dset, b_show=False):

		if isinstance(detector_exp_or_name, str):
			exp = self.init_detector(detector_exp_or_name)
		else:
			exp = detector_exp_or_name

		pipe = exp.construct_default_pipeline('test')

		pipe.tr_input += [
			TrChannelLoad(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
			TrChannelLoad(self.storage['gen_image'], 'gen_image'),
		]
		
		if b_show:
			pipe.tr_output.append(exp.tr_out_for_eval_show)
			pipe.execute(dset, b_one_batch=True)
		else:
			pipe.tr_output.append(exp.tr_out_for_eval)
			pipe.execute(dset, b_accumulate=False)
			dset.flush_hdf5_files()

	def run_detector_all(self, dset):
		detector_names = list(self.ANOMALY_DETECTORS_CLASSES.keys())

		for name in detector_names:
			log.info(f'Running detector {name}')

			score_field = f'anomaly_{name}'
			score_file = Path(self.storage[score_field].resolve_file_path(dset, dset.frames[0]))

			if score_file.is_file():
				log.info(f'Out file for {name} already exists - skipping')
			else:
				self.run_detector(name, dset)
				gc.collect()

	
	def run_demo_imgs(self, dset, b_show=False):

		tr_make_demo_imgs = TrsChain(
			TrChannelLoad(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
			TrChannelLoad(self.storage['gen_image'], 'gen_image'),
			TrChannelLoad(self.storage['anomaly_discrepancy_label_and_gen'], 'anomaly_discrepancy_label_and_gen'),
			TrChannelLoad(self.storage[f'anomaly_{self.uncertainty_variant}'], 'anomaly_uncertainty'),
			TrChannelLoad(self.storage['anomaly_rbm'], 'anomaly_rbm'),
			dset.tr_get_anomaly_gt,
			tr_draw_anomaly_contour,
			SemSegLabelsToColorImg([('pred_labels_trainIds', 'pred_labels_colorimg')]),
		)

		tr_make_demo_imgs += [
			TrBlend(anomaly_field, 'image', f'overlay_{anomaly_field}', 0.8)
			for anomaly_field in ['anomaly_discrepancy_label_and_gen', 'anomaly_uncertainty', 'anomaly_rbm']
		]

		tr_make_demo_imgs += [
			TrImgGrid([
					'image', 'pred_labels_colorimg', 
					'gen_image', 'overlay_anomaly_discrepancy_label_and_gen',
				],
				num_cols = 2,
				out_name = 'demo_with_labels',
			),
			TrImgGrid([
					'anomaly_contour', 'overlay_anomaly_discrepancy_label_and_gen', 
					'overlay_anomaly_uncertainty', 'overlay_anomaly_rbm',
				],
				num_cols = 2,
				out_name = 'demo_with_baselines',
			),
		]

		if b_show:
			tr_make_demo_imgs.append(TrShow('demo_with_labels', 'demo_with_baselines'))
			dset[0].apply(tr_make_demo_imgs)

		else:
			tr_make_demo_imgs += [
				TrChannelSave(self.storage['demo_with_labels'], 'demo_with_labels'),
				TrChannelSave(self.storage['demo_with_baselines'], 'demo_with_baselines'),
			]
			Frame.frame_list_apply(tr_make_demo_imgs, dset, n_threads=6, batch=4)


class ExpPSP_EnsebleReuse(ExpSemSegPSP):
	"""
	Load saved weights for PSP BDD from the enseble
	"""
	cfg = add_experiment(ExpSemSegPSP_Ensemble_BDD.cfg,
		name = '0121_PSPEns_BDD_00',
		net = dict(
			use_aux = True, # sadly there are some weights in the checkpoint associated with this
			apex_mode = False,
		)
	)

	def training_run(self):
		raise NotImplementedError()


class DiscrepancyJointPipeline:

	def __init__(self):
		self.construct_persistence()

		self.loader_args = dict(
			shuffle = False,
			batch_size = 4,
			num_workers = 4,
			drop_last = False,
		)

		self.tr_make_demo = TrImgGrid([
				'image', 'pred_labels_trainIds_colorimg', 
				'gen_image', 'pred_anomaly_prob',
			],
			num_cols = 2,
			out_name = 'demo',
		)


	def init_semseg(self):
		self.exp_sem_seg = ExpPSP_EnsebleReuse()
		self.exp_sem_seg.init_net('eval')

	def tr_apply_semseg(self, image, **_):
		
		logits = self.exp_sem_seg.net_mod(image)

		_, class_id = torch.max(logits, 1)

		return dict(
			pred_labels_trainIds = class_id,
		)

	def init_gan(self):
		self.pix2pix = Pix2PixHD_Generator()

		# # TODO context?
		# self.generator = Pix2PixHD_Generator('0405_nostyle_crop_ctc')

		# # self.mod_pix2pix = load_pix2pixHD_default('0405_nostyle_crop_ctc')
		# self.mod_pix2pix.cuda()

		# self.table_trainId_to_fullId = CityscapesLabelInfo.table_trainId_to_label
		# self.table_trainId_to_fullId_cuda = torch.from_numpy(self.table_trainId_to_fullId).byte().cuda()

		# self.tr_trainId_to_fullId = TrSemSegLabelTranslation(
		# 	self.table_trainId_to_fullId,
		# 	fields=[('pred_labels_trainIds', 'labels_source')],
		# ),

	

	def tr_apply_pix2pix2(self, pred_labels_trainIds, **_):

		labels = self.table_trainId_to_fullId_cuda[pred_labels_trainIds.reshape(-1).long()].reshape(pred_labels_trainIds.shape)

		labels_onehot = torch_onehot(
			labels,
			num_channels=self.mod_pix2pix.opt.label_nc, 
			dtype=torch.float32,
		)

		# labels = tr_trainId_to_fullId.forward(None, pred_labels_trainIds)
		inst = None
		img = None
		# log.debug(f'labels {labels_onehot.shape} {labels_onehot.dtype}')

		# log.debug(f'labels_onehot {labels_onehot.shape} {labels_onehot.dtype}')

		gen_out = self.mod_pix2pix.inference(labels_onehot, inst, img)

		# log.debug(f'gen out shape {gen_out.shape} from labels {labels.shape}')

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

	def tr_apply_discrepancy(self, image, pred_labels_trainIds, gen_image_raw=None, **_):
		
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

	def tr_make_demo_2(self, image, pred_labels_trainIds_colorimg, pred_anomaly_prob, gen_image = None, **_):

		EMPTY_IMG = np.zeros(image.shape, dtype=np.uint8)

		# labels_colorimg = self.tr_colorimg.forward(None, labels) if labels is not None else EMPTY_IMG

		# prob_based_img = (pred_class_trainId_colorimg.astype(np.float32) * (0.25 + 0.75*pred_class_prob[:, :, None])).astype(np.uint8)

		gen_image = EMPTY_IMG if gen_image is None else gen_image
		
		out = self.img_grid_2x2([image, pred_labels_trainIds_colorimg, gen_image, pred_anomaly_prob])
	
		return dict(
			demo = out,
		)


	def construct_persistence(self):

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
				self.pix2pix.tr_generator,
				self.tr_apply_discrepancy,
				# self.tr_postprocess_gen_image,
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
			),
			loader_args =self.loader_args,
		)

	def run_on_dset(self, dset, b_show=False):
		"""
		@param b_show: runs only one batch and displays the result in the notebook
		"""
		
		pipe = self.construct_pipeline()
		
		if b_show:
			pipe.tr_output.append(TrShow('demo'))
			pipe.execute(dset, b_one_batch=True)

		else:
			
			pipe.execute(dset, b_accumulate=False)
	
	

class DiscrepancyJointPipeline_LabelsOnly(DiscrepancyJointPipeline):
	def init_discrepancy(self):
		self.exp_discrepancy = Exp0517_Diff_SwapFgd_ImgVsLabels_semGT()
		self.exp_discrepancy.init_net('eval')

	def init_gan(self):
		pass

	def construct_pipeline(self):
		self.tr_pre_batch = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image')
		)

		self.tr_colorimg = TrColorimg('pred_labels_trainIds')

		return Pipeline(
			tr_input = TrsChain(),
			tr_batch_pre_merge = self.tr_pre_batch,
			tr_batch = TrsChain(
				TrCUDA(),
				self.tr_apply_semseg,
				self.tr_apply_discrepancy,
				tr_print,
				TrKeepFields('pred_anomaly_prob', 'pred_labels_trainIds'),
			),
			tr_output = TrsChain(
				TrNP(),
				self.tr_colorimg,
				self.tr_make_demo,
				TrChannelSave(self.chan_demo, 'demo'),
			),
			loader_args =self.loader_args,
		)

