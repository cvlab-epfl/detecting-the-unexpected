
import torch
import numpy as np
from pathlib import Path
from ..paths import DIR_EXP
from .transforms import *
from ..pipeline.pipeline import Pipeline
from ..pipeline.transforms import TrsChain, TrKeepFields
from ..pipeline.transforms_pytorch import torch_onehot, TrCUDA, TrNP
from ..datasets.cityscapes import CityscapesLabelInfo
from ..pipeline.evaluations import TrChannelLoad, TrChannelSave


# channel_reconstruction_p2pHD_stylevec_tCTC = ChannelResultImage('reconstruction/p2pHD-stylevec_tCTC', suffix='_pred_img_reconstr')
# channel_reconstruction_p2pHD_stylevec_mrcnn_tCTC = ChannelResultImage('reconstruction/p2pHD-stylevec-mrcnn_tCTC', suffix='_pred_img_reconstr')

class Pix2PixHD_Generator:

	pix2pixHD_VARIANTS = {
		'nostyle_pretrained': """
			--name label2city_1024p
			--checkpoints_dir /cvlabdata2/home/lis/pytorch_pretrained/pix2pixHD_checkpoints
		""",
		'0401_style_ctc': f"""
			--name 0401_pix2pixHD512_style_ctc --resize_or_crop none
			--checkpoints_dir {DIR_EXP}
			--label_feat
			--use_encoded_image
		""",
		'0404_style_crop_ctc': f"""
			--name 0404_pix2pixHD512_style_ctc_crop
			--checkpoints_dir {DIR_EXP}
			--instance_feat --label_feat
			--use_encoded_image
			--resize_or_crop crop --fineSize 384 --batchSize 4
		""",
		'0405_nostyle_crop_ctc': f"""
			--name 0405_pix2pixHD512_nostyle_ctc_crop
			--checkpoints_dir {DIR_EXP}
			--no_instance
			--resize_or_crop crop --fineSize 384 --batchSize 4
		""",
		'0407_NYU_nostyle_crop_ctc': f"""
			--name 0407_pix2pixHD512_nostyle_crop_NYU
			--checkpoints_dir {DIR_EXP}
			--no_instance
			--label_nc 41
			--resize_or_crop crop --fineSize 384 --batchSize 4
		""",
		'noBus': """
			--name 512p_style_noBus
			--checkpoints_dir /cvlabdata2/home/lis/exp/0302_pix2pixHD_noBus 
			--dataroot /cvlabdata2/cvlab/dataset_cityscapes_downsampled/for_pix2pixHD_training_noBus
			--use_encoded_image
			--instance_feat --label_feat
		""",
		'noBusNoStyle': """
			--name 512p_style_noBus 
			--checkpoints_dir /cvlabdata2/home/lis/exp/0302_pix2pixHD_noStylenoBus 
			--dataroot /cvlabdata2/cvlab/dataset_cityscapes_downsampled/for_pix2pixHD_training_noBus
		""",
	}

	NET_CACHE = dict()

	loader_args = dict(
		shuffle = False,
		batch_size = 4,
		num_workers = 2,
		drop_last = False,
	)

	@classmethod
	def load_pix2pixHD_default(cls, variant='0405_nostyle_crop_ctc'):
		if variant in cls.NET_CACHE:
			return cls.NET_CACHE[variant]

		import sys
		sys.path.append(str(Path(__file__).parent / 'pix2pixHD'))  # to make the "util" module loadable
		from .pix2pixHD.options.test_options import TestOptions
		from .pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel  # autoreload
		from .pix2pixHD.models.models import create_model

		# _feat in name to add style vectors
		opt = TestOptions().parse(save=False, override_args=cls.pix2pixHD_VARIANTS[variant].split())
		
		# --ngf 32
		opt.nThreads = 1  # test code only supports nThreads = 1
		opt.batchSize = 1  # test code only supports batchSize = 1
		opt.serial_batches = True  # no shuffle
		opt.no_flip = True  # no flip

		pix2pixHD_module = create_model(opt)

		cls.NET_CACHE[variant] = pix2pixHD_module
		return pix2pixHD_module


	def __init__(self, variant='0405_nostyle_crop_ctc'):
		self.mod_pix2pix = self.load_pix2pixHD_default(variant)
		self.table_trainId_to_fullId_cuda = torch.from_numpy(CityscapesLabelInfo.table_trainId_to_label).byte().cuda()


	@staticmethod
	def tr_untorch_img(gen_image, **_):
		return dict(gen_image = gen_image.transpose([1, 2, 0]))

	def tr_generator(self, pred_labels_trainIds, **_):
		# labels = self.table_trainId_to_fullId_cuda[pred_labels_trainIds.reshape(-1)].reshape(pred_labels_trainIds.shape)

		labels = self.table_trainId_to_fullId_cuda[pred_labels_trainIds.reshape(-1).long()].reshape(pred_labels_trainIds.shape)

		labels_onehot = torch_onehot(
			labels,
			num_channels = self.mod_pix2pix.opt.label_nc, 
			dtype = torch.float32,
		)

		# labels = tr_trainId_to_fullId.forward(None, pred_labels_trainIds)
		inst = None
		img = None
		gen_out = self.mod_pix2pix.inference(labels_onehot, inst, img)

		# log.debug(f'gen out shape {gen_out.shape} from labels {labels.shape}')

		desired_shape = labels_onehot.shape[2:]
		if gen_out.shape[2:] != desired_shape:
			gen_out = gen_out[:, :, :desired_shape[0], :desired_shape[1]]

		gen_image = (gen_out + 1) * 128
		gen_image = torch.clamp(gen_image, min=0, max=255)
		gen_image = gen_image.type(torch.uint8)

		return dict(
			gen_image_raw = gen_out,
			gen_image = gen_image,
		)

	def tr_generator_np(self, pred_labels_trainIds, **_):
		pred_labels_trainIds = torch.from_numpy(pred_labels_trainIds)
		pred_labels_trainIds = pred_labels_trainIds[None] # add batch dim
		
		out = self.tr_generator(pred_labels_trainIds = pred_labels_trainIds.cuda())

		gen_image = out['gen_image'][0].cpu()
		gen_image = gen_image.numpy().transpose(1, 2, 0)

		return dict(
			gen_image = gen_image,
		)


	def construct_pipeline(self):
		return Pipeline(
			tr_input = TrsChain(
			),
			tr_batch_pre_merge = TrsChain(),
			tr_batch = TrsChain(
				TrCUDA(),
				self.tr_generator,
				TrKeepFields('gen_image'),
			),
			tr_output = TrsChain(
				TrNP(),
				self.tr_untorch_img,
			),
			loader_args = self.loader_args,
		)


class TrPix2pixHD_Generator(TrBase):


	def __init__(self, generator, b_postprocess=False):
		if isinstance(generator, str):
			self.pix2pixHD_generator = Pix2PixHD_Generator.load_pix2pixHD_default(generator)
		else:
			self.pix2pixHD_generator = generator

		self.b_postprocess = b_postprocess

		self.tr_gen_and_show = TrsChain(
			self,
			TrShow(['image', 'gen_image']),
		)

	def __call__(self, labels_source, image=None, instances=None, **_):
		with torch.no_grad():
			labels = torch.from_numpy(labels_source[None, None, :, :])

			if image is not None:
				img = zero_center_img(image, means=IMG_MEAN_HALF, stds=IMG_STD_HALF)
				img = torch.from_numpy(img.transpose(2, 0, 1))[None, :, :, :].cuda()
			else:
				img = None

			if instances is not None:
				inst = torch.from_numpy(instances[None, None, :, :].astype(np.int32)).cuda()
			else:
				inst = None

			# gen_out = pix2pixHD_generator.inference(labels, inst)
			#gen_out_style = self.pix2pixHD_generator.reconstruction(labels, inst, img)
			gen_out_style = self.pix2pixHD_generator.inference(labels, inst, img)

			# gen_out_style = pix2pixHD_generator.reconstruction(labels, inst, img, dense_style=True)

			gen_image = gen_out_style[0]


			if self.b_postprocess:
				gen_image = postprocess_gen_img(gen_image)

			if gen_image.shape[:2] != labels_source.shape:
				gen_image = gen_image[:labels_source.shape[0], :labels_source.shape[1]]

			return dict(
				#	gen_image = gen_out[0],
				gen_image = gen_image,
				#	gen_image_stylevec_dense = gen_out_style[0],
			)

