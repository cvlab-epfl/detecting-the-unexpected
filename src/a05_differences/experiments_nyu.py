

import numpy as np
import torch
from ..pipeline import *
from ..pipeline.transforms import TrsChain, TrCopy, TrNoOp, tr_print
from ..datasets.NYU_depth_v2 import DatasetNYUDv2, NYUD_LabelInfo_Category40
from ..datasets.dataset import TrSaveChannelsAutoDset
from .experiments import Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT, Exp0516_Diff_SwapFgd_ImgVsGen_semGT
from .transforms import tr_swap_labels
from ..a01_sem_seg.experiments import ExpSemSegPSP_Ensemble, ExpSemSegBayes
from ..a01_sem_seg.transforms import tr_class_argmax, SemSegLabelsToColorImg, TrColorimg
from ..a04_reconstruction.experiments import TrPix2pixHD_Generator


class NyuDsetMixin:

	def init_default_datasets(self, b_threaded=False):
		dset_train = DatasetNYUDv2(split='nohuman_train', b_cache=b_threaded)
		dset_val = DatasetNYUDv2(split='nohuman_val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)

		self.frames_to_log = [self.datasets['val'].frames[i].fid for i in [0, 5, 10, 15, 20]]

	def tr_labels_to_trainIds(self, labels, **_):
		region_unlabeled = (labels == 0)

		labels_trainIds = labels - 1
		labels_trainIds[region_unlabeled] = 255

		return dict(
			labels = labels_trainIds,
		)

	def tr_class_argmax_from_trainIds(self, pred_prob, **_):
		""" Shift by 1 to get to normal ids from trainIds """
		return dict(
			pred_labels = np.argmax(pred_prob, axis=0) + 1
		)

	
	# def replace_class_argmax(self, transform):
	# 	for i in range(transform.__len__()):
	# 		if transform[i] == tr_class_argmax:
	# 			transform[i] = self.tr_class_argmax_from_trainIds
	# 			return
	# 	print('PSP-NYU: Failed to find tr_class_argmax in ', transform)

class NyuSemSegMixin(NyuDsetMixin):
	def init_transforms(self):
		super().init_transforms()

		self.tr_prepare_batch_train = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			self.tr_labels_to_trainIds,
			TrKeepFields('image', 'labels'),
		)

		self.tr_colorimg = SemSegLabelsToColorImg(
			colors_by_classid=NYUD_LabelInfo_Category40.colors, 
			fields=[('pred_labels', 'pred_labels_colorimg')],
		)

		self.tr_postprocess_log = TrsChain(
			TrNP(),
			self.tr_class_argmax_from_trainIds,
			self.tr_colorimg,
		)



class ExpSemSegPSP_Ensemble_NYU(NyuSemSegMixin, ExpSemSegPSP_Ensemble):
	cfg = add_experiment(ExpSemSegPSP_Ensemble.cfg,
		name='0123_PSPEns_NYU',
		epoch_limit=45,
		net = dict(
			num_classes = 40,
			batch_train = 3,
		),
		train = dict(
			# NYU images are [425, 560]
			crop_size = [384, 512],
		),
	)

	def construct_default_pipeline(self, role):
		if role != 'test':
			return super().construct_default_pipeline(role)

		return Pipeline(
			tr_input=TrsChain(
			),
			tr_batch_pre_merge=TrsChain(
				TrZeroCenterImgs(),
				tr_torch_images,
				TrKeepFields('image')
			),
			tr_batch=TrsChain(
				TrCUDA(),
				self.tr_ensemble,
				TrKeepFields('pred_prob', 'pred_var_ensemble'),
				TrNP(),
			),
			tr_output=TrsChain(
				self.tr_class_argmax_from_trainIds,
				TrAsType({'pred_labels': np.uint8}),
				self.tr_colorimg,
			),
			loader_args=self.loader_args_for_role(role),
		)



class ExpSemSegBayes_NYU(NyuSemSegMixin, ExpSemSegBayes):
	cfg = add_experiment(ExpSemSegBayes.cfg,
		name='0124_BayesSegNet_NYU',
		epoch_limit=30,
		net = dict(
			num_classes = 40,
			batch_train = 6,
		),
		train = dict(
			# NYU images are [425, 560]
			crop_size = [384, 512],
		),
	)

	def construct_default_pipeline(self, role):	
		if role == 'test':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = self.tr_prepare_batch_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net_with_uncertainty,
					TrKeepFields('pred_prob', 'pred_var_dropout'),
					TrNP(),
				),
				tr_output = TrsChain(
					self.tr_class_argmax_from_trainIds,
					TrAsType({'pred_labels': np.uint8}),
					self.tr_colorimg,
				),
				loader_args = self.loader_args_for_role(role),
			)

		else:
			return super().construct_default_pipeline(role)


def tr_synthetic_swap_labels_NyuGT(pred_labels_trainIds, instances, swap_fraction=0.2, **_):

	labels_swapped = tr_swap_labels(
		labels_source = pred_labels_trainIds, 
		instances = instances, 
		only_objects = False,
		fraction = swap_fraction,
		target_classes = np.arange(1, 41),
		invalid_class = 0,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(pred_labels_trainIds != labels_swapped),
	)


class TrNyuToTrainIds(TrByField):
	def forward(self, field_name, labels_with_0):
		region_unlabeled = (labels_with_0 == 0)

		labels_trainIds = labels_with_0 - 1
		labels_trainIds[region_unlabeled] = 255

		return labels_trainIds

class NyuDiffMixin(NyuDsetMixin):

	cfg_shared = dict(
		gen_name='0530_NYU_semGT__fakeSwap__genNoSty',
		swap_fraction=0.25,
		pix2pix_variant = '0407_NYU_nostyle_crop_ctc',
		net = dict(
			num_classes = 40,
			batch_eval = 8,
			batch_train = 5,

		),
		train = dict(
			class_weights = [1.68470027, 4.8392086 ],
			crop_size = [384, 512],
		),
		epoch_limit = 40,
	)

	def tr_labels_to_trainIds(self, labels_fakeErr_trainIds, **_):
		
		region_unlabeled = (labels_fakeErr_trainIds == 0)

		labels_trainIds = pred_labels_trainIds - 1
		labels_trainIds[region_unlabeled] = 255

		return dict(
			labels_fakeErr_trainIds = labels_trainIds,
		)

	def make_gt_labels(self, semseg_errors, **_):
		return dict(
			semseg_errors_label = semseg_errors.astype(np.int64),
		)

	def init_transforms(self):
		super().init_transforms()
		# need to set here (instead of making a method synthetic_mod) because super() sets this value
		self.synthetic_mod = partial(
			tr_synthetic_swap_labels_NyuGT,
			swap_fraction = self.cfg['swap_fraction'],
		)

		self.tr_input_train = self.make_gt_labels

		pre_merge = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
		)

		to_trainids = TrNyuToTrainIds('labels_fakeErr_trainIds')

		self.pre_merge_test = pre_merge.copy()
		
		if 'labels_fakeErr_trainIds' in self.fields_for_test:
			self.pre_merge_test.append(to_trainids)
		
		self.pre_merge_test += [
			TrKeepFields(*self.fields_for_test),
		]

		self.pre_merge_train = pre_merge.copy()
		if 'labels_fakeErr_trainIds' in self.fields_for_training:
			self.pre_merge_train.append(to_trainids)

		self.pre_merge_train += [
			TrKeepFields(*self.fields_for_training),
		]

	def prepare_labels_pred(self, dsets):
		# GT labels
		tr_copy_gt_labels = TrsChain(
			TrRenameKw(labels_category40 = 'pred_labels_trainIds'),
			TrSaveChannelsAutoDset(['pred_labels_trainIds']),
		)

		for dset in dsets:
			dset.set_channels_enabled('labels_category40')
			Frame.frame_list_apply(tr_copy_gt_labels, dset, ret_frames=False)

	def prepare_synthetic_changes(self, dsets, b_show=False):

		# initialize self.pix2pix earlier

		tr_disap_and_gen = TrsChain(
			self.synthetic_mod,
			TrCopy(labels_fakeErr_trainIds='labels_source'),
			TrPix2pixHD_Generator(self.cfg['pix2pix_variant'], b_postprocess=True),
		)

		tr_gen_and_show = TrsChain(
			tr_disap_and_gen,
			TrColorimg('pred_labels_trainIds', table=NYUD_LabelInfo_Category40.colors),
			TrColorimg('labels_fakeErr_trainIds', table=NYUD_LabelInfo_Category40.colors),
			TrShow(['image', 'gen_image'],
			       ['pred_labels_trainIds_colorimg', 'labels_fakeErr_trainIds_colorimg', 'semseg_errors']),
		)

		tr_gen_and_save = TrsChain(
			tr_disap_and_gen,
			TrByField('semseg_errors', lambda x: x.astype(np.uint8)),
			TrSaveChannelsAutoDset(['gen_image', 'semseg_errors', 'labels_fakeErr_trainIds']),
		)

		for dset in dsets:

			dset.set_channels_enabled('image', 'pred_labels_trainIds', 'instances')
			dset.discover()

			if b_show:
				dset[0].apply(tr_gen_and_show)

			else:
				Frame.frame_list_apply(tr_gen_and_save, dset, ret_frames=False)




class Exp0530_NYU_Swap_ImgVsLabelAndGen_semGT(NyuDiffMixin, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0530_NYU_Swap_ImgVsLabelAndGen_semGT',
		**NyuDiffMixin.cfg_shared,
    )

class Exp0531_NYU_SwapFgd_ImgVsLabel_semGT(NyuDiffMixin, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0531_NYU_Swap_ImgVsLabel_semGT',
		**NyuDiffMixin.cfg_shared,
    )

class Exp0532_NYU_SwapFgd_ImgVsGen_semGT(NyuDiffMixin, Exp0516_Diff_SwapFgd_ImgVsGen_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0532_NYU_Swap_ImgVsGen_semGT',
		**NyuDiffMixin.cfg_shared,
    )
