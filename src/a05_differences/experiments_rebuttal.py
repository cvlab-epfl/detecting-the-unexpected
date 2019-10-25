
import numpy as np
import torch
from functools import partial
from ..pipeline import *
from ..pipeline.transforms import TrsChain, TrCopy, TrNoOp, tr_print
from ..datasets.dataset import TrSaveChannelsAutoDset
from ..datasets.generic_sem_seg import TrSemSegLabelTranslation
from ..datasets.cityscapes import CityscapesLabelInfo
from .transforms import tr_synthetic_swapAll_labels
from .experiments import Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT, Exp0516_Diff_SwapFgd_ImgVsGen_semGT
from ..datasets.lost_and_found import DatasetLostAndFoundSmall
from ..a01_sem_seg.transforms import tr_class_argmax, SemSegLabelsToColorImg, TrColorimg
from ..a01_sem_seg.experiments import ExpSemSegPSP_Ensemble_BDD
from ..a04_reconstruction.experiments import TrPix2pixHD_Generator

from copy import copy

# Rebuttal - R3
# Swap not only objects but backgrounds too

class Rebuttal_SwapFgd_Mixin:

	cfg_shared = dict(
		gen_name='0541_semGT__fakeSwapAll__genNoSty',
		swap_fraction=0.25,
		train = dict(
			class_weights = [1.56987986, 7.190096 ],
		)
	)

	def init_transforms(self):
		super().init_transforms()
		self.synthetic_mod = partial(
			tr_synthetic_swapAll_labels,
			allow_road = False,
			swap_fraction = self.cfg['swap_fraction'],
		)

class Exp0540_Rebuttal_SwapFgd_ImgAndLabelsVsGen_semGT(Rebuttal_SwapFgd_Mixin, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0540_Rebuttal_SwapFgd_ImgVsLabelAndGen_semGT',
		**Rebuttal_SwapFgd_Mixin.cfg_shared,
    )

class Exp0541_Rebuttal_SwapFgd_ImgVsLabel_semGT(Rebuttal_SwapFgd_Mixin, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0541_Rebuttal_SwapFgd_ImgVsLabel_semGT',
		**Rebuttal_SwapFgd_Mixin.cfg_shared,
    )

class Exp0542_Rebuttal_SwapFgd_ImgVsGen_semGT(Rebuttal_SwapFgd_Mixin, Exp0516_Diff_SwapFgd_ImgVsGen_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0542_Rebuttal_SwapFgd_ImgVsGen_semGT',
		**Rebuttal_SwapFgd_Mixin.cfg_shared,
    )

class DatasetLostAndFoundSmall_SplitForSupervised(DatasetLostAndFoundSmall):
	def __init__(self, *args, split='supervised_train', only_interesting=False, **kwargs):
		self.supervised_split = split

		super().__init__(*args, split='train', only_interesting=only_interesting, **kwargs)

	def discover(self):
		super().discover()

		if self.supervised_split == 'supervised_train':
			self.frames = self.frames[:-103]
		elif self.supervised_split == 'supervised_val':
			self.frames = self.frames[-103:]
		else:
			raise NotImplementedError(f'Supervised LAF - wrong split: {self.supervised_split}')
		
		self.after_discovering_frames()
		print(f'Selecting {self.frames.__len__()} for split {self.supervised_split}')


class Rebuttal_SupervisedDiscrepancy_Mixin:

	cfg_shared = dict(
		gen_name='0545_supervised_discrepancy',
		net = dict(
			num_classes = 19,
		),
		train = dict(
			class_weights = [1.42336916, 47.91712529],
		),
	)

	@staticmethod
	def tr_laf_gt_anomaly(labels_source = None, **_):
		# only activate if we are loading labels
		if labels_source is None:
			return {}

		anomaly_gt = labels_source > 1

		return dict(
			anomaly_gt = anomaly_gt,
			semseg_errors = anomaly_gt,
		)

	def init_transforms(self):
		super().init_transforms()
		self.synthetic_mod = self.tr_laf_gt_anomaly

	def setup_dset(self, dset):
		super().setup_dset(dset)
		dset.channels['labels_fakeErr_trainIds'] = dset.channels['pred_labels_trainIds']

	def init_default_datasets(self, b_threaded=False):
		dset_train = DatasetLostAndFoundSmall_SplitForSupervised(split='supervised_train', b_cache=b_threaded)
		dset_val = DatasetLostAndFoundSmall_SplitForSupervised(split='supervised_val', b_cache=b_threaded)

		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)

		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)

		self.frames_to_log = [self.datasets['val'].frames[i].fid for i in [0, 25, 50, 75, 100]]

	def prepare_labels_pred(self, dsets, b_show=False):
		exp_sem = ExpSemSegPSP_Ensemble_BDD()
		exp_sem.load_subexps()
		exp_sem.init_net('master_eval')

		pipe_sem = exp_sem.construct_default_pipeline('test')

		if b_show:
			pipe_sem.tr_output += [
				TrRenameKw(dict(pred_labels='pred_labels_trainIds')),
				TrColorimg('pred_labels_trainIds'),
				TrShow(['image', 'pred_labels_trainIds_colorimg']),
			]

		else:
			pipe_sem.tr_output += [
				TrRenameKw(dict(pred_labels='pred_labels_trainIds')),
				TrSaveChannelsAutoDset(['pred_labels_trainIds']),
			]

		for dset in dsets:
			dset.set_channels_enabled(['image'])

			if b_show:
				bout, outfrs = pipe_sem.execute(dset, b_one_batch=True)
				del bout, outfrs

			else:
				pipe_sem.execute(dset, b_accumulate=False)

	def prepare_synthetic_changes(self, dsets, b_show=False):

		# initialize self.pix2pix earlier

		tr_disap_and_gen = TrsChain(
			self.synthetic_mod,
			TrSemSegLabelTranslation(
				# use predicted labels for gen
				fields=dict(pred_labels_trainIds='labels_source'),
			    table=CityscapesLabelInfo.table_trainId_to_label
			),
			TrPix2pixHD_Generator(self.cfg['pix2pix_variant'], b_postprocess=True),
		)

		tr_gen_and_show = TrsChain(
			tr_disap_and_gen,
			TrColorimg('pred_labels_trainIds'),
			# TrColorimg('labels_fakeErr_trainIds'),
			TrShow(['image', 'gen_image'],
			       ['pred_labels_trainIds_colorimg', 'semseg_errors']),
		)

		tr_gen_and_save = TrsChain(
			tr_disap_and_gen,
			TrByField('semseg_errors', lambda x: x.astype(np.uint8)),
			TrSaveChannelsAutoDset(['gen_image', 'semseg_errors']),
		)

		print(tr_gen_and_save)

		for dset in dsets:

			dset.set_channels_enabled('image', 'labels_source', 'pred_labels_trainIds', 'instances')
			# dset.discover()

			if b_show:
				dset[0].apply(tr_gen_and_show)

			else:
				Frame.frame_list_apply(tr_gen_and_save, dset, ret_frames=False)

class Exp0545_SupervisedDiscrepancy_ImgVsLabelsAndGen(Rebuttal_SupervisedDiscrepancy_Mixin, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT):
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name='0545_SupervisedDiscrepancy_ImgVsLabelsAndGen',
		**Rebuttal_SupervisedDiscrepancy_Mixin.cfg_shared,
    )

class Exp0546_SupervisedDiscrepancy_ImgVsLabel(Rebuttal_SupervisedDiscrepancy_Mixin, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT):
	cfg = add_experiment(Exp0517_Diff_SwapFgd_ImgVsLabels_semGT.cfg,
		name='0546_SupervisedDiscrepancy_ImgVsLabel',
		**Rebuttal_SupervisedDiscrepancy_Mixin.cfg_shared,
    )

class Exp0547_SupervisedDiscrepancy_ImgVsGen(Rebuttal_SupervisedDiscrepancy_Mixin, Exp0516_Diff_SwapFgd_ImgVsGen_semGT):
	cfg = add_experiment(Exp0516_Diff_SwapFgd_ImgVsGen_semGT.cfg,
		name='0547_SupervisedDiscrepancy_ImgVsGen',
		**Rebuttal_SupervisedDiscrepancy_Mixin.cfg_shared,
    )