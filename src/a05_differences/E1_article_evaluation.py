
import numpy as np
import torch
from pathlib import Path
import logging, gc
log = logging.getLogger('exp.eval')

from ..paths import DIR_EXP, DIR_DATA
from ..pipeline.config import add_experiment
from ..pipeline.frame import Frame
from ..pipeline.pipeline import Pipeline
from ..pipeline.evaluations import Evaluation, TrChannelLoad, TrChannelSave
from ..pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix, tr_print, TrRenameKw, TrRemoveFields
from ..pipeline.transforms_imgproc import TrZeroCenterImgs, TrShow
from ..pipeline.transforms_pytorch import tr_torch_images, TrCUDA, TrNP, torch_onehot
from ..pipeline.bind import bind
from ..datasets.dataset import imwrite, ImageBackgroundService, ChannelLoaderImage, ChannelLoaderHDF5, ChannelLoaderHDF5_NotShared, hdf5_read, hdf5_write
from ..datasets.generic_sem_seg import TrSemSegLabelTranslation
from ..datasets.cityscapes import CityscapesLabelInfo

from .metrics import binary_confusion_matrix, cmats_to_rocinfo

from ..a01_sem_seg.transforms import SemSegLabelsToColorImg, TrColorimg
from ..a01_sem_seg.experiments import ExpSemSegPSP_Ensemble_BDD, ExpSemSegBayes_BDD, ExpSemSegPSP
from ..a04_reconstruction.experiments import Pix2PixHD_Generator
from .experiments import Exp0516_Diff_SwapFgd_ImgVsGen_semGT, Exp0517_Diff_SwapFgd_ImgVsLabels_semGT, Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT
from ..a05_road_rec_baseline.experiments import Exp0525_RoadReconstructionRBM_LowLR

from .E0_article_evaluation import get_anomaly_net
from .E1_plot_utils import TrImgGrid, TrBlend, tr_draw_anomaly_contour, draw_rocinfos

# class EvaluationSemSeg(Evaluation):
# 	def __init__(self, exp):
# 		super().__init__(exp)

# 		self.tr_colorimg = TrColorimg('pred_labels')
# 		self.tr_colorimg_gt = TrColorimg('labels')

# 		self.tr_make_demo = TrImgGrid(
# 			['image', None, 'labels_colorimg', 'pred_labels_colorimg'],
# 			num_cols = 2,
# 			out_name = 'demo_semseg',
# 		)

# 	def construct_persistence(self):
# 		self.persistence_base_dir = '{channel.ctx.workdir}/{dset.name}_{dset.split}/'

# 		self.chan_pred_trainId = ChannelLoaderImage(self.persistence_base_dir+'semantic/{fid_no_slash}_predTrainId.png'),
# 		self.chan_pred_trainId.ctx = self
# 		self.chan_pred_colorimg = ChannelLoaderImage(self.persistence_base_dir+'semantic/{fid_no_slash}_colorimg.png')
# 		self.chan_pred_colorimg.ctx = self
# 		self.chan_demo = ChannelLoaderImage(self.persistence_base_dir+'semantic/demo/{fid_no_slash}_demo.webp')
# 		self.chan_demo.ctx = self

# 	def construct_transforms(self, dset):
# 		self.tr_output = TrsChain(
# 			self.tr_colorimg,
# 			self.tr_colorimg_gt,
# 			self.tr_make_demo,
# 			TrChannelSave(self.chan_pred_trainId, 'pred_labels'),
# 			TrChannelSave(self.chan_pred_colorimg, 'pred_labels_colorimg'),
# 			TrChannelSave(self.chan_demo, 'demo'),
# 		)


# class EvaluationSemSegWithUncertainty(EvaluationSemSeg):
# 	def __init__(self, exp):
# 		super().__init__(exp)

# 		self.tr_make_demo = TrImgGrid(
# 			['image', self.exp.uncertainty_field_name, 'labels_colorimg', 'pred_labels_colorimg'],
# 			num_cols = 2,
# 			out_name = 'demo',
# 		)

# 	def construct_persistence(self):
# 		super().construct_persistence()

# 		self.chan_demo = ChannelLoaderImage(self.persistence_base_dir+'anomaly/demo/{fid_no_slash}_anomaly_demo.webp')
# 		self.chan_demo.ctx = self

# 		self.chan_uncertainty = ChannelLoaderHDF5(
# 			self.persistence_base_dir+'anomaly/anomaly_score.hdf5',
# 			'{fid}',
# 			compression=6,
# 		)
# 		self.chan_uncertainty.ctx = self

# 	def construct_transforms(self, dset):
# 		super().construct_transforms(dset)

# 		self.tr_output.append(
# 			TrChannelSave(self.chan_uncertainty, self.exp.uncertainty_field_name),
# 		)

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


from collections import namedtuple

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

	PlotStyle = namedtuple('PlotStyle', ['display_name', 'display_fmt'])
	DEFAULT_PLOT_STYLES = {
		'discrepancy_label_and_gen': PlotStyle('Ours', dict(color='b', linestyle='-')),
		'discrepancy_gen_only': PlotStyle('Ours (Resynthesis only)', dict(color=(0.8, 0.3, 0.), linestyle='-.')),
		'discrepancy_label_only': PlotStyle('Ours (Labels only)', dict(color='g', linestyle='--')),
		'rbm': PlotStyle('RBM', dict(color='r', linestyle='--')),
		'dropout': PlotStyle('Uncertainty (Bayesian)', dict(color='k', linestyle=':')),
		'ensemble': PlotStyle('Uncertainty (Ensemble)', dict(color='k', linestyle=':')),
	}

	def __init__(self, sem_seg_variant):

		self.workdir =  DIR_EXP / '0500_Eval'

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

		out_dir_default = Path('{channel.ctx.workdir}') / '{dset.name}_{dset.split}' / f'sem_{self.sem_seg_variant}'
		out_dir = Path(out_dir_override or out_dir_default)

		self.persistence_base_dir = out_dir

		# outputs of the pipelines steps: 
		self.storage = dict(
			# semantic segmentation
			pred_labels_trainIds = ChannelLoaderImage(out_dir / 'sem_labels/{fid}_predTrainIds.png'),
			pred_labels_colorimg = ChannelLoaderImage(out_dir / 'sem_labels/{fid}_predColorImg.png'),

			# synthesized image
			gen_image = ChannelLoaderImage(out_dir / 'gen_image/{fid}_gen_image.webp'),

			# overview
			demo_with_labels = ChannelLoaderImage(out_dir / 'demo/{fid_no_slash}_pipeline.webp'),
			demo_with_baselines = ChannelLoaderImage(out_dir / 'demo/{fid_no_slash}_scores.webp'),
		)

		# anomaly/discrepancy scores
		self.storage.update({
			f'anomaly_{name}': ChannelLoaderHDF5(
				out_dir / f'anomaly_score/anomaly_{name}.hdf5',
				'{fid}',
				write_as_type = np.float16,
				read_as_type = np.float32,
			)
			for name in self.anomaly_detector_variants
		})

		for c in self.storage.values(): c.ctx = self


	def init_semseg(self):
		if self.sem_seg_variant == 'BaySegBdd':
			exp = ExpSemSegBayes_BDD()
			exp.init_net('eval')

			uncertainty_field_name = 'anomaly_dropout'

			tr_renames = TrRenameKw(
				pred_labels = 'pred_labels_trainIds', 
				pred_var_dropout = uncertainty_field_name,
			)

		elif self.sem_seg_variant == 'PSPEnsBdd':
			exp = ExpSemSegPSP_Ensemble_BDD()
			exp.load_subexps()
			exp.init_net('master_eval')

			uncertainty_field_name = 'anomaly_ensemble'

			tr_renames = TrRenameKw(
				pred_labels = 'pred_labels_trainIds', 
				pred_var_ensemble = uncertainty_field_name,
			)
			
		else:
			raise NotImplementedError(self.sem_seg_variant)
		
		tr_write_results = TrsChain(
			TrChannelSave(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
			TrChannelSave(self.storage['pred_labels_colorimg'], 'pred_labels_colorimg'),
			TrChannelSave(self.storage[uncertainty_field_name], uncertainty_field_name)
		)

		exp.tr_out_for_eval = TrsChain(
			tr_renames,
			tr_write_results
		)

		tr_show_results = TrsChain(
			# TrImgGrid(
			# 	['image', uncertainty_field_name, 'labels_colorimg', 'pred_labels_colorimg'],
			# 	num_cols = 2,
			# 	out_name = 'demo',
			# ),
			TrShow(['image', uncertainty_field_name], ['labels_colorimg', 'pred_labels_colorimg']),
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
			TrChannelLoad('image', 'image'),
			TrChannelLoad('labels_source', 'labels_source'),
			dset.tr_get_anomaly_gt,
			TrChannelLoad(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
			TrChannelLoad(self.storage['gen_image'], 'gen_image'),
			TrChannelLoad(self.storage['anomaly_discrepancy_label_and_gen'], 'anomaly_discrepancy_label_and_gen'),
			TrChannelLoad(self.storage[f'anomaly_{self.uncertainty_variant}'], 'anomaly_uncertainty'),
			TrChannelLoad(self.storage['anomaly_rbm'], 'anomaly_rbm'),
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
				# demo images are a cpu-bound operation, so we want to run with multiprocessing
				# the dset object contains HDF5 handles, which can not be sent between processes
				# so we remove them before results are sent back
				#TrRemoveFields('dset'), 
				# even better, send nothing back
				TrKeepFields(), 
			]
			Frame.frame_list_apply(tr_make_demo_imgs, dset.frames, n_proc=8, batch=4)

	def run_roc_curves_for_variant(self, dset, anomaly_variants=None, on_road=False):
		
		if anomaly_variants is None:
			anomaly_variants = self.anomaly_detector_variants

		roi_field = 'roi' if not on_road else 'roi_onroad'

		# Pipeline
		# (1) Load groundtruth
		tr_roc = TrsChain(
			TrChannelLoad('labels_source', 'labels_source'),
			dset.tr_get_anomaly_gt,
			dset.tr_get_roi,
		)

		# (2) calculate confusion matrices for each variant
		# by doing all at once we only load the groundtruth once
		for variant in anomaly_variants:
			score_field = f'anomaly_{variant}'
			cmat_field = f'cmats_{variant}'
			tr_roc += [
				TrChannelLoad(self.storage[score_field], score_field),
				bind(binary_confusion_matrix, prob=score_field, gt_label='anomaly_gt', roi=roi_field).outs(cmat=cmat_field),
			]

		# (3) return only the confusion matrices (by clearing all the rest)
		tr_roc += [
			TrKeepFieldsByPrefix('cmats_'),
		]
		
		# Execute pipeline
		dset.discover()
		dset.flush_hdf5_files()
		dset.set_channels_enabled()
		results = Frame.frame_list_apply(tr_roc, dset.frames, ret_frames=True, n_proc=8, batch=8)
		
		# Extract info from conf mats, such as fp / tp
		rocinfo_by_variant = {}
		for variant in anomaly_variants:
			out_variant_name = variant if not on_road else f'{variant}_onroad'
			cmat_field = f'cmats_{variant}'
			cmat_sum = np.sum([fr[cmat_field] for fr in results], axis=0)

			# rocinfo_by_variant[variant] = cmat_sum

			rocinfo_by_variant[out_variant_name] = cmats_to_rocinfo(
				name = out_variant_name,
				cmats = cmat_sum,
			)

		# Store
		self.roc_save_all(dset=dset, rocinfo_by_variant=rocinfo_by_variant)

		return rocinfo_by_variant

	def roc_path(self, variant_name, dset):
		path_tmpl = str(self.persistence_base_dir / 'anomaly_roc' / f'{variant_name}_roc.hdf5')
		path = Path(path_tmpl.format(dset=dset, channel = Frame(ctx=self)))		
		return path

	def roc_save(self, variant_name, dset, rocinfo):
		p = self.roc_path(variant_name, dset)
		log.info(f'Saving {p}')
		p.parent.mkdir(parents=True, exist_ok=True)
		hdf5_write(p, rocinfo)

	def roc_save_all(self, dset, rocinfo_by_variant):
		for variant_name, rocinfo in rocinfo_by_variant.items():
			self.roc_save(variant_name=variant_name, dset=dset, rocinfo=rocinfo)

	def roc_load(self, variant_name, dset):
		return hdf5_read(self.roc_path(variant_name, dset))

	def plot_path(self, dset):
		path_tmpl = str(self.persistence_base_dir / 'ROC_{dset.name}_{dset.split}')
		path = Path(path_tmpl.format(dset=dset, channel = Frame(ctx=self)))		
		return path

	def roc_plot_variants(self, dset, rocinfo_by_variant=None, variant_names=None, title=None):
		if rocinfo_by_variant is None:
			if variant_names is None:
				variant_names = self.anomaly_detector_variants

			rocinfo_by_variant = {
				name: self.roc_load(variant_name=name, dset=dset)
				for name in variant_names
			}

		# load default styles and names
		infos = []
		for info in rocinfo_by_variant.values():
			default_style = self.DEFAULT_PLOT_STYLES.get(info['name'], {})
			# load defaults but make it possible to overwrite them
			info_with_style = default_style._asdict()
			info_with_style.update(info)
			infos.append(info_with_style)
 
		fig = draw_rocinfos(infos, save=self.plot_path(dset))

		



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

	name = '0550_DetectingTheUnexpected_FullPipeline'

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

	def set_batch_size(self, batch_size):
		""" call before construct_pipeline """
		self.loader_args['batch_size'] = batch_size

	def init_semseg(self):
		self.exp_sem_seg = ExpPSP_EnsebleReuse()
		self.exp_sem_seg.init_net('eval')

	def tr_apply_semseg(self, image, **_):
		
		logits = self.exp_sem_seg.net_mod(image)

		_, class_id = torch.max(logits, 1)

		return dict(
			pred_labels_trainIds = class_id.byte(),
		)

	def init_gan(self):
		self.pix2pix = Pix2PixHD_Generator()

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

	def init_apex_optimization(self):
		"""
		Use the apex multi precision library to run the networks in half precision
		which is faster and takes less memory.
		"""
		try:
			import apex

			net_mods = [
				self.exp_sem_seg.net_mod,
				self.exp_discrepancy.net_mod,
			]

			if hasattr(self, 'pix2pix'):
				net_mods.append(self.pix2pix.mod_pix2pix)

			apex.amp.initialize(net_mods, opt_level='O1')
			log.info(f'APEX applied to {net_mods.__len__()} networks')

		except ImportError:
			log.warning('APEX can not be imported')

	def construct_persistence(self, dir_data=DIR_DATA):
		# self.persistence_base_dir = '{channel.ctx.exp.workdir}/pred/{dset.name}_{dset.split}/'
		self.workdir = dir_data / self.name
		out_dir = Path('{channel.ctx.workdir}/pred/{dset.name}_{dset.split}')
		self.persistence_base_dir = out_dir

		self.storage = dict(
			pred_labels_trainIds = ChannelLoaderImage(out_dir / 'sem_labels/{fid_no_slash}_predTrainIds.png'),
			pred_labels_colorimg = ChannelLoaderImage(out_dir / 'sem_labels/{fid_no_slash}_predColorImg.png'),
			gen_image = ChannelLoaderImage(out_dir / 'gen_image/{fid_no_slash}_gen_image.webp'),
			discrepancy = ChannelLoaderHDF5_NotShared(
				file_path_tmpl = out_dir / 'discrepancy/{fid_no_slash}_discrepancy.hdf5', 
				var_name_tmpl = 'discrepancy',
				write_as_type = np.float16, read_as_type = np.float32,
			),
			demo = ChannelLoaderImage(out_dir / 'demo/{fid_no_slash}_demo.webp'),
		)
		for c in self.storage.values():
			c.ctx = self

		self.tr_save_results = TrsChain(
			TrChannelSave(self.storage['pred_labels_trainIds'], 'pred_labels_trainIds'),
			TrChannelSave(self.storage['pred_labels_colorimg'], 'pred_labels_trainIds_colorimg'),
			TrChannelSave(self.storage['gen_image'], 'gen_image'),
			TrChannelSave(self.storage['discrepancy'], 'pred_anomaly_prob'),
			TrChannelSave(self.storage['demo'], 'demo'),
		)

	def construct_pipeline(self):
		self.tr_pre_batch = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
			TrKeepFields('image')
		)

		self.tr_colorimg = TrColorimg('pred_labels_trainIds')
		self.tr_colorimg.set_override(19, [0, 0, 0])

		return Pipeline(
			tr_input = TrsChain(),
			tr_batch_pre_merge = self.tr_pre_batch,
			tr_batch = TrsChain(
				TrCUDA(),
				self.tr_apply_semseg,
				self.pix2pix.tr_generator,
				self.tr_apply_discrepancy,
				TrKeepFields('gen_image', 'pred_anomaly_prob', 'pred_labels_trainIds'),
			),
			tr_output = TrsChain(
				TrNP(),
				self.tr_colorimg,
				lambda gen_image, **_: dict(gen_image = gen_image.transpose([1, 2, 0])),
				self.tr_make_demo,
			),
			loader_args =self.loader_args,
		)

	def run_on_dset(self, dset, b_show=False, **exec_kwargs):
		"""
		@param b_show: runs only one batch and displays the result in the notebook
		"""
		
		pipe = self.construct_pipeline()
		
		if b_show:
			pipe.tr_output.append(TrShow('demo'))
			pipe.execute(dset, b_one_batch=True, **exec_kwargs)

		else:
			pipe.tr_output.append(self.tr_save_results)
			pipe.execute(dset, b_accumulate=False, **exec_kwargs)
	

class DiscrepancyJointPipeline_LabelsOnly(DiscrepancyJointPipeline):

	name = '0551_DetectingTheUnexpected_LabelsOnly'

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
				TrKeepFields('pred_anomaly_prob', 'pred_labels_trainIds'),
			),
			tr_output = TrsChain(
				TrNP(),
				self.tr_colorimg,
				self.tr_make_demo,
			),
			loader_args =self.loader_args,
		)

