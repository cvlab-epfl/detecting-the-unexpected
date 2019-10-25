
from ..datasets.dataset import *
from ..datasets.cityscapes import *
from ..datasets.lost_and_found import DatasetLostAndFoundSmall
from ..datasets.road_anomaly import DatasetRoadAnomaly
from ..datasets.NYU_depth_v2 import DatasetNYUDv2, NYUD_LabelInfo_Category40
from .experiments import *
from .experiments_nyu import ExpSemSegPSP_Ensemble_NYU, ExpSemSegBayes_NYU
from .experiments_nyu import Exp0530_NYU_Swap_ImgVsLabelAndGen_semGT, Exp0531_NYU_SwapFgd_ImgVsLabel_semGT, Exp0532_NYU_SwapFgd_ImgVsGen_semGT
from .experiments_rebuttal import Exp0545_SupervisedDiscrepancy_ImgVsLabelsAndGen, Exp0546_SupervisedDiscrepancy_ImgVsLabel, Exp0547_SupervisedDiscrepancy_ImgVsGen
from .metrics import *
from ..a01_sem_seg.experiments import ExpSemSegBayes_BDD, ExpSemSegPSP_Ensemble_BDD
from ..a01_sem_seg.transforms import tr_semseg_detect_freespace_and_obstacles
from ..a05_road_rec_baseline import *
import gc, colorsys, re

ANOMALY_VARIANCES_BY_SEM_CAT = {
	'BaySegBdd': ['dropout'],
	'PSPEnsBdd': ['ensemble'],
	'BaySegNYU': ['dropout'],
	'PSPEnsNYU': ['ensemble'],
}

ANOMALY_DETECTORS_CTC = {
	#'gen_disap_gt': Exp0510_Difference_ImgVsGen_onGT,
	#'label_disap_gt': Exp0515_Diff_Disap_ImgVsLabels_semGT,
	'gen_swap_gt': Exp0516_Diff_SwapFgd_ImgVsGen_semGT,
	#'gen_swapAll_gt': Exp0511_Difference_LabelsVsGen_onGT,
	'label_swap_gt': Exp0517_Diff_SwapFgd_ImgVsLabels_semGT,
	#'gen_disap_PspBdd': Exp0512_Difference_ImgVsGen_onPredBDD,
	#'label_disap_PspBdd': Exp0513_Difference_LabelsVsGen_onPredBDD,
	#'gen_swap_PspBdd': Exp0518_Diff_SwapFgd_ImgVsGen_semPspBdd,
	#'gen_swapAll_PspBdd': Exp0514_Difference_ImgVsGen_Swap_onPredBDD,
	#'label_swap_PspBdd': Exp0519_Diff_SwapFgd_ImgVsLabels_semPspBdd,
	#'lag_disap_gt': Exp0520_Diff_ImgAndLabelsVsGen_semGT,
	'lag_swap_gt': Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT,
	#'lag_disap_PspBdd': Exp0522_Disap_ImgAndLabelsVsGen_semPspBdd,
	#'lag_swap_PspBdd': Exp0523_Disap_ImgAndLabelsVsGen_semPspBdd,
	'rbm': Exp0525_RoadReconstructionRBM_LowLR,
	#'lag_sNd_gt': Exp0527_SwapNDisap_ImgAndLabelsVsGen_semGT,
	'mrcnn_coco_r101': None,
}

ANOMALY_DETECTORS_NYU = {
	'lag_swap_gt': Exp0530_NYU_Swap_ImgVsLabelAndGen_semGT,
	'gen_swap_gt': Exp0532_NYU_SwapFgd_ImgVsGen_semGT,
	'label_swap_gt': Exp0531_NYU_SwapFgd_ImgVsLabel_semGT,
}

ANOMALY_DETECTORS_SUPERVISED = {
	'lag_swap_supervised': Exp0545_SupervisedDiscrepancy_ImgVsLabelsAndGen,
	'gen_swap_supervised': Exp0547_SupervisedDiscrepancy_ImgVsGen,
	'label_swap_supervised': Exp0546_SupervisedDiscrepancy_ImgVsLabel,
}

# Sadly we used the global in the functions here,
# to switch btw domains (CTC, NYU) we will edit this global
ANOMALY_DETECTORS = dict()

def set_anomaly_detectors(detector_dict):
	ANOMALY_DETECTORS.clear()
	ANOMALY_DETECTORS.update(detector_dict)


def get_all_detector_names_for_baseline(baseline_name):
	baselines = [baseline_name]
	all_detectors = baselines + list(ANOMALY_DETECTORS.keys())

	# maxSD_detectors = [f'{type}_maxSD_{sem}' for type in ('gen', 'label', 'lag') for sem in ('gt', 'PspBdd')]
	# all_detectors += maxSD_detectors

	# all_detectors = (list(set(
	# 	baselines + filter_names(r'.*swap.*', all_detectors)
	# 	+ filter_names(r'lag_disap.*', all_detectors)))
	# 	+ ['rbm']
	# )

	# all_detectors += [d + '_island' for d in all_detectors]
	all_detectors.sort()

	return all_detectors


def channels_for_eval(sem_category, anomaly_names):

	base = f'eval_{sem_category}'

	chans = dict(
		pred_labels_trainIds = ChannelResultImage(f'{base}/labels', suffix='_predTrainIds', img_ext='.png'),
		pred_labels_colorimg = ChannelResultImage(f'{base}/labels', suffix='_predColorImg', img_ext='.png'),
		gen_image = ChannelResultImage(f'{base}/gen_image', suffix='_gen_nostyle', img_ext='.jpg'),
	)

	chans.update({
		f'anomaly_{name}': ChannelLoaderHDF5(
			os.path.join('{dset.dir_out}', base, 'anomalyScore/{dset.split}/anomalyScore_' + name + '.hdf5'),
			'{fid}',
			compression=6,
		)
		for name in anomaly_names
	})

	return chans


def tr_LAF_preprocess_labels(labels_source = None, **_):
	# only activate if we are loading labels
	if labels_source is None:
		return {}

	return dict(
		anomaly_gt = labels_source > 1,
		valid = np.count_nonzero(labels_source) > 0,
	)


def make_dset_LAF(sem_cat, only_interesting=True, split='test'):
	anos = ANOMALY_VARIANCES_BY_SEM_CAT[sem_cat] + list(ANOMALY_DETECTORS.keys())
	chans = channels_for_eval(sem_cat, anos)

	dset = DatasetLostAndFoundSmall(split=split, only_interesting=only_interesting, b_cache=False)

	#dset.channels['labels_gt'] = dset.channels['labels_source']

	dset.discover()

	dset.add_channels(**chans)

	dset.tr_post_load_pre_cache = TrsChain(tr_LAF_preprocess_labels)

	return dset


def tr_Ano_preprocess_labels(labels_source = None, **_):
	# only activate if we are loading labels
	if labels_source is None:
		return {}

	return dict(
		anomaly_gt = labels_source > 1,
	)

def make_dset_Ano(sem_cat):
	anos = ANOMALY_VARIANCES_BY_SEM_CAT[sem_cat] + list(ANOMALY_DETECTORS.keys())
	chans = channels_for_eval(sem_cat, anos)

	dset = DatasetRoadAnomaly()
	dset.discover()

	dset.add_channels(**chans)

	dset.tr_post_load_pre_cache = TrsChain(tr_Ano_preprocess_labels)

	return dset

def tr_NYU_preprocess_labels(labels_category40 = None, **_):
	# only activate if we are loading labels
	if labels_category40 is None:
		return {}

	return dict(
		anomaly_gt = labels_category40 == NYUD_LabelInfo_Category40.name2id['person'],
	)

def tr_NYU_variants(labels_category40, **_):
	res = dict(
		anomaly_gt_unlabeled = labels_category40 == 0,
		anomaly_gt_human = labels_category40 == NYUD_LabelInfo_Category40.name2id['person'],
	)

	res.update(
		roi_labeled = ~res['anomaly_gt_unlabeled'],
		anomaly_gt_human_and_unlabeled = res['anomaly_gt_unlabeled'] | res['anomaly_gt_human'],
	)

	return res

def make_dset_NYU(sem_cat):
	anos = ANOMALY_VARIANCES_BY_SEM_CAT[sem_cat] + list(ANOMALY_DETECTORS_NYU.keys())
	chans = channels_for_eval(sem_cat, anos)

	dset = DatasetNYUDv2(split='nohuman_test', b_cache=False)

	#dset.channels['labels_gt'] = dset.channels['labels_source']

	dset.discover()

	dset.add_channels(**chans)

	dset.tr_post_load_pre_cache = TrsChain(tr_NYU_preprocess_labels)

	return dset


def get_sem_net(sem_cat):
	fields_out = ['pred_labels_trainIds', 'pred_labels_colorimg']

	if sem_cat == 'BaySegBdd':
		exp = ExpSemSegBayes_BDD()
		exp.init_net('eval')

		tr_renames = TrRenameKw(pred_labels = 'pred_labels_trainIds', pred_var_dropout='anomaly_dropout')
		fields_out += ['anomaly_dropout']

	elif sem_cat == 'PSPEnsBdd':
		exp = ExpSemSegPSP_Ensemble_BDD()
		exp.load_subexps()
		exp.init_net('master_eval')

		tr_renames = TrRenameKw(pred_labels='pred_labels_trainIds', pred_var_ensemble='anomaly_ensemble')
		fields_out += ['anomaly_ensemble']

	elif sem_cat == 'BaySegNYU':
		exp = ExpSemSegBayes_NYU()
		exp.init_net('eval')

		tr_renames = TrRenameKw(pred_labels = 'pred_labels_trainIds', pred_var_dropout='anomaly_dropout')
		fields_out += ['anomaly_dropout']

	elif sem_cat == 'PSPEnsNYU':
		exp = ExpSemSegPSP_Ensemble_NYU()
		exp.load_subexps()
		exp.init_net('master_eval')

		tr_renames = TrRenameKw(pred_labels='pred_labels_trainIds', pred_var_ensemble='anomaly_ensemble')
		fields_out += ['anomaly_ensemble']

	else:
		raise NotImplementedError(sem_cat)

	exp.tr_out_for_eval = TrsChain(
		tr_renames,
		TrSaveChannelsAutoDset(fields_out),
	)

	exp.tr_out_for_eval_show = TrsChain(
		tr_renames,
		TrShow(['image'] + fields_out),
	)

	return exp


def sem_run(exp, dset, b_show=False):
	pipe = exp.construct_default_pipeline('test')

	trout = exp.tr_out_for_eval_show if b_show else exp.tr_out_for_eval

	pipe.tr_output += trout

	dset.set_channels_enabled('image')

	if b_show:
		pipe.execute(dset, b_one_batch=True)
	else:
		pipe.execute(dset, b_accumulate=False)
		dset.flush_hdf5_files()


def gen_run(gen_variant, dset, b_show=True):

	tr_gen = TrsChain(
		TrSemSegLabelTranslation(
			fields=dict(pred_labels_trainIds='labels_source'),
			table=CityscapesLabelInfo.table_trainId_to_label
		),
		TrPix2pixHD_Generator(gen_variant, b_postprocess=True),
	)

	tr_gen_and_show = TrsChain(
		tr_gen,
		TrColorimg('pred_labels_trainIds'),
		TrShow(['image', 'gen_image', 'pred_labels_trainIds_colorimg'])
	)

	tr_gen_and_save = TrsChain(
		tr_gen,
		TrSaveChannelsAutoDset(['gen_image']),
	)

	dset.set_channels_enabled('pred_labels_trainIds', 'image')
	dset.discover()

	if b_show:
		dset[1].apply(tr_gen_and_show)
	else:
		Frame.frame_list_apply(tr_gen_and_save, dset, ret_frames=False)


def get_anomaly_net(variant):

	exp_class = ANOMALY_DETECTORS[variant]

	exp = exp_class()
	exp.init_net('eval')

	score_field = f'anomaly_{variant}'

	tr_renames = TrRenameKw(anomaly_p=score_field)

	exp.tr_out_for_eval = TrsChain(
		tr_renames,
		TrSaveChannelsAutoDset([score_field]),
	)

	exp.tr_out_for_eval_show = TrsChain(
		tr_renames,
		TrColorimg('pred_labels_trainIds'),
		TrShow(['image', 'gen_image', 'pred_labels_trainIds_colorimg', score_field]),
	)

	return exp

def anomaly_run(exp, dset, b_show=True):
	pipe = exp.construct_default_pipeline('test')

	trout = exp.tr_out_for_eval_show if b_show else exp.tr_out_for_eval

	pipe.tr_output += trout

	dset.set_channels_enabled('image', 'pred_labels_trainIds', 'gen_image')
	dset.discover()


	if b_show:
		pipe.execute(dset, b_one_batch=True)
	else:
		pipe.execute(dset, b_accumulate=False)
		dset.flush_hdf5_files()

def anomaly_run_all_detectors(dset, detectors=None):
	detectors = detectors or ANOMALY_DETECTORS.keys()

	for name in detectors:
		print(name)

		score_channel = dset.channels[f'anomaly_{name}']
		score_file = Path(score_channel.resolve_file_path(dset, dset.frames[0]))

		if score_file.is_file():
			print(f'Out file for {name} already exists - skipping')
		else:
			exp = get_anomaly_net(name)
			anomaly_run(exp, dset, False)
			del exp
			gc.collect()

	dset.flush_hdf5_files()
	gc.collect()

def tr_combine_swap(**fields):
	out = dict()

	for type in ('gen', 'label', 'lag'):
		for sem in ('gt', 'PspBdd'):
			disap = fields.get(f'anomaly_{type}_disap_{sem}')
			swap = fields.get(f'anomaly_{type}_swap_{sem}')

			if disap is not None and swap is not None:
				out[f'anomaly_{type}_maxSD_{sem}'] = np.maximum(disap, swap)

	return out

	# return {
	# 	f'anomaly_{type}_maxSD_{sem}': np.maximum(fields[f'anomaly_{type}_disap_{sem}'], fields[f'anomaly_{type}_swap_{sem}'])
	# 	for type in ('gen', 'label', 'lag') for sem in ('gt', 'PspBdd')
	# }


def filter_names(filter, detector_names):
	regexp = re.compile(filter)
	return [name for name in detector_names if regexp.match(name)]


def tr_island_detector(pred_labels_trainIds, **fields):
	res = tr_semseg_detect_freespace_and_obstacles(pred_labels_trainIds=pred_labels_trainIds)

	obstacle_mask = res['obstacle_mask_semseg']

	out = dict()

	for k, v in fields.items():
		if k.startswith('anomaly_'):
			# island detector outputs 1.0 anomaly score
			out[k+'_island'] = np.maximum(v, obstacle_mask)

	return out


def laf_rois(labels_source, dset, **_):
	"""
	0 - background
	1 - road
	2 and higher - anomaly, set to 2
	"""
	labels_normalized = np.minimum(labels_source, 2)

	road_mask_with_holes = labels_normalized >= 1

	#print(labels_normalized, road_mask_with_holes)

	_, contour_list, _ = cv2.findContours(
		road_mask_with_holes.astype(np.uint8), 
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE,
	)

	road_mask_full = cv2.drawContours(
		np.zeros(labels_source.shape, dtype=np.uint8),
		contour_list, -1, 1, -1,
	).astype(np.bool)

	return dict(
		labels_normalized=labels_normalized,
		roi_default = dset.roi,
		roi_road=road_mask_full,
	)


def replace_names(src_text, dest_text, names):
	return [n.replace(src_text, dest_text) for n in names]


DET_PLOT_NAMES_AND_FMT = {
	'lag_swap_gt': ('Ours', dict(color='b', linestyle='-')),
	'gen_swap_gt': ('Ours (Resynthesis only)', dict(color=(0.8, 0.3, 0.), linestyle='-.')),
	#'gen_swapAll_gt': Exp0511_Difference_LabelsVsGen_onGT,
	'label_swap_gt': ('Ours (Labels only)', dict(color='g', linestyle='--')),
	#'gen_disap_PspBdd': Exp0512_Difference_ImgVsGen_onPredBDD,
	#'label_disap_PspBdd': Exp0513_Difference_LabelsVsGen_onPredBDD,
	#'gen_swap_PspBdd': Exp0518_Diff_SwapFgd_ImgVsGen_semPspBdd,
	#'gen_swapAll_PspBdd': Exp0514_Difference_ImgVsGen_Swap_onPredBDD,
	#'label_swap_PspBdd': Exp0519_Diff_SwapFgd_ImgVsLabels_semPspBdd,
	#'lag_disap_gt': Exp0520_Diff_ImgAndLabelsVsGen_semGT,

	#'lag_disap_PspBdd': Exp0522_Disap_ImgAndLabelsVsGen_semPspBdd,
	#'lag_swap_PspBdd': Exp0523_Disap_ImgAndLabelsVsGen_semPspBdd,
	'rbm': ('RBM', dict(color='r', linestyle='--')),
	'dropout': ('Uncertainty (Bayesian)', dict(color='k', linestyle=':')),
	'ensemble': ('Uncertainty (Ensemble)', dict(color='k', linestyle=':')),
}


def det_specs_for_baseline(baseline, onroad=False, suffix = None, default_roi=None, common_channels=set(), gt_field='anomaly_gt'):
	names = get_all_detector_names_for_baseline(baseline)
	names.sort()

	specs = []

	for det_name in names:
		plot_name, plot_fmt = DET_PLOT_NAMES_AND_FMT.get(det_name, (None, None))

		chans = set([f'anomaly_{det_name}']).union(common_channels)
		name = det_name if suffix is None else f'{det_name}_{suffix}'

		det_base = DetSpec(
			name, 
			pred_field = f'anomaly_{det_name}',
			gt_field = gt_field,
			plot_fmt = plot_fmt, 
			plot_name = plot_name, 
			roi = default_roi,
			channels = chans,
		)
		specs.append(det_base)

		if onroad:
			specs.append(DetSpec(
				name + '_onroad',
				gt_field=det_base.gt_field,
				pred_field=det_base.pred_field,
				channels=det_base.channels,
				roi='roi_road',
				plot_fmt=plot_fmt, plot_name=plot_name,
			))

	return specs


def run_rocs(sem_cat, dset, detectors_specs, transform=TrsChain(), reload_dset=True):

	base = Path(dset.dir_out) / f'eval_{sem_cat}' / dset.split

	# rocs_again(dset, outdir, detector_specs, transform_custom=TrsChain(), reload_dset=True):
	rocs_calculate(dset, base / 'rocs', detectors_specs, transform_custom=transform, reload_dset=reload_dset)


LINESTYLES = {
	'gen': ':',
	'label': '--',
	'lag': '-.',
	'mrcnn': '-',
}

HUE_BY_SYNTH = {
	'disap': 0.0,
	'swap': 0.6,
	'swapAll': 0.8,
	'maxSD': 0.4,
	'sNd': 0.5,
	'coco': 0.2,
}

LIGHTNESS_BY_DSET = {
	'gt': 0.4,
	'PspBdd': 0.7,
	'r101': 0.6,
}

SATURATION = 1.0


def name_to_plot_format(name):

	parts = name.split('_')

	if parts.__len__() < 3:
		return 'k-'

	arch, synth, dset = parts[:3]

	return dict(
		linestyle=LINESTYLES[arch],
		color = colorsys.hls_to_rgb(HUE_BY_SYNTH.get(synth, 0.35), LIGHTNESS_BY_DSET.get(dset, 0.4), SATURATION),
	)


#	'-', '--', '-.', ':'
#	gen, label, lag

#	disap, swap, maxSD

#	gt, PspBdd

	# 'gen_disap_gt': Exp0510_Difference_ImgVsGen_onGT,
	# 'label_disap_gt': Exp0515_Diff_Disap_ImgVsLabels_semGT,
	# 'gen_swap_gt': Exp0516_Diff_SwapFgd_ImgVsGen_semGT,
	# 'gen_swapAll_gt': Exp0511_Difference_LabelsVsGen_onGT,
	# 'label_swap_gt': Exp0517_Diff_SwapFgd_ImgVsLabels_semGT,
	# 'gen_disap_PspBdd': Exp0512_Difference_ImgVsGen_onPredBDD,
	# 'label_disap_PspBdd': Exp0513_Difference_LabelsVsGen_onPredBDD,
	# 'gen_swap_PspBdd': Exp0518_Diff_SwapFgd_ImgVsGen_semPspBdd,
	# 'gen_swapAll_PspBdd': Exp0514_Difference_ImgVsGen_Swap_onPredBDD,
	# 'label_swap_PspBdd': Exp0519_Diff_SwapFgd_ImgVsLabels_semPspBdd,
	# 'lag_disap_gt': Exp0520_Diff_ImgAndLabelsVsGen_semGT,
	# 'lag_swap_gt': Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT,
	# 'lag_disap_PspBdd': Exp0522_Disap_ImgAndLabelsVsGen_semPspBdd,
	# 'lag_swap_PspBdd': Exp0523_Disap_ImgAndLabelsVsGen_semPspBdd,


def show_rocs(sem_cat, dset, detector_specs, save_as):
	base = Path(dset.dir_out) / f'eval_{sem_cat}' / dset.split / 'rocs'

	draw_rocs(detector_specs, base, base / save_as)