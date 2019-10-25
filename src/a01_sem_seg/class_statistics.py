
import numpy as np
from ..pipeline import *
from functools import partial
from ..datasets.cityscapes import CityscapesLabelInfo
from matplotlib import pyplot as plt

@FrameTransform
def class_stats_single(labels_source, num_classes = -1, **_):
	# p1 = plus one, because of the -1 class
	class_area = np.bincount(labels_source.reshape(-1), minlength=num_classes).astype(np.float64) / np.prod(labels_source.shape)
	class_presence = class_area > 0

	return dict(class_area=class_area, class_presence=class_presence)

@FrameTransform
def ctc_instances(instances, num_classes=0, **_):
	inst_unique = np.unique(instances)

	inst_class = inst_unique[inst_unique > 1000].copy()

	inst_class //= 1000

	#inst_class[above_1000] = inst_class[above_1000] // 1000

	inst_count = np.bincount(inst_class, minlength=num_classes)

	return dict(
		class_instances=inst_count,
	)

@FrameTransform
def apollo_instances(instance_contours, num_classes=0, **_):
	inst_class = [inst.label for inst in instance_contours]

	inst_count = np.bincount(inst_class, minlength=num_classes)

	return dict(
		class_instances=inst_count,
	)

# 	i_with_c = np.stack([inst_unique, inst_class])
# 	print(i_with_c.T)	return dict()

def class_stats_dset_compile(dset, instance_transform=ctc_instances):
	num_classes = np.max([l.id for l in dset.label_info.labels ])+1
	print('num cl', num_classes)

	if instance_transform is None:
		instance_transform = FrameTransform(lambda **_: dict(class_instances=np.zeros(num_classes)))

	tr = NTrChain(
		class_stats_single.partial(num_classes=num_classes),
		instance_transform.partial(num_classes=num_classes)
	)

	def f(frame):
		frame = tr(frame)

		return Frame(
			class_area = frame.class_area,
			class_presence = frame.class_presence,
			class_instances = frame.class_instances,
		)

	frames_out = Frame.parallel_process(
		f,
		dset,
		n_threads=8, batch=16,
	)

	stats = Frame(
		class_presence_by_frame = np.array([fr.class_presence for fr in frames_out]),
		class_area_by_frame = np.array([fr.class_area for fr in frames_out]),
		class_instances_by_frame =  np.array([fr.class_instances for fr in frames_out]),
	)

	stats.class_presence_total = np.sum(stats.class_presence_by_frame, axis=0)
	stats.class_area_total = np.sum(stats.class_area_by_frame, axis=0)
	stats.class_instances_total = np.sum(stats.class_instances_by_frame, axis=0)

	dset.class_statistics = stats
	return stats


def class_stats_draw_presence_plots(dset, save=None, stats=None):
	""" from dev_unknown """

	stats = stats or dset.class_statistics

	cat_names = []
	cat_is_in_eval = []
	cat_presence = []
	cat_area = []
	cat_insts = []

	for cl in dset.label_info.labels:
		if cl.id != 255:
			cat_names.append(cl.name)
			cat_is_in_eval.append(not cl.ignoreInEval)
			cat_presence.append(stats.class_presence_total[cl.id])
			cat_area.append(stats.class_area_total[cl.id])
			cat_insts.append(stats.class_instances_total[cl.id])

	cat_is_in_eval = np.array(cat_is_in_eval, dtype=np.bool)

	colors = np.array([
		[200, 50, 0],
		[0, 200, 50],
	]) / 255

	cat_colors = colors[cat_is_in_eval.astype(np.uint8)]

	plt.figure(figsize=(8, 8))
	plt.title('Class number of frames')
	plt.barh(cat_names, cat_presence, color=cat_colors, )
	plt.tight_layout()

	if save:
		os.makedirs(os.path.dirname(save), exist_ok=True)
		name, ext = os.path.splitext(save)
		plt.savefig(name + '_per_frame' + ext)

	plt.figure(figsize=(8, 8))
	plt.title('Class area [logscale]')
	plt.barh(cat_names, cat_area, color=cat_colors, log=True)
	plt.tight_layout()

	if save:
		plt.savefig(name + '_area_log' + ext)

	plt.figure(figsize=(8, 8))
	plt.title('Class instances')
	plt.barh(cat_names, cat_insts, color=cat_colors)
	plt.tight_layout()

	if save:
		plt.savefig(name + '_instances' + ext)


def find_frames_containing_classes(stats, classids):
	presence = stats.class_presence_by_frame[:, classids]
	if presence.shape.__len__() > 1:
		presence = np.any(presence, axis=1)

	idx_present = np.where(presence)[0]
	idx_not_present = np.where(~presence)[0]

	return idx_present, idx_not_present


def ctcEval_config(out_file, b_instances=True, b_anomaly=True):
	from cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling as ctc_eval_lib
	from copy import copy
	eval_args = copy(ctc_eval_lib.args)
	eval_args.exportFile = out_file
	eval_args.evalPixelAccuracy = True
	eval_args.evalInstLevelScore = b_instances
	if b_anomaly:
		eval_args.avgClassSize['ANOMALY'] = float(np.mean([eval_args.avgClassSize[c] for c in ('bus', 'train', 'truck')]))
	print(str(eval_args).replace(', ', '\n	'))

	return eval_args, ctc_eval_lib.evaluateFrameLists


def ctcEvalRun(out_file, frame_collection, b_instances=True, b_anomaly=True):
	cfg, func = ctcEval_config(out_file, b_instances=b_instances, b_anomaly=b_anomaly)
	return func(frame_collection, cfg)















def conf_matrix_bincount(gt_value, pred_value, gt_num_classes, pred_num_classes):
	return np.bincount(
		gt_value * pred_num_classes + pred_value,
		minlength=pred_num_classes * gt_num_classes,
	).reshape(
		(gt_num_classes, pred_num_classes),
	).astype(np.uint64)


class TrConfMatrix(TrBase):
	def __init__(self, gt_field, gt_num_classes, pred_field, pred_num_classes, out_field='confusion_matrix'):
		self.gt_field = gt_field
		self.gt_num_classes = gt_num_classes
		self.pred_field = pred_field
		self.pred_num_classes = pred_num_classes
		self.out_field = out_field

	# self.labels_for_api = np.arange(max(gt_num_classes, pred_num_classes));

	def __call__(self, **fields):
		gt_value = fields[self.gt_field].reshape(-1)
		pred_value = fields[self.pred_field].reshape(-1)

		return {
			# 			self.out_field: confusion_matrix(gt_value, pred_value),
			self.out_field: conf_matrix_bincount(gt_value, pred_value, self.gt_num_classes, self.pred_num_classes),
		}


def class_stats_draw_presence_plots(area_by_trainId, label_info, save=None, stats=None, relative=True, title=''):
	if relative:
		area_by_trainId = area_by_trainId / np.sum(area_by_trainId)

	order = np.argsort(area_by_trainId)

	cat_names = []
	cat_colors = []
	cat_areas = []

	for trainid in order:
		cl = label_info.trainId2label[trainid]
		cat_names.append(cl.name)

		cat_areas.append(area_by_trainId[trainid])
		cat_colors.append(label_info.colors_by_trainId[trainid])

	cat_colors = np.array(cat_colors) * (1. / 255.)

	plt.figure(figsize=(8, 8))
	plt.title(f'{title} predicted as:')
	plt.barh(cat_names, cat_areas, color=cat_colors, )
	plt.tight_layout()

	if save:
		out_dir = os.path.dirname(save)
		if out_dir: os.makedirs(out_dir, exist_ok=True)
		name, ext = os.path.splitext(save)
		plt.savefig(name + '_area' + ext)


def what_classified_as_what(
		frames,
		gt_field='labels',
		gt_num_classes=3,
		pred_field='pred_labels_trainIds',
		pred_num_classes=CityscapesLabelInfo.num_trainIds,
		label_info=CityscapesLabelInfo,
		tr_preprocess=TrsChain(),
		to_plot=[('Anomaly', 2, 'anomaly_classified_as.pdf')],
):
	tr_all_conf_matrix = TrsChain(
		tr_preprocess,
		TrConfMatrix(gt_field, gt_num_classes, pred_field, pred_num_classes),
		TrKeepFields('confusion_matrix'),
	)

	conf_matrix_per_frame = np.stack([
		fr.confusion_matrix for fr in
	    Frame.frame_list_apply(tr_all_conf_matrix, frames, ret_frames=True, n_threads=8)
	])

	conf_matrix = np.sum(conf_matrix_per_frame, axis=0)

	for name, gt_label, save_name in to_plot:
		class_stats_draw_presence_plots(
			conf_matrix[gt_label], label_info=label_info,
			title=name, save=save_name,
		)

	return dict(
		conf_matrix=conf_matrix,
		conf_matrix_per_frame=conf_matrix_per_frame,
	)