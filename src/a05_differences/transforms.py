import numpy as np
from math import floor
from scipy import stats
from random import choice
from ..paths import DIR_DATA
from ..pipeline.transforms import *
from ..pipeline.transforms_imgproc import *
from ..datasets.generic_sem_seg import TrSemSegLabelTranslation
from ..datasets.cityscapes import CityscapesLabelInfo
from ..a01_sem_seg.transforms import SemSegLabelsToColorImg
from ..a04_reconstruction.transforms import tr_instances_from_semantics

def tr_label_to_validEval(labels, dset, **_):
	v = dset.label_info.valid_for_eval_trainId[labels.reshape(-1)].reshape(labels.shape)
	return dict(
		labels_validEval = v,
	)


def tr_get_errors(labels, pred_labels, labels_validEval, **_):
	errs = (pred_labels != labels) & labels_validEval
	return dict(
		semseg_errors = errs,
	)


def tr_errors_to_gt(semseg_errors, labels_validEval, **_):
	errs = semseg_errors.astype(np.int64)
	errs[np.logical_not(labels_validEval)] = 255
	return dict(
		semseg_errors_label = errs,
	)

def tr_errors_to_gt_float(semseg_errors, labels_validEval, **_):
	errs = semseg_errors.astype(np.int64)
	errs[np.logical_not(labels_validEval)] = 255
	return dict(
		semseg_errors_label = errs,
	)


try:
	CTC_ROI = imageio.imread(DIR_DATA/'cityscapes/roi.png').astype(np.bool)
	CTC_ROI_neg = ~CTC_ROI
except:
	print('Cityscapes ROI file is not present', DIR_DATA/'cityscapes/roi.png')

DISAPPEAR_TRAINIDS = [CityscapesLabelInfo.name2trainId[n] for n in ['person', 'rider', 'car', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']]

FORBIDDEN_DISAPPEAR_ALTERNATIVE_IDS = [0, 1, 2, 3]
FORBIDDEN_DISAPPEAR_ALTERNATIVE_TRAINIDS = [CityscapesLabelInfo.trainId2label[n].id for n in FORBIDDEN_DISAPPEAR_ALTERNATIVE_IDS]


MORPH_KERNEL = np.ones((11, 11), np.uint8)

def tr_LAF_exclude_anomalies_from_difference_training(semseg_errors_label, labels_source, **_):
	anomaly_mask = labels_source > 1

	#print(np.count_nonzero(anomaly_mask), np.count_nonzero(semseg_errors_label == 255))


	# expand with a margin
	anomaly_mask = cv2.dilate(anomaly_mask.astype(np.uint8), MORPH_KERNEL).astype(np.bool)

	label_to_override = semseg_errors_label.copy()
	label_to_override[anomaly_mask] = 255

	#print(np.count_nonzero(anomaly_mask), np.count_nonzero(label_to_override == 255))

	return dict(
		semseg_errors_label = label_to_override,
	)

def tr_exclude_ROI_from_difference_training(semseg_errors_label, roi, **_):
	semseg_errors_label[~roi] = 255
	return dict(
		semseg_errors_label = semseg_errors_label,
	)

tr_show_errors = TrsChain(
	SemSegLabelsToColorImg([('labels', 'labels_colorimg'), ('pred_labels', 'pred_labels_colorimg')]),
	tr_label_to_validEval,
	tr_get_errors,
	TrShow(
		['labels_colorimg', 'pred_labels_colorimg'],
		['image', 'labels_validEval'],
		'semseg_errors',
	),
)


def tr_disappear_inst(labels_source, instances, inst_ids=None, clear_instance_map=False, only_objects=True, swap_fraction=0.5, **_):
	if inst_ids is None:
		inst_uniq = np.unique(instances)

		if only_objects:
			inst_uniq_objects = inst_uniq[inst_uniq >= 24000]
		else:
			inst_uniq_objects = inst_uniq[inst_uniq >= 1]

		if inst_uniq_objects.__len__() == 0:
			return dict(
				labels_fakeErr=labels_source.copy(),
			)

		inst_ids = np.random.choice(inst_uniq_objects, int(inst_uniq_objects.__len__() *swap_fraction), replace=False)
		#print(inst_uniq, 'remove', inst_ids)


	disappear_mask = np.any([instances == inst_id for inst_id in inst_ids], axis=0)

	obj_classes = np.unique(labels_source[disappear_mask])

	forbidden_classes = DISAPPEAR_TRAINIDS
	forbidden_class_mask = np.any([labels_source == cl for cl in forbidden_classes], axis=0)

	mask_dont_use_label = forbidden_class_mask | disappear_mask
	mask_use_label = np.logical_not(mask_dont_use_label)

	# 	show(forbidden_class_mask)

	# 	dis_mask_u8 = disappear_mask.astype(np.uint8)
	nearest_dst, nearest_labels = cv2.distanceTransformWithLabels(
		mask_dont_use_label.astype(np.uint8),
		distanceType=cv2.DIST_L2,
		maskSize=5,
		labelType=cv2.DIST_LABEL_PIXEL,
	)

	background_indices = nearest_labels[mask_use_label]
	background_labels = labels_source[mask_use_label]

	label_translation = np.zeros(labels_source.shape, dtype=np.uint8).reshape(-1)
	label_translation[background_indices] = background_labels

	label_reconstr = labels_source.copy()
	label_reconstr[disappear_mask] = label_translation[nearest_labels.reshape(labels_source.shape)[disappear_mask]]

	# 	label_reconstr = label_translation[nearest_labels].reshape(labels_source.shape)

	# 	show(label_reconstr)

	result = dict(
		# 		dist_lab = nearest_labels,
		labels_fakeErr=label_reconstr,
	)

	if clear_instance_map:
		inst_cleared = instances.copy()
		inst_cleared[disappear_mask] = 0
		result['instances'] = inst_cleared

	return result


def tr_synthetic_disappear_objects(pred_labels_trainIds, instances = None, disap_fraction=0.5, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	if instances is None:
		instances_encode_class = False
		instances = tr_instances_from_semantics(labels, min_size=750, allowed_classes=DISAPPEAR_TRAINIDS)['instances']
	else:
		instances_encode_class = True # gt instances

	labels_disap = tr_disappear_inst(
		pred_labels_trainIds,
		instances,
		only_objects = instances_encode_class,
		fraction = disap_fraction,
	)['labels_fakeErr']


	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_disap,
		semseg_errors=(pred_labels_trainIds != labels_disap) & CTC_ROI,
	)


def tr_synthetic_disappear_objects_onPred(pred_labels_trainIds, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	instance_map = tr_instances_from_semantics(labels, min_size=750, allowed_classes=DISAPPEAR_TRAINIDS)['instances']

	labels_disap = tr_disappear_inst(pred_labels_trainIds, instance_map, only_objects=False)['labels_fakeErr']

	return dict(
		instances=instance_map,
		labels_fakeErr_trainIds=labels_disap,
		semseg_errors=(labels != labels_disap) & CTC_ROI,
	)




def tr_swap_labels(labels_source, instances, inst_ids=None, only_objects=False, fraction=0.2, target_classes=np.arange(19), invalid_class=255, **_):
	if inst_ids is None:
		inst_uniq = np.unique(instances)

		if only_objects:
			inst_uniq_objects = inst_uniq[inst_uniq >= 24000]
		else:
			inst_uniq_objects = inst_uniq[inst_uniq >= 1]

		if inst_uniq_objects.__len__() == 0:
			return dict(
				labels_fakeErr=labels_source.copy(),
			)

		inst_ids = np.random.choice(inst_uniq_objects, floor(inst_uniq_objects.__len__() * fraction), replace=False)

	# print(inst_uniq, 'remove', inst_ids)

	labels = labels_source.copy()

	for inst_id in inst_ids:
		inst_mask = instances == inst_id

		inst_view = labels[inst_mask]

		obj_class = stats.mode(inst_view, axis=None).mode[0]

		if obj_class != invalid_class:
			tc = list(target_classes)
			try:
				tc.remove(obj_class)
			except ValueError:
				print(f'Instance class {obj_class} not in set of classes {target_classes}')
			new_class = choice(tc)

			labels[inst_mask] = new_class

	result = dict(
		# 		dist_lab = nearest_labels,
		labels_fakeErr=labels
	)

	return result


def tr_synthetic_swapAll_labels_onPred(pred_labels_trainIds, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	instance_map = tr_instances_from_semantics(labels, min_size=750)['instances']

	labels_swapped = tr_swap_labels(pred_labels_trainIds, instance_map, only_objects=False)['labels_fakeErr']

	return dict(
		instances=instance_map,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)


def tr_synthetic_swapFgd_labels(pred_labels_trainIds, instances=None, swap_fraction=0.2, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	if instances is None:
		instances_encode_class = False
		instances = tr_instances_from_semantics(labels, min_size=750, allowed_classes=DISAPPEAR_TRAINIDS)['instances']
	else:
		instances_encode_class = True # gt instances

	labels_swapped = tr_swap_labels(
		pred_labels_trainIds,
		instances,
		only_objects=instances_encode_class,
		fraction = swap_fraction,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)

def tr_synthetic_swapFgd_labels_onGT(pred_labels_trainIds, instances, only_objects=True, swap_fraction=0.2, **_):
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	labels_swapped = tr_swap_labels(
		pred_labels_trainIds, 
		instances, 
		only_objects=only_objects,
		fraction = swap_fraction,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)


def tr_synthetic_swapAll_labels(pred_labels_trainIds, instances, swap_fraction=0.2, allow_road=False, min_size=500, **_):
	"""
	Swap background connected-components in addition to object instances.
	"""
	
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255
	
	obj_mask = instances >= 24000
	
	labels_for_cc = pred_labels_trainIds.copy()
	labels_for_cc[CTC_ROI_neg] = 255 # exclude outside of ROI
	labels_for_cc[obj_mask] = 255 # exclude objects, they have their own instances
	
	if not allow_road:
		labels_for_cc[labels_for_cc == 0] = 255
	
	stuff_instances = tr_instances_from_semantics(
		labels_for_cc, 
		min_size=min_size,
		forbidden_classes=[255],
	)['instances']
	
# 	show([stuff_instances, stuff_instances == 0])
	
	new_instances = stuff_instances
	new_instances[obj_mask] = instances[obj_mask]

	# Pass this instance map to the standard swapper
	
	labels = pred_labels_trainIds.copy()
	labels[CTC_ROI_neg] = 255

	labels_swapped = tr_swap_labels(
		pred_labels_trainIds, 
		new_instances, 
		only_objects = False,
		fraction = swap_fraction,
	)['labels_fakeErr']

	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels_swapped,
		semseg_errors=(labels != labels_swapped) & CTC_ROI,
	)


def tr_synthetic_swapNdisap(pred_labels_trainIds, instances, **_):
	labels_orig = pred_labels_trainIds.copy()
	labels_orig[CTC_ROI_neg] = 255
	
	P_SWAP = 0.3
	P_DISAP = 0.3

	inst_uniq = np.unique(instances)
	inst_uniq_objects = inst_uniq[inst_uniq >= 24000]

	num_obj = inst_uniq_objects.__len__()
	
	if num_obj == 0:
		labels = labels_orig
		
	else:

		num_disap = int(np.floor(P_DISAP * num_obj))
		num_swap = int(np.floor(P_SWAP * num_obj))

		inst_uniq_objects = np.random.permutation(inst_uniq_objects)

		inst_to_disap = inst_uniq_objects[:num_disap]
		inst_to_swap = inst_uniq_objects[num_disap:num_disap+num_swap]


		labels_orig = pred_labels_trainIds.copy()
		labels_orig[CTC_ROI_neg] = 255

		labels = tr_disappear_inst(
			labels_orig,
			instances,
			inst_ids=inst_to_disap
		)['labels_fakeErr']


		labels = tr_swap_labels(
			labels,
			instances,
			inst_ids=inst_to_swap,
		)['labels_fakeErr']
	
	return dict(
		instances=instances,
		labels_fakeErr_trainIds=labels,
		semseg_errors=(labels != labels_orig) & CTC_ROI,
	)

	
	
	
	
	
	
	
tr_show_dis = TrsChain(
	tr_disappear_inst,
	SemSegLabelsToColorImg(
		{'labels_fakeErr': 'labels_fakeErr_colorimg', 'labels_source': 'labels_source_colorimg'},
		CityscapesLabelInfo.colors_by_id,
	),
	TrShow(['labels_source_colorimg', 'labels_fakeErr_colorimg']),
	TrSemSegLabelTranslation(
		fields={'labels_fakeErr': 'pred_labels'},
		table=CityscapesLabelInfo.table_label_to_trainId,
	),
	tr_label_to_validEval,
	tr_get_errors,
	TrShow('semseg_errors'),
)

BUS_UNK_TRAINID = [CityscapesLabelInfo.name2label[n].trainId for n in ['bus', 'truck', 'train']]
BUS_UNK_ID = [CityscapesLabelInfo.name2label[n].id for n in ['bus', 'truck', 'train']]

CAR_TRAINID = CityscapesLabelInfo.name2label['car'].trainId

OUT_OF_ROI_ID = [CityscapesLabelInfo.name2label[n].id for n in ['ego vehicle', 'rectification border', 'out of roi']]

def tr_bus_errors(pred_labels, labels, labels_validEval, **_):
	bus_mask = np.any([labels == c for c in BUS_UNK_TRAINID], axis=0)

	pred_car_mask = pred_labels == CAR_TRAINID

	bus_pred_as_nocar_mask = bus_mask & (~pred_car_mask)

	bus_as_car = labels.copy()
	bus_as_car[bus_mask] = CAR_TRAINID

	semseg_errors_busiscar = (bus_as_car != pred_labels) & labels_validEval

	return dict(
		bus_mask = bus_mask,
		bus_pred_as_nocar_mask = bus_pred_as_nocar_mask,
		semseg_errors_busiscar = semseg_errors_busiscar,
	)

def tr_bus_error_simple(labels, **_):
	return dict(
		bus_mask = np.any([labels == c for c in BUS_UNK_TRAINID], axis=0),
	)


def tr_unlabeled_error_simple(labels_source, labels_validEval, **_):

	mask_out_of_roi = np.any([labels_source == c for c in OUT_OF_ROI_ID], axis=0)

	unlabeled_mask = ~( mask_out_of_roi | labels_validEval )

	return dict(
		unlabeled_mask = unlabeled_mask,
		labels_validEval = mask_out_of_roi,
	)
