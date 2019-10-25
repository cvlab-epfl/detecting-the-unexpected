
import numpy as np
import cv2, torch
from ..pipeline.transforms import TrsChain
from ..pipeline.transforms_imgproc import *
from ..datasets.generic_sem_seg import apply_label_translation_table
from ..datasets.cityscapes import CityscapesLabelInfo, DatasetCityscapesSmall


class TrApplyRoiToPredictedLabels(TrByField):
	def __init__(self,
			example_label_map,
			fields='labels_source',
			labels_to_copy=['unlabeled', 'ego vehicle', 'rectification border', 'out of roi'],
			label_info=CityscapesLabelInfo,
		):
		super().__init__(fields)

		self.label_ids_to_copy = [label_info.name2id[label_name] for label_name in labels_to_copy]

		self.mask_to_copy = np.any(
			np.stack([example_label_map == labid for labid in self.label_ids_to_copy]),
			axis=0,
		)
		self.value_to_copy = example_label_map[self.mask_to_copy]

	def forward(self, field_name, label_map):
		label_map_replaced = label_map.copy()
		label_map_replaced[self.mask_to_copy] = self.value_to_copy
		return label_map_replaced

	@staticmethod
	def default_cityscapes_roi_tr():
		dset_sample = DatasetCityscapesSmall(split='val')
		dset_sample.discover()
		return TrApplyRoiToPredictedLabels(dset_sample[0].labels_source)


def tr_laf_convert_labels(labels_semantic, **_):
	"""
	convert labels to original ids and upscale to 1024p
	"""
	lab_origid = apply_label_translation_table(CityscapesLabelInfo.table_trainId_to_label, labels_semantic)

	# 	lab_1024p[EGO_VEHICLE_MASK] = LABEL_EGO
	# lab_1024p[MASK_ROI] = LABEL_OUT_OF_ROI

	return dict(
		labels_source=lab_origid,
		instances=np.zeros([512, 1024], dtype=np.int32),
	)


def tr_instances_from_semantics(labels_source, min_size=None, allowed_classes=None, forbidden_classes=None, **_):
	# max_label = np.max(labels_source + 1)
	label_set = set(np.unique(labels_source))

	if allowed_classes is not None:
		label_set = label_set.intersection(set(allowed_classes))

	if forbidden_classes is not None:
		label_set = label_set.difference(set(forbidden_classes))

	label_set = list(label_set)
	label_set.sort()

	instance_idx = 1
	out_inst_map = np.zeros(labels_source.shape, dtype=np.int32)

	for l in label_set:
		this_label_map = labels_source == l
		num_pixels = np.count_nonzero(this_label_map)

		# print(l, num_pixels)

		if num_pixels > 0:
			num_cc, cc_map = cv2.connectedComponents(this_label_map.astype(np.uint8))

			for cc_idx in range(1, num_cc):
				inst_mask = cc_map == cc_idx

				if (min_size is None) or np.count_nonzero(inst_mask) > min_size:
					out_inst_map[inst_mask] = instance_idx
					instance_idx += 1

	return dict(
		instances=out_inst_map,
	)

def tr_instances_from_objectdetection(labels_source, mrcnn_masks, roi, high_ids_for_obj_instances=False, **_):
	if high_ids_for_obj_instances:
		instance_idx = 24001
	else:
		instance_idx = 1

	out_inst_map = np.zeros(labels_source.shape, dtype=np.int32)

	# instances from mrcnn_masks
	for mask in mrcnn_masks.transpose((2, 0, 1)):
		if np.count_nonzero(roi[mask]) > 0.9 * np.count_nonzero(mask):
			# at least 90% in roi
			out_inst_map[mask] = instance_idx
			instance_idx += 1

	return dict(
		instances=out_inst_map,
	)


def tr_instances_from_semantics_and_objectdetection(
		labels_source, roi, instances=None,
		high_ids_for_obj_instances=False, **_
	):
	"""
		high_ids_for_obj_instances - in cityscapes, object instances are above 24000
		while buildings roads etc are lower
	"""

	out_inst_map = instances if instances is not None else np.zeros(labels_source.shape, dtype=np.int32)

	if high_ids_for_obj_instances:
		instance_idx = 1
	else:
		instance_idx = np.max(out_inst_map)

	# instances from connected components
	area_to_be_filled = (out_inst_map == 0)

	max_label = np.max(labels_source + 1)

	if high_ids_for_obj_instances:
		instance_idx = 1

	for l in range(max_label):
		this_label_map = (labels_source == l) & area_to_be_filled
		num_pixels = np.count_nonzero(this_label_map)

		# print(l, num_pixels)

		if num_pixels > 0:
			num_cc, cc_map = cv2.connectedComponents(this_label_map.astype(np.uint8))

			for cc_idx in range(1, num_cc):
				out_inst_map[cc_map == cc_idx] = instance_idx
				instance_idx += 1

	out_inst_map[~roi] = 0

	return dict(
		instances=out_inst_map,
	)


def postprocess_gen_img(gen_image):
	gen_image = gen_image.cpu().numpy().transpose([1, 2, 0])
	gen_image = (gen_image + 1) * 128
	gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)

	return gen_image


