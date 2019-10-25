
import numpy as np
import torch
import cv2
from torch import nn
from ..pipeline import *
from ..datasets.cityscapes import CityscapesLabelInfo


def tr_class_argmax(pred_prob, **_):
	return dict(
		pred_labels = np.argmax(pred_prob, axis=0)
	)


class SemSegLabelsToColorImg(TrByField):
	def __init__(self, fields=[('pred_labels', 'pred_labels_colorimg')], colors_by_classid=CityscapesLabelInfo.colors_by_trainId):
		"""
		colors_by_classid : [num_classes x 3] uint8
		"""
		super().__init__(fields=fields)
		self.set_class_colors(colors_by_classid)

	def set_class_colors(self, colors_by_classid):
		self.colors_by_classid = colors_by_classid
		# extend to 255 to allow the "unlabeled" areas
		# if we don't specify the dtype, it will be float64 and tensorboard will display it wrongly
		self.colors_by_classid_ext = np.zeros((256, 3), dtype=self.colors_by_classid.dtype)
		self.colors_by_classid_ext[:self.colors_by_classid.shape[0]] = self.colors_by_classid

	def set_override(self, class_id, color):
		self.colors_by_classid_ext[class_id] = color

	def forward(self, field_name, pred_labels):
		fr_sh = pred_labels.shape
		return self.colors_by_classid_ext[pred_labels.reshape(-1)].reshape((fr_sh[0], fr_sh[1], 3))


class TrColorimg(SemSegLabelsToColorImg):
	def __init__(self, *fields, table=CityscapesLabelInfo.colors_by_trainId):
		super().__init__(fields=[(f, f'{f}_colorimg') for f in fields], colors_by_classid=table)


class SemSegColorImgtoLabels(TrByField):
	def __init__(self, fields=[('colorimg_source', 'labels_source')], colors_by_classid=CityscapesLabelInfo.colors_by_id, default_label=0):
		"""
		colors_by_classid : [num_classes x 3] uint8
		"""
		super().__init__(fields=fields)
		self.set_class_colors(colors_by_classid)
		self.default_label = default_label

	def set_class_colors(self, colors_by_classid):
		self.colors_by_classid = colors_by_classid

	def forward(self, field_name, colorimg):
		fr_sh = colorimg.shape

		out_labels = np.zeros((fr_sh[0], fr_sh[1]), dtype=np.uint8)
		out_labels.fill(self.default_label)

		for cid in range(self.colors_by_classid.__len__()-1, -1, -1):
			color = self.colors_by_classid[cid]
			mask = np.all(colorimg == color[None, None, :], axis=2)
			out_labels[mask] = cid

		return out_labels


class SemSegProbabilityToEntropy(TrBase):
	def __call__(self, pred_prob, **other):
		"""
		pred_prob: softmax probability
		"""
		#pred_labels = np.argmax(pred_prob, axis=0)


		entropy_elems = np.log(pred_prob)
		entropy_elems *= pred_prob
		pred_entropy = -np.sum(entropy_elems, axis=0)
		return dict(
			pred_entropy = pred_entropy,
		)


def contour_hierarchy_get_child_list(hierarchy, parent_idx):
	# hierarchy[i] = [next, previous, child, parent]

	if parent_idx >= 0:
		cidx = hierarchy[parent_idx, 2]
	else:
		cidx = 0

	children = []
	while cidx != -1:
		children.append(cidx)
		cidx = hierarchy[cidx, 0]

	return np.array(children, dtype=np.int)




def tr_road_mask(labels_semantic, **_):
	road_mask_raw = labels_semantic == CityscapesLabelInfo.name2id['road']

	# road_mask = cv2.erode(road_mask_raw.astype(np.uint8), EROSION_MASK)
	road_mask = road_mask_raw.astype(np.uint8)
	road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, EROSION_MASK, iterations=2)
	road_mask = road_mask.astype(np.bool)

	return dict(
		road_mask_raw=road_mask_raw,
		road_mask=road_mask,
	)

def tr_detect_semantic_road_holes(frame, labels_semantic, road_mask, obstacle_contours=[], obstacle_scores=[],
								  b_show=False, **_):
	# detect hierarchical contours for the road class
	img, contour_list, hierarchy = cv2.findContours(
		road_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1,
	)

	if contour_list.__len__() == 0:
		return dict(
			valid=False,
			obstacle_contours=[],
			obstacle_scores=np.zeros(0),
		)

	hierarchy = hierarchy[0]  # obsolete dimension

	# hierarchy[i] is [next, previous, child, parent]

	# calc area for each contour
	contour_areas = np.array([cv2.contourArea(cnt) for cnt in contour_list])

	# find the biggest continuous road area - this will be the freespace in front of the car
	top_cidx = np.argmax(contour_areas)
	top_contour = contour_list[top_cidx]

	# get holes inside of the freespace
	hole_contour_indices = contour_hierarchy_get_child_list(hierarchy, top_cidx)
	hole_contour_areas = contour_areas[hole_contour_indices]

	if b_show:
		print('--', frame.fid, '--')
		# 	for h, cnt in zip(hierarchy[0], contour_list):
		# 		area = cv2.contourArea(cnt)
		# 		print(h, area)

		img = np.stack(3 * [img], axis=2)
		img = cv2.drawContours(img, contour_list, top_cidx, (0, 255, 0))

		for cidx in hole_contour_indices:
			print(cidx, contour_areas[cidx])
			img = cv2.drawContours(img, contour_list, cidx, (255, 0, 0))

		show(img)

	out = dict(
		valid=True,
		#		road_contours_all = contour_list,
		#		road_hierarchy_all = hierarchy,
		road_contour=top_contour[:, 0, :],  # obsolete dimension
		road_hole_contours=[contour_list[i][:, 0, :] for i in hole_contour_indices],
		road_hole_areas=hole_contour_areas,
	)

	out['obstacle_contours'] = obstacle_contours + out['road_hole_contours']
	out['obstacle_scores'] = obstacle_scores + list(out['road_hole_areas'])

	return out

def tr_obstacle_contours_to_mask(road_mask, obstacle_contours, obstacle_scores, **_):

	out_mask = np.zeros(road_mask.shape, dtype=np.uint8)

	for cnt in obstacle_contours:
		out_mask = cv2.drawContours(out_mask, [cnt], 0, 1, -1)

	return dict(
		obstacle_mask = out_mask,
	)

# Combined sem seg free space and hole detection

EROSION_MASK = np.ones((5, 5), np.uint8)


def tr_semseg_detect_freespace_and_obstacles(pred_labels_trainIds, road_label_ids=[CityscapesLabelInfo.name2trainId[c] for c in ['road', 'sidewalk']], **_):
	road_mask_raw = np.any([pred_labels_trainIds == label_id for label_id in road_label_ids], axis=0)

	# expand freespace by
	road_mask = road_mask_raw.astype(np.uint8)
	road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, EROSION_MASK, iterations=1)
	road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, EROSION_MASK, iterations=1)

	road_mask = road_mask.astype(np.bool)

	# detect hierarchical contours for the road class
	img, contour_list, hierarchy = cv2.findContours(
		road_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1,
	)
	hierarchy = hierarchy[0]  # obsolete dimension

	# calc area for each contour
	contour_areas = np.array([cv2.contourArea(cnt) for cnt in contour_list])

	# find the biggest continuous road area - this will be the freespace in front of the car
	top_cidx = np.argmax(contour_areas)
	top_contour = contour_list[top_cidx]

	freespace = cv2.drawContours(np.zeros(pred_labels_trainIds.shape, dtype=np.uint8), [top_contour], 0, 1, -1).astype(
		np.bool)
	freespace_area = np.count_nonzero(freespace)

	# get holes inside of the freespace
	hole_contour_indices = contour_hierarchy_get_child_list(hierarchy, top_cidx)
	hole_contour_areas = contour_areas[hole_contour_indices]
	scores = hole_contour_areas / freespace_area

	obstacle_mask = cv2.drawContours(
		np.zeros(pred_labels_trainIds.shape, dtype=np.uint8),
		[contour_list[ci] for ci in hole_contour_indices],
		-1, 1, -1,
	).astype(np.bool)

	# obstacle_mask2 = road_mask != freespace
	# show(image, [road_mask_raw, freespace], [obstacle_mask, obstacle_mask2])

	return dict(
		freespace_mask = freespace,
		obstacle_mask_semseg = obstacle_mask,
	)

def tr_bayesseg_sum_unc(pred_uncertainty, **_):
	return dict(
		pred_uncertainty_sum = np.sum(pred_uncertainty, axis=0)
	)

def tr_probout_sum_unc(pred_var, **_):
	return dict(
		pred_var_sum = np.sum(np.abs(pred_var), axis=0)
	)
