
from .dataset import *
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrByField
import numpy as np
from functools import partial
from collections import namedtuple


def binary_color_to_rgb():
	masks = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.int32)
	shifts = np.array([16, 8, 0], dtype=np.int32)

	def binary_color_to_rgb(binary_color):
		return (binary_color & masks) >> shifts

	return binary_color_to_rgb

binary_color_to_rgb = binary_color_to_rgb()

# num classes
# translation tables
#	 train id to id
# id to color

# class statistics
# class color table

def apply_label_translation_table(table, labels_source):
	mx =  np.max(labels_source)
	if mx >= table.shape[0]-1:
		if mx != 255:
			print('Overflow in labels', mx)
		labels_source = labels_source.copy()
		labels_source[labels_source >= table.shape[0]] = 0

	return table[labels_source.reshape(-1)].reshape(labels_source.shape)

class TrSemSegLabelTranslation(TrByField):
	def __init__(self, table, fields=[('labels_source', 'labels')]):
		super().__init__(fields=fields)
		self.table = table

	def forward(self, field_name, value):
		return apply_label_translation_table(self.table, value)

class DatasetLabelInfo:
	def __init__(self, label_list):
	
		self.labels = label_list
				
		# name to label object
		self.name2label = { label.name: label for label in self.labels }
		self.name2id = { label.name: label.id for label in self.labels }
		self.name2trainId = { label.name: label.trainId for label in self.labels }
		# id to label object
		self.id2label = { label.id: label for label in self.labels }
		# trainId to label object
		self.trainId2label = { label.trainId: label for label in reversed(self.labels) }

		self.table_label_to_trainId = self.build_translation_table([
			(label.id, label.trainId) for label in self.labels
		])
		self.table_trainId_to_label = self.invert_translation_table(self.table_label_to_trainId)

		self.tr_labelSource_to_trainId = TrSemSegLabelTranslation(self.table_label_to_trainId)
		self.tr_trainId_to_labelSource = TrSemSegLabelTranslation(self.table_trainId_to_label)

		self.valid_for_eval = self.build_bool_table([
			(label.id, not label.ignoreInEval) for label in self.labels
		], False)
		self.valid_for_eval_trainId = self.build_bool_table([
			(label.trainId, not label.ignoreInEval) for label in self.labels
		], False)


		self.num_labels = self.labels.__len__()
		self.num_trainIds = self.trainId2label.__len__() - 1 #-1 because 255->unlabeled is in the list
	
		self.build_colors()
	
	@staticmethod
	def build_color_table(index_color_pairs):	
		table_size = max(index for (index, color) in index_color_pairs)+1
		table_size = max(table_size, 256)

		color_table = np.zeros(
			(table_size, 3),
			dtype=np.uint8,
		)
		color_table[:] = (0, 255, 255) # "missing color"
		
		for idx, color in reversed(index_color_pairs):
			color_table[idx] = color
		
		return color_table

	@staticmethod
	def build_bool_table(pairs, default_val):
		table_size = max(src for (src, dest) in pairs) + 1
		table_size = max(table_size, 256)

		table = np.zeros(
			table_size,
			dtype=np.bool,
		)
		table[:] = default_val

		for (src, dest) in pairs:
			table[src] = dest

		return table

	@staticmethod
	def build_translation_table(pairs):
		table_size = max(src for (src, dest) in pairs)+1
		table_size = max(table_size, 256)

		table = np.zeros(
			table_size,
			dtype=np.uint8,
		)
		table[:] = 255

		for (src, dest) in pairs:
			table[src] = dest

		return table
	
	@staticmethod
	def invert_translation_table(table):
		
		table_inv = np.zeros(np.max(table)+1, dtype=table.dtype)
		
		for src, dest in reversed(list(enumerate(table))):
			table_inv[dest] = src
			
		return table_inv
	
	def build_colors(self):
		
		# fix binary colors (apollo)
		labels_replace = []
		for label in self.labels:
			if not isinstance(label.color, tuple):
				color_decoded = binary_color_to_rgb(label.color)

				try:
					label.color = color_decoded
				except AttributeError: # for namedtuples which don't like assignment
					label = label._replace(color = color_decoded)

			labels_replace.append(label)

		self.labels = labels_replace
				
		self.colors_by_id = self.build_color_table([
			(label.id, label.color) for label in self.labels 
		])
		
		self.colors_by_trainId = self.build_color_table([
			(label.trainId, label.color) for label in self.labels 
		])

def calculate_class_distribution_tr(chan_name, num_classes, **fields):
	label_map = fields[chan_name]
	area_inv = 1. / np.product(label_map.shape)

	return {
		'class_distrib': np.bincount(label_map.reshape(-1), minlength=num_classes) * area_inv,
		chan_name: None, # clear to save mem
	}

def calculate_class_distribution(chan_name, num_classes, dset):
	dset.set_channels_enabled(chan_name)
	dset.discover()

	worker = partial(
		calculate_class_distribution_tr,
		chan_name = chan_name,
		num_classes = num_classes,
	)

	frs = Frame.frame_list_apply(worker, dset, n_threads=16, ret_frames=True)

	class_distribs = np.mean(np.array([fr['class_distrib'] for fr in frs]), axis=0)
	return class_distribs

def class_weights_from_class_distrib(class_distrib):
	class_distrib = class_distrib / np.sum(class_distrib)
	class_weights = 1. / np.log(1.02 + class_distrib)
	return class_weights
