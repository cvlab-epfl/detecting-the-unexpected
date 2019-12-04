
import logging
from pathlib import Path
import numpy as np
import os, re
from .dataset import DatasetBase, ChannelLoaderImage, ChannelLoaderHDF5, imread
from ..paths import DIR_DSETS, DIR_DATA

log = logging.getLogger('exp')

DIR_LOST_AND_FOUND = Path(os.environ.get('DIR_LAF', DIR_DSETS / 'dataset_LostAndFound' / '2048x1024'))
DIR_LOST_AND_FOUND_SMALL = Path(os.environ.get('DIR_LAF_SMALL', DIR_DSETS / 'dataset_LostAndFound' / '1024x512'))

class DatasetLostAndFound(DatasetBase):

	# invalid frames are those where np.count_nonzero(labels_source) is 0
	INVALID_LABELED_FRAMES = {
		'train': [44,  67,  88, 109, 131, 614],
		'test': [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793],
	}

	name = 'lost_and_found'

	IMG_FORMAT_TO_CHECK = ['.png', '.webp', '.jpg']

	def __init__(self, dir_root=DIR_LOST_AND_FOUND, split='train', only_interesting=True, only_valid=True, b_cache=True):
		"""
		:param split: Available splits: "train", "test"
		:param only_interesting: means we only take the last frame from each sequence:
			in that frame the object is the closest to the camera
		"""
		super().__init__(b_cache=b_cache)

		self.dir_root = Path(dir_root)
		self.split = split
		self.only_interesting = only_interesting
		self.only_valid = only_valid

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/leftImg8bit/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtCoarse/{dset.split}/{fid}_gtCoarse_labelIds{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtCoarse/{dset.split}/{fid}_gtCoarse_instanceIds{channel.img_ext}',
			),
		)

	def load_roi(self):
		"""
		Load a ROI which excludes the ego-vehicle and registration artifacts
		"""
		self.roi = imread(self.dir_root / 'LAF_roi.png') > 0
		self.roi_frame = dict(
			roi = self.roi
		)

	@staticmethod
	def tr_get_anomaly_gt(labels_source, **_):
		return dict(
			anomaly_gt = labels_source >= 2,
			roi_onroad = labels_source == 1,
		)
	
	def tr_get_roi(self, **_):
		return self.roi_frame

	@staticmethod
	def calc_dir_img(dset):
		return dset.dir_root / 'leftImg8bit' / dset.split

	@staticmethod
	def calc_dir_label(dset):
		return dset.dir_root / 'gtCoarse' / dset.split

	def discover(self):
		for img_ext in self.IMG_FORMAT_TO_CHECK:
			self.frames_all = self.discover_directory_by_suffix(
				self.dir_root / 'leftImg8bit' / self.split,
				suffix = f'_leftImg8bit{img_ext}',
			)
			if self.frames_all:
				log.info(f'LAF: found images in {img_ext} format')
				break

		self.channels['image'].img_ext = img_ext

		# LAF's PNG images contain a gamma value which makes them washed out, ignore it
		if img_ext == '.png':
			self.channels['image'].opts['ignoregamma'] = True

		# parse names to determine scenes, sequences and timestamps
		for fr in self.frames_all:
			fr.apply(self.laf_name_to_sc_seq_t)

		if self.only_valid:
			invalid_indices = self.INVALID_LABELED_FRAMES[self.split]
			valid_indices = np.delete(np.arange(self.frames_all.__len__()), invalid_indices)
			self.frames_all = [self.frames_all[i] for i in valid_indices]

		# orgnize frames into hierarchy:
		# fr = scenes_by_id[fr.scene_id][fr.scene_seq][fr.scene_time]
		scenes_by_id = dict()

		for fr in self.frames_all:
			scene_seqs = scenes_by_id.setdefault(fr.scene_id, dict())

			seq_times = scene_seqs.setdefault(fr.scene_seq, dict())

			seq_times[fr.scene_time] = fr

		self.frames_interesting = []

		# Select the last frame in each sequence, because thats when the object is the closest
		for sc_name, sc_sequences in scenes_by_id.items():
			for seq_name, seq_times in sc_sequences.items():
				#ts = list(seq_times.keys())
				#ts.sort()
				#ts_sel = ts[-1:]
				#self.frames_interesting += [seq_times[t] for t in ts_sel]

				t_last = max(seq_times.keys())
				self.frames_interesting.append(seq_times[t_last])

		# set self.frames to the appropriate collection
		self.use_only_interesting(self.only_interesting)

		self.load_roi()

		super().discover()

	RE_LAF_NAME = re.compile(r'([0-9]{2})_.*_([0-9]{6})_([0-9]{6})')

	@staticmethod
	def laf_name_to_sc_seq_t(fid, **_):
		m = DatasetLostAndFound.RE_LAF_NAME.match(fid)

		return dict(
			scene_id = int(m.group(1)),
			scene_seq = int(m.group(2)),
			scene_time = int(m.group(3))
		)

	def use_only_interesting(self, only_interesting):
		self.only_interesting = only_interesting
		self.frames = self.frames_interesting if only_interesting else self.frames_all

class DatasetLostAndFoundSmall(DatasetLostAndFound):
	def __init__(self, dir_root=DIR_LOST_AND_FOUND_SMALL, **kwargs):
		super().__init__(dir_root=dir_root, **kwargs)

	def load_roi(self):
		"""
		Load a ROI which excludes the ego-vehicle and registration artifacts
		"""
		self.roi = imread(DIR_DATA / 'cityscapes/roi.png').astype(np.bool)
