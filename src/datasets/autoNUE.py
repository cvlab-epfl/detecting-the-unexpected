
from functools import partial
import numpy as np
import os
pp = os.path.join

from .dataset import DatasetBase, ChannelLoaderImage
from .generic_sem_seg import *
from ..paths import DIR_DSETS

# Labels as defined by the dataset
from .autoNUE_labels import labels as anue_labels
AutoNUELabelInfo = DatasetLabelInfo(anue_labels)

DIR_AUTONUE = Path(os.environ.get('MY_DIR_AUTONUE', DIR_DSETS / 'dataset_AutoNUE/anue/'))

# Loader
class DatasetAutoNUE(DatasetBase):
	name = 'autoNUE'

	def __init__(self, dir_root=DIR_AUTONUE, split='train', b_cache=True):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split
		self.label_info = AutoNUELabelInfo

		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/leftImg8bit/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_labelids{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_instanceids{channel.img_ext}',
			),
		)

		#self.channel_disable('instances')

		self.tr_post_load_pre_cache.append(
			self.label_info.tr_labelSource_to_trainId,
		)

	def discover(self):
		self.frames = self.discover_directory_by_suffix(
			pp(self.dir_root, 'leftImg8bit', self.split),
			suffix = '_leftImg8bit' + self.channels['image'].img_ext,
		)

		super().discover()
