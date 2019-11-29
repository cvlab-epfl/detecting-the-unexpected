
import logging
from pathlib import Path
import os
from .dataset import DatasetBase, ChannelLoaderImage
from .generic_sem_seg import DatasetLabelInfo
from ..paths import DIR_DSETS

# Labels as defined by the dataset
from .cityscapes_labels import labels as cityscapes_labels
CityscapesLabelInfo = DatasetLabelInfo(cityscapes_labels)

log = logging.getLogger('exp')

DIR_CITYSCAPES = Path(os.environ.get('DIR_CITYSCAPES', DIR_DSETS / 'dataset_Cityscapes' / '2048x1024'))
DIR_CITYSCAPES_SMALL = Path(os.environ.get('DIR_CITYSCAPES_SMALL', DIR_DSETS / 'dataset_Cityscapes' / '1024x512'))

# Loader
class DatasetCityscapes(DatasetBase):
	name = 'cityscapes'
	label_info = CityscapesLabelInfo
	IMG_FORMAT_TO_CHECK = ['.png', '.webp', '.jpg']

	def __init__(self, dir_root=DIR_CITYSCAPES, split='train', b_cache=True):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl = '{dset.dir_root}/images/leftImg8bit/{dset.split}/{fid}_leftImg8bit{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_labelIds{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/gtFine/{dset.split}/{fid}_gtFine_instanceIds{channel.img_ext}',
			),
		)

		self.channel_disable('instances')

		self.tr_post_load_pre_cache.append(
			self.label_info.tr_labelSource_to_trainId,
		)

	def discover(self):
		for img_ext in self.IMG_FORMAT_TO_CHECK:
			self.frames = self.discover_directory_by_suffix(
				self.dir_root / 'images' / 'leftImg8bit' / self.split,
				suffix = f'_leftImg8bit{img_ext}',
			)
			if self.frames:
				log.info(f'Cityscapes: found images in {img_ext} format')
				break

		self.channels['image'].img_ext = img_ext
		super().discover()


class DatasetCityscapesSmall(DatasetCityscapes):
	def __init__(self, dir_root=DIR_CITYSCAPES_SMALL, split='train', b_cache=True):
		super().__init__(dir_root=dir_root, split=split, b_cache=b_cache)
