
import numpy as np
from pathlib import Path
import os

from .dataset import *
from .generic_sem_seg import *
from ..paths import DIR_DSETS, DIR_DATA
from ..pipeline.transforms import TrRenameKw

# Labels as defined by the dataset
from .cityscapes_labels import labels as cityscapes_labels
CityscapesLabelInfo = DatasetLabelInfo(cityscapes_labels)

DIR_NYUDv2 = Path(os.environ.get('MY_DIR_NYUDv2', DIR_DSETS / 'dataset_NYU_Depth_v2'))

class NYUD_LabelInfo_Category40:
	names = ["unlabeled", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop"]

	colors = np.array([
		[0, 0, 0],
		[165,66,104],
		[83,194,76],
		[172,80,199],
		[142,188,58],
		[120,101,223],
		[65,146,51],
		[207,67,168],
		[88,197,124],
		[219,62,131],
		[88,200,169],
		[208,63,54],
		[59,188,195],
		[222,112,55],
		[91,123,222],
		[188,176,57],
		[121,84,172],
		[110,141,48],
		[218,127,219],
		[53,124,65],
		[157,79,149],
		[219,151,53],
		[85,99,164],
		[106,111,27],
		[179,146,214],
		[107,146,81],
		[206,70,94],
		[94,169,121],
		[221,131,174],
		[35,100,63],
		[222,127,120],
		[89,191,236],
		[162,84,46],
		[112,151,218],
		[142,111,44],
		[59,144,191],
		[213,161,107],
		[46,138,112],
		[139,74,95],
		[163,181,109],
		[98,106,50],
	], dtype=np.uint8)

	name2id = { name: idx for idx, name in enumerate(names) }

# Loader
class DatasetNYUDv2(DatasetBase):
	name = 'NYUv2'

	def __init__(self, dir_root=DIR_NYUDv2, split=None, b_cache=True):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split
		
		# self.split = split
		self.label_info = NYUD_LabelInfo_Category40()

		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/images/{fid:04d}_rgb{channel.img_ext}',
			),
			labels_category40 = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/labels_gupta2013_category40/{fid:04d}_category40{channel.img_ext}',
			),
			instances = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/instances_gupta2013/{fid:04d}_instances{channel.img_ext}',
			),
		)

		self.tr_post_load_pre_cache = TrsChain(
			TrRenameKw(labels_category40 = 'labels'),
		)

		self.channel_disable('instances')

	def discover(self):
		if self.split is not None:
			with (self.dir_root / 'splits_nohuman.json').open('r') as fin:
				split_info = json.load(fin)
			
			split_name, split = self.split.split('_')

			fids = split_info[split_name][split]

		else:
			fids = range(1, 1449+1)

		self.frames = [Frame(fid=fid) for fid in fids]

		super().discover()

	def load_class_statistics(self, path_override=None):
		print('NYU: No class statistics')
