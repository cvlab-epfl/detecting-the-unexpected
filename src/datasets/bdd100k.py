
from .dataset import *
from .generic_sem_seg import *
from ..paths import DIR_DSETS

# Labels as defined by the dataset
from .bdd100k_labels import labels as bdd_labels
BDDLabelInfo = DatasetLabelInfo(bdd_labels)

DIR_BDD_SEG = Path(os.environ.get('MY_DIR_BDD_SEG', DIR_DSETS / 'dataset_BDD100k/bdd100k/seg'))

class DatasetBDD_Segmentation(DatasetBase):
	name = 'bdd100k'

	def __init__(self, dir_root=DIR_BDD_SEG, split='train', b_cache=True, b_recreate_original_ids=False):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split
		self.label_info = BDDLabelInfo

		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = '.jpg',
				file_path_tmpl = '{dset.dir_root}/images/{dset.split}/{fid}{channel.img_ext}',
				#opts={'ignoregamma': True},
			),
			# The label values are the train ids, because that is the only provided format
			labels = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/labels/{dset.split}/{fid}_train_id{channel.img_ext}',
			),
		)

		# The dataset only contains trainIds in the 'labels' channel
		# We can recreate the 'labels_source' channel with the following option:
		if b_recreate_original_ids:
			self.tr_post_load_pre_cache.append(
				TrSemSegLabelTranslation(self.label_info.table_trainId_to_label, fields = [('labels', 'labels_source')]),
			)

	def discover(self):
		self.frames = self.discover_directory_by_suffix(
			self.dir_root / 'images' / self.split,
			suffix = self.channels['image'].img_ext,
		)
		super().discover()
