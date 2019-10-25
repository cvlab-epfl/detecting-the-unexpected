
from .dataset import *
from ..paths import DIR_DSETS
from .generic_sem_seg import *
import json, re

DIR_APOLLO = Path(os.environ.get('MY_DIR_APOLLO', DIR_DSETS / 'dataset_ApolloScape/scenes'))

# Labels as defined by the dataset
from .apolloscape_labels import labels as apolloscape_labels
ApolloScapeLabelInfo = DatasetLabelInfo(apolloscape_labels)

class ApolloContoursJSONChannel(ChannelLoaderFileCollection):

	@staticmethod
	def apo_instances_parse_object(obj):
		contours = [np.array(poly) for poly in obj['polygons']]
		label = obj['label']
		return Frame(label=label, contours=contours)

	@classmethod
	def apo_instances_parse_frame(cls, json_frame):
		return [cls.apo_instances_parse_object(obj) for obj in json_frame['objects']]

	def read_file(self, path):
		with open(path, 'r') as fin:
			return self.apo_instances_parse_frame(json.load(fin))



class DatasetApolloScape(DatasetBase):
	name = 'apollo'

	def __init__(self, dir_root=DIR_APOLLO, split='train', b_cache=True):
		super().__init__(b_cache=b_cache)

		self.dir_root = dir_root
		self.split = split
		self.label_info = ApolloScapeLabelInfo

		self.add_channels(
			image = ChannelLoaderImage(
				img_ext = '.jpg',
				file_path_tmpl = '{dset.dir_root}/{frame.scene}/ColorImage/Record{frame.seq:03d}/Camera {frame.cam}/{fid}{channel.img_ext}',
			),
			labels_source = ChannelLoaderImage(
				img_ext = '.png',
				file_path_tmpl = '{dset.dir_root}/{frame.scene}/Label/Record{frame.seq:03d}/Camera {frame.cam}/{fid}{frame.sem_suffix}{channel.img_ext}',
			),
			lane_labels_source=ChannelLoaderImage(
				img_ext='.png',
				file_path_tmpl='{dset.dir_root}/{frame.scene}/Lane/Record{frame.seq:03d}/Camera {frame.cam}/{fid}_bin{channel.img_ext}',
			),
			instance_contours = ApolloContoursJSONChannel(
				file_path_tmpl = '{dset.dir_root}/{frame.scene}/Label/Record{frame.seq:03d}/Camera {frame.cam}/{fid}.json',
			),
		)

		self.channel_disable('instances')

		self.tr_post_load_pre_cache.append(
			self.label_info.tr_labelSource_to_trainId,
		)

	def discover(self, scenes=['01', '02', '03']):
		dir_lists = self.dir_root / '..' / 'public_image_lists'

		RE_img = re.compile(r'(.+)/ColorImage/Record([0-9]+)/Camera ([0-9]+)/(.+)\.jpg')

		frs = []

		for scn in scenes:
			list_path = dir_lists / 'road{scn}_ins_{split}.lst'.format(scn=scn, split=self.split)
			with open(list_path, 'r') as fin:
				for line in fin:
					img_path = line.split('	')[0]
					scene, seqid, cam, fid = re.match(RE_img, img_path).groups()
					seqid = int(seqid)
					cam = int(cam)

					fr = Frame(
						fid = fid,
						scene = scene,
						cam = cam,
						seq = seqid,
						sem_suffix = '' if scn == '01' else '_bin',
					)
					frs.append(fr)

		self.frames = frs
		super().discover()


class ApolloContoursHDF(ChannelLoaderFileCollection):
	pass

# from collections import namedtuple
#
# def apo_instances_to_hdf5(json_frame):
# 	print(json_frame.keys())
# 	objects = [apo_instances_parse_object(obj) for obj in json_frame['objects']]
#
# 	rect_record = namedtuple('start_index', 'length')
# 	rect_stack = dict(
# 		pts = [],
# 		pt_count = 0,
# 	)
# 	def add_rect(pts):
# 		rect_stack['pts'].append(pts)
#
# 		count = rect_stack['pt_count']
# 		new_count = count + pts.__len__()
#
# 		rect_count['pt_count'] = new_count
#
# 		return rect_record(
# 			start_index = count,
# 			length = pts.__len__(),
# 		)
#
#
# 	for obj in objects:
# 		rect_records = [add_rect(obj)]


#class DatasetApolloScapeSmall(DatasetApolloScape):
#
#	cropping: 