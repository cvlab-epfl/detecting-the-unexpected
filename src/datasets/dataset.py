
import logging
log = logging.getLogger('exp')
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset as PytorchDatasetInterface
import os, inspect, types
from functools import partial
from ..common.util import hdf5_load, hdf5_save
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrsChain
import h5py, json
import operator
try:
	import scipy.io
except:
	log.warning('No scipy, will not load .mat files')
import threading
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def imread(path):
	return np.asarray(Image.open(path))

IMWRITE_OPTS = dict(
	webp = dict(quality = 90),
)

def imwrite(path, data):
	path = Path(path)
	Image.fromarray(data).save(
		path, 
		**IMWRITE_OPTS.get(path.suffix.lower()[1:], {}),
	)

LOCK_HDF_CREATION = threading.Lock()

class DatasetBase(PytorchDatasetInterface):

	def __init__(self,  b_cache=True):
		# cfg?
		self.b_cache = b_cache
		self.frames = []
		self.frame_idx_by_fid = {}
		self.channels = dict()
		self.channels_enabled = set()

		# Transform chains
		# self.tr_pre_load
		# self.tr_post_load
		self.tr_post_load_pre_cache = TrsChain()
		self.tr_output = TrsChain()

		self.hdf5_files = dict()

		self.fake_num_frames = None


	def after_discovering_frames(self):
		self.frame_idx_by_fid = {fr.fid: idx for (idx, fr) in enumerate(self.frames)}
		for fr in self.frames:
			fr['dset'] = self

	def discover(self):
		""" Discovers available frames and stores them in self.frames
			Each frame should have the frame id "fid" field
			uniquely identifying it within the dataset
		"""
		self.after_discovering_frames()
		log.info(f'Discovered {self.__len__()} frames - {self}')
		return self

	def discover_directory_by_suffix(self, src_dir, suffix=''):
		suffix_len = suffix.__len__()

		return [
			Frame(fid = fn[:-suffix_len]) # remove suffix
			for fn in listdir_recursive(src_dir, ext=suffix, relative_paths=True)
		]

	def add_channels(self, **chs):
		for name, loader in chs.items():
			self.channels[name] = loader
			self.channels_enabled.add(name)

	def check_channel(self, ch):
		if ch not in self.channels:
			raise KeyError('Requested to enable channgel {ch} but the dset does not have it (channels = {chs})'.format(
				ch=ch,
				chs = ', '.join(self.channels.keys())
			))

	def set_channels_enabled(self, *chns):
		"""
		For example ds.set_channels_enabled(["image", "labels"])

		set_channels_enabled('*') enables all
		"""
		if '*' in chns:
			chns = list(self.channels.keys())
		elif chns.__len__() == 1 and (isinstance(chns[0], list) or isinstance(chns[0], tuple)):
			chns = chns[0]

		self.channels_enabled = set()
		self.channel_enable(*chns)

	def channel_enable(self, *chns):
		for ch in chns:
			self.check_channel(ch)
			self.channels_enabled.add(ch)

	def channel_disable(self, *chns):
		self.channels_enabled.difference_update(chns)

	def load_frame(self, frame):
		for ch_name in self.channels_enabled:
			if ch_name not in frame:
				self.channels[ch_name].load(self, frame, ch_name)

		frame.apply(self.tr_post_load_pre_cache)

	def save_frame_channel(self, frame, chn):
		self.channels[chn].save(self, frame, chn)

	def __getitem__(self, index):
		if isinstance(index, str):
			fr = self.get_frame_by_fid(index)
		else:
			fr = self.frames[index]
		#elif isinstance(index, slice):
			#index = range(index.start, index.stop, intex.step)
			#return [self[i] for i in index]

		#self.load_frame(fr)

		if self.b_cache:
			if not fr.get('cached', False):
				self.load_frame(fr)
				fr.cached = True
			out_fr = fr.copy()
			del out_fr['cached']
		else:
			out_fr = fr.copy()
			self.load_frame(out_fr)

		out_fr.apply(self.tr_output)

		return out_fr

	def get_frame_by_fid(self, fid):
		return self[self.frame_idx_by_fid[fid]]

	def get_idx_by_fid(self, fid):
		return self.frame_idx_by_fid[fid]

	def __len__(self):
		return self.fake_num_frames or self.frames.__len__()

	def __iter__(self):
		for idx in range(self.__len__()):
			yield self[idx]

	def __repr__(self):
		dr = getattr(self, 'dir_root', None)
		split = getattr(self, 'split', None)
		part = getattr(self, 'part', None)

		return '{cn}({nf} frames{rd}{split}{part})'.format(
			cn = self.__class__.__name__,
			nf = self.__len__(),
			rd = ', ' + str(dr) if dr is not None else '',
			split = ', s=' + split if split is not None else '',
			part = ', part=' + part if part is not None else '',
		)

	def path_for_channel(self, chan_name, fr_or_idx_or_fid):
		if isinstance(fr_or_idx_or_fid, Frame):
			frame = fr_or_idx_or_fid
		elif isinstance(fr_or_idx_or_fid, str):
			frame = self.frames[self.get_idx_by_fid(fr_or_idx_or_fid)]
		else:
			frame = self.frames[fr_or_idx_or_fid]
			
		return self.channels[chan_name].resolve_file_path(self, frame)

	def get_hdf5_file(self, file_path, write=False):
		file_path = Path(file_path)

		with LOCK_HDF_CREATION:
			handle = self.hdf5_files.get(file_path, None)

			if handle is None:
				if write:
					file_path.parent.mkdir(exist_ok=True)

				mode = (
					('a' if file_path.is_file() else 'w') if write
					else 'r'
				)

				handle = h5py.File(file_path, mode)
				self.hdf5_files[file_path] = handle

			return handle

	def flush_hdf5_files(self):
		for handle in self.hdf5_files.values():
			handle.close()
		self.hdf5_files = dict()

	def original_dataset(self):
		return self

	def set_fake_length(self, num_frames=None):
		""" Pretend we just have 0...n frames. None resets """
		self.fake_num_frames = num_frames

	def apply_fid_mask_from_json(self, json_path):
		with open(json_path, 'r') as fin:
			fids = json.load(fin)

		fids = set(fids)
		self.frames = [fr for fr in self.frames if fr.fid in fids]
		self.after_discovering_frames()
		log.info(f'Filtered to {self.frames.__len__()} frames')

	def load_class_statistics(self, path_override=None):
		path = path_override or '{dset.dir_out}/class_stats/{dset.split}_stats.hdf5'.format(dset=self)
		self.class_statistics = Frame(hdf5_load(path))

	def save_class_statistics(self, path_override=None):
		path = path_override or '{dset.dir_out}/class_stats/{dset.split}_stats.hdf5'.format(dset=self)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		hdf5_save(path, self.class_statistics)

class DatasetFrameList(DatasetBase):
	def __init__(self, frames, **kw):
		super().__init__(**kw)
		self.frames = frames
		self.discover()

	def __repr__(self):
		return 'DatasetFrameList()'

class ChannelLoader:
	def load(self, dset, frame, field_name):
		raise NotImplementedError()

class ChannelLoaderFileCollection:
	"""
	The channel's value for each frame is in a separate file, for example an image.
	@param file_path_tmpl: a template string or a function(channel, dset, fid)
		Example:

	"""
	def __init__(self, file_path_tmpl):
		# convert Path to str so that we can .format it
		self.file_path_tmpl = str(file_path_tmpl) if isinstance(file_path_tmpl, Path) else file_path_tmpl

	def resolve_template(self, template, dset, frame):
		if isinstance(template, str):
			# string template
			return template.format(dset=dset, channel=self, frame=frame, fid=frame['fid'], fid_no_slash=frame['fid'].replace('/', '__'))
		else:
			# function template
			return template(dset=dset, channel=self, frame=frame, fid=frame['fid'])

	def resolve_file_path(self, dset, frame):
		return self.resolve_template(self.file_path_tmpl, dset, frame)

	def load(self, dset, frame, field_name):
		path = self.resolve_file_path(dset, frame)
		frame[field_name] = self.read_file(path)

	def save(self, dset, frame, field_name):
		path = self.resolve_file_path(dset, frame)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		self.write_file(path, frame[field_name])

	def read_file(self, path):
		raise NotImplementedError('read_file for {c}'.format(c=self.__class__.__name__))

	def write_file(self, path, data):
		raise NotImplementedError('write_file for {c}'.format(c=self.__class__.__name__))

	def discover_files(self, dset):
		pattern = self.resolve_file_path(dset, Frame(fid='**'))
		return glob.glob(pattern)

	def __repr__(self):
		return '{cls}({tp})'.format(cls=self.__class__.__name__, tp=self.file_path_tmpl)


class ImageBackgroundService:
	IMWRITE_BACKGROUND_THREAD = ThreadPoolExecutor(max_workers=3)

	@classmethod
	def imwrite(cls, path, data):
		cls.IMWRITE_BACKGROUND_THREAD.submit(imwrite, path, data)


class ChannelLoaderImage(ChannelLoaderFileCollection):
	def __init__(self, file_path_tmpl=None, dir_root=None, suffix='', img_ext='.jpg'):
		"""
		Specify a function turning (channel, dset, frame.fid) into path
		or use the generic function:
			(channel.dir_root or dset.dir_root)/fid + channel.suffix + channel.img_ext
		"""
		super().__init__(file_path_tmpl or self.file_path_tmpl_generic)

		self.img_ext = img_ext
		self.opts = dict()

		self.dir_root = dir_root
		self.suffix = suffix

	@staticmethod
	def file_path_tmpl_generic(channel, dset, fid):
		""" generic path based on self.dir_root and self.suffix """
		return os.path.join(
			channel.dir_root or dset.dir_root,
			fid + channel.suffix + channel.img_ext,
		)

	def read_file(self, path):
		return np.asarray(imread(path, **self.opts))

	def write_file(self, path, data):
		#imwrite(path, data)
		ImageBackgroundService.imwrite(path, data)

class ChannelLoaderNpy(ChannelLoaderFileCollection):
	""" Single arrays stored in numpy's .npy files """
	def read_file(self, path):
		return np.load(path, allow_pickle=False)

	def write_file(self, path, data):
		np.save(path, data)


class ChannelLoaderMat(ChannelLoaderFileCollection):

	def __init__(self, *args, deserialize_func=lambda x: x, serialize_func=lambda x: x, **kwargs):
		super().__init__(*args, **kwargs)
		
		self.deserialize_func = deserialize_func
		self.serialize_func = serialize_func

	def read_file(self, path):
		return self.deserialize_func(scipy.io.loadmat(path))

	def write_file(self, path, data):
		scipy.io.savemat(path, self.serialize_func(data))


class ChannelLoaderNpyShared(ChannelLoaderFileCollection):

	def __init__(self, file_path_tmpl, index_func=operator.itemgetter('idx')):
		super().__init__(file_path_tmpl)

		self.index_func = index_func
		self.cache = dict()

	def get_file_contents(self, filepath):
		return self.cache.setdefault(filepath, np.load(filepath))

	def load(self, dset, frame, field_name):
		fpath = self.resolve_file_path(dset, frame)
		fvalue = self.get_file_contents(fpath)
		index = self.index_func(frame)
		frame[field_name] = fvalue[index]


class ChannelLoaderHDF5(ChannelLoaderFileCollection):
	def __init__(self, file_path_tmpl, var_name_tmpl='{fid}/{field_name}', index_func=None, compression=None):
		super().__init__(file_path_tmpl)

		self.var_name_tmpl = var_name_tmpl
		self.index_func = index_func
		self.compression = compression

	def resolve_var_name(self, dset, frame, field_name):
		return self.var_name_tmpl.format(dset=dset, channel=self, frame=frame, fid=frame['fid'], field=field_name)

	def resolve_index(self, dset, frame, field_name):
		if self.index_func:
			return self.index_func(dset=dset, channel=self, frame=frame, fid=frame['fid'], field=field_name)
		else:
			return None

	@staticmethod
	def read_hdf5_variable(variable):
		if variable.shape.__len__() > 0:
			return variable[:]
		else:
			return variable

	def load(self, dset, frame, field_name):
		hdf5_file_path = self.resolve_file_path(dset, frame)
		hdf5_file_handle = dset.get_hdf5_file(hdf5_file_path, write=False)
		var_name = self.resolve_var_name(dset, frame, field_name)
		if self.index_func:
			index = self.resolve_index(dset, frame, field_name)
			frame[field_name] = self.read_hdf5_variable(hdf5_file_handle[var_name][index])
		else:
			try:
				frame[field_name] = self.read_hdf5_variable(hdf5_file_handle[var_name])
			except KeyError as e:
				raise KeyError(f'Failed to read {var_name} from {hdf5_file_path}: {e}')

	def save(self, dset, frame, field_name):
		hdf5_file_path = self.resolve_file_path(dset, frame)
		hdf5_file_handle = dset.get_hdf5_file(hdf5_file_path, write=True)
		var_name = self.resolve_var_name(dset, frame, field_name)
		value_to_write = frame[field_name]

		if var_name in hdf5_file_handle:
			if self.index_func:
				index = self.resolve_index(dset, frame, field_name)
				hdf5_file_handle[var_name][index][:] = value_to_write
			else:
				hdf5_file_handle[var_name][:] = value_to_write
		else:
			if self.index_func:
				raise NotImplementedError('Writing to new HDF5 dataset with index_func')

			hdf5_file_handle.create_dataset(var_name, data=value_to_write, compression=self.compression)
			#hdf5_file_handle[var_name] = value_to_write

		#except Exception as e:
		#	print('HDF5 bla bla, key=', hdf5_path, '\n', e)


class ChannelResultImage(ChannelLoaderImage):
	def __init__(self, name, *args,  file_path_tmpl = '{dset.dir_out}/{channel.name}/{dset.split}/{fid}{channel.suffix}{channel.img_ext}', **kwargs):
		self.name = name

		super().__init__(
			*args,
			file_path_tmpl = file_path_tmpl,
			**kwargs,
		)

	def __repr__(self):
		return '{cls} "{n}"'.format(cls=self.__class__.__name__, n=self.name)


def listdir_recursive(base_dir, ext=None, relative_paths=False):
	results = []
	for (dir_cur, dirs, files) in os.walk(base_dir):
		if relative_paths:
			dir_cur = os.path.relpath(dir_cur, base_dir)
		if dir_cur == '.':
			dir_cur = ''

		for fn in files:
			if ext is None or fn.endswith(ext):
				results.append(os.path.join(dir_cur, fn))

	results.sort()
	return results


class DatasetImageDir(DatasetBase):
	def __init__(self, dir_root, img_ext='.jpg', name=None, split='', **kw):
		super().__init__(**kw)

		dir_root = Path(dir_root)
		self.name = name or dir_root.name

		self.dir_root = dir_root
		self.img_ext = img_ext

		if name is not None:
			from ..paths import DIR_DATA
			self.name = name
			self.dir_out = DIR_DATA / name

		self.add_channels(
			image = ChannelLoaderImage(
				file_path_tmpl='{dset.dir_root}/{fid}{channel.img_ext}',
				img_ext=img_ext
			),
		)

	def discover(self):
		self.frames = self.discover_directory_by_suffix(self.dir_root, suffix = self.img_ext)
		super().discover()


class TrSaveChannels:
	def __init__(self, dset, channels, ignore_none=False):
		self.dset = dset
		if isinstance(channels, str):
			self.channels = [channels]
		else:
			self.channels = channels

		self.ignore_none = ignore_none

	def __call__(self, frame, **_):
		if self.ignore_none:
			chs_to_save = [ch for ch in self.channels if ch in frame]
		else:
			chs_to_save = self.channels

		for ch in chs_to_save:
			self.dset.save_frame_channel(frame, ch)

class TrSaveChannelsAutoDset:
	def __init__(self, channels, ignore_none=False):
		if isinstance(channels, str):
			self.channels = [channels]
		else:
			self.channels = channels

		if isinstance(ignore_none, str):
			raise RuntimeError(f'channel name {ignore_none} in ignore_none')
		
		self.ignore_none = ignore_none

	def __call__(self, frame, dset, **_):
		if self.ignore_none:
			chs_to_save = [ch for ch in self.channels if ch in frame]
		else:
			chs_to_save = self.channels

		for ch in chs_to_save:
			dset.save_frame_channel(frame, ch)

	def __repr__(self):
		return '{clsn}({args})'.format(
			clsn = self.__class__.__name__,
			args = ', '.join(self.channels),
		)


def tr_print_paths(dset, frame, **_):
	for chn in dset.channels_enabled:
		log.info(f'	{chn}	{dset.path_for_channel(chn, frame)}')


#class DatasetWrapped:
	#""" Wrap a dataset with preprocessing func for pytorch.DataLoader """
	#def __init__(self, dset, f):
		#self.dset = dset
		#self.f = f

	#def __len__(self):
		#return self.dset.__len__()

	#def __getitem__(self, key):
		#return self.f(self.dset[key])

#class DatasetWrappedTransform(DatasetWrapped):
	#""" apply a frame transform before returning the frame """
	#def __init__(self, dset, transform):
		#super.__init__(dset, partial(self.preproc_func, transform))

	#@staticmethod
	#def preproc_func(transform, frame):
		#frame.apply(transform)
		#return frame

