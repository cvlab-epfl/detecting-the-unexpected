
import logging
log = logging.getLogger('exp')
import numpy as np
from .frame import *
from .transforms import TrsChain
from ..common.util_notebook import ProgressBar
import torch
from torch.utils.data.dataloader import DataLoader
from functools import partial
import gc
from concurrent.futures import ThreadPoolExecutor
# from tqdm.autonotebook import tqdm
from tqdm import tqdm

# TODO Transforms:
# input_pre_batch - wrapping the dataset
#  train and eval transforms
#  or transforms which apply conditionally on the type of batch
#  (type of batch would also determine how variables are formed)
# input_post_batch -> cuda
# 

from torch.utils.data._utils.collate import string_classes, int_classes, np_str_obj_array_pattern
import re, collections

def default_collate_edited(batch):
	r"""Puts each data field into a tensor with outer dimension batch size"""

	elem = batch[0]
	elem_type = type(elem)
	if isinstance(elem, torch.Tensor):
		out = None
		if torch.utils.data.get_worker_info() is not None:
			# If we're in a background process, concatenate directly into a
			# shared memory tensor to avoid an extra copy
			numel = sum([x.numel() for x in batch])
			storage = elem.storage()._new_shared(numel)
			out = elem.new(storage)
		return torch.stack(batch, 0, out=out)
	elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
		if elem_type.__name__ == 'ndarray':
			# array of string classes and object
			if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
				raise TypeError(f'Unusual numpy dtype {elem.dtype}')

			return default_collate_edited([torch.as_tensor(b) for b in batch])
		elif elem.shape == ():  # scalars
			return torch.as_tensor(batch)
	elif isinstance(elem, float):
		return torch.tensor(batch, dtype=torch.float64)
	elif isinstance(elem, int_classes):
		return torch.tensor(batch)
	elif isinstance(elem, string_classes):
		return batch
	elif isinstance(elem, collections.Mapping):
		return {key: default_collate_edited([d[key] for d in batch]) for key in elem}
	elif isinstance(elem, collections.Sequence):
		transposed = zip(*batch)
		return [default_collate_edited(samples) for samples in transposed]

	# if the objects did not match, return a list
	return batch

class Pipeline:


	default_loader_args = dict(
		shuffle=False,
		batch_size = 4,
		num_workers = 0,
		drop_last = False,
	)

	def __init__(self, tr_input=None, tr_batch_pre_merge=None, tr_batch=None, tr_output=None, loader_args=default_loader_args):
		self.tr_input = tr_input or TrsChain()
		self.tr_batch_pre_merge = tr_batch_pre_merge or TrsChain()
		self.tr_batch = tr_batch or TrsChain()
		self.tr_output = tr_output or TrsChain()

		self.loader_class = DataLoader
		self.loader_args = loader_args

		# TODO explicit batch size

	def __repr__(self):
		return ('Piepeline(\n' + '\n'.join(
			f + ' = ' + str(v)
			for (f, v) in self.__dict__.items() if f.startswith('tr_')
		) + '\n)')


	@staticmethod
	def pipline_collate(tr_input, tr_batch_pre_merge, frames):
		""" Applies transforms """
		result_frame = Frame()

		to_batch = []

		for fr in frames:
			fr.apply(tr_input)
			fr_to_batch = fr.copy()
			fr_to_batch.apply(tr_batch_pre_merge)
			to_batch.append(fr_to_batch)

		result_frame = Frame(default_collate_edited(to_batch))
		return dict(batch=result_frame, input_frames = frames)

	@staticmethod
	def unbatch_value(value, idx):
		# those pesky 0-dim tensors

		if not hasattr(value, "__getitem__"): # not indexable can't be split
			return value
		elif isinstance(value, torch.Tensor) and value.shape.__len__() == 0:
			return value
		else:
			return value[idx]
	
	@classmethod
	def unbatch(cls, batch, input_frames):
		return [
			in_fr.copy().update({
				field: cls.unbatch_value(value, idx) for field, value in batch.items()
			})
			for (idx, in_fr) in enumerate(input_frames)
		]

		#batch_size = batch['input_frame'].__len__()
		#unbatched_frames = [
			#Frame(**{	field: value[idx] for field, value in batch.items()})
			#for idx in range(batch_size)
		#]
		#return unbatched_frames

	def get_loader(self, dset):
		collate_fn = partial(self.pipline_collate, self.tr_input, self.tr_batch_pre_merge)
		return self.loader_class(dset, collate_fn=collate_fn, **self.loader_args)

	def apply_tr_output(self, frame):
		frame.apply(self.tr_output)
		return frame

	def execute(self, dset, b_accumulate=True, b_grad=False, b_one_batch=False, b_pbar=True, log_progress_interval=None, short_epoch=None):
		loader = self.get_loader(dset)

		out_frames_all = [] if b_accumulate else None

		if b_pbar and (not b_one_batch):
			pbar = ProgressBar(dset.__len__())
		else:
			pbar = 0

		frames_processed = 0
		batches_processed = 0

		with ThreadPoolExecutor(max_workers=max(self.loader_args['batch_size'], 1)) as pool:
			with torch.set_grad_enabled(b_grad):
				for loader_item in loader:
					batch = loader_item['batch']
					input_frames = loader_item['input_frames']

					batch.apply(self.tr_batch)
					out_frames = self.unbatch(batch, input_frames)

					# parallelize output
					# that damned executor returns a generator !
					out_frames = list(pool.map(self.apply_tr_output, out_frames))

					if b_one_batch:
						return batch, out_frames

					del batch

					if b_accumulate:
						out_frames_all += out_frames

					# progress recording
					frames_processed += out_frames.__len__()
					pbar += out_frames.__len__()

					if log_progress_interval and frames_processed % log_progress_interval < out_frames.__len__():
						frame_num = dset.__len__()
						if short_epoch is not None: frame_num = min(short_epoch, frame_num)
						log.debug('	{p}	/	{t}'.format(p=frames_processed, t=frame_num))

					if short_epoch is not None and frames_processed >= short_epoch:
						return out_frames_all

					# GC periodically
					batches_processed += 1
					if batches_processed % 8 == 0:
						gc.collect()

		return out_frames_all

class SamplerThreaded:
	def __init__(self, dset, batch_size, collate_fn, shuffle = True, num_workers=2, drop_last=False):
		self.dset = dset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.collate_fn = collate_fn

		self.thread_executor = ThreadPoolExecutor(num_workers)

	def __len__(self):
		return self.dset.__len__()

	def build_batch(self, idx_group):
		return self.collate_fn([self.dset[idx] for idx in idx_group])

	def __iter__(self):
		num_frames = self.dset.__len__()

		if self.shuffle:
			perm = np.random.permutation(num_frames)
		else:
			perm = np.arange(num_frames)


		batch_count = np.ceil(num_frames / self.batch_size)
		batch_remainder = num_frames % self.batch_size
		if batch_remainder != 0 and self.drop_last:
			perm = perm[batch_remainder:]
			batch_count -= 1

		batch_index_groups = np.array_split(perm, batch_count)

		# start loading batch 0
		future_batch = self.thread_executor.submit(self.build_batch, batch_index_groups[0])

		for idx_group in batch_index_groups[1:]:
			# retrieve batch which was loaded in background
			materialized_batch = future_batch.result()

			# start loading next batch in background
			future_batch = self.thread_executor.submit(self.build_batch, idx_group)

			# return the loaded batch
			# now the batch is being consumed while we are loading the next one on other thread
			yield materialized_batch

		# retrieve the last batch
		yield future_batch.result()

