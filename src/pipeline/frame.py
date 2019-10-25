
import logging
log = logging.getLogger('exp')
import numpy as np
import inspect, types
import multiprocessing, multiprocessing.dummy
from functools import partial
from ..common.util_notebook import ProgressBar

class Frame(dict):
	def __getattr__(self, key):
		"""
		Note that if the attribute is found through the normal mechanism, __getattr__() is not called.
		https://docs.python.org/3/reference/datamodel.html
		"""
		try:
			return self[key]
		except KeyError as e:
			raise AttributeError(e)

	def __setattr__(self, key, value):
		self[key] = value

	def apply(self, f):
		"""
		Calls the function, giving our data as kwargs.
		Updates our data with values returned by function
		"""
		try:
			result = f(frame=self, **self)

			if result:
				self.update(result)
			return result
			
		except KeyError as e:
			log.error('Missing field {field} for transform {t}'.format(field=e, t=f))
		#except TypeError as e:
		#	print(f'Missing field for transform {f}:\n{e}')

	def __call__(self, f):
		return self.apply(f)

	def copy(self):
		return Frame(super().copy())

	def update(self, *args, **kwargs):
		""" make this method chainable """
		super().update(*args, **kwargs)
		return self

	@staticmethod
	def repr_field(val):
		#is_np = isinstance(val, np.ndarray)
		#is_torch = isinstance(val, torch.Tensor)
		#if is_np or is_torch:

		if isinstance(val, dict):
			return '{' + ''.join(f'\n		{k}: {Frame.repr_field(v)}'  for k, v in val.items()) + '}'

		if isinstance(val, list):
			val_to_show = val[:10] if val.__len__() > 10 else val
			return '[' + '\n'.join(map(Frame.repr_field, val_to_show)) + ']'

		val_shape = getattr(val, 'shape', None)
		val_dtype = getattr(val, 'dtype', None)
		if not (val_shape is None or val_dtype is None):
			if val_shape.__len__() > 0:
				return '{tp}[{dims}]'.format(
					dims = ', '.join(map(str, val_shape)),
					tp = val_dtype,
				)

		return str(val)


	def __repr__(self):
		return ''.join(
			["Frame(\n"]
			+
			[
			'	{fn} = {fval}\n'.format(fn=fn, fval = self.repr_field(fval))
			for fn, fval in self.items()
			]
			+
			[')']
		)

	@staticmethod
	def frame_list_apply_worker(func, fr):
		fr.apply(func)
		return fr

	@staticmethod
	def build_pool(n_proc, n_threads):
		if n_proc > 1:
			return multiprocessing.Pool(n_proc)
		if n_threads > 1:
			return multiprocessing.dummy.Pool(n_threads)

	@classmethod
	def frame_list_apply(cls, func, frames, n_proc=1, n_threads=1, batch = 2, ret_frames=False, pbar=True):
		if ret_frames:
			out_frames = []

		if pbar:
			pbar = ProgressBar(frames.__len__())
		else:
			pbar = 0

		if n_proc == 1 and n_threads == 1:
			for fr in frames:
				fr.apply(func)
				pbar += 1

				if ret_frames:
					out_frames.append(fr)
		else:
			task = partial(cls.frame_list_apply_worker, func)

			with cls.build_pool(n_proc, n_threads) as pool:
				for result in pool.imap(task, frames, chunksize=batch):
					pbar += 1
					
					if ret_frames:
						out_frames.append(result)

		if ret_frames:
			return out_frames

	@classmethod
	def parallel_process(cls, func, frames, n_proc=1, n_threads=4, batch = 4, ret_frames=True, pbar=True):
		if ret_frames:
			out_frames = []

		if pbar:
			pbar = ProgressBar(frames.__len__())
		else:
			pbar = 0

		if n_proc == 1 and n_threads == 1:
			for fr in frames:
				fr = func(fr)
				pbar += 1

				if ret_frames:
					out_frames.append(fr)
		else:
			with cls.build_pool(n_proc, n_threads) as pool:
				for result in pool.imap(func, frames, chunksize=batch):
					pbar += 1
					if ret_frames:
						out_frames.append(result)

		if ret_frames:
			return out_frames

"""
def process_frame(image, labels=None, **other):
	print('img shape', image.shape)
	if labels is None:
		print('Setting labels')
		labels = np.zeros((5, 5))

	print('other args', other.keys())

	return dict(labels=labels)

fr(process_frame)
"""
