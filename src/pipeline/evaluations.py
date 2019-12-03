
import numpy as np
from ..common.jupyter_show_image import adapt_img_data
from .transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix


class TrChannelIOBase(TrBase):
	def __init__(self, channel, field):
		self.field_name = field
		self.channel = channel
		self.channel_from_frame = isinstance(self.channel, str)

	def get_channel(self, dset_from_frame):
		if self.channel_from_frame:
			if dset_from_frame is None:
				raise ValueError(f"""
Tried to load channel {self.channel} from default dataset but frame has no "dset" field. 
Transform: {self}.
""")
			return dset_from_frame.channels[self.channel]
		else:
			return self.channel

	def __repr__(self):
		return f'{self.__class__.__name__}({self.field_name} ~ channel {self.channel})'


class TrChannelLoad(TrChannelIOBase):
	def __call__(self, frame, dset=None, **_):
		self.get_channel(dset).load(dset, frame, self.field_name)


class TrChannelSave(TrChannelIOBase):
	def __call__(self, frame, dset=None, **_):
		self.get_channel(dset).save(dset, frame, self.field_name)


class Evaluation:
	def __init__(self, exp):
		self.exp = exp
		self.tr_batch = TrsChain()
		self.tr_ouput = TrsChain()
		self.workdir = exp.workdir / 'pred'

		self.construct_persistence()

	def construct_persistence(self):
		pass

	def get_dset(self):
		return self.exp.datasets['val']

	@staticmethod
	def img_grid_2x2(imgs):
		imgs = [adapt_img_data(img) for img in imgs]
		return np.concatenate([
			np.concatenate(imgs[i::2], axis=1)
			for i in range(2)
		], axis=0)