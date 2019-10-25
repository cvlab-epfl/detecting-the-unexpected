
import numpy as np
from ..common.jupyter_show_image import adapt_img_data
from .transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix


class TrChannelLoad(TrBase):
	def __init__(self, channel, field):
		self.field_name = field
		self.channel = channel

	def __call__(self, dset, frame, **_):
		self.channel.load(dset, frame, self.field_name)

class TrChannelSave(TrBase):
	def __init__(self, channel, field):
		self.field_name = field
		self.channel = channel

	def __call__(self, dset, frame, **_):
		self.channel.save(dset, frame, self.field_name)

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