
import numpy as np
import torch
from torch.nn import functional as torch_functional


class Padder:
	""" Pad image so that the size is divisible by `divisor`, unpad the result later """

	def __init__(self, shape, divisor):
		sz = np.array(shape[-2:])
		sz_deficit = np.mod(divisor - np.mod(sz, divisor), divisor)

		self.needs_padding = np.any(sz_deficit)

		self.size_orig = sz
		self.padding_tl = sz_deficit // 2
		self.padding_br = (sz_deficit + 1) // 2
		self.unpad_end = self.padding_tl + self.size_orig

		# print('Padder', self.padding_tl, self.padding_br)

	def pad(self, tensor):
		if self.needs_padding:

			# padding needs [B, C, H, W] but so we ensure a C dimension exists even if its just 1
			expand_chans = tensor.shape.__len__() == 3
			if expand_chans:
				tensor = torch.unsqueeze(tensor, 1)
			
			# pytorch can not do reflect for uint8
			mode = 'reflect' if tensor.dtype != torch.uint8 else 'constant'

			res = torch_functional.pad(
				tensor, 
				(self.padding_tl[1], self.padding_br[1], self.padding_tl[0], self.padding_br[0]),
				mode=mode,
			)

			# undo the adding of a C dimension
			if expand_chans:
				res = torch.squeeze(res, 1)

			# print(f'Padded from {tuple(tensor.shape)} to {tuple(res.shape)}')
			return res
		else:
			return tensor

	def unpad(self, tensor):
		if self.needs_padding:
			if tensor.shape.__len__() == 4:
				res =  tensor[:, :, self.padding_tl[0]:self.unpad_end[0], self.padding_tl[1]:self.unpad_end[1]]
			elif  tensor.shape.__len__() == 3:
				res =  tensor[:, self.padding_tl[0]:self.unpad_end[0], self.padding_tl[1]:self.unpad_end[1]]
			else:
				raise ValueError(f'Unpadding wants a 3 or 4 dim value, but we got shape {tuple(tensor.shape)}')
			# print(f'Unpadded from {tuple(tensor.shape)} to {tuple(res.shape)}')
			return res
		else:
			return tensor

