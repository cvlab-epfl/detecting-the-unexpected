import logging
log = logging.getLogger('exp')
from .transforms import TrBase, TrByField
import numpy as np
import torch


class TrCUDA(TrByField):
	""" Transform which sends the given fields to a GPU """
	# TODO allow selecting which GPU
	
	def forward(self, field_name, val):
		if not torch.cuda.is_available():
			return val

		if isinstance(val, torch.Tensor):
			return val.cuda()
		elif isinstance(val, np.ndarray):
			return torch.from_numpy(val).cuda()
		else:
			self.conditionally_complain_about_type(field_name, val, 'torch.Tensor')
			return val

class TrPytorchNoGrad(TrByField):
	def forward(self, field_name, val):
		val.requires_grad = False
		return val

class TrNP(TrByField):
	""" Transform which retrieves the given fields from a GPU """
	def forward(self, field_name, val):
		if isinstance(val, torch.Tensor):
			if val.requires_grad:
				return val.detach().cpu().numpy()
			else:
				return val.cpu().numpy()
		else:
			self.conditionally_complain_about_type(field_name, val, 'torch.Tensor')
			return val

class TrTorch(TrByField):
	""" NumPy -> torch.Tensor """
	def forward(self, field_name, val):
		if isinstance(val, np.ndarray):
			return torch.from_numpy(val)
		elif isinstance(val, torch.Tensor):
			return val
		else:
			self.conditionally_complain_about_type(field_name, val, 'np.ndarray')
			return val

def tr_torch_images(**fields):
	""" change shape from [H, W, 3] to [3, H, W] """
	result = dict()
	for field, value in fields.items():
		if isinstance(value, np.ndarray) and value.shape.__len__() == 3 and value.shape[2] == 3:
			img_tr = torch.from_numpy(value.transpose(2, 0, 1))
			img_tr.requires_grad = False
			result[field] = img_tr
	return result

def tr_untorch_images(**fields):
	result = dict()
	for field, value in fields.items():
		if hasattr(value, 'shape') and value.shape.__len__() == 3 and value.shape[0] == 3:
			if isinstance(value, torch.Tensor):
				value = value.cpu().numpy()

			result[field] = value.transpose(1, 2, 0)

	return result


def torch_onehot(index, num_channels, dtype=torch.uint8):
	"""
	Everything above num_channels will be all-0
	"""
	
	roi = index < num_channels
	index = index.byte().clone()
	index *= roi.byte()
	
	onehot = torch.zeros( 
		index.shape[:1] + (num_channels, ) + index.shape[1:], # add the channel dimension
		device = index.device,
		dtype = dtype,
	)
	# log.debug(f'one {onehot.shape} idx {index.shape} roi {roi.shape}')
	onehot.scatter_(1, index[:, None].long(), roi[:, None].type(dtype))
	return onehot

def onehot(index, num_channels):
	return torch_onehot(index[:, None], num_channels, dtype=torch.float32)

class TrOnehotTorch(TrByField):
	def __init__(self, num_channels, dtype, fields):
		super().__init__(fields)
		self.num_channels = num_channels
		self.dtype = dtype
	
	def forward(self, _, index):
		return torch_onehot(index, num_channels=self.num_channels, dtype=self.dtype)
	

#def transfer_cuda(frame, fields=None, **_):
	#if torch.cuda.is_available():
		#for fname in (fields if fields is not None else frame.keys()):
			#val = frame[fname]
			#if isinstance(val, torch.Tensor):
				#frame[fname] = val.cuda()
			#elif isinstance(val, np.ndarray):
				#frame[fname] = torch.from_numpy(val).cuda()
			#elif fields is not None:
				#only complain if the field was specifically requested
				#print('Warning: requested CUDAing field {fname} which is not a torch.Tensor but a {cn}'.format(
					#fname=fname,
					#cn=type(val),
				#))
	#else:
		#print('ensure_cuda: CUDA is not loaded!')
		
	#return frame

#def transfer_np(frame, fields=None, **_):
	#for fname in (fields if fields is not None else frame.keys()):
		#val = frame[fname]
		#if isinstance(val, torch.Tensor):
			#frame[fname] = val.cpu().numpy()
		#elif isinstance(val, torch.autograd.Variable):
			#frame[fname] = val.data.cpu().numpy()
		#elif fields is not None:
			#only complain if the field was specifically requested
			#print('Warning: requested numpy-ing field {fname} which is not a torch.Tensor but a {cn}'.format(
				#fname=fname,
				#cn=type(val),
			#))

	#return frame
