
import torch
import numpy as np
from torch import nn
from torch.nn import functional
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrsChain
from ..pytorch_semantic_segmentation import utils as ptseg_utils
from ..pytorch_semantic_segmentation import models as ptseg_models
from ..common.util_networks import Padder

class LossCrossEntropy2d(nn.Module):
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super().__init__()
		self.log_sfm = nn.LogSoftmax(dim=1) # along channels
		self.nll_loss = nn.NLLLoss(
			weight,
			reduction='mean' if size_average else 'none',
			ignore_index=ignore_index,
		)

	def forward(self, pred_logits, labels, **other):
		return dict(loss = self.nll_loss(
			self.log_sfm(pred_logits),
			labels,
		))


class ClassifierSoftmax(nn.Module):
	def __init__(self):
		super().__init__()
		self.softmax = torch.nn.Softmax2d()

	#def __call__(self, pf, net_output):
		#pred_softmax = self.softmax(net_output[None, :, :, :])
		#pf.pred_prob = pred_softmax[0, :, :, :].data.cpu().numpy()
		#classify(pf)

	def forward(self, pred_logits, **_):
		if pred_logits.shape.__len__() == 4:
			pred_softmax = self.softmax(pred_logits)
		else:
			# single sample, with no batch dimension
			pred_softmax = self.softmax(pred_logits[None])[0]

		return dict(
			pred_prob = pred_softmax
		)

class PerspectiveSceneParsingNet(ptseg_models.PSPNet):
	def forward(self, image, **_):
		pred_raw = super().forward(image)
		return dict(
			pred_logits = pred_raw,
		)

class LossPSP(nn.Module):
	def __init__(self, weight=None, size_average=True, ignore_index=255):
		super().__init__()
		self.log_sfm = nn.LogSoftmax(dim=1) # along channels
		self.nll_loss = nn.NLLLoss(weight, reduction='mean' if size_average else 'none', ignore_index=ignore_index)
		self.cel = LossCrossEntropy2d(weight, size_average, ignore_index)

	def forward(self, pred_logits, labels, **other):
		if isinstance(pred_logits, tuple):
			pred_raw_main, pred_raw_aux = pred_logits

			loss_main = self.nll_loss(self.log_sfm(pred_raw_main), labels)
			loss_aux = self.nll_loss(self.log_sfm(pred_raw_aux), labels)

			return dict(
				loss = loss_main * (1.0/1.4) + loss_aux * (0.4/1.4),
				loss_main = loss_main,
				loss_aux = loss_aux,
			)
		else:
			return self.cel(pred_logits, labels, **other)


class BayesianSegNet(ptseg_models.SegNetBayes):

	def forward(self, img):

		# the network fails if the image size is not divisible by 32
		# pad to 32  -> run net -> remove padding

		padder = Padder(img.shape, 32)
		img = padder.pad(img)

		result = super().forward(img)

		return padder.unpad(result)
