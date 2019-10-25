
import numpy as np
import torch
from torch import nn
from torch.nn import functional


class RoadReconstructionRBM(nn.Module):
	"""
	"Real-time small obstacle detection on highways using compressive RBM road reconstruction"
	Clement Creusot and Asim Munawar
	Intelligent Vehicles Symposium 2015
	"""
	#def __init__(self, patch_shape=(50, 50, 3), num_hidden=20):
	def __init__(self, patch_shape=(8, 8, 3), num_hidden=20, local_normalize=True):

		super().__init__()

		self.patch_shape = patch_shape
		self.num_hidden = num_hidden

		in_dim = np.prod(patch_shape)

		# W and bias_hidden
		self.linear_layer = nn.Linear(in_dim, num_hidden)

		# bias_visible
		self.bias_visible = nn.Parameter(torch.zeros(in_dim))

		self.loss_mod = nn.MSELoss(reduction='mean')

		self.local_normalize = local_normalize

	def pre_normalize_and_flatten(self, image_patches, **_):
		if self.local_normalize:
			orig_shape = image_patches.shape
			batch_size = orig_shape[0]
			#patches_flattened = image_patches.reshape(batch_size, -1)

			# mean = torch.mean(patches_flattened, dim=0)
			#patches_flattened = patches_flattened - mean[None, :]

			# zero center
			means = torch.mean(image_patches, dim=1)
			image_patches_normalized = image_patches - means[:, None]
			# normalize
			image_patches_normalized *= (1./255.)

			return dict(
				image_patches_normalized = image_patches_normalized,
				image_patches_means = means,
			)
		else:
			return dict(
				image_patches_normalized=image_patches,
				image_patches_means = None,
			)


	def post_denormalize_reconstruction(self, reconstructed_patches_normalized, image_patches_means, **_):
		if self.local_normalize:
			return reconstructed_patches_normalized * 255.  + image_patches_means[:, None]
		else:
			return reconstructed_patches_normalized


	def reconstruct_normalized(self, image_patches_normalized, **_):
		encoded = torch.sigmoid(self.linear_layer(image_patches_normalized))
		decoded = functional.linear(encoded, self.linear_layer.weight.t(), self.bias_visible)

		#decoded *= 0

		#print('decoded range', torch.min(decoded), torch.max(decoded))

		discrepancy = torch.abs(image_patches_normalized-decoded)

		# pool
		discrepancy[:, :] = torch.mean(discrepancy, dim=1)[:, None]


		return dict(
			reconstructed_patches_normalized = decoded,
			discrepancy_flat = discrepancy,
		)

	def forward(self, image_patches, **_):
		"""
		#:param image_patches: [N x 192]
		#:param image_patches: [N x 3 x Ph x Pw]
		:return:
		"""

		#orig_shape = image_patches.shape
		bs =  image_patches.shape[0]
		normalized = self.pre_normalize_and_flatten(image_patches)

		result = self.reconstruct_normalized(normalized['image_patches_normalized'])

		# sum discrepancy along color dimension
		#discrepancy_patches = torch.sum(result['discrepancy_flat'].reshape(orig_shape), dim=3)

		# sum along the color dimension
		discrepancy_patches = torch.sum(result['discrepancy_flat'].reshape((bs, 3, -1)), dim=1).reshape((bs, -1)) # without the last color dimension

		# remap the intensity level of [0, mean] to [0, 1] so as to
		mn = torch.mean(discrepancy_patches)
		# print('discrepancy mean', mn)
		# print('disc range', torch.min(discrepancy_patches), torch.max(discrepancy_patches))

		# discrepancy_patches *= (2./ mn )
		# discrepancy_patches = torch.clamp(discrepancy_patches, 0., 1.)

		discrepancy_patches = mn / discrepancy_patches
		discrepancy_patches = torch.clamp(discrepancy_patches, 0., 1.)
		discrepancy_patches = 1. - discrepancy_patches


		reconstructed_patches = self.post_denormalize_reconstruction(result['reconstructed_patches_normalized'], **normalized)
		reconstructed_patches = torch.clamp(reconstructed_patches, 0., 255.)

		# print('recp range', torch.min(reconstructed_patches), torch.max(reconstructed_patches))


		return dict(
			discrepancy_patches = discrepancy_patches,
			reconstructed_patches = reconstructed_patches,
		)

	def loss(self, image_patches, noise_var, **_):

		normalized = self.pre_normalize_and_flatten(image_patches)

		normalized_patches = normalized['image_patches_normalized']

		if self.training:
			normalized_patches_noised = normalized_patches + torch.normal(torch.zeros_like(image_patches), noise_var)
		else:
			normalized_patches_noised = normalized_patches

		result = self.reconstruct_normalized(normalized_patches_noised)

		loss = self.loss_mod(normalized_patches, result['reconstructed_patches_normalized'])

		return dict(
			loss = loss,
		)
