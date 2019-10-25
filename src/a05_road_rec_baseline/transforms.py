import numpy as np
import torch
from torch.nn import functional
from ..pipeline.transforms import TrByField
from ..datasets.cityscapes import CityscapesLabelInfo

# ROAD_TRAIN_IDS = [CityscapesLabelInfo.name2label[n].trainId for n in ['road', 'sidewalk']]
ROAD_TRAIN_IDS = [CityscapesLabelInfo.name2label[n].trainId for n in ['road']]


def patches_check_road(patches_label, road_train_ids = ROAD_TRAIN_IDS):
	if isinstance(patches_label, torch.Tensor):
		patches_label = patches_label.numpy()

	#patches_label = patches_label.reshape((patches_label.shape[0], -1))

	patches_are_road = patches_label == road_train_ids[0]
	for c in road_train_ids[1:]:
		patches_are_road |= patches_label == c

	mask_road = np.all(patches_are_road, axis=1)
	mask_not_road = np.all(~patches_are_road, axis=1)

	return np.where(mask_road)[0], np.where(mask_not_road)[0]

def extract_square_patches(image, patch_size=8, stride=6):
	# convert to dims: [1 chans W H]

	# there is no line here!!!

	if image.shape.__len__() > 2:
		image = image.transpose(2, 0, 1)
	else:
		image = image[None, :] # fake 1-sized channel

	img_torch = torch.from_numpy(image.astype(np.float32))[None, :]
	patches = functional.unfold(img_torch, kernel_size=(patch_size, patch_size), stride=stride)
	return patches[0].permute(1, 0)

# def extract_square_patches(image, patch_size=8, stride=6):
# 	img = torch.from_numpy(image)
# 	extra_channels = img.shape[2:]
# 	patches_2d = img.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
# 	patches_2d = patches_2d.reshape((-1,) + patches_2d.shape[2:]) # collapse 2 first dimensions
# 	# out = [Patch, H, W, channels]
# 	patches_2d = patches_2d.numpy().transpose((0, -2, -1) + tuple(range(1, patches_2d.shape.__len__()-2)))
# 	return patches_2d


# def extract_square_patches(image, patch_size=50, stride=25):
#   # extract_patches_2d is broken because it does a random permutation of the patches
# 	img_size = np.array(image.shape[:2])
# 	num_patches = np.prod((img_size - patch_size // 2) // stride)
# 	return extract_patches_2d(image, patch_size=(patch_size, patch_size), max_patches=num_patches, random_state=0)

def tr_extract_patches_all(image, patch_size=8, stride=6, **_):
	return dict(
		image_patches = extract_square_patches(image, patch_size=patch_size, stride=stride)
	)

def tr_extract_patches_road(image, labels, patch_size=8, stride=6, **_):
	patches_img = extract_square_patches(image, patch_size=patch_size, stride=stride)
	patches_labels = extract_square_patches(labels, patch_size=patch_size, stride=stride)

	idx_road, idx_not_road = patches_check_road(patches_labels)

	return dict(
		patches_road=patches_img[idx_road],
		patches_not_road=patches_img[idx_not_road],
	)


class TrRebuildFromPatches(TrByField):

	OVERLAP_INV_WEIGHTS_CACHE = dict()

	@classmethod
	def calculate_overlap_inv_weights(cls, patch_size, stride, image_h_w):
		img = torch.ones((1, 1) + image_h_w)
		patches = functional.unfold(img, kernel_size=(patch_size, patch_size), stride=stride)
		overlap = functional.fold(patches, output_size=image_h_w, kernel_size=(patch_size, patch_size), stride=stride)
		#overlap[overlap == 0] = 1 #otherwise we get NANs

		overlap_inv =  1. / overlap
		overlap_inv[overlap == 0] = 0.

		return overlap_inv

	@classmethod
	def get_overlap_inv_weights(cls, patch_size, stride, image_h_w):

		key = (patch_size, stride, image_h_w)

		weights = cls.OVERLAP_INV_WEIGHTS_CACHE.get(key, None)
		if weights is None:
			weights = cls.calculate_overlap_inv_weights(patch_size, stride, image_h_w)
			cls.OVERLAP_INV_WEIGHTS_CACHE[key] = weights

		return weights

	def __init__(self,
			fields=dict(reconstructed_patches = 'reconstructed', discrepancy_patches = 'discrepancy'),
			patch_size=8,
	        stride=6,
	    ):
		super().__init__(fields)

		self.patch_size = patch_size
		self.kernel_size = (patch_size, patch_size)
		self.stride = stride


	def forward(self, patches, h_w):
		if isinstance(patches, np.ndarray):
			patches = torch.from_numpy(patches)

		img_reconstr_torch = functional.fold(
			patches.permute(1, 0)[None, :],
			output_size=h_w,
			kernel_size=self.kernel_size,
			stride=self.stride,
		)
		img_reconstr_torch *= self.get_overlap_inv_weights(self.patch_size, self.stride, h_w).to(device=img_reconstr_torch.device)
		img_reconstr_torch = img_reconstr_torch[0].permute(1, 2, 0).squeeze()

		return img_reconstr_torch

	def __call__(self, image, **fields):

		h_w = image.shape[:2]

		return {
			fi_out: self.forward(fields[fi_in], h_w)
			for(fi_in, fi_out) in self.field_pairs
		}

