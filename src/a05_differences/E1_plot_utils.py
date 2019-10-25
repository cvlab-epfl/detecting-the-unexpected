import numpy as np
import cv2 as cv
from ..common.jupyter_show_image import adapt_img_data
from ..pipeline.transforms import TrBase
from ..pipeline.transforms_imgproc import TrShow

import torch
def ensure_numpy_image(img):
	if isinstance(img, torch.Tensor):
		img = img.cpu().numpy().transpose([1, 2, 0])
	return img


def image_grid(imgs, num_cols=2, downsample=1):
	num_imgs = imgs.__len__()
	num_rows = int(np.ceil(num_imgs / num_cols))
	
	row_cols = np.array([num_rows, num_cols])

	img_size = np.array(imgs[0].shape[:2]) // downsample

	full_size = (num_rows * img_size[0], num_cols * img_size[1], 3)

	out = np.zeros(full_size, dtype=np.uint8)

	row_col_pos = np.array([0, 0])

	for img in imgs:
		# none means black section
		if img is not None:
			img = ensure_numpy_image(img)
			if downsample != 1:
				img = img[::downsample, ::downsample]
			img = adapt_img_data(img)

			tl = img_size * row_col_pos
			br = tl + img_size

			out[tl[0]:br[0], tl[1]:br[1]] = img
		
		row_col_pos[1] += 1
		if row_col_pos[1] >= num_cols:
			row_col_pos[0] += 1
			row_col_pos[1] = 0

	# TODO captions

	return out


class TrImgGrid(TrShow):
	def __init__(self, channel_names, out_name='demo', num_cols=2, downsample=1):
		super().__init__(*channel_names)
		self.out_name = out_name
		self.num_cols = num_cols
		self.downsample = downsample

	def __call__(self, **fields):
		imgs = self.retrieve_channel_values(fields, self.channel_names)
		grid = image_grid(imgs, num_cols = self.num_cols, downsample = self.downsample)

		return {
			self.out_name: grid,
		}

class TrBlend(TrBase):
	def __init__(self, field_a, field_b, field_out, alpha_a=0.8):
		self.field_a = field_a
		self.field_b = field_b
		self.field_out = field_out
		self.alpha_a = alpha_a

	def __call__(self, **fields):
		img_a = adapt_img_data(fields[self.field_a])
		img_b = adapt_img_data(fields[self.field_b])
		img_blend = cv.addWeighted(img_a, self.alpha_a, img_b, 1-self.alpha_a, 0.0)

		return {
			self.field_out: img_blend,
		}

def tr_draw_anomaly_contour(image, anomaly_gt, **_):
	contour_list, _ = cv.findContours(
		anomaly_gt.astype(np.uint8),
		cv.RETR_LIST,
		cv.CHAIN_APPROX_TC89_KCOS,
	)

	img_with_gt_contours = cv.drawContours(
		image.copy(), contour_list, -1, (0, 255, 0), 2,
	)

	return dict(
		anomaly_contour = img_with_gt_contours,
	)

