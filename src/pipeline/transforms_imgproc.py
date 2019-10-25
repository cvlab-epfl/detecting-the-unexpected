import logging
log = logging.getLogger('exp')
import numpy as np
import imageio, os
from pathlib import Path
from .frame import *
from .transforms import TrBase, TrByField
from ..common.util import image_grid_Nx2
from ..common.jupyter_show_image import show

try:
	import cv2 as cv
except Exception as e:
	log.warning('OpenCV import failed:', e)

IMG_MEAN_DEFAULT = np.array([0.485, 0.456, 0.406])[None, None, :]
IMG_STD_DEFAULT = np.array([0.229, 0.224, 0.225])[None, None, :]

IMG_MEAN_HALF = np.array([0.5, 0.5, 0.5])[None, None, :]
IMG_STD_HALF = np.array([0.5, 0.5, 0.5])[None, None, :]

def zero_center_img(image, means = IMG_MEAN_DEFAULT, stds = IMG_STD_DEFAULT):
	img_float = image.astype(np.float32) #this also allocated new array
	img_float *= (1./255.)
	img_float -= means
	img_float /= stds
	return img_float

def zero_center_img_undo(image, means = IMG_MEAN_DEFAULT, stds = IMG_STD_DEFAULT):
	img_new = image * stds #this also allocated new array
	img_new += means
	img_new *= 255
	return img_new.astype(np.uint8)

class ZeroCenterBase(TrByField):
	def __init__(self, fields='*', means = IMG_MEAN_DEFAULT, stds = IMG_STD_DEFAULT):
		super().__init__(fields=fields)
		self.means = np.array(means, dtype=np.float32).reshape(-1)[None, None, :]
		self.stds = np.array(stds, dtype=np.float32).reshape(-1)[None, None, :]

	def func(self, image):
		raise NotImplementedError()

	def forward(self, field_name, value):
		if isinstance(value, np.ndarray):
			# check if it is not already 8bit and that shape is [H, W, 3]
			if value.shape.__len__() == 3 and value.shape[2] == 3:
				return self.func(value)
			else:
				return value
		else:
			self.conditionally_complain_about_type(field_name, value, 'np.ndarray')
			return value

class TrZeroCenterImgs(ZeroCenterBase):
	def func(self, image):
		return zero_center_img(image, self.means, self.stds)

class TrZeroCenterImgsUndo(ZeroCenterBase):
	def func(self, image):
		# if its 8bit, it can't be zero centered
		if image.dtype != np.uint8:
			return zero_center_img_undo(image, self.means, self.stds)
		else:
			return image

def blend_img(ia, ib, w=0.5):
	return (ia*(1-w) + ib*w).astype(np.uint8)

def make_demo_grid_simple(imgs, captions=None, downsample=1):

	img_sz = np.array(imgs[0].shape[:2])

	num_ds = 0
	if downsample != 1:
		img_sz //= downsample
		num_ds = int(np.log2(downsample))

	grid = np.zeros((img_sz[0]*2, img_sz[1]*2, 3), dtype=np.uint8)

	top_left_corners = np.array([
		[0, 0],
		[0, img_sz[1]],
		[img_sz[0], 0],
		img_sz,
	], dtype=np.int)

	for img, pos in zip(imgs, top_left_corners):
		for idx in range(num_ds):
			img = cv.pyrDown(img)

		if img.shape.__len__() == 2:
			img = img[:, :, None]
		grid[pos[0]:pos[0]+img_sz[0], pos[1]:pos[1]+img_sz[1]] = img

	if captions:
		for txt, pos in zip(captions, top_left_corners + np.array([32, 32])[None, :]):
			cv.putText(grid, txt, tuple(pos[::-1]),
				cv.FONT_HERSHEY_SIMPLEX, 1.5, (200, 150, 30), 2,
			)

	return grid

class TrRandomlyFlipHorizontal(TrBase):
	""" For Numpy images """

	def __init__(self, fields = ['image', 'labels']):
		self.fields = fields

	def __call__(self, **frame_values):
		if np.random.random() > 0.5:
			return {
				fi: frame_values[fi][:, ::-1].copy()
				# .copy to prevent this pytorch error:
				# ValueError: some of the strides of a given numpy array are negative. This is currently not supported, but will be added in future releases.
				for fi in self.fields
			}
	
	def __repr__(self):
		return self.repr_with_args(self.object_name(self), self.fields)

class TrRandomCrop(TrBase):
	""" For Numpy images, example:

		rc = TrRandomCrop([256, 512], ['image', 'labels'])
		si = TrsChain([
			SemSegLabelsToColorImg(fields=[('labels', 'labels_colorimg')]),
			TrShow(['image', 'labels_colorimg'])
		])

		dset_bdd_val[0].apply(TrsChain(
			si, tr_print,
			rc,
			si,	tr_print,
		))
	"""

	def __init__(self, crop_size = [512, 1024], fields = ['image', 'labels'], b_flip_horizontal=False):
		self.fields = fields
		self.crop_size = np.array(crop_size, dtype=np.int)

	#def apply(top_left, bot_right, b_flip, image):


	def __call__(self, **frame_values):

		shape = np.array(frame_values[self.fields[0]].shape[:2])

		if np.any(shape < self.crop_size):
			raise ValueError('Image size is {sh} but cropping requested to size {cr}'.format(sh=shape, cr=self.crop_size))

		space = shape - self.crop_size
		#try:
		top_left = np.array([np.random.randint(0, space[i]) for i in range(2)])
		#except ValueError:
		#	print('space {s}'.format(s=space))

		bot_right = top_left + self.crop_size

		return {
			fi: frame_values[fi][top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
			for fi in self.fields
		}

class TrSaveImages(TrBase):
	"""
	Example
	```
	TrSaveImages(
		dict(gen_image = 'laf_reconstr.jpg', image = 'laf_image.jpg', labels_colorimg = 'laf_labels.png'),
		out_dir = 'results/pix2pixHD_basic_1024p/laf_gen',
	))
	```
	"""
	def __init__(self, field_to_filename, out_dir='.'):
		self.field_to_filename = field_to_filename
		self.out_dir = out_dir

	def __call__(self, **fields):
		os.makedirs(self.out_dir, exist_ok=True)

		for field, filename in self.field_to_filename.items():
			path = os.path.join(self.out_dir, filename)
			imageio.imwrite(path, fields[field])


class TrSaveImagesTmpl(TrBase):
	"""
	Example
	```
	TrSaveImages(
		dict(gen_image = 'laf_reconstr.jpg', image = 'laf_image.jpg', labels_colorimg = 'laf_labels.png'),
		out_dir = 'results/pix2pixHD_basic_1024p/laf_gen',
	))
	```
	"""
	def __init__(self, field_to_filename_tmpl, out_dir='.'):
		self.field_to_filename = field_to_filename_tmpl
		self.out_dir = Path(out_dir)

	def __call__(self, **fields):
		os.makedirs(self.out_dir, exist_ok=True)

		for field, filename in self.field_to_filename.items():
			path = self.out_dir / filename.format(
				fid = fields['fid'], 
				frame = fields['frame'],
				fid_no_slash = str(fields['fid']).replace('/', '__'),
			)
			path.parent.mkdir(exist_ok=True, parents=True)
			imageio.imwrite(path, fields[field])


class TrShow(TrBase):
	def __init__(self, *channel_names):
		self.channel_names = channel_names

	@classmethod
	def retrieve_channel_values(cls, fields, name_or_names):
		if isinstance(name_or_names, str):
			return fields.get(name_or_names, None)
		elif name_or_names is None:
			return None
		else:
			return list(map(partial(cls.retrieve_channel_values, fields),  name_or_names))

	def __call__(self, **fields):
		show(*self.retrieve_channel_values(fields, self.channel_names))


class TrImgGrid(TrShow):
	def __init__(self, channel_names, b_show=False, save=None):
		super().__init__(*channel_names)
		self.b_show = b_show
		self.save = save

	def __call__(self, **fields):
		imgs = self.retrieve_channel_values(fields, self.channel_names)
		grid = image_grid_Nx2(imgs)

		if self.b_show:
			show(grid)

		if self.save:
			if isinstance(self.save, str):
				save_path = self.save
			else:
				save_path = self.save(**fields)

			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			imageio.imwrite(save_path, grid)

		return dict(
			grid = grid,
		)




