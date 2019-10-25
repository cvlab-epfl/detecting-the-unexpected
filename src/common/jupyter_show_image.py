
import numpy as np
import PIL.Image
from matplotlib import cm
from io import BytesIO
from binascii import b2a_base64
from IPython.display import display_html

def adapt_img_data(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed()):
	num_dims = img_data.shape.__len__()

	if num_dims == 3:
		# if img_data.shape[2] > 3:
		# 	img_data = img_data[:, :, :3]

		if img_data.dtype != np.uint8:
			if np.max(img_data) < 1.1:
				img_data = img_data * 255
			img_data = img_data.astype(np.uint8)

	elif num_dims == 2:
		if img_data.dtype == np.bool:
			img_data = img_data.astype(np.uint8)*255
			#c = 'png'

		else:
			vmax = np.max(img_data)
			if img_data.dtype == np.uint8 and vmax == 1:
				img_data = img_data * 255

			else:
				vmin = np.min(img_data)

				if vmin >= 0:
					img_data = (img_data - vmin) * (1 / (vmax - vmin))
					img_data = cmap_pos(img_data, bytes=True)[:, :, :3]

				else:
					vrange = max(-vmin, vmax)
					img_data = img_data / (2 * vrange) + 0.5
					img_data = cmap_div(img_data, bytes=True)[:, :, :3]

	return img_data 


class ImageHTML:
	CONTENT_TMPL = """<div style="width:100%;"><img src="data:image/{fmt};base64,{data}" /></div>"""
	
	def __init__(self, image, fmt='webp', adapt=True):
		self.fmt = fmt
		image = adapt_img_data(image) if adapt else image
		self.data_base64 = self.encode_image(image, fmt)
		
	@staticmethod
	def encode_image(image, fmt):
		with BytesIO() as buffer:
			PIL.Image.fromarray(image).save(buffer, format=fmt)
			image_base64 = str(b2a_base64(buffer.getvalue()), 'utf8')
		return image_base64
		
	def _repr_html_(self):
		return self.CONTENT_TMPL.format(fmt=self.fmt, data=self.data_base64)

	def show(self):
		display_html(self)
	

class ImageGridHTML:
	ROW_START = """<div style="display:flex; justify-content: space-evenly;">"""
	ROW_END = """</div>"""
	
	def __init__(self, *rows, fmt='webp', adapt=True):
		"""
		`show(img_1, img_2)` will draw each image on a separate row
		`show([img_1, img_2])` will draw both images in one row
		`show([img_1, img_2], [img_3, img_4])` will draw two rows

		@param fmt: image format, usually png jpeg webp
		@param adapt: whether to try converting unusual shapes and datatypes to the needed RGB
		"""
		self.fmt = fmt
		self.adapt = adapt
		self.rows = [self.encode_row(r) for r in rows]
		
	def encode_row(self, row):
		if isinstance(row, (list, tuple)):
			return [ImageHTML(img, fmt=self.fmt, adapt=self.adapt) for img in row if img is not None]
		elif row is None:
			return []
		else:
			return [ImageHTML(row, fmt=self.fmt, adapt=self.adapt)]
	
	def _repr_html_(self):
		fragments = []
		
		for row in self.rows:
			fragments.append(self.ROW_START)
			fragments += [img._repr_html_() for img in row]
			fragments.append(self.ROW_END)
		
		return '\n'.join(fragments)
	
	def show(self):
		display_html(self)

	@staticmethod
	def show_image(*images, **options):
		"""
		`show(img_1, img_2)` will draw each image on a separate row
		`show([img_1, img_2])` will draw both images in one row
		`show([img_1, img_2], [img_3, img_4])` will draw two rows

		@param fmt: image format, usually png jpeg webp
		@param adapt: whether to try converting unusual shapes and datatypes to the needed RGB
		"""
		ImageGridHTML(*images, **options).show()

show = ImageGridHTML.show_image
