
try:
	import cv2
except Exception as e:
	print('OpenCV import fail:', e)

import numpy as np
import colorsys
from os.path import join as pp
from pathlib import Path
from functools import partial
import os, glob, datetime, math, time, gc
import h5py
from functools import partial
import multiprocessing, subprocess
from imageio import imread, imwrite

try:
	import torch
	def ensure_numpy_image(img):
		if isinstance(img, torch.Tensor):
			img = img.cpu().numpy().transpose([1, 2, 0])
		return img

except:
	def ensure_numpy_image(img):
		return img


def mat_info(a, name=None):
	if name:
		print(name, a.dtype, a.shape, np.nanmin(a), np.nanmax(a))
	else:
		print(a.dtype, a.shape, np.nanmin(a), np.nanmax(a))

def byted_to_real(data):
	return data.astype(np.float32) / 255

def real_to_byted(data):
	c = data.copy()
	c -= c.min()
	c *= 255/c.max()
	return c.astype(np.uint8)

def nth_color(color_idx, color_count, scale=255, tp=int):
	"""
		Find `color_count` unique colors along the HSV circle
	"""
	# end at 300/360 so its red --> purple,
	return tuple(tp(scale*v) for v in colorsys.hsv_to_rgb( 0.83 * color_idx / color_count, 1, 1))

def ensure_file_removed(filepath):
	if os.path.exists(filepath):
		os.remove(filepath)

def save_plot(fig, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path)
	fn, ext = os.path.splitext(path)
	if ext == '.ps':
		cmd = ['/usr/bin/ps2pdf', path, fn + '.pdf']
		#print(cmd)
		subprocess.run(cmd)

def load_image_any_ext(base_path):
	potential = glob.glob(base_path + '.*')
	if potential:
		return imread(potential[0])
	else:
		print('No image for', base_path)

def shard_array(ar, per_row):
	num_elems = len(ar)
	num_rows = math.ceil(num_elems / per_row)

	idx = 0
	rows = []

	while idx < num_elems:
		end_idx = min(idx+per_row, num_elems)
		rows.append(ar[idx:end_idx])
		idx = end_idx

	return rows


###################################################################################################
# Thread-parallel loops
###################################################################################################
from multiprocessing.dummy import Pool as thread_Pool

if not ('POOL' in globals()):
	POOL = thread_Pool(8)

def parallel_process(func, tasks, threading=True, disp_progress=True, step_size=1, pbar_class=None, ret=True):

	results = []

	if disp_progress and pbar_class:
		#from util_notebook import ProgressBar
		pbar = pbar_class(len(tasks))
	else:
		pbar = 0

	if threading:
		for result in POOL.imap(func, tasks, chunksize=step_size):
			pbar += 1
			if ret:
				results.append(result)

	else:
		for t in tasks:
			result = func(t)
			pbar += 1
			if ret:
				results.append(result)

	if ret:
		return results

###################################################################################################
# Drawing
###################################################################################################

def to_cv_pt(v):
	""" Row vector to tuple of ints, which is used as input in some OpenCV functions """
	vi = np.rint(v).astype(np.int)
	return (vi.flat[0], vi.flat[1])

def to_cv_color(c):
	return tuple(map(int, c))

def kpts_to_cv(pt_positions, pt_sizes, pt_orientations):
	""" Arrays of keypoint positions, sizes and orientations to list of cv2.KeyPoint """
	sz = pt_sizes * 2
	angs = pt_orientations * (180/np.pi)

	return [
		cv2.KeyPoint(pt_positions[idx, 0], pt_positions[idx, 1], sz[idx], angs[idx])
		for idx in range(pt_positions.shape[0])
	]

def matches_to_cv(pair_array):
	""" Nx2 int array to list of cv2.DMatch """
	return [
		cv2.DMatch(row[0], row[1], 0)
		for row in pair_array
	]

def draw_keypoints(photo, pt_positions, pt_sizes, pt_orientations, color=None):
	""" Draw keypoints, with sizes and orientations , on an image """

	pt_objs = kpts_to_cv(pt_positions, pt_sizes, pt_orientations)
	return cv2.drawKeypoints(photo, pt_objs, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=color)

def frame_draw_kpt_subset(frame, indices, img = None, color=(0, 255, 0)):
	return draw_keypoints(
		img if img is not None else frame.photo,
		frame.kpt_locs[indices, :],
		frame.kpt_sizes[indices, :],
		frame.kpt_orientations[indices, :],
		color=color
	)

def draw_keypoint_matches(frame_src, frame_dest, pairs):

	pts_cv_src = kpts_to_cv(frame_src.kpt_locs, frame_src.kpt_sizes, frame_src.kpt_orientations)
	pts_cv_dest = kpts_to_cv(frame_dest.kpt_locs, frame_dest.kpt_sizes, frame_dest.kpt_orientations)
	matches_cv = matches_to_cv(pairs)

	return cv2.drawMatches(
		frame_src.photo, pts_cv_src, frame_dest.photo, pts_cv_dest, matches_cv, None,
		#matchColor = (0, 255, 0),
		#singlePointColor = (25, 50, 255),
		flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
	)


def draw_contour_around_mask(mask, canvas, cnt_color, cnt_thickness, overlay_opacity=0, overlay_color=None):
	color_ocv = tuple(map(int, cnt_color))
	_, contours, __ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	canvas = cv2.drawContours(canvas, contours, -1, color_ocv, cnt_thickness)

	if overlay_opacity > 0:
		color_overlay_ocv = tuple(map(int, overlay_color)) if overlay_color is not None else color_ocv
		overlay = np.zeros(canvas.shape, canvas.dtype)
		overlay = cv2.drawContours(overlay, contours, -1, color_overlay_ocv, -1)
		canvas = cv2.addWeighted(overlay, overlay_opacity, canbas, 1-overlay_opacity)

	return canvas


###################################################################################################
# Files
###################################################################################################

import natsort

def discover_files(directory):
	fns = os.listdir(directory)
	fns = natsort.natsorted(fns)
	return [pp(directory, fn) for fn in fns]

def index_list(lst, indices):
	return [lst[i] for i in indices]


##### VIDEO

def sparse_sample_video(vidfile, stride=1):
	vr = cv2.VideoCapture(vidfile)
	fridx = 0
	proceed = True
	frs = []

	proceed, fr = vr.read()

	while proceed:
		if fridx % stride == 0:
			frs.append(fr[:, :, ::-1])

		proceed, fr = vr.read()

	return np.stack(frs)

# Context manager for changing directory
# http://ralsina.me/weblog/posts/BB963.html

from contextlib import contextmanager
from pathlib import Path
import os
@contextmanager
def current_dir(new_cwd):
	old_cwd = Path.cwd()
	os.chdir(new_cwd)
	try:
		yield
	finally:
		os.chdir(old_cwd)

from matplotlib import cm

def img_convert_to_displayable(img_data, cmap_pos=cm.get_cmap('magma'), cmap_div=cm.get_cmap('Spectral').reversed()):
	num_dims = img_data.shape.__len__()

	if num_dims == 3:
		if img_data.shape[2] > 3:
			img_data = img_data[:, :, :3]

		if img_data.dtype != np.uint8 and np.max(img_data) < 1.1:
			img_data = (img_data * 255).astype(np.uint8)


	elif num_dims == 2:
		if img_data.dtype == np.bool:

			img_data = img_data.astype(np.uint8)*255

			img_data = np.stack([img_data]*3, axis=2)

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

# Image grid

def image_grid_2x2(imgs):
	sh = np.array(imgs[0].shape[:2])


	out = np.zeros((*(sh*2), 3), dtype=np.uint8)
	grid = np.array([
		[0, 0], [0, 1],
		[1, 0], [1, 1],
	])

	for img, g in zip(imgs, grid):
		tl = g*sh
		br = tl+sh

		if img is not None:
			out[tl[0]:br[0], tl[1]:br[1]] = img

	return out

def image_grid_Nx2(imgs):
	num_rows = imgs.__len__()
	sh = np.array(imgs[0][0].shape[:2])
	shh = sh // 2

	# print(f'img grid sh: {sh} sh-half: {shh}')

	h, w = sh
	hh, wh = shh

	# out = np.zeros((*(sh * 2), 3), dtype=np.uint8)
	out = np.zeros((hh * num_rows, w, 3), dtype=np.uint8)

	for rid, row in enumerate(imgs):
		for cid, col in enumerate(row):
			tl = np.array([hh*rid, wh*cid])
			br = tl + shh

			if col is not None:
				# half of img
				disp = img_convert_to_displayable(col[:-1:2, :-1:2])
				out[tl[0]:br[0], tl[1]:br[1]] = disp
				
	return out

#def hdf5_save_group(group, data_dict):


def hdf5_save(path, data):

	with h5py.File(path, 'w') as fout:
		for name, value in data.items():

			if isinstance(value, np.ndarray) and np.prod(value.shape) > 1 and value.dtype == np.bool:
				fout.create_dataset(name, data=value, compression=7)
			else:
				fout[name] = value

def hdf5_load(path):
	with h5py.File(path, 'r') as fin:

		result = dict()

		def visit(name, obj):
			if isinstance(obj, h5py.Dataset):
				result[name] = obj[()]

		fin.visititems(visit)

		return result

# def repackage_to_hdf5(npz_path, hdf5_path):
# 	with np.load(npz_path, 'r') as fin:
# 		data = dict(fin)
#
# 	with h5py.File(hdf5_path, 'w') as fout:
# 		for name, value in data.items():
# 			fout[name] = value
#
# 	p, ext = os.path.splitext(hdf5_path)
# 	with h5py.File(p + '_comp' + ext, 'w') as fout:
# 		for name, value in data.items():
# 			if isinstance(value, np.ndarray) and np.prod(value.shape) > 1:
# 				fout.create_dataset(name, data=value, compression=7)
# 			else:
# 				fout[name] = value
#
#
# repackage_to_hdf5(DIR_DATA / 'lost_and_found/rocs_fakeErr_LAFref/diff_roc.npz',
# 				  DIR_DATA / 'lost_and_found/rocs_fakeErr_LAFref/diff_roc.hdf5')
