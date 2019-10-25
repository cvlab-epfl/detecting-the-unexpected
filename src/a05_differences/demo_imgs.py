import numpy as np
import cv2
from os.path import join as pp
from matplotlib import cm
from functools import partial

from ..common.util import image_grid_Nx2
from ..pipeline.transforms import TrsChain, TrRename, TrKeepFields
from ..a01_sem_seg.transforms import SemSegLabelsToColorImg
from ..datasets.dataset import ChannelLoaderImage, TrSaveChannelsAutoDset

IMAGE_OVERLAY_ALPHA = 0.2

def demo_chans(sem_cat):
	dir_base = pp('{dset.dir_out}', 'eval_' + sem_cat, '{dset.name}_demo_{dset.split}_'+sem_cat)

	tmpl_part = pp(dir_base, 'parts', '{frame.fid_no_subdir}_{channel.suffix}{channel.img_ext}')
	tmpl_fused = pp(dir_base, '{frame.fid_no_subdir}_{channel.suffix}{channel.img_ext}')

	return {
		'demo_image': ChannelLoaderImage(tmpl_part, suffix='image', img_ext='.jpg'),
		'demo_gt_contour': ChannelLoaderImage(tmpl_part, suffix='gt_contour', img_ext='.jpg'),
		'demo_pred_labels': ChannelLoaderImage(tmpl_part, suffix='pred_labels', img_ext='.png'),
		'demo_gen_image': ChannelLoaderImage(tmpl_part, suffix='gen_image', img_ext='.jpg'),
		'demo_anomaly_uncertainty': ChannelLoaderImage(tmpl_part, suffix='anomaly_uncertainty', img_ext='.jpg'),
		'demo_anomaly_ours': ChannelLoaderImage(tmpl_part, suffix='anomaly_ours', img_ext='.jpg'),
		'demo_anomaly_rbm': ChannelLoaderImage(tmpl_part, suffix='anomaly_rbm', img_ext='.jpg'),

		'demo_pipeline': ChannelLoaderImage(tmpl_fused, suffix='pipeline', img_ext='.jpg'),
		'demo_scores': ChannelLoaderImage(tmpl_fused, suffix='scores', img_ext='.jpg'),
	}


label_colorizer_tr = SemSegLabelsToColorImg()
label_colorizer = lambda labels: label_colorizer_tr.forward('', labels)


def score_colorize(score, cmap = cm.get_cmap('magma'), image=None):
	score_colorized = cmap(score, bytes=True)[:, :, :3]

	if image is not None and IMAGE_OVERLAY_ALPHA > 0:
		score_colorized = cv2.addWeighted(score_colorized, 1-IMAGE_OVERLAY_ALPHA, image, IMAGE_OVERLAY_ALPHA, 0.0)

	return score_colorized


def tr_demo_imgs(fid, image, anomaly_gt, pred_labels_trainIds, gen_image, anomaly_ours, anomaly_uncertainty, anomaly_rbm, blend_image=True, **_):

	_, contour_list, _ = cv2.findContours(
		anomaly_gt.astype(np.uint8),
		cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE,
	)

	img_with_gt_contours = cv2.drawContours(
		image.copy(), contour_list, -1, (0, 255, 0), 2,
	)

	pred_labels_colorimg = label_colorizer(pred_labels_trainIds)

	if blend_image:
		score_colorizer = partial(score_colorize, image=image)
	else:
		score_colorizer = score_colorize

	anomaly_uncertainty_color = score_colorizer(anomaly_uncertainty)
	anomaly_ours_color = score_colorizer(anomaly_ours)
	anomaly_rbm_color = score_colorizer(anomaly_rbm)

	return {
		'fid_no_subdir': str(fid).replace('/', '__'),

		'demo_image': image,
		'demo_gt_contour': img_with_gt_contours,
		'demo_pred_labels': pred_labels_colorimg,
		'demo_gen_image': gen_image,
		'demo_anomaly_uncertainty': anomaly_uncertainty_color,
		'demo_anomaly_ours': anomaly_ours_color,
		'demo_anomaly_rbm': anomaly_rbm_color ,

		'demo_pipeline': image_grid_Nx2([
			[image, pred_labels_colorimg],
			[gen_image, anomaly_ours_color],
		]),
		'demo_scores': image_grid_Nx2([
			[img_with_gt_contours, anomaly_ours_color],
			[anomaly_uncertainty_color, anomaly_rbm_color],
		]),
	}

def _demo_get_uncertainty_bounds_for_plots_worker(unc_name, **fields):
	value = fields[f'anomaly_{unc_name}']

	percentiles = np.percentile(value, [0.05, 0.5, 0.95])

	return {
		f'percentiles_{unc_name}': percentiles,
	}


def demo_get_uncertainty_bounds_for_plots(dset, baseline_uncertainty_name):

	# tr = TrsChain([
	# 	partial(_demo_get_uncertainty_bounds_for_plots_worker, unc_name=baseline_uncertainty_name)
	# 	TrKeepFields([f'percentiles_{baseline_uncertainty_name}'])
	# ])

	anomaly_field = f'anomaly_{baseline_uncertainty_name}'
	anomaly_concat = np.concatenate([f[anomaly_field] for f in dset])
	anomaly_percentile = np.percentile(anomaly_concat, 95)


	return anomaly_percentile


# def tr_rescale_uncertainty(unc_name, factor, **fields):
#
# 	unc_fn = f'anomaly_{unc_name}'
#
# 	return {
# 		unc_fn: fields(unc_fn) * factor,
# 	}

def tr_rescale_uncertainty(factor, anomaly_uncertainty, **_):
	return {
		'anomaly_uncertainty': anomaly_uncertainty*factor
	}

def demo_tr_apply_roi(roi_field, score_fields, **fields):
	roi_val = fields[roi_field]

	return {
		f: fields[f] * roi_val
		for f in score_fields
	}

def tr_rbm_remove_extra_channel(anomaly_rbm, **_):

	if anomaly_rbm.shape.__len__() > 2:
		return dict(
			anomaly_rbm = anomaly_rbm[:, :, 0]
		)
	else:
		return dict()

def demo_pipeline_prepare(sem_cat, dset, baseline_uncertainty_name, reload_dset=True, roi_field=None, roi_transform=None):
	demo_chs = demo_chans(sem_cat)
	demo_chs_names = list(demo_chs.keys())

	dset.add_channels(**demo_chs)
	#dset.channel_disable(*demo_chs_names)

	chan_anomaly_unc = f'anomaly_{baseline_uncertainty_name}'
	chan_anomaly_ours = 'anomaly_lag_swap_gt'
	chan_anomaly_rbm = 'anomaly_rbm'

	dset.set_channels_enabled(
		'image', 'labels_source', 'pred_labels_trainIds', 'gen_image',
		chan_anomaly_ours, chan_anomaly_unc, chan_anomaly_rbm,
	)
	if reload_dset:
		dset.discover()

	unc_95_percentile = demo_get_uncertainty_bounds_for_plots(dset, baseline_uncertainty_name)
	unc_scale = 1. / unc_95_percentile

	print(baseline_uncertainty_name, '- uncertainty scale:', unc_scale)

	roi_trs = []

	if roi_field is not None:

		if roi_transform is not None:
			roi_trs.append(roi_transform)

		roi_trs.append(
			partial(demo_tr_apply_roi, roi_field = roi_field, score_fields = [
				'anomaly_uncertainty', 'anomaly_ours', 'anomaly_rbm',
			])
		)

	demo_out_pipeline = TrsChain(*[
		TrRename({
			chan_anomaly_unc: 'anomaly_uncertainty',
			chan_anomaly_ours: 'anomaly_ours',
			chan_anomaly_rbm: 'anomaly_rbm',
		}),
		tr_rbm_remove_extra_channel,
		partial(tr_rescale_uncertainty, unc_scale),
		] + roi_trs + [
		tr_demo_imgs,
		TrSaveChannelsAutoDset(demo_chs_names),
	])

	return demo_out_pipeline


def synth_demo_chans(gen_name):
	dir_base = pp('{dset.dir_out}', gen_name, 'demo', gen_name + '{dset.split}')

	tmpl_part = pp(dir_base, 'parts', '{frame.fid_no_subdir}_{channel.suffix}{channel.img_ext}')
	tmpl_fused = pp(dir_base, '{frame.fid_no_subdir}_{channel.suffix}{channel.img_ext}')

	return {
		# 'demo_image': ChannelResultImage(f'{gen_name}/demo/parts', suffix='_image', img_ext='.jpg')
		'demo_image_contour': ChannelLoaderImage(tmpl_part, suffix='image_contour', img_ext='.jpg'),
		'demo_gen_image_contour': ChannelLoaderImage(tmpl_part, suffix='gen_image_contour', img_ext='.jpg'),
		'demo_pred_labels': ChannelLoaderImage(tmpl_part, suffix='pred_labels', img_ext='.png'),
		'demo_fake_labels': ChannelLoaderImage(tmpl_part, suffix='fake_labels', img_ext='.png'),
		'demo_synth_grid': ChannelLoaderImage(tmpl_fused, suffix='grid', img_ext='.jpg'),
	}


def tr_demo_synthetic_training_sample(fid, image, gen_image, pred_labels_trainIds, labels_fakeErr_trainIds, semseg_errors, **_):
	_, contour_list, _ = cv2.findContours(
		semseg_errors.astype(np.uint8),
		cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE,
	)

	pred_labels_colorimg = label_colorizer(pred_labels_trainIds)
	fake_labels_colorimg = label_colorizer(labels_fakeErr_trainIds)

	img_with_gt_contours = cv2.drawContours(
		image.copy(), contour_list, -1, (0, 255, 0), 2,
	)
	gen_image_with_gt_contours = cv2.drawContours(
		gen_image.copy(), contour_list, -1, (0, 255, 0), 2,
	)

	return {
		'fid_no_subdir': fid.replace('/', '__'),

		# 'demo_image': image,
		# 'demo_gen_image': gen_image,
		'demo_image_contour': img_with_gt_contours,
		'demo_gen_image_contour': gen_image_with_gt_contours,
		'demo_pred_labels': pred_labels_colorimg,
		'demo_fake_labels': fake_labels_colorimg,

		'demo_synth_grid': image_grid_Nx2([
			[img_with_gt_contours, pred_labels_colorimg],
			[gen_image, fake_labels_colorimg],
		]),
	}


def synth_demo_pipeline_prepare(gen_name, dset, reload_dset=True):
	demo_chs = synth_demo_chans(gen_name)
	demo_chs_names = list(demo_chs.keys())

	dset.add_channels(**demo_chs)
	#dset.channel_disable(*demo_chs_names)


	dset.set_channels_enabled(
		'image', 'labels_source', 'pred_labels_trainIds', 'gen_image', 'labels_fakeErr_trainIds', 'semseg_errors',
	)
	if reload_dset:
		dset.discover()

	demo_out_pipeline = TrsChain(
		tr_demo_synthetic_training_sample,
		TrSaveChannelsAutoDset(demo_chs_names),
	)

	return demo_out_pipeline


