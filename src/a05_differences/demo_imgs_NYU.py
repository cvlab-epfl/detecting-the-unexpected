from .demo_imgs import *
from ..datasets.NYU_depth_v2 import NYUD_LabelInfo_Category40

def demo_chans_nyu(sem_cat):
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

		'demo_pipeline': ChannelLoaderImage(tmpl_fused, suffix='pipeline', img_ext='.jpg'),
		'demo_scores': ChannelLoaderImage(tmpl_fused, suffix='scores', img_ext='.jpg'),
	}


label_colorizer_nyu_tr = SemSegLabelsToColorImg(
	colors_by_classid=NYUD_LabelInfo_Category40.colors, 
)
label_colorizer_nyu = lambda labels: label_colorizer_nyu_tr.forward('', labels)


def tr_demo_imgs_nyu(fid, image, anomaly_gt, pred_labels_trainIds, gen_image, anomaly_ours, anomaly_uncertainty,  blend_image=True, **_):

	_, contour_list, _ = cv2.findContours(
		anomaly_gt.astype(np.uint8),
		cv2.RETR_LIST,
		cv2.CHAIN_APPROX_SIMPLE,
	)

	# fix gen image
	if gen_image.shape != image.shape:
		# loc = locals()
		# print(loc.keys())
		# print(' | '.join(f'{name}: {loc[name].shape}' for name in ['image', 'gen_image', 'pred_labels_trainIds', 'anomaly_ours']))
		gen_image = gen_image[:image.shape[0], :image.shape[1]]
		#print('gen_img shape', gen_image.shape)

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
	#anomaly_rbm_color = score_colorizer(anomaly_rbm)

	return {
		'fid_no_subdir': str(fid).replace('/', '__'),

		'demo_image': image,
		'demo_gt_contour': img_with_gt_contours,
		'demo_pred_labels': pred_labels_colorimg,
		'demo_gen_image': gen_image,
		'demo_anomaly_uncertainty': anomaly_uncertainty_color,
		'demo_anomaly_ours': anomaly_ours_color,
		#'demo_anomaly_rbm': anomaly_rbm_color ,

		'demo_pipeline': image_grid_Nx2([
			[image, pred_labels_colorimg],
			[gen_image, anomaly_ours_color],
		]),
		'demo_scores': image_grid_Nx2([
			[img_with_gt_contours, anomaly_ours_color],
			[anomaly_uncertainty_color, np.zeros(img_with_gt_contours.shape, dtype=np.uint8)],
		]),
	}


def demo_pipeline_prepare_nyu(sem_cat, dset, baseline_uncertainty_name, reload_dset=True, roi_field=None, roi_transform=None):
	demo_chs = demo_chans_nyu(sem_cat)
	demo_chs_names = list(demo_chs.keys())

	dset.add_channels(**demo_chs)
	#dset.channel_disable(*demo_chs_names)

	chan_anomaly_unc = f'anomaly_{baseline_uncertainty_name}'
	chan_anomaly_ours = 'anomaly_lag_swap_gt'
	
	dset.set_channels_enabled(
		'image', 'labels_category40', 'pred_labels_trainIds', 'gen_image',
		chan_anomaly_ours, chan_anomaly_unc,
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
				'anomaly_uncertainty', 'anomaly_ours',
			])
		)

	demo_out_pipeline = TrsChain(*[
		TrRename({
			chan_anomaly_unc: 'anomaly_uncertainty',
			chan_anomaly_ours: 'anomaly_ours',
		}),
		partial(tr_rescale_uncertainty, unc_scale),
		] + roi_trs + [
		tr_demo_imgs_nyu,
		TrSaveChannelsAutoDset(demo_chs_names),
	])

	return demo_out_pipeline
