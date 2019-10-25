
BUS_ROC_FIELDS = [
	('error', 'semseg_errors', 'anomaly_p', ':r'),
	('bus', 'bus_mask', 'anomaly_p', '-g'),
	('error bus not car', 'semseg_errors_busiscar', 'anomaly_p', '-b'),
	('bus not car', 'bus_pred_as_nocar_mask', 'anomaly_p', '--m'),
]


def roc_calc(frames, roi, pred_field, gt_field, num_levels=512, scale=1, transform=None):
	# roi_area = np.count_nonzero(roi) if roi is not None else 1

	def f_pre(frame):

		if transform is not None:
			frame.apply(transform)

		prob = frame[pred_field]
		gt_label = frame[gt_field]

		cmats = binary_confusion_matrix(prob=prob*scale, gt_label=gt_label, roi=roi, levels=num_levels)['cmat']

		return dict(
			cmats=cmats,
		)

	results = Frame.parallel_process(f_pre, frames, ret_frames=True, n_threads=6, batch=8)
	cmats = np.array([r['cmats'] for r in results])
	# cmats indexing [frame, threshold, predicted, gt]

	tp = np.sum(cmats[:, :, 0, 0], axis=0)
	fp = np.sum(cmats[:, :, 0, 1], axis=0)

	fn = np.sum(cmats[:, :, 1, 0], axis=0)
	tn = np.sum(cmats[:, :, 1, 1], axis=0)

	tp_rates = tp / (tp+fn)
	fp_rates = fp / (fp+tn)

	return dict(
		num_levels=num_levels,
		tp_rates=tp_rates,
		fp_rates=fp_rates,
		cmats=cmats,
		roi=roi,
	)


def rocs_all(frs, outdir, roi, spec, scales=dict(), transforms=dict()):
	os.makedirs(outdir, exist_ok=True)

	fig, plott = plt.subplots(1)

	for variant in spec:
		name, gtf, predf, fmt = variant

		scale = scales.get(name, 1)

		rocinfo = roc_calc(frs, roi=roi, pred_field=predf, gt_field=gtf, transform=transforms.get(name))

		np.savez_compressed(outdir / (name + '_roc.npz'), **rocinfo)

		roc_plot_additive(rocinfo, label=name, plot=plott, fmt=fmt)

	fig.legend()
	fig.tight_layout()

	fig.savefig(outdir / 'rocs.pdf')

def rocs_calc_all_simple(frs, outdir, detector_names, roi, transform=TrsChain(), validate=False):
	os.makedirs(outdir, exist_ok=True)

	frs = Frame.frame_list_apply(transform, frs, ret_frames=True, n_threads=6, batch=8)

	if validate:
		num_all = frs.__len__()
		frs = [fr for fr in frs if fr.valid]
		print(f'Valid {frs.__len__()}/{num_all}')

	for name in detector_names:
		print(name)
		out_file_path = outdir / f'anomalyScore_{name}_roc.npz'

		if out_file_path.exists():
			print(out_file_path, 'already exists')
		else:
			rocinfo = roc_calc(frs, roi=roi, pred_field=f'anomaly_{name}', gt_field='anomaly_gt')
			rocinfo['name'] = name
			np.savez_compressed(out_file_path, **rocinfo)


def rocs_plot_all_simple(outdir, detector_names, save_as, name_to_plot_format=lambda x: None):

	spec_comparison = [
		(name, f'anomalyScore_{name}_roc.npz', name_to_plot_format(name)) for name in detector_names
	]

	# fp_rates, tp_rates, num_levels = (roc_info[k] for k in ['fp_rates', 'tp_rates', 'num_levels'])
	#
	# if plot is None:
	# 	fig, plot = plt.subplots(1)
	# 	plot.set_xlim([0, 1])
	#
	# fmt_args = []
	# fmt_kwargs = {}
	#
	# if fmt is not None:
	# 	if isinstance(fmt, str):
	# 		# one string specifier
	# 		fmt_args = [fmt]
	# 	elif isinstance(fmt, dict):
	# 		fmt_kwargs = fmt
	# 	elif isinstance(fmt, tuple):
	# 		fmt_kwargs = dict(color=fmt[0], linestyle=fmt[1])
	# 	else:
	# 		raise NotImplementedError(f"Format object {fmt}")

	# last_valid = np.searchsorted(fp_rates, 1.001)
	# print(last_valid)
	# area_under = np.trapz(tp_rates[:last_valid], fp_rates[:last_valid])
	#area_under = np.trapz(tp_rates, fp_rates)

	draw_rocs_from_files(spec_comparison, save=outdir / save_as, base_dir=outdir, max_fpr=0.3)



def laf_roc_calc(frames, roi, pred_field, gt_field, processor=None, num_levels=512):
	"""
	In LAf, the false-positive rate is defined as:
		fp / area_of_gt_freespace
	but the ROC  by default calculates
		fp / area of non-anomaly

	So we rescale:
		fp_rates = fp_rates * area_non_anomaly / area_gt_freespace
	"""


	roi_area = np.count_nonzero(roi) if roi is not None else 1

	def f_pre(frame):
		prob = frame[pred_field]
		gt_label = frame[gt_field]
		labels_source = frame['labels_source']

		if processor is not None:
			prob = processor(frame, prob)

		gt_freespace_fraction = np.count_nonzero(labels_source[roi] == 1) / roi_area
		cmats = binary_confusion_matrix(prob=prob, gt_label=gt_label, roi=roi, levels=num_levels)['cmat']

		return dict(
			cmats=cmats,
			gt_freespace_fraction=gt_freespace_fraction,
		)

	results = Frame.parallel_process(f_pre, frames, ret_frames=True, n_threads=6, batch=8)
	cmats = np.array([r['cmats'] for r in results])
	gt_freespace_fractions = np.array([r['gt_freespace_fraction'] for r in results])

	# cmats indexing [frame, threshold, predicted, gt]

	tp = np.sum(cmats[:, :, 0, 0], axis=0)
	fp = np.sum(cmats[:, :, 0, 1], axis=0)

	fn = np.sum(cmats[:, :, 1, 0], axis=0)
	#tn = np.sum(cmats[:, :, 1, 1], axis=0)

	freespace_fraction_sum = np.sum(gt_freespace_fractions)

	tp_rates = tp / (tp + fn)
	fp_rates = fp / freespace_fraction_sum

	return dict(
		num_levels=num_levels,
		tp_rates=tp_rates,
		fp_rates=fp_rates,
		cmats=cmats,
		roi=roi,
	)

## LAF

# semseg:
#	tr_semseg_detect_freespace_and_obstacles
# diff:
# 	eval net
# 	diff on road: zero everything outside of road
# 	diff + semseg: p[semseg obstacles] = 1
# 	diff + semseg on road: diff on road -> fuse with semseg

def laf_detection_variants(freespace_mask, obstacle_mask_semseg, anomaly_p, roi, **_):
	obstacle_mask_semseg = obstacle_mask_semseg.astype(np.bool)

	out = dict()

	anomaly_p = anomaly_p * roi

	# diff
	out['anomaly_p'] = anomaly_p

	# diff on road
	out['anomaly_p_onroad'] = anomaly_p * freespace_mask

	# diff + semseg
	anomaly_p_with_sem = anomaly_p.copy()
	anomaly_p_with_sem[obstacle_mask_semseg] = 1
	out['anomaly_p_with_sem'] = anomaly_p_with_sem

	# duff + semseg on road
	anomaly_p_with_sem_onroad = anomaly_p_with_sem * freespace_mask
	out['anomaly_p_with_sem_onroad'] = anomaly_p_with_sem_onroad

	return out


def processor_roadonly(frame, anomaly_p):
	return anomaly_p * frame.freespace_mask

def processor_add_sem(frame, anomaly_p):
	anomaly_p = anomaly_p.copy()
	anomaly_p *= frame.freespace_mask
#	anomaly_p += frame.obstacle_mask_semseg * 10

	anomaly_p[frame.obstacle_mask_semseg] = 1

	return anomaly_p


def laf_rocs_all(frs, outdir, roi, spec=LAF_ROC_FIELDS):
	os.makedirs(outdir, exist_ok=True)

	rocinfo_sem = laf_roc_calc(frs, roi=roi, pred_field='obstacle_mask_semseg', gt_field='anomaly_gt')
	np.savez_compressed(outdir / 'semonly_roc.npz', **rocinfo_sem)

	tpr_sem = rocinfo_sem['tp_rates'][0]
	fpr_sem = rocinfo_sem['fp_rates'][0]

	fig, plott = plt.subplots(1)

	plott.scatter([fpr_sem], [tpr_sem], marker='x', label='sem')

	for variant in spec:
		name, field, fmt = variant

		if name.endswith('_wsem'):
			proc = processor_add_sem
		else:
			proc = processor_roadonly

		rocinfo = laf_roc_calc(frs, roi=roi, pred_field=field, gt_field='anomaly_gt', processor=proc)

		np.savez_compressed(outdir / (name + '_roc.npz'), **rocinfo)

		roc_plot_additive(rocinfo, label=name, plot=plott, fmt=fmt)

	fig.legend(loc='lower right')
	fig.tight_layout()

	fig.savefig(outdir / 'rocs.pdf')


def draw_rocs_from_files(spec, save=None, point_spec=[], base_dir=None, title=None, max_fpr=None):
	"""
	bdir = DIR_DATA/'lost_and_found'
	dir_ctc = bdir / 'rocs_fakeErrCtc'
	dir_lafref =  bdir / 'rocs_fakeErr_LAFref'
	dir_ctc_i = bdir / 'rocs_fakeErrCtc_interesting_only'
	dir_lafref_i =  bdir / 'rocs_fakeErr_LAFref_interesting_only'

	spec_semonly = [
		('sem', dir_ctc/'semonly_roc.npz', 'x'),
	]

	spec = [
		('diff', 'diff_roc.npz', ':'),
		('diff+sem', 'diff+sem_roc.npz', '-'),
		('diff (road)', 'diff (road)_roc.npz', '--'),
		('diff+sem (road)', 'diff+sem (road)_roc.npz', '^'),
	]
	draw_rocs(spec, point_spec=spec_semonly, base_dir = dir_lafref)

	draw_rocs(spec, point_spec=spec_semonly, base_dir = dir_ctc)

	spec = [
		('diff + sem (road)', dir_ctc/'diff+sem (road)_roc.npz', '-'),
		('diff + sem LAF refine (road)', dir_lafref/'diff+sem (road)_roc.npz', '--'),
	]
	draw_rocs(spec, point_spec=spec_semonly)
	"""

	fig, plott = plt.subplots(1, figsize=(12, 8))

	areas = []

	for row in spec:
		name, path, fmt = row

		if base_dir:
			path = base_dir / path

		if path.is_file():
			with np.load(path, 'r') as rocinfo:
				aoc = roc_plot_additive(rocinfo, label=name, plot=plott, fmt=fmt)
				areas.append(aoc)
		else:
			print('No file at', path)

	for row in point_spec:
		name, path, fmt = row

		if base_dir:
			path = base_dir / path

		with np.load(path, 'r') as rocinfo:
			#rocinfo = dict(fin)
			tpr = rocinfo['tp_rates'][0]
			fpr = rocinfo['fp_rates'][0]
			plott.scatter([fpr], [tpr], marker=fmt, label=name)

	plott.set_xlabel('false positive rate')
	plott.set_ylabel('true positive rate')

	if max_fpr is not None:
		plott.set_xlim([0, max_fpr])

	if title:
		plott.set_title(title)

	permutation = np.argsort(areas)[::-1]
	handles, labels = plott.get_legend_handles_labels()
	handles = [handles[i] for i in permutation]
	labels = [labels[i] for i in permutation]

	fig.legend(handles, labels, loc='lower right')
	fig.tight_layout()

	if save:
		save = Path(save)
		fig.savefig(save.with_suffix('.png'))
		fig.savefig(save.with_suffix('.pdf'))


def run_rocs(sem_cat, dset, detector_names, roi=None, transform=tr_combine_swap, chans_to_load = [], validate=False):

	base = Path(dset.dir_out) / f'eval_{sem_cat}' / dset.split

	if roi is None:
		roi = dset.get_roi()

	chans = ['labels_source'] + [f'anomaly_{name}' for name in detector_names] + chans_to_load

	dset.set_channels_enabled([])
	chans_missing = []
	for ch in chans:
		try:
			dset.channel_enable(ch)
		except Exception as e:
			chans_missing.append(ch)

	if chans_missing:
		print('Missing channels:', ', '.join(chans_missing))

	# dset.discover()

	rocs_calc_all_simple(dset, base, detector_names, roi, transform=transform, validate=validate)