
import numpy as np
from matplotlib import pyplot as plt
from ..pipeline.frame import Frame
from ..pipeline.transforms import TrBase, TrsChain, TrKeepFields
from ..datasets.dataset import DatasetBase
import cv2, os
from pathlib import Path
from types import SimpleNamespace

NUM_LEVELS = 1024
# TODO nonlinear levels

def binary_confusion_matrix(prob, gt_label, roi=None, levels=NUM_LEVELS):
	if roi is not None:
		prob = prob[roi]
		area = prob.__len__()
		gt_label = gt_label[roi]
	else:
		area = np.prod(prob.shape)

	gt_label_bool = gt_label.astype(np.bool)

	gt_area_true = np.count_nonzero(gt_label)
	gt_area_false = area - gt_area_true

	prob_at_true = prob[gt_label_bool]
	prob_at_false = prob[~gt_label_bool]

	tp, _ = np.histogram(prob_at_true, levels, range=[0, 1])
	tp = np.cumsum(tp[::-1])

	fn = gt_area_true - tp

	fp, _ = np.histogram(prob_at_false, levels, range=[0, 1])
	fp = np.cumsum(fp[::-1])

	tn = gt_area_false - fp

	cmat = np.array([
		[tp, fp],
		[fn, tn],
	]).transpose(2, 0, 1)

	cmat = cmat.astype(np.float64) / area

	return dict(
		cmat=cmat,
	)


def test_binary_confusion_matrix():
	pp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	gt = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=np.bool)
	cmat = binary_confusion_matrix(pp, gt, levels=20)['cmat']
	cmat_all_p = np.sum(cmat[:, :, 0], axis=1)
	cmat_all_n = np.sum(cmat[:, :, 1], axis=1)

	print(cmat_all_p, cmat_all_n)

	plt.plot(cmat[:, 0, 1] / cmat_all_n, cmat[:, 0, 0] / cmat_all_p)


class DetSpec(SimpleNamespace):
	def __init__(self,
	        name, gt_field='anomaly_gt', pred_field=None, cmats_field=None,
	        channels=None, roi=None,
	        roc_transform=None, pre_transform=None, plot_fmt=None, plot_name=None,
	    ):

		from .E0_article_evaluation import name_to_plot_format

		super().__init__(
			name = name,
			gt_field = gt_field,
			pred_field = pred_field or f'anomaly_{name}',
			cmats_field = cmats_field or f'cmats_{name}',
			roi=roi,
			# channels to load
			channels = channels if (channels is not None) else [f'anomaly_{name}'],
			roc_transform = roc_transform,
			pre_transform = pre_transform,
			plot_fmt = plot_fmt or name_to_plot_format(name),
			plot_name = plot_name or name,
			filename = f'anomalyScore_{name}_roc.npz',
		)

	def __repr__(self):
		return f'DetSpec({self.name},   {self.gt_field}, {self.pred_field}, {self.roi})'


class TrRocFrame(TrBase):
	def __init__(self, name, gt_field, pred_field, roi_field=None, cmats_field=None, num_levels=NUM_LEVELS):
		self.name = name
		self.gt_field = gt_field
		self.pred_field = pred_field
		self.roi_field = roi_field
		self.cmats_field = cmats_field or f'cmats_{self.name}'
		self.num_levels = num_levels

	def __call__(self, **fields):
		p_anomaly = fields[self.pred_field]
		gt_label = fields[self.gt_field]
		roi = fields[self.roi_field] if self.roi_field is not None else None

		cmats_by_thr = binary_confusion_matrix(
			prob=p_anomaly,
			gt_label=gt_label,
			roi=roi,
			levels=self.num_levels,
		)['cmat']

		return {
			self.cmats_field: cmats_by_thr,
		}

	def __repr__(self):
		return self.repr_with_args(self.__class__.__name__, (self.name, self.gt_field, self.pred_field))

	@staticmethod
	def from_det_spec(ds):
		return TrRocFrame(
			ds.name,
			gt_field=ds.gt_field,
			pred_field=ds.pred_field,
			roi_field=ds.roi,
			cmats_field = ds.cmats_field,
		)


def cmats_to_rocinfo(name, cmats):
	# roi_area = np.count_nonzero(roi) if roi is not None else 1

	num_levels = cmats.shape[0]
	tp = cmats[:, 0, 0]
	fp = cmats[:, 0, 1]
	fn = cmats[:, 1, 0]
	tn = cmats[:, 1, 1]

	tp_rates = tp / (tp+fn)
	fp_rates = fp / (fp+tn)

	area_under = np.trapz(tp_rates, fp_rates)

	return dict(
		name = name,
		num_levels=num_levels,
		tp_rates=tp_rates,
		fp_rates=fp_rates,
		cmats=cmats,
		auroc = area_under,
#		roi=roi,
	)


def rocs_calculate(dset, outdir, detector_specs, transform_custom=TrsChain(), reload_dset=True):
	os.makedirs(outdir, exist_ok=True)

	# transforms needed to preprocess and calculate the ROCs
	roc_trs = TrsChain(
		transform_custom,
	)

	roc_fields = []
	# chans_to_load = set(['labels_source'])
	chans_to_load = set()
	ds_to_save = []

	for ds in detector_specs:
		out_file_path = outdir / ds.filename

		if out_file_path.exists():
			print(f'Skip {ds.name}: {out_file_path} already exists')
		else:
			ds_to_save.append(ds)

			if ds.pre_transform:
				roc_trs.append(ds.pre_transform)

			roc_transform = ds.roc_transform or TrRocFrame.from_det_spec(ds)
			roc_trs.append(roc_transform)

			roc_fields.append(roc_transform.cmats_field)
			chans_to_load.update(ds.channels)


	# remove fields other than CMats to clear memory
	roc_trs.append(TrKeepFields(*roc_fields))

	# can't set channel list if we have a sequence of frames instead of a Dset
	if isinstance(dset, DatasetBase):
		dset.set_channels_enabled(list(chans_to_load))

		if reload_dset:
			dset.discover()


	# perform calculation
	results = Frame.frame_list_apply(roc_trs, dset, ret_frames=True, n_threads=6, batch=8)

	# fuse cmats
	#cmats_summed = {k: cmat for k, cmat in results[0].items() if k.startswith('cmats_')}
	cmats_summed = {k: results[0][k] for k in roc_fields}

	for fr in results[1:]:
		for k, sm in cmats_summed.items():
			sm += fr[k]

	# write ROCs
	for ds in ds_to_save:
		out_file_path = outdir / ds.filename

		rocinfo = cmats_to_rocinfo(ds.name, cmats_summed[ds.cmats_field])
		np.savez_compressed(out_file_path, **rocinfo)



def roc_plot_single(roc_info, thrs=[0.1, 0.5, 0.75, 0.8, 0.9]):
	if isinstance(roc_info, (str, Path)):
		with np.load(roc_info) as fin:
			roc_info = dict(fin)

	fp_rates, tp_rates, num_levels = (roc_info[k] for k in ['fp_rates', 'tp_rates', 'num_levels'])

	plt.figure()
	# plt.plot(fp_rates[:-1], tp_rates[:-1])
	plt.plot(fp_rates[:-1], tp_rates[:-1])

	#plt.xlim([0, 0.1])

	# 	thrs = [0.1, 0.5, 0.75, 0.8, 0.9, 0.95]
	thrs_levels = [int(round((1 - t) * num_levels)) for t in thrs]

	for t, tl in zip(thrs, thrs_levels):
		plt.scatter(fp_rates[tl], tp_rates[tl], marker='x', label='{0:.02f}'.format(t))

	plt.legend(loc=4)

	plt.xlabel('FP rate')
	plt.ylabel('TP rate')
	plt.tight_layout()


def roc_plot_additive(roc_info, label, plot=None, fmt=None):

	fp_rates, tp_rates, num_levels = (roc_info[k] for k in ['fp_rates', 'tp_rates', 'num_levels'])

	if plot is None:
		fig, plot = plt.subplots(1)
		plot.set_xlim([0, 1])

	fmt_args = []
	fmt_kwargs = {}

	if fmt is not None:
		if isinstance(fmt, str):
			# one string specifier
			fmt_args = [fmt]
		elif isinstance(fmt, dict):
			fmt_kwargs = fmt
		elif isinstance(fmt, tuple):
			fmt_kwargs = dict(color=fmt[0], linestyle=fmt[1])
		else:
			raise NotImplementedError(f"Format object {fmt}")

	#last_valid = np.searchsorted(fp_rates, 1.001)
	#print(last_valid)
	#area_under = np.trapz(tp_rates[:last_valid], fp_rates[:last_valid])
	area_under = np.trapz(tp_rates, fp_rates)

	#print(label, '\nfp:', fp_rates[:5], fp_rates[-5:], '\ntp:', tp_rates[:5], tp_rates[-5:])

	#plot.plot(fp_rates[:-1], tp_rates[:-1],
	plot.plot(fp_rates, tp_rates,
		*fmt_args,
		# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
		label='{lab}  {a:.02f}'.format(lab=label, a=area_under),
		**fmt_kwargs,
	)

	#print('xlim ', plot.get_xlim())
	xlimit = max(fp_rates[-2] + 0.05, plot.get_xlim()[1])
	plot.set_xlim([0, xlimit])

	# plot.plot(
	# 	fp_rates[-2:], tp_rates[-2:],
	# 	linestyle = ':',
	# 	color=(0.7, 0.7, 0.7),
	# )
	return area_under

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

COLORS = dict(
	tp=np.array((131, 255, 82), dtype=np.uint8),
	fp=np.array((255, 143, 0), dtype=np.uint8),
	fn=np.array((249, 119, 99), dtype=np.uint8),
)


def contour_inset(mask):
	eroded = cv2.erode(mask.astype(np.uint8), MORPH_KERNEL).astype(np.bool)
	return eroded != mask


def meld(canvas, color, mask):
	canvas[mask] = (0.5 * canvas[mask] + 0.5 * color).astype(np.uint8)


def classification_contours_binary(pred_mask, gt_mask, background=None, roi=None, colors=COLORS):
	if background is None:
		canvas = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
	else:
		canvas = background

	if roi is not None:
		pred_mask = pred_mask & roi

	tp = pred_mask & gt_mask
	meld(canvas, colors['tp'], tp)
	# 	canvas[contour_inset(tp)] = (0.5*canvas[tp] + 0.5*colors['tp']).astype(np.uint8)

	fp = pred_mask & (~gt_mask)
	meld(canvas, colors['fp'], fp)

	# 	canvas[contour_inset(fp)] = colors['fp']

	fn = (~pred_mask) & gt_mask
	meld(canvas, colors['fn'], fn)
	# 	canvas[contour_inset(fn)] = colors['fn']

	return canvas


# LAF ROC plotting

LAF_ROC_FIELDS = [
	('diff', 'anomaly_p', ':r'),
	('diff+sem', 'anomaly_p_with_sem', '-r'),
	('diff (road)', 'anomaly_p_onroad', '--m'),
	('diff+sem (road)', 'anomaly_p_with_sem_onroad', '-m'),
]


def draw_rocs(detector_specs, base_dir, save=None, title=None, max_fpr=None, figsize=(4, 4)):
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

	fig, plott = plt.subplots(1, figsize=figsize)

	areas = []

	base_dir = Path(base_dir)


	for ds in detector_specs:
		data_file_path = base_dir / ds.filename

		if data_file_path.is_file():
			with np.load(data_file_path, 'r') as rocinfo:
				aoc = roc_plot_additive(rocinfo, label=ds.plot_name, plot=plott, fmt=ds.plot_fmt)
				areas.append(aoc)
		else:
			print('No file at', data_file_path)

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

# 	legend = fig.legend(handles, labels, loc='right')
	legend = plott.legend(handles, labels, loc=(0.2, 0.05))


	# shift = max([t.get_window_extent().width for t in legend.get_texts()])
	# for t in legend.get_texts():
	# 	t.set_ha('right')   # horizontal alignment
	# 	t.set_position((shift, 0))

	fig.tight_layout()

	if save:
		save = Path(save)
		fig.savefig(save.with_suffix('.png'))
		fig.savefig(save.with_suffix('.pdf'))
