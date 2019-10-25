
from ..datasets.dataset import *
from ..datasets.cityscapes import *
from ..datasets.lost_and_found import DatasetLostAndFoundSmall

BUS_UNK_ID = [CityscapesLabelInfo.name2label[n].id for n in ['bus', 'truck', 'train']]

def tr_bus_to_anomaly(labels_source, **_):
	bus_mask = np.any([labels_source == c for c in BUS_UNK_ID], axis=0),

	labels_with_anomaly = labels_source.copy()
	labels_with_anomaly[bus_mask] = 27
	return dict(
		labels_source=labels_with_anomaly,
	)

tr_prepare_for_ctcEval = TrsChain(
	TrSemSegLabelTranslation(CityscapesLabelInfo.table_trainId_to_label, fields=[('pred_labels_trainIds', 'pred_labels_ids')]),
	tr_bus_to_anomaly,
)


class TrThresholdAnomaly:
	def __init__(self, anomaly_field, thr, anomaly_class=27):

		self.anomaly_field = anomaly_field
		self.thr = thr
		self.anomaly_class = anomaly_class

	def __call__(self, pred_labels_ids, **fields):

		anomaly_p = fields[self.anomaly_field]

		labels = pred_labels_ids.copy()
		labels[anomaly_p > self.thr] = self.anomaly_class

		return dict(
			pred_labels_ids = labels,
		)


DenseEpi_var_file = '{dset.dir_out}/eval_DenseEpi/uncertainty/uncertainty_{dset.split}.hdf5'

ch_DenseEpi_sem = ChannelResultImage('eval_DenseEpi/labels', suffix='_trainIds', img_ext='.png')
ch_DenseEpi_sem_color = ChannelResultImage('eval_DenseEpi/labels', suffix='_color', img_ext='.png')
ch_DenseEpi_var_alea = ChannelLoaderHDF5(DenseEpi_var_file, '{fid}/var_alea', compression=6)
ch_DenseEpi_var_dropout = ChannelLoaderHDF5(DenseEpi_var_file, '{fid}/var_dropout', compression=6)
ch_DenseEpi_gen_image_genAll = ChannelResultImage('eval_DenseEpi/gen_image_genAll', suffix='_gen', img_ext='.jpg')
ch_DenseEpi_anomaly_p_genAll = ChannelLoaderHDF5(
	'{dset.dir_out}/eval_DenseEpi/uncertainty/anomaly_genAll_{dset.split}.hdf5',
	'{fid}/var_dropout', compression=6,
)

def get_val_dset_DenseEpi():
	dset = DatasetCityscapesSmall(split='val')
	dset.discover()
	dset.load_class_statistics()

	dset.add_channels(
		pred_labels_trainIds = ch_DenseEpi_sem,
		pred_labels_colorimg = ch_DenseEpi_sem_color,
		pred_var_alea = ch_DenseEpi_var_alea,
		pred_var_dropout = ch_DenseEpi_var_dropout,
		gen_image = ch_DenseEpi_gen_image_genAll,
		anomaly_p = ch_DenseEpi_anomaly_p_genAll,
	)

	return dset

BaySegNet_var_file = '{dset.dir_out}/eval_BaySegNet/uncertainty/uncertainty_{dset.split}.hdf5'

ch_BaySegNet_sem = ChannelResultImage('eval_BaySegNet/labels', suffix='_trainIds', img_ext='.png')
ch_BaySegNet_sem_color = ChannelResultImage('eval_BaySegNet/labels', suffix='_color', img_ext='.png')
ch_BaySegNet_var_dropout = ChannelLoaderHDF5(BaySegNet_var_file, '{fid}/var_dropout', compression=6)
ch_BaySegNet_gen_image_genAll = ChannelResultImage('eval_BaySegNet/gen_image_genAll', suffix='_gen', img_ext='.jpg')
ch_BaySegNet_anomaly_p_genAll = ChannelLoaderHDF5(
	'{dset.dir_out}/eval_BaySegNet/uncertainty/anomaly_genAll_{dset.split}.hdf5',
	'{fid}/var_dropout', compression=6,
)

def get_val_dset_BaySegNet():
	dset = DatasetCityscapesSmall(split='val')
	dset.discover()
	dset.load_class_statistics()

	dset.add_channels(
		pred_labels_trainIds = ch_BaySegNet_sem,
		pred_labels_colorimg = ch_BaySegNet_sem_color,
		pred_var_dropout = ch_BaySegNet_var_dropout,
		gen_image = ch_BaySegNet_gen_image_genAll,
		anomaly_p = ch_BaySegNet_anomaly_p_genAll,
	)

	return dset


def get_laf_BaySegNet(b_interesting=True):
	dset = DatasetLostAndFoundSmall(split='test', only_interesting=b_interesting)
	dset.discover()
	# dset.load_class_statistics()

	dset.add_channels(
		pred_labels_trainIds=ch_BaySegNet_sem,
		pred_labels_colorimg=ch_BaySegNet_sem_color,
		pred_var_dropout=ch_BaySegNet_var_dropout,
		gen_image=ch_BaySegNet_gen_image_genAll,
		anomaly_p=ch_BaySegNet_anomaly_p_genAll,
	)

	return dset


PSP_var_file = '{dset.dir_out}/eval_PSP/uncertainty/uncertainty_{dset.split}.hdf5'

ch_PSP_sem = ChannelResultImage('eval_PSP/labels', suffix='_trainIds', img_ext='.png')
ch_PSP_sem_color = ChannelResultImage('eval_PSP/labels', suffix='_color', img_ext='.png')
ch_PSP_var_ensemble = ChannelLoaderHDF5(PSP_var_file, '{fid}/var_ensemble', compression=6)
ch_PSP_gen_image_genAll = ChannelResultImage('eval_PSP/gen_image_genAll', suffix='_gen', img_ext='.jpg')
ch_PSP_anomaly_p_genAll = ChannelLoaderHDF5(
	'{dset.dir_out}/eval_PSP/uncertainty/anomaly_genAll_{dset.split}.hdf5',
	'{fid}/anomaly_p', compression=6,
)


def get_val_dset_PSP():
	dset = DatasetCityscapesSmall(split='val')
	dset.discover()
	dset.load_class_statistics()

	dset.add_channels(
		pred_labels_trainIds = ch_PSP_sem,
		pred_labels_colorimg = ch_PSP_sem_color,
		pred_var_ensemble = ch_PSP_var_ensemble,
		gen_image = ch_PSP_gen_image_genAll,
		anomaly_p = ch_PSP_anomaly_p_genAll,
	)

	return dset


def get_laf_PSP(b_interesting=True):
	dset = DatasetLostAndFoundSmall(split='test', only_interesting=b_interesting)
	dset.discover()
	#dset.load_class_statistics()

	dset.add_channels(
		pred_labels_trainIds=ch_PSP_sem,
		pred_labels_colorimg=ch_PSP_sem_color,
		pred_var_ensemble=ch_PSP_var_ensemble,
		gen_image=ch_PSP_gen_image_genAll,
		anomaly_p=ch_PSP_anomaly_p_genAll,
	)
	
	return dset


def laf_is_valid(fr):
	return np.count_nonzero(fr.labels_source == 1) > 0


def tr_laf_is_valid(labels_source, **_):
	return dict(
		valid=np.count_nonzero(labels_source) > 0
	)


def tr_laf_calc_anomaly_gt(labels_source, **_):
	return dict(
		anomaly_gt=labels_source > 1,
	)


tr_laf_preprocess = TrsChain(
	tr_laf_is_valid,
	tr_laf_calc_anomaly_gt,
)
