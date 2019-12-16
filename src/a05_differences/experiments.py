import numpy as np
import torch
from pathlib import Path
from ..pipeline import *
from .networks import *
from .transforms import *
from ..paths import DIR_DATA
from ..datasets.dataset import imwrite, ChannelLoaderImage, ChannelResultImage, ChannelLoaderHDF5, TrSaveChannelsAutoDset
from ..datasets.cityscapes import DatasetCityscapesSmall
from ..datasets.lost_and_found import DatasetLostAndFoundSmall
DatasetLostAndFoundWithSemantics = DatasetLostAndFoundSmall
from ..pipeline.bind import bind
from ..pipeline.evaluations import TrChannelLoad, TrChannelSave
from ..a01_sem_seg.networks import ClassifierSoftmax, LossCrossEntropy2d, PerspectiveSceneParsingNet
from ..a01_sem_seg.experiments import ExpSemSegPSP
from ..a01_sem_seg.transforms import TrColorimg

from ..a04_reconstruction.experiments import Pix2PixHD_Generator

from matplotlib import pyplot as plt
CMAP_MAGMA = plt.get_cmap('magma') 

channel_reconstruction_trCTC_ssBDD = ChannelResultImage('reconstr_p2phd-s_trained-ctc_semseg-bdd', suffix='_reconstr')

channel_labels_fakeErr01 = ChannelResultImage('fakeErr01/labels', suffix='_trainIds', img_ext='.png')
channel_reconstruction_trCTC_ssFakeErr = ChannelResultImage('fakeErr01/reconstr_p2phd-s_trained-ctc', suffix='_reconstr')

channel_reconstruction_trCTC_ssBus = ChannelResultImage('reconstr/bus/reconstr_p2phd-s_trained-ctc', suffix='_reconstr')

channel_anomalyp_bus_fakeErr01 = ChannelLoaderHDF5(
	'{dset.dir_out}/err/bus/fakeErr01_p_anomaly.hdf5',
	var_name_tmpl = '{fid}',
	compression = 5,
)

ch_BaySegNet_sem = ChannelResultImage('eval_BaySegNet/labels', suffix='_trainIds', img_ext='.png')

class ExperimentDifference01(ExperimentBase):
	cfg = add_experiment(
		name='corrdiff_01_errors-ctc',
		net=dict(
			batch_eval=5,
			batch_train=3,
		),
		train=dict(
			class_weights=[1.50660602, 10.70138633],
			optimizer=dict(
				lr_patience=5,
			)
		),
		epoch_limit = 50,
	)

	def init_transforms(self):
		super().init_transforms()

		self.class_softmax = ClassifierSoftmax()
		self.cuda_modules(['class_softmax'])

		self.tr_preprocess = TrsChain(
			tr_label_to_validEval,
			tr_get_errors,
			tr_errors_to_gt,
		)

		self.tr_postprocess_log = TrsChain(
			TrNP(),
		)

	def init_loss(self):
		# TODO by class name from cfg
		
		class_weights = self.cfg['train'].get('class_weights', None)
		if class_weights is not None:
			print('	class weights:', class_weights)
			class_weights = torch.Tensor(class_weights)
		else:
			print('	no class weights')
		self.loss_mod = LossCrossEntropy2d(weight=class_weights)
		self.cuda_modules(['loss_mod'])

	def tr_net(self, image, gen_image, **_):
		return dict(
			pred_anomaly_logits = self.net_mod(image, gen_image)
		)

	def tr_loss(self, semseg_errors_label, pred_anomaly_logits, **_):
		return self.loss_mod(pred_anomaly_logits, semseg_errors_label)

	def tr_classify(self, pred_anomaly_logits, **_):
		return dict(
			anomaly_p = self.class_softmax(pred_anomaly_logits)['pred_prob'][:, 1, :, :]
			# get anomaly class prob which is label=1,
		)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		self.net_mod = CorrDifference01()

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])



	def init_log(self, frames_to_log=None):
		if frames_to_log is not None:
			self.frames_to_log = set(frames_to_log)

		super().init_log()

		ds = self.datasets['val']

		chans_backup = ds.channels_enabled
		ds.set_channels_enabled('image', 'semseg_errors')

		# Write the ground-truth for comparison
		for fid in self.frames_to_log:
			fid_no_slash = str(fid).replace('/', '__')
			fr = ds.get_frame_by_fid(fid)

			fr.apply(self.tr_preprocess)

			imwrite(self.train_out_dir / f'gt_image_{fid_no_slash}.webp', fr.image)
			imwrite(self.train_out_dir / f'gt_labels_{fid_no_slash}.png', (fr.semseg_errors > 0).astype(np.uint8) * 255)
			
			self.tboard_img.add_image(
				'{0}_img'.format(fid),
				fr.image.transpose((2, 0, 1)),
				0,
			)

			self.tboard_gt.add_image(
				'{0}_gt'.format(fid),
				fr.semseg_errors[None, :, :],
				0,
			)

		ds.set_channels_enabled(*chans_backup)


	def tr_eval_batch_log(self, frame, fid, anomaly_p, **_):
		if fid in self.frames_to_log:
			frame.apply(self.tr_postprocess_log)

			fid_no_slash = str(fid).replace('/', '__')
			epoch = self.state['epoch_idx']

			# drop the alpha channel
			pred_colorimg = CMAP_MAGMA(frame.anomaly_p, bytes=True)[:, :, :3]  
			imwrite(self.train_out_dir / f'e{epoch:03d}_anomalyP_{fid_no_slash}.webp', pred_colorimg)

			self.tboard.add_image(
				'{0}_class'.format(fid),
				frame.anomaly_p[None, :, :],
				self.state['epoch_idx'],
			)


	def construct_default_pipeline(self, role):

		# TrRandomlyFlipHorizontal(['image', 'labels']),

		pre_merge = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
		)

		pre_merge_test = pre_merge.copy()
		pre_merge_test.append(
			TrKeepFields('image', 'gen_image'),
		)

		fields_for_training = ['image', 'gen_image', 'semseg_errors_label']

		pre_merge_train = pre_merge.copy()
		pre_merge_train.append(
			TrKeepFields(*fields_for_training),
		)


		if role == 'test':
			return Pipeline(
				tr_input = TrsChain(
				),
				tr_batch_pre_merge = pre_merge_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net,
					self.tr_classify,
					TrKeepFields('anomaly_p'),
					TrNP(),
				),
				tr_output = TrsChain(

				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				tr_input = self.tr_preprocess,
				tr_batch_pre_merge = pre_merge_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net,
					self.tr_loss,
					self.tr_classify,
					TrKeepFieldsByPrefix('loss', 'anomaly_p'),
				),
				tr_output = TrsChain(
					self.tr_eval_batch_log,
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				tr_input = TrsChain(
					self.tr_preprocess,
					TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [384, 768]), fields = self.fields_for_training),
					TrRandomlyFlipHorizontal(fields_for_training),
				),
				tr_batch_pre_merge = pre_merge_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.training_start_batch,
					self.tr_net,
					self.tr_loss,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

	def setup_dset(self, dset):
		dset.add_channels(gen_image=channel_reconstruction_trCTC_ssBDD)
		dset.discover()

	def init_default_datasets(self, b_threaded=False):
		dir_sem_ctc = DIR_DATA / 'cityscapes/sem_seg'
		dir_sem_ctc_bdd = dir_sem_ctc / 'psp01_trained_on_bdd'
		dset_ctc_train = DatasetCityscapesSmall_PredictedSemantics(
			split='train',
			dir_semantics=dir_sem_ctc_bdd,
			b_cache=b_threaded,
		)
		dset_ctc_val = DatasetCityscapesSmall_PredictedSemantics(
			split='val',
			dir_semantics=dir_sem_ctc_bdd,
			b_cache=b_threaded,
		)
		dsets_ctc = [dset_ctc_train, dset_ctc_val]
		for dset in dsets_ctc:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_ctc_val.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', dset_ctc_train)
		self.set_dataset('val', dset_ctc_val)


def count_error_fraction(semseg_errors, **_):
	return dict(
		error_fraction=np.count_nonzero(semseg_errors) / np.prod(semseg_errors.shape),
	)


def calc_label_balance(dset, pred_labels_is_trainId=True):
	# TODO case for many classes
	ch_en = dset.channels_enabled
	dset.set_channels_enabled(['labels_source', 'pred_labels'])

	tr_proc = TrsChain(
		tr_label_to_validEval,
		tr_get_errors,
		count_error_fraction,
		TrKeepFields('error_fraction'),
	)
	
	if not pred_labels_is_trainId:
		tr_proc.insert(
			0,
			TrSemSegLabelTranslation(fields=['pred_labels'], table=CityscapesLabelInfo.table_label_to_trainId),
		)

	frs = Frame.frame_list_apply(tr_proc, dset, n_threads=16, ret_frames=True)

	errors_fractions = np.array([fr.error_fraction for fr in frs])
	fraction_mean = np.mean(errors_fractions)

	dset.set_channels_enabled(list(ch_en))

	return fraction_mean, errors_fractions

def label_balance_to_weights(label_prob):
	# https://github.com/fregu856/deeplabv3/blob/master/utils/preprocess_data.py#L184
	class_weights = 1. / np.log(1.02 + label_prob)
	return class_weights

def label_balance_to_weights_2class(p_c1):
	prob = np.array([1 - p_c1, p_c1])
	return label_balance_to_weights(prob)

#ct_err_fr_mean, ct_err_fr = calc_label_balance(dset_ctc_train)
# https://github.com/fregu856/deeplabv3/blob/master/utils/preprocess_data.py#L184
# prob = np.array([1-ct_err_fr_mean, ct_err_fr_mean])
# class_weights = 1. / np.log(1.02 + prob)
#
# prob, class_weights - cityscapes bdd errors
# (array([0.92204887, 0.07795113]), array([ 1.50660602, 10.70138633]))



class ExperimentDifference02_fakeErr(ExperimentDifference01):
	cfg = add_experiment(
		name='corrdiff_02_fakeErr-ctc',
		net=dict(
			batch_eval=5,
			batch_train=3,
		),
		train=dict(
			class_weights=[1.45693524, 19.18586532],
			optimizer=dict(
				lr_patience=5,
			)
		)
	)

	def setup_dset(self, dset):
		dset.add_channels(
			pred_labels = channel_labels_fakeErr01,
			gen_image = channel_reconstruction_trCTC_ssFakeErr,
		)
		dset.tr_post_load_pre_cache.append(
			# accidentally saved fakeErr labels as sourceIds, so translate them to trainIds for consistency
			TrSemSegLabelTranslation(fields=['pred_labels'], table=CityscapesLabelInfo.table_label_to_trainId),
		)
		dset.discover()

	def init_default_datasets(self, b_threaded=False):
		# Cityscapes with prediction channel
		dset_train = DatasetCityscapesSmall(split='train', b_cache=b_threaded)
		dset_val = DatasetCityscapesSmall(split='val', b_cache=b_threaded)
		dsets = [dset_train, dset_val]
		for dset in dsets:
			self.setup_dset(dset)
			
		self.frames_to_log = set([dset_val.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', dset_train)
		self.set_dataset('val', dset_val)



class ExperimentDifferenceBin_fakeErr(ExperimentDifference02_fakeErr):
	cfg = add_experiment(ExperimentDifference02_fakeErr.cfg,
		name='0504_CorrDiffBin_fakeErr',
		net=dict(
			batch_eval=5,
			batch_train=2,
		),
		train=dict(
			class_weights=[1.45693524, 19.18586532],
			optimizer=dict(
				lr_patience=5,
			)
		)
	)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		self.net_mod = CorrDifference01(num_outputs=1, freeze=False)
		
		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def init_loss(self):
		class_weights = self.cfg['train'].get('class_weights', None)
		if class_weights is not None:
			print('	class weights:', class_weights)
			class_weights = torch.Tensor(class_weights)
		else:
			print('	no class weights')

		self.loss_mod = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights[1])

		self.cuda_modules(['loss_mod'])

	def tr_net(self, image, gen_image, **_):
		return dict(
			pred_anomaly_logit = self.net_mod(image, gen_image)
		)

	def tr_loss(self, semseg_errors_label, pred_anomaly_logit, **_):
		#print(semseg_errors_label.shape, pred_anomaly_logit.shape)
		loss_val = self.loss_mod(pred_anomaly_logit[:, 0], semseg_errors_label)
		#print(loss_val.shape, loss_val)
		return dict(
			loss = loss_val,
		)

	def tr_classify(self, pred_anomaly_logit, **_):
		return dict(
			anomaly_p = torch.nn.functional.sigmoid(pred_anomaly_logit)
			# get anomaly class prob which is label=1,
		)

	def setup_dset(self, dset):
		dset.add_channels(
			pred_labels=channel_labels_fakeErr01,
			gen_image=channel_reconstruction_trCTC_ssFakeErr,
		)
		dset.tr_post_load_pre_cache.append(
			# accidentally saved fakeErr labels as sourceIds, so translate them to trainIds for consistency
			TrSemSegLabelTranslation(fields=['pred_labels'], table=CityscapesLabelInfo.table_label_to_trainId),
		)
		dset.discover()


ch_labels_fakePredErrBayes = ChannelResultImage('eval_BaySegNet/fakePredErr/labels', suffix='_trainIds', img_ext='.png')
ch_discrepancy_mask_fakePredErrBayes = ChannelResultImage('eval_BaySegNet/fakePredErr/labels', suffix='_errors', img_ext='.png')
ch_reconstruction_fakePredErrBayes = ChannelResultImage('eval_BaySegNet/fakePredErr/gen_image', suffix='_gen')

class ExperimentDifferenceBin_fakePredErrBDD(ExperimentDifferenceBin_fakeErr):
	ch_labelsPred_fakePredErrBDD = ChannelResultImage('0508_fakePredErrBDD/labels', suffix='_predTrainIds', img_ext='.png')
	ch_labelsFake_fakePredErrBDD = ChannelResultImage('0508_fakePredErrBDD/labels', suffix='_fakeTrainIds', img_ext='.png')
	ch_discrepancy_mask_fakePredErrBDD = ChannelResultImage('0508_fakePredErrBDD/labels', suffix='_errors', img_ext='.png')
	ch_reconstruction_fakePredErrBDD = ChannelResultImage('0508_fakePredErrBDD/gen_image', suffix='_gen')

	cfg = add_experiment(ExperimentDifference02_fakeErr.cfg,
		name='0508_CorrDiffBin_fakePredErrBDD',
	)

	# def preprocess_fakeerr(self, dset, labels_source, semseg_errors):
	# 	labels_validEval = tr_label_to_validEval(labels_source, dset)["labels_validEval"]
	# 	return tr_errors_to_gt(semseg_errors, labels_validEval)

	def init_transforms(self):
		super().init_transforms()
		
		self.tr_preprocess = TrsChain(
#			TrRenameKw(semseg_errors = 'semseg_errors_label'),

		)

	def setup_dset(self, dset):
		super().setup_dset(dset)
		dset.add_channels(
			pred_labels_trainIds = self.ch_labelsPred_fakePredErrBDD,
			labels_fakeErr_trainIds = self.ch_labelsFake_fakePredErrBDD,
			semseg_errors = self.ch_discrepancy_mask_fakePredErrBDD,
			gen_image = self.ch_reconstruction_fakePredErrBDD,
		)
		dset.tr_post_load_pre_cache = TrsChain()
		dset.set_channels_enabled('image', 'gen_image', 'semseg_errors')


class ExperimentDifference_Auto_Base(ExperimentDifferenceBin_fakeErr):
	cfg = add_experiment(ExperimentDifferenceBin_fakePredErrBDD.cfg,
		name='0510_DiffImgToLabel_',
		gen_name = '051X_semGT__fakeDisp__genNoSty',
		gen_img_ext = '.jpg',
	    pix2pix_variant = '0405_nostyle_crop_ctc',
		net=dict(
			batch_eval=3,
			batch_train=2,  # to train on small gpu
			num_classes=19, # num semantic classes
		),
	    disap_fraction = 0.5,
	    epoch_limit = 50,
    )

	# def preprocess_fakeerr(self, dset, labels_source, semseg_errors):
	# 	labels_validEval = tr_label_to_validEval(labels_source, dset)["labels_validEval"]
	# 	return tr_errors_to_gt(semseg_errors, labels_validEval)

	fields_for_test = ['image', 'gen_image']
	fields_for_training = ['image', 'gen_image', 'semseg_errors_label']


	def init_transforms(self):
		super().init_transforms()

		self.init_discrepancy_dataset_channels()

		# the function which alters labels to create synthetic discrepancies
		self.synthetic_mod = partial(tr_synthetic_disappear_objects, disap_fraction = self.cfg['disap_fraction'])

		self.roi_outside = np.logical_not(CTC_ROI)

		self.tr_preprocess = TrsChain()
		self.tr_input_train = self.tr_semseg_errors_to_label
		self.tr_input_test = TrsChain()

		pre_merge = TrsChain(
			TrZeroCenterImgs(),
			tr_torch_images,
		)

		self.pre_merge_test = pre_merge.copy()
		self.pre_merge_test.append(
			TrKeepFields(*self.fields_for_test),
		)

		self.pre_merge_train = pre_merge.copy()
		self.pre_merge_train += [
			TrKeepFields(*self.fields_for_training),
		]

	def init_discrepancy_dataset_channels(self):
		gen_name = self.cfg['gen_name']
		dir_disrepancy_dset = DIR_DATA / 'discrepancy_dataset' / '{dset.name}' / gen_name
		
		# Channels of the synthetic discrepancy dataset 

		# the labels with changed instances
		self.ch_labelsFake = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_fakeTrainIds.png')
		self.ch_labelsFake_colorimg = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_fakeTrainIds_colorimg.png')
		self.ch_discrepancy_mask = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_errors.png')
		self.ch_reconstruction = ChannelLoaderImage(dir_disrepancy_dset / 'gen_image' / '{dset.split}' / '{fid}_gen{channel.img_ext}', img_ext=self.cfg['gen_img_ext'])
		
		# the "correct" labels for the frame
		# usually this are the trainIDs of the Cityscapes groundtruth, but alternatively those could be predictions of a sem-seg network
		self.ch_labelsPred = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_predTrainIds.png')
		self.ch_labelsPred_colorimg = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_predTrainIds_colorimg.png')


		# dir_disrepancy_dset = self.workdir / 'discrepancy_dset'
		# self.storage = dict(
		# 	disc_dset_labels_fake = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_fakeTrainIds.png'),
		# 	disc_dset_discrepancy_mask = ChannelLoaderImage(dir_disrepancy_dset / 'labels' / '{dset.split}' / '{fid}_errors.png'),
		# 	disc_dset_gen_image = ChannelLoaderImage(dir_disrepancy_dset / 'gen_image' / '{dset.split}' / '{fid}_gen{channel.img_ext}', img_ext='.jpg'),
		# )

	def tr_semseg_errors_to_label(self, semseg_errors, **_):
		errs = (semseg_errors > 0).astype(np.int64)
		errs[self.roi_outside] = 255
		return dict(
			semseg_errors_label=errs,
		)

	def setup_dset(self, dset):
		super().setup_dset(dset)
		dset.add_channels(
			# pred_labels_trainIds=self.ch_labelsPred,
			labels_fakeErr_trainIds=self.ch_labelsFake,
			semseg_errors=self.ch_discrepancy_mask,
			gen_image=self.ch_reconstruction,
		)
		dset.tr_post_load_pre_cache = TrsChain()
		dset.set_channels_enabled('image', 'gen_image', 'semseg_errors')
		dset.discover()

	def init_default_datasets(self, b_threaded=False):
		dset_ctc_train = DatasetCityscapesSmall(
			split='train',
			b_cache=b_threaded,
		)
		dset_ctc_val = DatasetCityscapesSmall(
			split='val',
			b_cache=b_threaded,
		)
		dsets_ctc = [dset_ctc_train, dset_ctc_val]
		for dset in dsets_ctc:
			self.setup_dset(dset)

		self.frames_to_log = set([dset_ctc_val.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', dset_ctc_train)
		self.set_dataset('val', dset_ctc_val)

	def prepare_labels_pred(self, dsets):

		# GT labels
		tr_copy_gt_labels = TrsChain(
			TrSemSegLabelTranslation(CityscapesLabelInfo.table_label_to_trainId, [('labels_source', 'pred_labels_trainIds')]),
			TrSaveChannelsAutoDset(['pred_labels_trainIds']),
		)

		for dset in dsets:
			dset.set_channels_enabled('labels_source')
			Frame.frame_list_apply(tr_copy_gt_labels, dset, ret_frames=False)

	def discrepancy_dataset_init_pipeline(self, use_gt_labels=True, write_orig_label=False):
		"""
		@param use_gt_labels: True: the starting labels which we will be altering are the GT semantics of Cityscapes
			False: Load starting labels from ch_labelsPred
		"""
		self.pix2pix = Pix2PixHD_Generator(self.cfg['pix2pix_variant'])

		if use_gt_labels:
			self.tr_load_correct_labels = TrsChain(
				# load cityscapes labels and instances
				TrChannelLoad('labels_source', 'labels_source'),
				TrChannelLoad('instances', 'instances'),
				# convert to trainIDs
				TrSemSegLabelTranslation(fields=dict(labels_source='pred_labels_trainIds'),	table=CityscapesLabelInfo.table_label_to_trainId),
			)
		else:
			self.tr_load_correct_labels = TrChannelLoad(self.ch_labelsPred, 'pred_labels_trainIds'),

		self.tr_alter_labels_and_gen_image = TrsChain(
			# load original labels
			self.tr_load_correct_labels,
			# alter labels
			self.synthetic_mod,
			# synthetize image
			bind(self.pix2pix.tr_generator_np, pred_labels_trainIds='labels_fakeErr_trainIds').outs(gen_image='gen_image'),
		)

		self.tr_synthetic_and_show = TrsChain(
			self.tr_alter_labels_and_gen_image,
			TrColorimg('pred_labels_trainIds'),
			TrColorimg('labels_fakeErr_trainIds'),
			TrChannelLoad('image', 'image'),
			TrShow(
				['image', 'gen_image'],
				['pred_labels_trainIds_colorimg', 'labels_fakeErr_trainIds_colorimg', 'semseg_errors'],
			),
		)

		self.tr_synthetic_and_save = TrsChain(
			self.tr_alter_labels_and_gen_image,
			# saving as image does not like np.bool
			TrByField('semseg_errors', lambda x: (x > 0).astype(np.uint8)*255),
			# write to disk
			TrChannelSave(self.ch_discrepancy_mask, 'semseg_errors'),
			TrChannelSave(self.ch_labelsFake, 'labels_fakeErr_trainIds'),
			TrChannelSave(self.ch_reconstruction, 'gen_image'),

			# fake labels colorimg
			TrColorimg('labels_fakeErr_trainIds'),
			TrChannelSave(self.ch_labelsFake_colorimg, 'labels_fakeErr_trainIds_colorimg'),
		)

		# write the original labesl (as they were before alteration)
		if write_orig_label:
			self.tr_synthetic_and_save += [
				TrColorimg('pred_labels_trainIds'),
				TrChannelSave(self.ch_labelsPred, 'pred_labels_trainIds'),
				TrChannelSave(self.ch_labelsPred_colorimg, 'pred_labels_trainIds_colorimg'),
			]

	def discrepancy_dataset_generate(self, dsets=None, b_show=False, write_orig_label=False):
		self.discrepancy_dataset_init_pipeline(write_orig_label=write_orig_label)

		dsets = dsets or self.datasets.values()

		for dset in dsets:
			# disable default loading
			dset.set_channels_enabled()
			# clear cache
			dset.discover()

			if b_show:
				dset[0].apply(self.tr_synthetic_and_show)
			else:
				Frame.frame_list_apply(self.tr_synthetic_and_save, dset, n_proc=1, n_threads=1, ret_frames=False)


	def prepare_synthetic_changes(self, dsets, b_show=False):
		self.discrepancy_dataset_init_pipeline()

		for dset in dsets:
			dset.set_channels_enabled('image', 'pred_labels_trainIds', 'instances')
			dset.discover()

			if b_show:
				dset[0].apply(tr_gen_and_show)

			else:
				Frame.frame_list_apply(tr_gen_and_save, dset, ret_frames=False)

	def calc_class_statistics(self, dset=None):
		if dset is None:
			dset = self.datasets['train']

		class_distrib = calculate_class_distribution('semseg_errors', 2, dset)

		# ENet section 5.2
		class_weights = 1. / np.log(1.02 + class_distrib)

		print('Class distribution', class_distrib)
		print('Class weights', class_weights)

		with (Path(dset.dir_out) / self.cfg['gen_name'] / 'class_stats.json').open('w') as fout:
			json.dump(dict(
				class_distribution = list(map(float, class_distrib)),
				class_weights = list(map(float, class_weights)),
			), fout, indent='	')


	def prepare(self):

		dsets = [self.datasets['train'], self.datasets['val']]


		# self.prepare_labels_pred(dsets)
		# self.prepare_synthetic_changes(self, dsets)

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		self.net_mod = CorrDifference01(num_outputs=2, freeze=True)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])


	def init_loss(self):
		class_weights = self.cfg['train'].get('class_weights', None)
		if class_weights is not None:
			print('	class weights:', class_weights)
			class_weights = torch.Tensor(class_weights)
		else:
			print('	no class weights')

		self.loss_mod = LossCrossEntropy2d(weight=class_weights)
		self.cuda_modules(['loss_mod'])


	def tr_net(self, image, gen_image, **_):
		return dict(
			pred_anomaly_logits=self.net_mod(image, gen_image)
		)

	def tr_loss(self, semseg_errors_label, pred_anomaly_logits, **_):
		return self.loss_mod(pred_anomaly_logits, semseg_errors_label)

	def tr_classify(self, pred_anomaly_logits, **_):
		return dict(
			anomaly_p = self.class_softmax(pred_anomaly_logits)['pred_prob'][:, 1, :, :]
		)

	def construct_default_pipeline(self, role):

		# TrRandomlyFlipHorizontal(['image', 'labels']),


		if role == 'test':
			return Pipeline(
				tr_input = self.tr_input_test,
				tr_batch_pre_merge = self.pre_merge_test,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net,
					self.tr_classify,
					TrKeepFields('anomaly_p'),
					TrNP(),
				),
				tr_output = TrsChain(

				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				tr_input = self.tr_input_train,
				tr_batch_pre_merge = self.pre_merge_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.tr_net,
					self.tr_loss,
					self.tr_classify,
					TrKeepFieldsByPrefix('loss', 'anomaly_p'),
				),
				tr_output = TrsChain(
					self.tr_eval_batch_log,
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				tr_input = TrsChain(
					self.tr_input_train,
					TrRandomCrop(crop_size = self.cfg['train'].get('crop_size', [384, 768]), fields = self.fields_for_training),
					TrRandomlyFlipHorizontal(self.fields_for_training),
				),
				tr_batch_pre_merge = self.pre_merge_train,
				tr_batch = TrsChain(
					TrCUDA(),
					self.training_start_batch,
					self.tr_net,
					self.tr_loss,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				),
				tr_output = TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args = self.loader_args_for_role(role),
			)


class Exp0510_Difference_ImgVsGen_onGT(ExperimentDifference_Auto_Base):
	cfg = add_experiment(ExperimentDifference_Auto_Base.cfg,
		name='0510_DiffImgVsGen_onGT',
    )

	def tr_net(self, image, gen_image, **_):
		return dict(
			pred_anomaly_logits = self.net_mod(image, gen_image)
		)



class Exp0511_Difference_LabelsVsGen_onGT(ExperimentDifference_Auto_Base):
	cfg = add_experiment(ExperimentDifference_Auto_Base.cfg,
		name='0511_DiffLabelVsGen_onGT',
    )

	fields_for_test = ['labels_fakeErr_trainIds', 'image']
	fields_for_training = ['labels_fakeErr_trainIds', 'image', 'semseg_errors_label']

	def init_transforms(self):
		super().init_transforms()	

	def setup_dset(self, dset):
		super().setup_dset(dset)

		dset.channel_enable('labels_fakeErr_trainIds')
		dset.channel_disable('gen_image', 'pred_labels_trainIds')

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		self.net_mod = ComparatorImageToLabels(
			num_outputs=2, freeze=True,
			num_sem_classes=self.cfg['net']['num_classes'],
		)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def tr_net(self, labels_fakeErr_trainIds, image, **_):
		return dict(
			pred_anomaly_logits = self.net_mod(labels_fakeErr_trainIds, image)
		)

	def construct_default_pipeline(self, role):

		pipe = super().construct_default_pipeline(role)

		if role == 'test':
			pipe.tr_batch_pre_merge.insert(0, TrRenameKw(pred_labels_trainIds = 'labels_fakeErr_trainIds'))

		return pipe


class Exp0512_Difference_ImgVsGen_onPredBDD(ExperimentDifference_Auto_Base):
	cfg = add_experiment(ExperimentDifference_Auto_Base.cfg,
		name='0512_DiffImgVsGen_onPredBDD',
		gen_name='051X_semBDD__fakeDisp__genNoSty',
	)

	def init_transforms(self):
		super().init_transforms()

		self.synthetic_mod = tr_synthetic_disappear_objects

	def prepare_labels_pred(self, dsets, b_show=False):
		from ..a01_sem_seg.experiments import ExpSemSegPSP_BDD
		exp_sem = ExpSemSegPSP_BDD()
		exp_sem.init_net('eval')

		# from ..a01_sem_seg.experiments import ExpSemSegPSP_Ensemble_BDD
		# exp_sem = ExpSemSegPSP_Ensemble_BDD()
		# exp_sem.init_net('master_eval')

		pipe_sem = exp_sem.construct_default_pipeline('test')

		if b_show:
			pipe_sem.tr_output += [
				TrRenameKw(dict(pred_labels='pred_labels_trainIds')),
				TrColorimg('pred_labels_trainIds'),
				TrShow(['image', 'pred_labels_trainIds_colorimg']),
			]

		else:
			pipe_sem.tr_output += [
				TrRenameKw(dict(pred_labels='pred_labels_trainIds')),
				TrSaveChannelsAutoDset(['pred_labels_trainIds']),
			]

		for dset in dsets:
			dset.set_channels_enabled(['image'])

			if b_show:
				bout, outfrs = pipe_sem.execute(dset, b_one_batch=True)
				del bout, outfrs

			else:
				pipe_sem.execute(dset, b_accumulate=False)


class Exp0516_Diff_SwapFgd_ImgVsGen_semGT(ExperimentDifference_Auto_Base):
	cfg = add_experiment(ExperimentDifference_Auto_Base.cfg,
		name='0516_Diff_SwapFgd_ImgVsGen_semGT',
		gen_name='051X_semGT__fakeSwapFgd__genNoSty',
		swap_fraction = 0.5,
	)

	def init_transforms(self):
		super().init_transforms()
		self.synthetic_mod = partial(tr_synthetic_swapFgd_labels, swap_fraction = self.cfg['swap_fraction'])


class Exp0517_Diff_SwapFgd_ImgVsLabels_semGT(Exp0511_Difference_LabelsVsGen_onGT):
	cfg = add_experiment(ExperimentDifference_Auto_Base.cfg,
		name='0517_Diff_SwapFgd_ImgVsLabels_semGT',
		gen_name='051X_semGT__fakeSwapFgd__genNoSty',
    )

	def init_transforms(self):
		super().init_transforms()
		self.synthetic_mod = None # gen using 0516


class Exp0520_Diff_ImgAndLabelsVsGen_semGT(Exp0511_Difference_LabelsVsGen_onGT):
	cfg = add_experiment(Exp0511_Difference_LabelsVsGen_onGT.cfg,
		name='0520_Diff_Disap_ImgAndLabelVsGen_semGT',
		gen_name='051X_semGT__fakeDisp__genNoSty',
		net = dict(
			num_classes = 19,
		)
    )

	fields_for_test = ['labels_fakeErr_trainIds', 'gen_image', 'image']
	fields_for_training = ['labels_fakeErr_trainIds', 'gen_image', 'image', 'semseg_errors_label']

	def init_transforms(self):
		super().init_transforms()

	def setup_dset(self, dset):
		super().setup_dset(dset)

		dset.channel_enable('labels_fakeErr_trainIds', 'gen_image')
		dset.channel_disable('pred_labels_trainIds')

	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		self.net_mod = ComparatorImageToGenAndLabels(
			num_outputs=2, freeze=True,
			num_sem_classes=self.cfg['net']['num_classes'],
		)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def tr_net(self, labels_fakeErr_trainIds, image, gen_image, **_):
		return dict(
			pred_anomaly_logits = self.net_mod(image, gen_image, labels_fakeErr_trainIds)
		)


class Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT(Exp0520_Diff_ImgAndLabelsVsGen_semGT):
	cfg = add_experiment(Exp0520_Diff_ImgAndLabelsVsGen_semGT.cfg,
		name='0521_Diff_SwapFgd_ImgAndLabelVsGen_semGT',
		gen_name='051X_semGT__fakeSwapFgd__genNoSty',
		swap_fraction=0.5,
    )

	def init_transforms(self):
		super().init_transforms()
		self.synthetic_mod = partial(tr_synthetic_swapFgd_labels, swap_fraction = self.cfg['swap_fraction'])


class Exp0552_NewDiscrepancyTraining(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT):
	"""
	An example new experiment variant using the provided discrepancy training set.
	
	Training code:
	from src.a05_differences.experiments import Exp0552_NewDiscrepancyTraining
	Exp0552_NewDiscrepancyTraining.training_procedure()

	Weights will be written to $DIR_EXP/0552_NewDiscrepancyTraining
	Checkpoints are saved every epoch:
		chk_best.pth - checkpoint with the lowest loss on eval set
		chk_last.pth - checkpoint after the most recent epoch
		optimizer.pth - optimizer data (momentum etc) after the most recent epoch
	
	The directory will also contain:
	*	`[date]_log` - predictions for sample evaluation frames indexed by epoch
	*	`training.log` - logs from the logging module, if the training procedure failed, the exception will be written there

	The loss is written to tensorboard:
		tensorboard --logdir $DIR_EXP/0552_NewDiscrepancyTraining
	"""
	cfg = add_experiment(Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT.cfg,
		name = '0552_NewDiscrepancyTraining',
		gen_name = '051X_semGT__fakeSwapFgd__genNoSty',
    )

	
# Experiments used in article
# 'gen_swap_gt': Exp0516_Diff_SwapFgd_ImgVsGen_semGT,
# 'label_swap_gt': Exp0517_Diff_SwapFgd_ImgVsLabels_semGT,
# 'lag_swap_gt': Exp0521_SwapFgd_ImgAndLabelsVsGen_semGT,



