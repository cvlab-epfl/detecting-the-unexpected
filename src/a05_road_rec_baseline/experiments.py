import numpy as np
import torch
from ..pipeline import *
from .networks import *
from .transforms import *
from ..paths import DIR_DATA
from ..datasets.cityscapes import DatasetCityscapesSmall

# from collections import namedtuple
# NoCollate = namedtuple('NoCollate', 'value')

class NoCollate:
	def __init__(self, value):
		self.value = value


class Exp0505_RoadReconstructionRBM(ExperimentBase):
	cfg = add_experiment(
		name='0505_RBM_road',
		net=dict(
			batch_eval = 50,
			batch_train = 50,
			num_hidden = 20,
		),
		train=dict(
			optimizer=dict(
				#learn_rate=1e-5,
				learn_rate = 2e-4,
				lr_patience=5,
			),
			corruption_noise_variance = 0.005,
			num_workers = 1,
			#short_epoch_val = 200,
			#short_epoch_train = 500,
		),
		patch_size=8,
		patch_stride=6,
		epoch_limit=200,
	)


	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		print('Building net')

		ps = self.cfg['patch_size']

		self.net_mod = RoadReconstructionRBM(
			patch_shape = (ps, ps, 3),
			num_hidden = self.cfg['net']['num_hidden'],
		)

		if chk is not None:
			print('Loading weights from checkpoint')
			self.net_mod.load_state_dict(chk['weights'])

		self.cuda_modules(['net_mod'])

	def join_patch_lists_nocollate(self, image_patches, **_):
		return dict(
			image_patches = np.concatenate([v.value for v in image_patches], axis=0),
		)

	def join_patch_lists(self, image_patches, **_):
		"""

		"""
		return dict(
			num_frames_in_batch = image_patches.shape[0],
			image_patches = image_patches.reshape((-1, ) + image_patches.shape[2:]),
		)

	def unjoin_patch_lists(self, num_frames_in_batch, **fields):
		out = dict(
			num_frames_in_batch = None,
		)

		for k, v in fields.items():
			if k.endswith('_patches'):
				out[k] = v.reshape((num_frames_in_batch, -1) + v.shape[1:])

		return out

	def init_log(self):
		super().init_log()

	def init_transforms(self):
		super().init_transforms()
		self.preprocess_test = TrsChain(
			tr_extract_patches_all,
			# TrAsType(dict(image_patches=np.float32)),
		)

	def tr_preprocess_train_val(self, image, labels, **_):
		patches_road = tr_extract_patches_road(
			image, labels, self.cfg['patch_size'], self.cfg['patch_stride']
		)['patches_road']

		#patches_road = patches_road.astype(np.float32)

		return dict(
			image_patches = NoCollate(patches_road),
		)

	def tr_loss(self, image_patches, **_):
		return self.net_mod.loss(image_patches, noise_var = self.cfg['train']['corruption_noise_variance'])


	def construct_default_pipeline(self, role):
		# preprocess_train_val = TrsChain(
		# 	self.tr_preprocess_train_val,
		# 	TrKeepFields('image_patches'),
		# )


		if role == 'test':
			return Pipeline(
				tr_input= self.preprocess_test,
				tr_batch_pre_merge=TrKeepFields('image_patches'),
				tr_batch=TrsChain(
					self.join_patch_lists,
					TrCUDA(),
					self.net_mod,
					TrKeepFields('num_frames_in_batch', 'reconstructed_patches', 'discrepancy_patches'),
					self.unjoin_patch_lists,
				),
				tr_output=TrsChain(
					TrRebuildFromPatches(fields=dict(
						reconstructed_patches = 'reconstructed',
						discrepancy_patches = 'anomaly_p',
					)),
					TrNP(),
					TrAsType(dict(reconstructed=np.uint8))
				),
				loader_args=self.loader_args_for_role(role),
			)

		elif role == 'val':
			return Pipeline(
				#tr_input=self.tr_preprocess_train_val,
				tr_batch_pre_merge=TrKeepFields('image_patches'),
				tr_batch=TrsChain(
					self.join_patch_lists_nocollate,
					TrCUDA(),
					self.tr_loss,
					TrKeepFieldsByPrefix('loss'),
				),
				tr_output=TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args=self.loader_args_for_role(role),
			)

		elif role == 'train':
			return Pipeline(
				# tr_input=self.tr_preprocess_train_val,
				tr_batch_pre_merge=TrKeepFields('image_patches'),
				tr_batch=TrsChain(
					self.join_patch_lists_nocollate,
					TrCUDA(),
					self.tr_loss,
					self.training_backpropagate,
					TrKeepFieldsByPrefix('loss'),
				),
				tr_output=TrsChain(
					TrKeepFieldsByPrefix('loss'),
					TrNP(),
				),
				loader_args=self.loader_args_for_role(role),
			)
		else:
			raise NotImplementedError(role)

	def init_pipelines(self):
		super().init_pipelines()

		self.pipelines['train'].loader_class = SamplerThreaded
		self.pipelines['val'].loader_class = SamplerThreaded

	def setup_dset(self, dset):
		dset.discover()

	def init_default_datasets(self, b_threaded=True):
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

		dset_ctc_train.tr_post_load_pre_cache.append(self.tr_preprocess_train_val)
		dset_ctc_val.tr_post_load_pre_cache.append(self.tr_preprocess_train_val)


		self.frames_to_log = set([dset_ctc_val.frames[i].fid for i in [0, 1, 2, 3, 6, 8, 9]])

		self.set_dataset('train', dset_ctc_train)
		self.set_dataset('val', dset_ctc_val)


class Exp0525_RoadReconstructionRBM_LowLR(Exp0505_RoadReconstructionRBM):
	cfg = add_experiment(Exp0505_RoadReconstructionRBM.cfg,
		name='0525_RBM_road_lowLR',
		train=dict(
			optimizer=dict(
				learn_rate = 1e-5,
			)
		)
	)



class Exp0526_RoadReconstructionRBM_GlobalNorm(Exp0505_RoadReconstructionRBM):
	cfg = add_experiment(Exp0505_RoadReconstructionRBM.cfg,
		name='0526_RBM_road_globalNorm',
         net=dict(
             batch_eval=50,
             batch_train=50,
             num_hidden=20,
         ),
         train=dict(
             optimizer=dict(
                 # learn_rate=1e-5,
                 learn_rate=2e-4,
                 lr_patience=5,
             ),
             corruption_noise_variance=0.005,
         ),
	)

	def init_transforms(self):
		super().init_transforms()
		self.preprocess_test = TrsChain(
			tr_extract_patches_all,
			self.tr_normalize_img_patches,
		)

	def build_net(self, role, chk=None, chk_optimizer=None):
		super().build_net(role, chk, chk_optimizer)

		self.net_mod.local_normalize = False

	def tr_normalize_img_patches(self, image_patches, **_):

		mean = torch.mean(image_patches, dim=1)
		image_patches = image_patches - mean[:, None]
		std = torch.std(image_patches, dim=1)
		image_patches *= (1/std[:, None])

		return dict(
			image_patches = image_patches,
			image_patches_mean = mean,
			image_patches_std = std,
		)

	def tr_preprocess_train_val(self, image, labels, **_):
		patch_size = self.cfg['patch_size']
		stride = self.cfg['patch_stride']
		
		patches_img = extract_square_patches(image, patch_size=patch_size, stride=stride)
		patches_labels = extract_square_patches(labels, patch_size=patch_size, stride=stride)

		idx_road, idx_not_road = patches_check_road(patches_labels)

		norm_res = self.tr_normalize_img_patches(patches_img)

		return dict(
			patches_road = NoCollate(norm_res['image_patches'][idx_road]),
			image_patches_mean = norm_res['image_patches_mean'],
			image_patches_std = norm_res['image_patches_std'],
		)
