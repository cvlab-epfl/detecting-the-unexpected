
from .frame import *
from .pipeline import *
from .config import *
from .transforms import *
from .transforms_pytorch import *

import logging
log = logging.getLogger('exp')
from .log import log_config_file
from pathlib import Path

from ..common.util_notebook import ProgressBar
import numpy as np
import torch
import os, json, gc, datetime, time, shutil

from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
	from apex.optimizers import FusedAdam as AdamOptimizer
except ImportError:
	from torch.optim import Adam as AdamOptimizer

class GrumpyDict(dict):
	def __setitem__(self, key, value):
		if key not in self:
			log.warning('Dict Warning: setting key [{k}] which was not previously set'.format(k=key))
		super().__setitem__(key, value)

def train_state_init():
	train_state = GrumpyDict(
		epoch_idx = 0,
		best_loss_val = 1e5,
		#histories = {},
		run_name = 'training_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
	)
	return train_state

class ExperimentBase():
	# TODO train and val as pipelines
	
	def __init__(self, cfg=None):
		cfg = cfg or self.cfg # try reading from static variable

		self.datasets = {}
		self.pipelines = {}
		self.init_config(cfg)
		self.init_transforms()
	
	def init_config(self, cfg):
		self.cfg = cfg
		self.workdir = Path(cfg['dir_checkpoint'])

		self.workdir.mkdir(exist_ok=True, parents=True)

		(self.workdir / 'config.json').write_text(cfg_json_encode(self.cfg))
	
	def print_cfg(self):
		print_cfg(self.cfg)

	def init_transforms(self):
		""" Init and store transforms that take time to construct 
			The transforms will be used in self.construct_default_pipeline
		"""
		# Lifecycle of a frame
		# in dataset:
		# 	dset.tr_post_load_pre_cache
		# 	dset.tr_output
		# in experiment:
		pass

	#def sampler_for_dset(self, role, dset):
		#tr_role = self.tr_input_per_role.get(role, None)
		#tr_in = self.tr_input if tr_role is None else TrsChain(tr_role, self.tr_input)

		#collate_fn = partial(self.dataloader_collate, tr_in, self.tr_input_post_batch)

		#args = self.loader_args_for_role(role)

		#return DataLoader(dset, collate_fn=collate_fn, **args)

	def set_dataset(self, role, dset):
		"""
			role "train" or "val"
		"""

		self.datasets[role] = dset

	def load_checkpoint(self, chk_name = 'chk_best.pth'):
		dir_chk = Path(self.cfg['dir_checkpoint'])
		path_chk = dir_chk / chk_name

		if path_chk.is_file():
			log.info(f'Loading checkpoint found at {path_chk}')
			return torch.load(path_chk, map_location='cpu')
		else:
			log.info(f'No checkpoint at at {path_chk}')
			return None

	def init_net(self, role):
		""" Role: val or train - determines which checkpoint is loaded"""
		
		if role == 'train':
			chk = self.load_checkpoint(chk_name='chk_last.pth')
			chk_opt = self.load_checkpoint(chk_name='optimizer.pth')
			
			self.build_net(role, chk=chk)
			self.build_optimizer(role, chk_optimizer=chk_opt)
			self.net_mod.train()

		elif role == 'eval':
			chk = self.load_checkpoint(chk_name='chk_best.pth')
			self.build_net(role, chk=chk)
			self.net_mod.eval()

		else:
			raise NotImplementedError(f'role={role}')
		
		if chk is not None:
			self.state = GrumpyDict(chk['state'])
		else:
			self.state = train_state_init()
			
	def build_net(self, role, chk=None, chk_optimizer=None):
		""" Build net and optimizer (if we train) """
		log.info('Building net')


	@staticmethod
	def load_checkpoint_to_net(net_mod, chk_object):
		(missing_keys, superfluous_keys) = net_mod.load_state_dict(chk_object['weights'], strict=False)
		if missing_keys:
			log.warning(f'Missing keys when loading a checkpoint: {missing_keys}')
		if superfluous_keys:
			log.warning(f'Missing keys when loading a checkpoint: {superfluous_keys}')

	def build_optimizer(self, role, chk_optimizer=None):
		log.info('Building optimizer')

		cfg_opt = self.cfg['train']['optimizer']

		network = self.net_mod
		self.optimizer = AdamOptimizer(
			[p for p in network.parameters() if p.requires_grad],
			lr=cfg_opt['learn_rate'],
			weight_decay=cfg_opt.get('weight_decay', 0),
		)
		self.learn_rate_scheduler = ReduceLROnPlateau(
			self.optimizer,
			patience=cfg_opt['lr_patience'],
			min_lr = cfg_opt['lr_min'],
		)

		if chk_optimizer is not None:
			self.optimizer.load_state_dict(chk_optimizer['optimizer'])


	def init_loss(self):
		log.info('Building loss_mod')

	def init_log(self, fids_to_display=[]):
		"""
		:param fids_to_display: ids of frames to show in tensorboard
		"""
		# log for the current training run
		self.tboard = SummaryWriter(self.workdir / f"tb_{self.state['run_name']}")
		# save ground truth here to compare in tensorboard
		self.tboard_gt = SummaryWriter(self.workdir / 'tb_gt')
		self.tboard_img = SummaryWriter(self.workdir / 'tb_img')

		self.train_out_dir = self.workdir / f"imgs_{self.state['run_name']}"
		self.train_out_dir.mkdir(exist_ok=True, parents=True)

		self.txt_log = os

		# names of the frames to display
		def short_frame_name(fn):
			# remove directory path
			if '/' in fn:
				fn = os.path.basename(fn)
			return fn
			
		self.fids_to_display = set(fids_to_display)
		self.short_frame_names = {
			fid: short_frame_name(fid)
			for fid in self.fids_to_display
		}

	def log_selected_images(self, fid, frame, **_):
		if fid in self.fids_to_log:
			log.warning('log_selected_images: not implemented')

	def init_default_datasets(self):
		pass

	def init_pipelines(self):
		for role in ['train', 'val', 'test']:
			self.pipelines[role] = self.construct_default_pipeline(role)

	def get_epoch_limit(self):
		return self.cfg['train'].get('epoch_limit', None)

	def cuda_modules(self, attr_names):
		if torch.cuda.is_available():
			attr_names = [attr_names] if isinstance(attr_names, str) else attr_names

			for an in attr_names:
				setattr(self, an, getattr(self, an).cuda())

	
	def training_start_batch(self, **_):
		self.optimizer.zero_grad()
	
	def training_backpropagate(self, loss, **_):
		#if torch.any(torch.isnan(loss)):
		#	print('Loss is NAN, cancelling backpropagation in batch')

			#raise Exception('Stopping training so we can investigate where the nan is coming from')
		#else:
		loss.backward()
		self.optimizer.step()

	def training_epoch_start(self, epoch_idx):
		self.net_mod.train() # set train mode for dropout and batch-norm

	def training_epoch(self, epoch_idx):
		self.training_epoch_start(epoch_idx)

		out_frames = self.pipelines['train'].execute(
			dset = self.datasets['train'], 
			b_grad = True,
			b_pbar = False,
			b_accumulate = True,
			log_progress_interval = self.cfg['train'].get('progress_interval', None),
			short_epoch=self.cfg['train'].get('short_epoch_train', None),
		)
		gc.collect()

		results_avg = Frame({
			# the loss may be in fp16, let's average it at high precision to avoid NaN
			fn: np.mean(np.array([pf[fn] for pf in out_frames], dtype = np.float64))
			for fn in out_frames[0].keys() if fn.lower().startswith('loss')
		})

		self.training_epoch_finish(epoch_idx, results_avg)

		return results_avg['loss']

	def training_epoch_finish(self, epoch_idx, results_avg):
		for name, loss_avg in results_avg.items():
			self.tboard.add_scalar('train_'+name, loss_avg, epoch_idx)
	
	def val_epoch_start(self, epoch_idx):
		self.net_mod.eval()

	def val_epoch_finish(self, epoch_idx, results_avg):
		self.learn_rate_scheduler.step(results_avg['loss'])

		for name, loss_avg in results_avg.items():
			self.tboard.add_scalar('val_'+name, loss_avg, epoch_idx)

	def val_epoch(self, epoch_idx):
		self.val_epoch_start(epoch_idx)
		
		out_frames = self.pipelines['val'].execute(
			dset = self.datasets['val'], 
			b_grad = False,
			b_pbar = False,
			b_accumulate = True,
			short_epoch=self.cfg['train'].get('short_epoch_val', None),
		)
		gc.collect()

		results_avg = Frame({
			fn: np.mean([pf[fn] for pf in out_frames])
			for fn in out_frames[0].keys() if fn.lower().startswith('loss')
		})

		self.val_epoch_finish(epoch_idx, results_avg)

		return results_avg['loss']

	def run_epoch(self, epoch_idx):

		gc.collect()
		epoch_limit = self.get_epoch_limit()
		log.info('E {ep:03d}{eplimit}\n	train start'.format(
			ep=epoch_idx,
			eplimit=f' / {epoch_limit}' if epoch_limit is not None else '',
		))
		t_train_start = time.time()
		loss_train = self.training_epoch(epoch_idx)
		gc.collect()
		t_val_start = time.time()
		log.info('	train finished	t={tt}s	loss_t={ls}, val starting'.format(
			tt=t_val_start - t_train_start,
			ls=loss_train,
		))

		gc.collect()
		loss_val = self.val_epoch(epoch_idx)
		gc.collect()
		log.info('	val finished	t={tt}s	loss_e={ls}'.format(
			tt=time.time() - t_val_start,
			ls=loss_val,
		))

		is_best = loss_val < self.state['best_loss_val']
		if is_best:
			self.state['best_loss_val'] = loss_val
		is_chk_scheduled = epoch_idx % self.cfg['train']['checkpoint_interval'] == 0

		if is_best or is_chk_scheduled:
			self.save_checkpoint(epoch_idx, is_best, is_chk_scheduled)

	def save_checkpoint(self, epoch_idx, is_best, is_scheduled):

		# TODO separate methods for saving various parts of the experiment
		chk_dict = dict()
		chk_dict['weights'] = self.net_mod.state_dict()
		chk_dict['state'] = dict(self.state)

		path_best = self.workdir / 'chk_best.pth'
		path_last = self.workdir / 'chk_last.pth'

		if is_scheduled:
			pytorch_save_atomic(chk_dict, path_last)

			pytorch_save_atomic(dict(
					epoch_idx = epoch_idx,
					optimizer = self.optimizer.state_dict()
				), 
				self.workdir / 'optimizer.pth',
			)

		if is_best:
			log.info('	New best checkpoint')
			if is_scheduled:
				# we already saved to chk_last.pth
				shutil.copy(path_last, path_best)
			else:
				pytorch_save_atomic(chk_dict, path_best)

	def training_run(self, b_initial_eval=True):
		name = self.cfg['name']
		log.info(f'Experiment {name} - train')

		path_stop = self.workdir / 'stop'

		if b_initial_eval:
			log.info('INIT\n	initial val')
			loss_val = self.val_epoch(self.state['epoch_idx'])

			log.info('	init loss_e={le}'.format(le=loss_val))
			self.state['best_loss_val'] = loss_val
		else:
			self.state['best_loss_val'] = 1e4

		b_continue = True

		while b_continue:			
			self.state['epoch_idx'] += 1
			self.run_epoch(self.state['epoch_idx'])

			if path_stop.is_file():
				log.info('Stop file detected')
				path_stop.unlink() # remove file
				b_continue = False

			epoch_limit = self.get_epoch_limit()
			if (epoch_limit is not None) and (self.state['epoch_idx'] >= epoch_limit):
				log.info(f'Reached epoch limit {epoch_limit}')
				b_continue = False

	@classmethod
	def training_procedure(cls):
		print(f'-- Training procesure for {cls.__name__} --')
		exp = cls()
		log_config_file(exp.workdir / 'training.log')
		
		log.info(f'Starting training job for {cls.__name__}')		
		exp.print_cfg()

		try:
			exp.init_default_datasets()

			exp.init_net("train")
			exp.init_transforms()
			exp.init_loss()
			exp.init_log()
			exp.init_pipelines()
			exp.training_run()
		# if training crashes, put the exception in the log
		except Exception as e:
			log.exception(f'Exception in taining procedure: {e}')

	def predict_sequence(self, dset, consumer=None, pbar=True):
		"""
		If consumer is specified, it will be used for online processing:
		frames will be given to it instead of being accumulated
		"""
		self.net_mod.eval()
		out_frames = self.pipelines['test'].execute(
			dset = self.datasets['test'], 
			b_grad = False,
			b_pbar = pbar,
			b_accumulate = True,
		)
		return out_frames

	def loader_args_for_role(self, role):
		if role == 'train':
			return  dict(
				shuffle = True,
				batch_size = self.cfg['net']['batch_train'],
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = True,
			)
		elif role == 'val' or role == 'test':
			return  dict(
				shuffle = False,
				batch_size = self.cfg['net']['batch_eval'],
				num_workers = self.cfg['train'].get('num_workers', 0),
				drop_last = False,
			)
		else:
			raise NotImplementedError("role: " + role)

	def construct_default_pipeline(self, role):
		if role == 'train':
			tr_batch = TrsChain([
				TrCUDA(),
				self.training_start_batch,
				self.net_mod,
				self.loss_mod,
				self.training_backpropagate,
			])
			tr_output = TrsChain([
				TrKeepFieldsByPrefix('loss'),  # save loss for averaging later
				TrNP(), # clear away the gradients if any are left
			])
			
		elif role == 'val':
			tr_batch = TrsChain([
				TrCUDA(),
				self.net_mod,
				self.loss_mod,
			])
			tr_output = TrsChain([
				self.log_selected_images,
				TrKeepFieldsByPrefix('loss'), # save loss for averaging later
				TrNP(), # clear away the gradients if any are left
			])

		elif role == 'test':
			tr_batch = TrsChain([
				TrCUDA(),
				self.net_mod,
			])
			tr_output = TrsChain([
				TrNP(),
				tr_untorch_images,
			])
	
		return Pipeline(
			tr_batch = tr_batch,
			tr_output = tr_output,
			loader_args = self.loader_args_for_role(role),
		)

	def run_evaluation(self, eval_obj, dset=None, b_one_batch=False):
		pipe_test = self.construct_default_pipeline('test')
		dset = dset or eval_obj.get_dset()

		eval_obj.construct_transforms(dset)

		pipe_test.tr_batch.append(eval_obj.tr_batch)
		pipe_test.tr_output.append(eval_obj.tr_output)

		log.info(f'Test pipeline: {pipe_test}')
		pipe_test.execute(dset, b_accumulate=False, b_one_batch=b_one_batch)


def pytorch_save_atomic(data, filepath):
	""" Don't lose the previous checkpoint if we get interrupted while writing """
	filepath = Path(filepath)
	filepath_tmp = filepath.with_suffix('.tmp')
	torch.save(data, filepath_tmp)
	shutil.move(filepath_tmp, filepath)

def experiment_class_by_path(path):
	modname, _, objname = path.rpartition('.')
	return getattr(import_module(modname), objname)
