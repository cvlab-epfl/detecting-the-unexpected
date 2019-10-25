
import logging
log = logging.getLogger('exp')
import json
from pathlib import Path
from functools import partial
from ..paths import DIR_EXP

def extend_config(base, diff, warn=False):
	result = base.copy()

	for k, val in diff.items():
		if warn and k not in result:
			log.warning(f'Warning: overriding key [{k}] which is not present in the base config')
		if isinstance(val, dict):
			baseval = base.get(k, None)
			if isinstance(baseval, dict):
				# merge a dict
				result[k] = extend_config(baseval, val, warn=warn)
			else:
				# overwrite any other type of value
				result[k] = val
		else:
			result[k] = val
	return result

EXPERIMENT_CONFIGS = dict()

CONFIG_BASE = dict(
	name = 'DEFAULT',
	dir_checkpoint = '/dev/null/invalid_path',

	net = dict(
		type = '<none>',
		batch_train = 4,
		batch_eval = 16,
	),

	train = dict(
		optimizer = dict(
			type = 'adam',
			learn_rate = 1e-4,
			lr_patience = 5,
			lr_min = 1e-8,
			weight_decay = 0,
		),
		checkpoint_interval = 1,
	),
)

def add_experiment(base=CONFIG_BASE, **kwargs):
	cfg = kwargs
	name = cfg['name']
	if 'dir_checkpoint' not in cfg:
		cfg['dir_checkpoint'] = str(DIR_EXP / name)

	result = extend_config(base, cfg)
	EXPERIMENT_CONFIGS[name] = result
	return result

def config_from_file(cfg_file_path):
	cfg_file_path = Path(cfg_file_path)

	log.info('Loading config from {cfg_file_path}')

	with cfg_file_path.open('r') as fin:
		cfg = json.load(fin)

	return cfg

def config_from_name(exp_name):
	return config_from_file(DIR_EXP / exp_name / 'config.json')

NUM_CLASSES_CITYSCAPES = 19

CONFIG_PSP = extend_config(CONFIG_BASE, dict(

	net = dict(
		type = 'psp',
		num_classes = NUM_CLASSES_CITYSCAPES,
		classfunc = 'softmax',
		loss = 'psp_aux',
		batch_train = 2,
		batch_eval = 16,
		backbone_freeze = False,
	),

	train = dict(
		optimizer = dict(
			type = 'adam',
			learn_rate = 1e-4,
			lr_patience = 5,
			lr_min = 1e-8,
		),
		epoch_limit = 50,
	),
), warn=False)

class MyJSONEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, Path):
			return str(o)
		return json.JSONEncoder.default(self, o)
ENCODER = MyJSONEncoder(indent='	')

cfg_json_encode = ENCODER.encode

def print_cfg(cfg):
	log.info(cfg_json_encode(cfg))
