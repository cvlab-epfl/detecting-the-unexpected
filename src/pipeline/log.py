import logging, sys

def log_config_default(log_name):
	# write INFO and below to stdout
	handler_stdout = logging.StreamHandler(sys.stdout)
	handler_stdout.setFormatter(
		logging.Formatter(fmt='{message}', style='{'),
	)
	handler_stdout.addFilter(lambda r: r.levelno <= logging.INFO)
	
	# write WARNING and above to stderr, resulting in red text in jupyter
	handler_stderr = logging.StreamHandler(sys.stderr)
	handler_stderr.setFormatter(
		logging.Formatter(fmt='{levelname:<7}|{filename}:{lineno}| {message}', style='{'),
	)
	handler_stderr.setLevel(logging.WARNING)
	
	# we don't use the root log, because all libraries are writing trash to it
	# instead all our code uses the 'exp' for "experiment" log space
	log = logging.getLogger(log_name)
	log.setLevel(logging.DEBUG)
	for h in [handler_stdout, handler_stderr]: log.addHandler(h)
	return log

log = log_config_default('exp')

def log_config_file(filepath, log_obj = log):
	handler_file = logging.FileHandler(filepath)
	handler_file.setFormatter(logging.Formatter(
		fmt = '{asctime}|{levelname:<7}| {message}',
		style = '{',
		datefmt = '%m-%d %H:%M:%S'
	))
	
	# add handlers to root
	log_obj.addHandler(handler_file)
	log_obj.info(f'Log file {filepath} initialized')
