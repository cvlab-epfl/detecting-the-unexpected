"""
	Initialization for jupyter-notebook files
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg') # prevent loading Qt which crashes
import matplotlib.pyplot as plt
from .jupyter_show_image import show
from tqdm import tqdm
import os, datetime

# Init display style
#get_ipython().magic('matplotlib inline')
#get_ipython().magic('matplotlib notebook')
np.set_printoptions(suppress=True, linewidth=180)

# IPY Progress Bar
# class ProgressBar:
# 	def __init__(self, goal):
# 		self.bar = IntProgress(min=0, max=goal, value=0)
# 		self.label = HTML()
# 		box = HBox(children=[self.bar, self.label])

# 		self.goal = goal
# 		self.value = 0
# 		self.template = '{{0}} / {goal}'.format(goal=goal)

# 		self.set_value(0)

# 		display(box)

# 	def set_value(self, new_val):
# 		self.value = new_val
# 		self.bar.value = new_val
# 		self.label.value = self.template.format(new_val)

# 		if new_val >= self.goal:
# 			self.bar.bar_style = 'success'

# 	def __iadd__(self, change):
# 		self.set_value(self.value + change)
# 		return self


class ProgressBar:
	def __init__(self, goal):
		self.goal = goal
		self.value = 0
		self.bar = tqdm(total=goal)

	def __iadd__(self, change):
		self.value += change
		self.bar.update(change)

		if self.value >= self.goal:
			self.bar.close()
		
		return self

def print_time():
	print(datetime.datetime.now().isoformat(sep='_'))

def get_memory_used():
	import psutil
	process = psutil.Process(os.getpid())
	mem = process.memory_info().rss

	print(mem / (1 << 20), 'MB')

	return mem

def memory_diag():
	import gc
	import psutil

	print('RAM:')
	process = psutil.Process(os.getpid())
	mem = process.memory_info().rss
	print('	before:', mem / (1 << 20), 'MB')
	print('	GC:', gc.collect())

	try:
		import torch
		torch.cuda.empty_cache()
	except Exception as e:
		print('	torch:', e)

	process = psutil.Process(os.getpid())
	mem = process.memory_info().rss
	print('	after: ', mem / (1 << 20), 'MB')
