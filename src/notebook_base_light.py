
from IPython import get_ipython
ipython = get_ipython()

ipython.run_cell("""
%load_ext autoreload
%aimport -numpy -cv2 -torch -matplotlib -matplotlib.pyplot
%autoreload 2
""")

from src.common.util_notebook import *
if not globals().get('PLT_STYLE_OVERRIDE'):
	plt.style.use('dark_background')
else:
	print('No plt style set')
plt.rcParams['figure.figsize'] = (12, 8)
ipython.run_cell("""
%matplotlib inline
%config InlineBackend.figure_format = 'png'
""")

import numpy as np
import cv2 as cv
from IPython.display import display

from src.pipeline.log import log
from src.pipeline.frame import Frame
from src.common.jupyter_show_image import show

from src.pipeline.transforms import TrByField, TrBase, TrsChain, TrKeepFields, TrAsType, TrKeepFieldsByPrefix, tr_print
