
import os
from pathlib import Path

dir_root = Path(__file__).parent / '..'

DIR_EXP = Path(os.environ.get('DIR_EXPERIMENTS', dir_root / 'exp'))
DIR_DATA = Path(os.environ.get('DIR_DATA', dir_root / 'data'))
DIR_DSETS = Path(os.environ.get('DIR_DATASETS', dir_root / 'datasets'))

