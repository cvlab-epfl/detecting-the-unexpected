
import os
from pathlib import Path

DIR_HERE = Path(__file__).absolute().parent

os.environ.update({
	'DIR_EXPERIMENTS': str(DIR_HERE.parents[1] / 'exp'),
	'DIR_LAF_SMALL': '/cvlabsrc1/cvlab/dataset_LostAndFound/1024x512_webp',
	'DIR_CITYSCAPES_SMALL': '/cvlabsrc1/cvlab/dataset_Cityscapes/1024x512',
	'DIR_ROAD_ANOMALY': '/cvlabsrc1/cvlab/dataset_RoadAnomaly',
})
