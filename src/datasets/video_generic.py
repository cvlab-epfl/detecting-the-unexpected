import logging
log = logging.getLogger('exp.dset_video')

from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2 as cv
from .dataset import DatasetBase
from ..paths import DIR_DATA
from ..pipeline.frame import Frame

class DatasetVideo(DatasetBase):
		
	def __init__(self, name, video_path, interval=30):
		super().__init__(b_cache=True)
		
		self.name = name
		self.split = ''
		self.dir_out = DIR_DATA / self.name
		self.video_path = video_path
		self.interval = 30
		

	def discover(self):
		self.frames = self.load_video_frames(self.video_path, interval=self.interval)
		super().discover()

	def load_video_frames(self, video_path, interval=30):
		reader = cv.VideoCapture(str(video_path))
		num_frames = int(reader.get(cv.CAP_PROP_FRAME_COUNT))
		log.info(f'video: {num_frames} frames in {video_path}')
		
		frames = []
		
		for fridx in tqdm(range(0, num_frames, interval)):
			reader.set(cv.CAP_PROP_POS_FRAMES, fridx)
			
			success, image = reader.read()
			
			if not success:
				log.warning(f'success is false in cv.VideoCapture.read for file {video_path}')
				return frames
			
			frames.append(Frame(
				fid = f'{fridx:05d}',
				image = image[:, :, [2,1,0]],
				dset = self,
			))
		
		return frames
	