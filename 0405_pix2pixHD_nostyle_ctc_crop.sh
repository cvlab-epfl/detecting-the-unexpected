#!/bin/bash
cd /home/lis/dev/unknown_dangers/pix2pixHD

# Crop config:
# 	data/base_dataset.py#L35
#	base_options.py#:25 - #L32
#	--fineSize 		= size of square crop
#	--resize_or_crop	
#	

Name='0405_pix2pixHD512_nostyle_ctc_crop'

python3 train.py --name $Name \
	--checkpoints_dir /cvlabdata2/home/lis/exp --dataroot /cvlabdata2/cvlab/dataset_cityscapes_downsampled/for_pix2pixHD_training \
	--no_instance \
	--resize_or_crop crop --fineSize 384 --batchSize 4 \
	--tf_log
