# Configure pytorch
import os, shutil

# Which GPU to use
def select_gpus(gpu_list):
	print('Select GPUs:', gpu_list)
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))

if 'PYTORCH_GPU' in globals() and PYTORCH_GPU is not None:
	select_gpus(PYTORCH_GPU)

print('CUDA GPU selection:', os.environ.get('CUDA_VISIBLE_DEVICES', '<no selection>'))

# Location for pretrained models

#os.environ['TORCH_MODEL_ZOO'] = "/cvlabdata1/cvlab/datasets_lis/nets/pytorch"
#os.environ['TORCH_MODEL_ZOO'] = "/home/adynathos/dev/phd/torch_pretrained"

if 'TORCH_MODEL_ZOO' not in os.environ:
	os.environ['TORCH_MODEL_ZOO'] = os.path.join(os.getcwd(), '..', 'pretrained')

print('PyTorch model zoo:', os.environ['TORCH_MODEL_ZOO'])

import torch
import torchvision
CUDA = torch.cuda.is_available()
print("CUDA:", CUDA)

def to_cuda(val):
	return val.cuda() if CUDA else val

def image_torch_to_numpy(img_torch):
	return img_torch.numpy().transpose((1, 2, 0))



