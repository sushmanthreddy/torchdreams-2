from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

totensor = ToTensor()


default_model_input_size = (224, 224)
default_img_size = (512, 512)
default_model_input_range = (-2, 2)
