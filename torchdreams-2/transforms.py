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


def box_crop_2(image_tensor, box_min_size=0.05, box_max_size=0.99):
    """
    Crop a random box from an image tensor.

    Args:
        image_tensor (torch.Tensor): The input image tensor.
        box_min_size (float, optional): The minimum size of the box. Defaults to 0.05.
        box_max_size (float, optional): The maximum size of the box. Defaults to 0.99.

    Returns:
        torch.Tensor: The cropped image tensor.
    """
    image = image_tensor
    batch_size, num_channels, image_width, image_height = image.shape

    box_width_fraction = torch.rand(1) * (box_max_size - box_min_size) + box_min_size
    box_height_fraction = box_width_fraction

    max_x0_fraction = 1 - box_width_fraction
    max_y0_fraction = 1 - box_height_fraction
    x0_fraction = torch.rand(1) * max_x0_fraction
    y0_fraction = torch.rand(1) * max_y0_fraction

    x_start = int(x0_fraction * image_width)
    x_end = int((x0_fraction + box_width_fraction) * image_width)
    y_start = int(y0_fraction * image_height)
    y_end = int((y0_fraction + box_height_fraction) * image_height)

    cropped_image = image[:, :, x_start:x_end, y_start:y_end]

    return cropped_image
