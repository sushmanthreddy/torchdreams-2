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


def box_crop_2(
    image_tensor, box_min_size=0.05, box_max_size=0.99, aspect_ratio_range=(0.5, 2.0)
):
    """
    Crop a random box from an image tensor.

    Args:
        image_tensor (torch.Tensor): The input image tensor.
        box_min_size (float, optional): The minimum size of the box as a fraction of the image dimensions. Defaults to 0.05.
        box_max_size (float, optional): The maximum size of the box as a fraction of the image dimensions. Defaults to 0.99.
        aspect_ratio_range (tuple, optional): The range of aspect ratios for the crop box. Defaults to (0.5, 2.0).

    Returns:
        torch.Tensor: The cropped image tensor.
    """
    # Unpack dimensions
    batch_size, num_channels, image_height, image_width = image_tensor.shape

    # Generate random box size within the specified range
    box_area_fraction = (
        torch.rand(1, device=image_tensor.device) * (box_max_size - box_min_size)
        + box_min_size
    )

    # Generate random aspect ratio within the specified range
    aspect_ratio = (
        torch.rand(1, device=image_tensor.device)
        * (aspect_ratio_range[1] - aspect_ratio_range[0])
        + aspect_ratio_range[0]
    )

    # Calculate box dimensions in pixels
    box_area = box_area_fraction * image_height * image_width
    box_height = torch.sqrt(box_area / aspect_ratio).int().item()
    box_width = (box_height * aspect_ratio).int().item()

    # Ensure the box dimensions are within image dimensions
    box_height = min(box_height, image_height)
    box_width = min(box_width, image_width)

    # Generate random top-left corner position for the box
    x0 = torch.randint(
        0, image_width - box_width + 1, (1,), device=image_tensor.device
    ).item()
    y0 = torch.randint(
        0, image_height - box_height + 1, (1,), device=image_tensor.device
    ).item()

    # Crop the image
    cropped_image = image_tensor[:, :, y0 : y0 + box_height, x0 : x0 + box_width]

    return cropped_image
