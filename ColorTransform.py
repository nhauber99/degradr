import random

import torch

from CamRgbMat import cam_rgb_matrices


def random_color_transform():
    return random.choice(cam_rgb_matrices)


def random_cam_white_balance():
    # values obtained by analysis of a few measurements for different cameras
    return [random.normalvariate(2.3, 0.4), 1, random.normalvariate(1.6, 0.16)]


def apply_color_matrix(image: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ij,jkl->ikl', color_matrix, image)


def apply_white_balance(image: torch.Tensor, white_balance: torch.Tensor) -> torch.Tensor:
    return image * white_balance.view(3, 1, 1)
