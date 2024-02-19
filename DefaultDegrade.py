import os
import random
import typing

import albumentations
import albumentations.pytorch
import torch

import CFA
import Convolve
import FileIO
from ColorTransform import random_color_transform, random_cam_white_balance, apply_color_matrix, apply_white_balance
import Noise
from CFA import BayerPattern, demosaic, DemosaicMethod, create_bayer_matrix
from Convolve import gaussian_kernel, apply_kernel, circular_kernel, apply_color_kernel
from JpgDegrade import jpg_degrade
from Random import random_bool, random_range


def degrade(image, kernels,
            demosaic_method: DemosaicMethod = DemosaicMethod.AHD,
            bayer_pattern: BayerPattern = BayerPattern.RGGB,
            gain: float = 16,
            bit_depth: int = 14,
            compression_quality: int = 50,
            read_noise: typing.Tuple[Noise.GaussianParams] = None,
            row_noise: Noise.GaussianParams = Noise.GaussianParams(0, 0.163),
            col_noise: Noise.GaussianParams = Noise.GaussianParams(0, 0.38),
            discard_input: bool = False):
    """
        Applies common degradations on an image.
    """
    if read_noise is None:
        read_noise = (Noise.GaussianParams(0, random_range(3, 40, 2), 0.999),
                      Noise.GaussianParams(0, random_range(30, 80, 2), 0.001))

    if not discard_input:
        image = image.clone()
    max_val = image.max()
    image /= max_val
    pedestal = 0

    color_transform = random_color_transform()
    rgb2cam = torch.tensor(color_transform[0])
    cam2rgb = torch.tensor(color_transform[1])
    wb = torch.tensor(random_cam_white_balance())

    image = image * (2 ** bit_depth)
    if gain > 0:
        image /= gain

    # inverse color transformations
    image = apply_color_matrix(image, rgb2cam)
    image = apply_white_balance(image, 1. / wb)

    # blurring
    for kernel in kernels:
        image = apply_color_kernel(image.unsqueeze(0), kernel.unsqueeze(0)).squeeze(0)

    # noise
    if gain > 0:
        image = torch.poisson(torch.nan_to_num(image.clip_(0.), 0, 0, 0)) * gain
        noise_tensor = Noise.gaussian_sample_combination_like(image, read_noise)
        row_noise_tensor = Noise.row_noise_like(image, row_noise)
        col_noise_tensor = Noise.col_noise_like(image, col_noise)
        image = torch.round(image + pedestal + noise_tensor + row_noise_tensor + col_noise_tensor)  # scale read noise?

    # color transformation 1
    image = apply_white_balance(image, wb)

    # demosaic
    image = demosaic(image, bayer_pattern, demosaic_method)

    # color transformation 2
    image = apply_color_matrix(image, cam2rgb)

    # normalize
    image = (image / (2 ** bit_depth))

    # jpg compression
    if compression_quality < 100:
        image = jpg_degrade((image / 3).clip_(0, 1), compression_quality) * 3

    image *= max_val
    return image


def random_degrade(image: torch.Tensor, blur_kernels, jpg_chance: float = 0.5, discard_input: bool = False):
    kernels = [Convolve.gaussian_kernel(3, 0.75).squeeze(0)]  # anti aliasing filter

    if random_bool(0.9):
        kernels.append(random.choice(blur_kernels))
        if random_bool(0.3):
            kernels.append(Convolve.gaussian_kernel(5, random_range(0.5, 2.5)).squeeze(0))
        if random_bool(0.3):
            kernels.append(Convolve.circular_kernel(7, random_range(1.5, 5)).squeeze(0))

    return degrade(image, kernels,
                   demosaic_method=CFA.DemosaicMethod.No if random_bool(0.15) else CFA.DemosaicMethod(random.randint(0, 2)),
                   bayer_pattern=CFA.BayerPattern(random.randint(0, 3)),
                   gain=0 if random_bool(0.15) else random_range(0.1, 64, 2),
                   compression_quality=random.randint(50, 99) if random_bool(jpg_chance) else 100,
                   discard_input=discard_input)


def load_kernels(path: str):
    blur_kernels = []
    for kernel_path in os.listdir(path):
        kernel = albumentations.pytorch.ToTensorV2()(image=FileIO.read_image_rgb(os.path.join(path, kernel_path)))["image"] ** 2.2
        kernel /= kernel.sum(dim=(1, 2), keepdims=True)
        blur_kernels.append(kernel)
    return blur_kernels
