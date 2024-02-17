import typing
import torch
from ColorTransform import random_color_transform, random_cam_white_balance, apply_color_matrix, apply_white_balance
import Noise
from CFA import BayerPattern, demosaic, DemosaicMethod, create_bayer_matrix
from Convolve import gaussian_kernel, apply_kernel, circular_kernel, apply_color_kernel
from JpgDegrade import jpg_degrade


def degrade(image, kernels,
            demosaic_method: DemosaicMethod = DemosaicMethod.AHD,
            bayer_pattern: BayerPattern = BayerPattern.RGGB,
            gain: float = 16,
            bit_depth: int = 14,
            compression_quality: int = 50,
            read_noise: typing.Tuple[Noise.GaussianParams] = (Noise.GaussianParams(2047.98 - 2048, 11.536, 0.9997),
                                                              Noise.GaussianParams(2055.5 - 2048, 43.837, 0.0003)),
            row_noise: Noise.GaussianParams = Noise.GaussianParams(0, 0.163),
            col_noise: Noise.GaussianParams = Noise.GaussianParams(0, 0.38)):
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
