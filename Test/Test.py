import ctypes

import numpy as np
import torch
import PyIPP

from Analysis.Utilities import FileIO
from Analysis.Utilities.TorchUtil import np_to_torch
from Data.ColorTransform import random_color_transform, random_cam_white_balance
from Operations import Noise
from Operations.CFA import demosaic_bilinear_
from Operations.ColorConvert import sensor_to_srgb_matrix, apply_color_matrix, apply_white_balance
from Operations.Convolve import gaussian_kernel, apply_kernel, circular_kernel


def demosaic(tensor: torch.Tensor):
    PyIPP.DemosaicIPP(ctypes.c_void_p(tensor.contiguous().data_ptr()).value, tensor.shape, tensor.dtype)


if __name__ == "__main__":
    PyIPP.fast_tanh(5)
    bit_depth = 14
    pedestal = 0
    images = 1
    gain = 16
    read_noise = (Noise.GaussianParams(2047.98 - 2048, 11.536 / np.sqrt(images), 0.9997),
                  Noise.GaussianParams(2055.5 - 2048, 43.837 / np.sqrt(images), 0.0003))
    row_noise = Noise.GaussianParams(0, 0.163)
    col_noise = Noise.GaussianParams(0, 0.38)

    aberration_kernels = [gaussian_kernel(3, 0.5), circular_kernel(9, 3)]
    color_transform = random_color_transform()
    rgb2cam = torch.tensor(color_transform[0])
    cam2rgb = torch.tensor(color_transform[1])
    wb = torch.tensor(random_cam_white_balance())

    # reading
    clean = np_to_torch(FileIO.read_image('in0.tif')) ** 2.2 * (2 ** bit_depth) / gain
    clean = apply_color_matrix(clean, rgb2cam)
    clean = apply_white_balance(clean, 1. / wb)

    # degradation
    for aberration_kernel in aberration_kernels:
        clean = apply_kernel(clean, aberration_kernel)

    clean = torch.poisson(clean) * gain
    noise_tensor = Noise.gaussian_sample_combination_like(clean, read_noise)
    row_noise_tensor = Noise.row_noise_like(clean, row_noise)
    col_noise_tensor = Noise.col_noise_like(clean, col_noise)

    noisy = (clean + pedestal + noise_tensor + row_noise_tensor + col_noise_tensor) #scale read noise?
    noisy = demosaic_bilinear_(noisy, 1)

    noisy = apply_white_balance(noisy, wb)
    noisy = apply_color_matrix(noisy, cam2rgb)

    # writing
    noisy *= 1 / (2 ** bit_depth)
    noisy = noisy.clip_(0)
    FileIO.write_image_tensor('noise.tif', noisy ** 0.4545, np.uint16)
