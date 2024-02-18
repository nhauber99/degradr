import numpy as np
import torch

import DefaultDegrade
import FileIO
from Analysis.Utilities.TorchUtil import np_to_torch
from ColorTransform import random_color_transform, random_cam_white_balance, apply_color_matrix, apply_white_balance
import Noise
from CFA import BayerPattern, demosaic, DemosaicMethod, create_bayer_matrix
from Convolve import apply_color_kernel
from JpgDegrade import jpg_degrade
from Prep.PrepKernels import prep_kernels
from Prep.ZernikePSF import gen_zernike_kernels

if __name__ == "__main__":
    bit_depth = 14
    pedestal = 0
    images = 1
    gain = 16
    read_noise = (Noise.GaussianParams(2047.98 - 2048, 11.536 / np.sqrt(images), 0.9997),
                  Noise.GaussianParams(2055.5 - 2048, 43.837 / np.sqrt(images), 0.0003))
    row_noise = Noise.GaussianParams(0, 0.163)
    col_noise = Noise.GaussianParams(0, 0.38)

    kernel = np_to_torch(FileIO.read_image_rgb("Kernels/4.tif")).unsqueeze(0) ** 2.2
    kernel /= kernel.sum(dim=(2, 3), keepdims=True)
    aberration_kernels = [kernel]
    # aberration_kernels = [gaussian_kernel(3, 0.5), circular_kernel(9, 3)]
    color_transform = random_color_transform()
    rgb2cam = torch.tensor(color_transform[0])
    cam2rgb = torch.tensor(color_transform[1])
    wb = torch.tensor(random_cam_white_balance())
    upsample = torch.nn.UpsamplingNearest2d(scale_factor=4)

    # reading
    clean = np_to_torch(FileIO.read_image_rgb('Test/in.tif')) ** 2.2 * (2 ** bit_depth) / gain

    FileIO.write_image_tensor('Test/in.png', upsample(np_to_torch(FileIO.read_image_rgb('Test/in.tif')).unsqueeze(0)).squeeze(0), np.uint16)
    clean = apply_color_matrix(clean, rgb2cam)
    clean = apply_white_balance(clean, 1. / wb)

    # degradation
    blurry = clean
    for aberration_kernel in aberration_kernels:
        blurry = apply_color_kernel(blurry.unsqueeze(0), aberration_kernel).squeeze(0)
    noisy = torch.poisson(blurry) * gain
    noise_tensor = Noise.gaussian_sample_combination_like(noisy, read_noise)
    row_noise_tensor = Noise.row_noise_like(noisy, row_noise)
    col_noise_tensor = Noise.col_noise_like(noisy, col_noise)

    noisy = torch.round(noisy + pedestal + noise_tensor + row_noise_tensor + col_noise_tensor)  # scale read noise?
    blurry = apply_white_balance(blurry * gain, wb)
    noisy = apply_white_balance(noisy, wb)
    noisy_bayer = noisy * create_bayer_matrix((noisy.shape[1], noisy.shape[2]), noisy.device)
    noisy_ahd = demosaic(noisy, BayerPattern.GRBG, DemosaicMethod.AHD)
    noisy_vng = demosaic(noisy, BayerPattern.GRBG, DemosaicMethod.VNG)
    noisy_leg = demosaic(noisy, BayerPattern.GRBG, DemosaicMethod.Legacy)

    noisy = apply_color_matrix(noisy, cam2rgb)
    blurry = apply_color_matrix(blurry, cam2rgb)
    noisy_ahd = apply_color_matrix(noisy_ahd, cam2rgb)
    noisy_vng = apply_color_matrix(noisy_vng, cam2rgb)
    noisy_leg = apply_color_matrix(noisy_leg, cam2rgb)

    # writing
    blurry = (blurry / (2 ** bit_depth)).clip_(0)
    noisy = (noisy / (2 ** bit_depth)).clip_(0)
    noisy_bayer = (noisy_bayer / (2 ** bit_depth)).clip_(0)
    noisy_ahd = (noisy_ahd / (2 ** bit_depth)).clip_(0, 1)
    noisy_vng = (noisy_vng / (2 ** bit_depth)).clip_(0, 1)
    noisy_leg = (noisy_leg / (2 ** bit_depth)).clip_(0, 1)
    noisy_ahd_jpg = jpg_degrade(noisy_ahd, 50)
    FileIO.write_image_tensor('Test/blur.png', upsample((blurry ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
    FileIO.write_image_tensor('Test/noise_blur.png', upsample((noisy ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
    FileIO.write_image_tensor('Test/noisy_blur_bayer.png', upsample((noisy_bayer ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
    FileIO.write_image_tensor('Test/noise_blur_ahd.png', upsample((noisy_ahd ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
    FileIO.write_image_tensor('Test/noise_blur_vng.png', upsample((noisy_vng ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
    FileIO.write_image_tensor('Test/noise_blur_legacy.png', upsample((noisy_leg ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
    FileIO.write_image_tensor('Test/noise_blur_ahd_jpg.png', upsample((noisy_ahd_jpg ** 0.4545).unsqueeze(0)).squeeze(0), np.uint16)
