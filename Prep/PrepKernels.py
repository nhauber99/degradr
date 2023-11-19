import os
import random

import torch.nn
import torchvision.transforms.functional

from Analysis.Utilities import FileIO
from Analysis.Utilities.TorchUtil import np_to_torch
import Convolve


def center_kernel(kernel):
    patch_size = kernel.shape[1]
    ks = kernel.mean(dim=0)
    x, y = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))
    xm = -int(round((float((ks * x).sum()) - patch_size / 2 - 0.5)))
    ym = -int(round((float((ks * y).sum()) - patch_size / 2 - 0.5)))
    kt = torch.zeros_like(kernel)
    kt[:, max(0, 0 + xm):min(patch_size, patch_size + xm), max(0, 0 + ym):min(patch_size, patch_size + ym)] = \
        kernel[:, max(0, 0 - xm):min(patch_size, patch_size - xm), max(0, 0 - ym):min(patch_size, patch_size - ym)]
    return kt


def prep_kernels(in_path: str, out_path: str):
    for kernel_path in os.listdir(in_path):
        kernel = np_to_torch(FileIO.read_image(os.path.join(in_path, kernel_path))) ** 2.2
        kernel = torchvision.transforms.functional.rotate(kernel, random.uniform(0, 360), torchvision.transforms.InterpolationMode.BILINEAR)
        kernel = Convolve.apply_kernel(kernel, Convolve.gaussian_kernel(15, 1.5))
        kernel = (kernel - 2 * kernel.quantile(0.5)).clip_(0)
        kernel = torch.nn.UpsamplingBilinear2d((15 * 9, 15 * 9))(kernel.unsqueeze(0)).squeeze(0)
        kernel /= kernel.sum(dim=(1, 2), keepdim=True)
        kernel = center_kernel(kernel)
        kernel = torch.nn.AvgPool2d(9)(kernel.unsqueeze(0)).squeeze(0)
        kernel /= kernel.max()
        FileIO.write_image_tensor(os.path.join(out_path, kernel_path), kernel ** 0.45)


if __name__ == "__main__":
    prep_kernels("ZernikeKernels", "Kernels")
