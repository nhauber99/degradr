import ctypes
import typing
from enum import Enum

import torch
import os

os.add_dll_directory(r"C:\SPRJ\degradr")

import PyIPP


class BayerPattern(Enum):
    BGGR = 0
    RGGB = 1
    GBRG = 2
    GRBG = 3


class DemosaicMethod(Enum):
    No = -1
    AHD = 0  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/demosaicahd.html
    VNG = 1  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/cfatobgra.html
    Legacy = 2  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/cfatorgb.html


def demosaic(image: torch.Tensor, pattern: BayerPattern = BayerPattern.RGGB, method: DemosaicMethod = DemosaicMethod.VNG) -> torch.Tensor:
    """
    Demosaics an image. The image is assumed to be a normal RGB image without the bayer matrix applied to it.
    """
    if method == DemosaicMethod.No:
        return image
    result = torch.zeros_like(image).contiguous()
    temp = bayer_filter(image.contiguous().clip(0, 2 ** 16 - 0.001), pattern)
    PyIPP.Demosaic(
        ctypes.c_void_p(temp.data_ptr()).value,
        ctypes.c_void_p(result.data_ptr()).value,
        image.shape[0],
        image.shape[1],
        image.shape[2],
        int(method.value),
        int(pattern.value)
    )
    return result


def create_bayer_matrix(shape: typing.Tuple, device: str = "cuda") -> torch.Tensor:
    """
    Creates an RGGB bayer matrix (mainly used for debugging/visualization)
    """
    bayer_tensor = torch.zeros((3, shape[0], shape[1]), device=device)
    bayer_tensor[0, ::2, ::2] = 1
    bayer_tensor[1, ::2, 1::2] = 1
    bayer_tensor[1, 1::2, ::2] = 1
    bayer_tensor[2, 1::2, 1::2] = 1
    return bayer_tensor


def bayer_filter(image: torch.Tensor, pattern: BayerPattern = BayerPattern.RGGB) -> torch.Tensor:
    """
    Applies a bayer matrix to an image
    """
    bayer_tensor = torch.zeros((1, image.shape[1], image.shape[2]), device=image.device)
    bayer_tensor[0, ::2, ::2] = image[0 if pattern == BayerPattern.RGGB else 2 if pattern == BayerPattern.BGGR else 1, ::2, ::2]
    bayer_tensor[0, ::2, 1::2] = image[0 if pattern == BayerPattern.GRBG else 2 if pattern == BayerPattern.GBRG else 1, ::2, 1::2]
    bayer_tensor[0, 1::2, ::2] = image[0 if pattern == BayerPattern.GBRG else 2 if pattern == BayerPattern.GRBG else 1, 1::2, ::2]
    bayer_tensor[0, 1::2, 1::2] = image[0 if pattern == BayerPattern.BGGR else 2 if pattern == BayerPattern.RGGB else 1, 1::2, 1::2]
    return bayer_tensor
