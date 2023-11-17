import ctypes
import typing
from enum import Enum

import torch
import PyIPP


class BayerPattern(Enum):
    BGGR = 0
    RGGB = 1
    GBRG = 2
    GRBG = 3


class DemosaicMethod(Enum):
    AHD = 0  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/demosaicahd.html
    VNG = 1  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/cfatobgra.html
    Legacy = 2  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/cfatorgb.html


def demosaic(tensor: torch.Tensor, pattern: BayerPattern = BayerPattern.RGGB, method: DemosaicMethod = DemosaicMethod.VNG) -> torch.Tensor:
    result = torch.zeros_like(tensor).contiguous()
    temp = bayer_filter(tensor.contiguous().clip(0, 2 ** 16 - 0.001), pattern)
    PyIPP.Demosaic(
        ctypes.c_void_p(temp.data_ptr()).value,
        ctypes.c_void_p(result.data_ptr()).value,
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
        int(method.value),
        int(pattern.value)
    )
    return result


def get_channel_index_from_char(ch):
    if ch == 'r':
        return 0
    if ch == 'g':
        return 1
    if ch == 'b':
        return 2
    return -1


def create_bayer_matrix(shape: typing.Tuple, device: str = "cuda") -> torch.Tensor:
    bayer_tensor = torch.zeros((3, shape[0], shape[1]), device=device)
    bayer_tensor[0, ::2, ::2] = 1
    bayer_tensor[1, ::2, 1::2] = 1
    bayer_tensor[1, 1::2, ::2] = 1
    bayer_tensor[2, 1::2, 1::2] = 1
    return bayer_tensor


def bayer_filter(tensor: torch.Tensor, pattern: BayerPattern = BayerPattern.RGGB) -> torch.Tensor:
    bayer_tensor = torch.zeros((1, tensor.shape[1], tensor.shape[2]), device=tensor.device)
    bayer_tensor[0, ::2, ::2] = tensor[0 if pattern == BayerPattern.RGGB else 2 if pattern == BayerPattern.BGGR else 1, ::2, ::2]
    bayer_tensor[0, ::2, 1::2] = tensor[0 if pattern == BayerPattern.GRBG else 2 if pattern == BayerPattern.GBRG else 1, ::2, 1::2]
    bayer_tensor[0, 1::2, ::2] = tensor[0 if pattern == BayerPattern.GBRG else 2 if pattern == BayerPattern.GRBG else 1, 1::2, ::2]
    bayer_tensor[0, 1::2, 1::2] = tensor[0 if pattern == BayerPattern.BGGR else 2 if pattern == BayerPattern.RGGB else 1, 1::2, 1::2]
    return bayer_tensor


def demosaic_bilinear_(image: torch.Tensor, factor: float = 1) -> torch.Tensor:
    r_tl = image[0, ::2, ::2]
    r_tr = torch.nn.functional.pad(image[0, ::2, 2::2].unsqueeze(0), (0, 1, 0, 0), 'replicate').squeeze(0)
    r_bl = torch.nn.functional.pad(image[0, 2::2, ::2].unsqueeze(0), (0, 0, 0, 1), 'replicate').squeeze(0)
    r_br = torch.nn.functional.pad(image[0, 2::2, 2::2].unsqueeze(0), (0, 1, 0, 1), 'replicate').squeeze(0)

    b_br = image[2, 1::2, 1::2]
    b_bl = torch.nn.functional.pad(b_br[:, 0:-1].unsqueeze(0), (1, 0, 0, 0), 'replicate').squeeze(0)
    b_tr = torch.nn.functional.pad(b_br[0:-1, :].unsqueeze(0), (0, 0, 1, 0), 'replicate').squeeze(0)
    b_tl = torch.nn.functional.pad(b_br[0:-1, 0:-1].unsqueeze(0), (1, 0, 1, 0), 'replicate').squeeze(0)

    g_rt = image[1, ::2, 1::2]
    g_bl = image[1, 1::2, ::2]
    g_t = torch.nn.functional.pad(g_bl[0:-1, :].unsqueeze(0), (0, 0, 1, 0), 'replicate').squeeze(0)
    g_l = torch.nn.functional.pad(g_rt[:, 0:-1].unsqueeze(0), (1, 0, 0, 0), 'replicate').squeeze(0)
    g_b = torch.nn.functional.pad(image[1, 2::2, 1::2].unsqueeze(0), (0, 0, 0, 1), 'replicate').squeeze(0)
    g_r = torch.nn.functional.pad(image[1, 1::2, 2::2].unsqueeze(0), (0, 1, 0, 0), 'replicate').squeeze(0)

    image[0, ::2, 1::2] = (1 - factor) * image[0, ::2, 1::2] + factor * (r_tl + r_tr) / 2
    image[0, 1::2, ::2] = (1 - factor) * image[0, 1::2, ::2] + factor * (r_tl + r_bl) / 2
    image[0, 1::2, 1::2] = (1 - factor) * image[0, 1::2, 1::2] + factor * (r_tl + +r_tr + r_bl + r_br) / 4

    image[2, ::2, 1::2] = (1 - factor) * image[2, ::2, 1::2] + factor * (b_tr + b_br) / 2
    image[2, 1::2, ::2] = (1 - factor) * image[2, 1::2, ::2] + factor * (b_bl + b_br) / 2
    image[2, ::2, ::2] = (1 - factor) * image[2, ::2, ::2] + factor * (b_tl + b_tr + b_bl + b_br) / 4

    image[1, ::2, ::2] = (1 - factor) * image[1, ::2, ::2] + factor * (g_rt + g_bl + g_t + g_l) / 4
    image[1, 1::2, 1::2] = (1 - factor) * image[1, 1::2, 1::2] + factor * (g_rt + g_bl + g_b + g_r) / 4

    return image
