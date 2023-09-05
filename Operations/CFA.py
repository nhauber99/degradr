import typing

import torch


def create_bayer_matrix(shape: typing.Tuple, device: str = "cuda"):
    bayer_tensor = torch.zeros((3, shape[0], shape[1]), device=device)
    bayer_tensor[0, ::2, ::2] = 1
    bayer_tensor[1, ::2, 1::2] = 1
    bayer_tensor[1, 1::2, ::2] = 1
    bayer_tensor[2, 1::2, 1::2] = 1
    return bayer_tensor


def demosaic_bilinear(image: torch.Tensor, factor: float = 1):
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
