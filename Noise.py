import typing

import numpy as np
import torch


class GaussianParams:
    def __init__(self, mean: float, std: float, amplitude: float = 1):
        self.mean = mean
        self.std = std
        self.amplitude = amplitude


def gaussian_dist(x: np.array, params: GaussianParams):
    return params.amplitude * np.exp(-((x - params.mean) ** 2 / (2 * params.std ** 2))) / (
            np.sqrt(2 * np.pi) * params.std)


def gaussian_dist_combination(x: np.array, params: typing.Iterable[GaussianParams]):
    res = np.zeros_like(x)
    for p in params:
        res += gaussian_dist(x, p)
    return res


def gaussian_sample_like(tensor: torch.Tensor, params: GaussianParams):
    return params.mean + torch.randn_like(tensor) * params.std


def gaussian_sample_combination_like(tensor: torch.Tensor, params: typing.Iterable[GaussianParams]):
    choices = torch.multinomial(torch.Tensor([p.amplitude for p in params], device=tensor.device), tensor.numel(), replacement=True).view(*tensor.shape)
    mean = torch.zeros_like(tensor)
    std = torch.zeros_like(tensor)
    for i, p in enumerate(params):
        mean[choices == i] = p.mean
        std[choices == i] = p.std
    return mean + torch.randn_like(tensor) * std


def row_noise_like(tensor: torch.Tensor, params: GaussianParams):
    return (params.mean + torch.randn((1, tensor.shape[-2], 1), device=tensor.device) * params.std).repeat((tensor.shape[-3], 1, tensor.shape[-1]))


def col_noise_like(tensor: torch.Tensor, params: GaussianParams):
    return (params.mean + torch.randn((1, 1, tensor.shape[-1]), device=tensor.device) * params.std).repeat((tensor.shape[-3], tensor.shape[-2], 1))
