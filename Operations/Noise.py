import typing

import numpy as np


class GaussianParams:
    def __init__(self, mean: float, std: float, amplitude: float):
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
