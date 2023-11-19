import random
from typing import Optional

import numpy as np
import prysm
import prysm.polynomials
import prysm.propagation
import prysm.coordinates
import prysm.geometry

from Analysis.Utilities import FileIO


class Aperture:
    def __init__(self, lens_diameter: float, samples: int = 256):
        aperture_grid_x, aperture_grid_y = prysm.coordinates.make_xy_grid(samples, diameter=1)
        self.spacing = lens_diameter / samples
        self.rho, self.phi = prysm.coordinates.cart_to_polar(aperture_grid_x, aperture_grid_y)
        self.geometry = prysm.geometry.circle(0.5, self.rho)


def gen_zernike_kernel(focal_length, aperture: Aperture, total_weight: Optional[float] = None, weights: Optional[np.ndarray] = None,
                       wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generates a psf from zernike polynomials.
    :param focal_length: focal length of the lens
    :param aperture: aperture length of the lens
    :param total_weight: weight to be multiplied by the individual weights of the zernike polynomials
    :param weights: individual weights of the zernike polynomials
    :param wavelengths: wavelengths to be sampled
    :return: a monochrome image of the psf
    """
    if weights is None:
        weight_std = np.array([0, 0.1, 0.1, 0.5, 0.3, 0.3, 0.7, 0.7, 0.1, 0.1, 0.5, 0.2, 0.2, 0.05, 0.05])
        weights = weight_std * np.random.randn(*weight_std.shape)
        weights /= np.sum(np.abs(weights))
    if wavelengths is None:
        wavelengths = np.linspace(0.4, 0.7, 11)
    if total_weight is None:
        total_weight = 3 * np.random.uniform(0., 1.)

    weights = total_weight * weights

    nms = [prysm.polynomials.noll_to_nm(i + 1) for i in range(weights.shape[0])]
    ps = np.array(list(prysm.polynomials.zernike_nm_sequence(nms, aperture.rho, aperture.phi)))

    phi = np.sum(np.expand_dims(weights, (1, 2)) * ps, 0)
    phi100 = phi * 1000

    components = []
    for wvl in wavelengths:
        wf = prysm.propagation.Wavefront.from_amp_and_phase(aperture.geometry, phi100, wvl, aperture.spacing)
        focused = wf.focus_fixed_sampling(focal_length, 1, 128)
        components.append(focused.intensity.data)

    psf = np.sum(components, 0)
    psf /= np.sum(psf)

    return psf


if __name__ == "__main__":
    a = Aperture(50 / 2.8)
    # generate randomly sampled kernels in the specified folder
    for i in range(1000):
        k = gen_zernike_kernel(100, a, random.uniform(1, 2))
        k /= np.max(k)
        FileIO.write_image(f"Kernels/{i}.tif", k ** 0.45, np.uint16)
