import albumentations.pytorch
import numpy as np

import DefaultDegrade
import FileIO
from Prep.PrepKernels import prep_kernels
from Prep.ZernikePSF import gen_zernike_kernels


def demo():
    # ---- KERNEL GENERATION (only necessary once) ----
    # generate high resolution zernike kernels (ideally use more than 10)
    gen_zernike_kernels("ZernikeKernels", 10)
    # post process and down-sample generated kernels
    prep_kernels("ZernikeKernels", "Kernels")

    # ---- LOAD KERNELS (once during startup) ----
    kernels = DefaultDegrade.load_kernels("Kernels")

    # ---- DEGRADE SINGLE IMAGE ----
    # load image
    target_image = FileIO.read_image_rgb("Test/in.tif")
    # transform it to a tensor in linear srgb space (pow 2.2)
    target_tensor = albumentations.pytorch.ToTensorV2()(image=target_image)["image"].cpu() ** 2.2
    # degrade the image with randomly chosen values
    degraded_tensor = DefaultDegrade.random_degrade(target_tensor, kernels, jpg_chance=1)
    # save the degraded image for testing
    FileIO.write_image_tensor('Test/degraded.png', degraded_tensor ** 0.4545, np.uint16)


if __name__ == "__main__":
    demo()
