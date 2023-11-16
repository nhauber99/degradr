import os

import cv2
import numpy as np
import torch


def read_image(path: str, convert_to_float: bool = True) -> np.array:
    if not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not image.dtype == np.float32 and convert_to_float:
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    if len(image.shape) == 2:
        image = np.stack([image, image, image], 2)
    else:
        image = image[:, :, (2, 1, 0)]
    return image


def write_image(path: str, image: np.array, dtype=np.uint8):
    factor = np.iinfo(dtype).max if dtype != np.float32 else 1
    np_image = (np.clip(image, 0, 1) * factor).astype(dtype)
    cv2.imwrite(path, np_image)


def write_image_tensor(path: str, image: torch.Tensor, dtype=np.uint8):
    factor = np.iinfo(dtype).max if dtype != np.float32 else 1
    np_image = (torch.clip(image.float().permute(1, 2, 0), 0, 1) * factor).numpy().astype(dtype)
    if image.shape[0] == 1:
        cv2.imwrite(path, np_image)
    else:
        cv2.imwrite(path, np_image[:, :, (2, 1, 0)])
