import os

import cv2
import numpy as np


def read_image(path: str, convert_to_float: bool = True) -> np.array:
    if not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not image.dtype == np.float32 and convert_to_float:
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    return image


def write_image(path: str, image: np.array, dtype=np.uint8):
    factor = np.iinfo(dtype).max if dtype != np.float32 else 1
    np_image = (np.clip(image, 0, 1) * factor).astype(dtype)
    cv2.imwrite(path, np_image)
