import albumentations
import albumentations.pytorch
import numpy as np
import torch


def np_to_torch(image: np.array) -> torch.Tensor:
    return albumentations.pytorch.ToTensorV2()(image=image.astype(np.float32))["image"]
