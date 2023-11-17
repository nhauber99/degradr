import ctypes

import torch
import PyIPP


def demosaic(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(tensor).contiguous()
    PyIPP.DemosaicAHD(
        ctypes.c_void_p(tensor.contiguous().data_ptr()).value,
        ctypes.c_void_p(result.data_ptr()).value,
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
    )


if __name__ == "__main__":
    PyIPP.fast_tanh(1.5)
    tensor = torch.randn((3, 256, 256)).clip_(0, 1) * (2**16 - 1)
    demosaic(tensor)

    print("hi")