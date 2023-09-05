import torch


def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.Tensor(
        [[[(x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2 for y in range(kernel_size)]
          for x in range(kernel_size)]]
    )
    kernel = torch.exp(-kernel / (2 * sigma ** 2))
    kernel = (kernel / torch.sum(kernel)).unsqueeze(0)
    return kernel


def apply_kernel(image_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    smoothed_list = []
    for i in range(image_tensor.shape[1]):
        padded = torch.nn.functional.pad(image_tensor[:, i], (
            kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2), mode='replicate')
        smoothed_list.append(torch.nn.functional.conv2d(padded.unsqueeze(0), kernel, dilation=1))
    smoothed = torch.cat(smoothed_list, dim=1)
    return smoothed


def apply_color_kernel(image_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    smoothed_list = []
    for i in range(image_tensor.shape[1]):
        current_kernel = kernel[:, i, None, :, :]
        padded = torch.nn.functional.pad(image_tensor[:, i], (
            kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2), mode='replicate')
        smoothed_list.append(torch.nn.functional.conv2d(padded.unsqueeze(0), current_kernel))
    smoothed = torch.cat(smoothed_list, dim=1)
    return smoothed
