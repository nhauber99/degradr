import torch


def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.Tensor(
        [[[(x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2 for y in range(kernel_size)]
          for x in range(kernel_size)]]
    )
    kernel = torch.exp(-kernel / (2 * sigma ** 2))
    kernel = (kernel / torch.sum(kernel)).unsqueeze(0)
    return kernel


def circular_kernel(kernel_size: int, sigma: float):
    kernel_size *= 5
    sigma *= 5
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).unsqueeze(1)
    y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).unsqueeze(0)
    grid = torch.sqrt(x * x + y * y)
    kernel = torch.where(grid <= sigma / 2, torch.ones_like(grid), torch.zeros_like(grid))
    kernel = torch.nn.functional.interpolate(kernel.unsqueeze(0).unsqueeze(0), scale_factor=1 / 5, mode='area')
    kernel = kernel / torch.sum(kernel)
    return kernel


def apply_kernel(image_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    unsqueezed = False
    if len(image_tensor.shape) == 3:
        unsqueezed = True
        image_tensor = image_tensor.unsqueeze(0)
    smoothed_list = []
    for i in range(image_tensor.shape[1]):
        padded = torch.nn.functional.pad(image_tensor[:, i], (
            kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2), mode='replicate')
        smoothed_list.append(torch.nn.functional.conv2d(padded.unsqueeze(0), kernel, dilation=1))
    smoothed = torch.cat(smoothed_list, dim=1)
    if unsqueezed:
        smoothed = smoothed.squeeze(0)
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
