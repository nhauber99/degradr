import torch


def sensor_to_srgb_matrix():
    # placeholder example matrix
    return torch.tensor([[1.67, -0.79, 0.11],
                         [-0.71, 1.72, -0.03],
                         [-0.14, -0.37, 1.52]], device="cuda")


def apply_color_matrix(image: torch.Tensor, color_matrix: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ij,jkl->ikl', color_matrix, image)
