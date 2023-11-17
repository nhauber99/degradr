import torch
import io
import imageio


def jpg_degrade(tensor: torch.Tensor, quality: int) -> torch.Tensor:
    buf = io.BytesIO()
    imageio.imwrite(buf, ((tensor ** 0.45)*255).to(torch.uint8).permute(1, 2, 0).numpy(), format='jpg', quality=quality)
    s = buf.getbuffer()
    return (torch.tensor(imageio.imread(s, format='jpg')).permute(2, 0, 1) / 255) ** 2.2
