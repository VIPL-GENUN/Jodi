import torch
from .utils import check_tensor


class EdgePostProcessor:
    def __call__(self, image: torch.Tensor):
        check_tensor(image, ndim=4, min_value=-1, max_value=1)
        image = torch.mean(image.float(), dim=1)  # (B, H, W)
        image = (image + 1) / 2
        image = image.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return image
