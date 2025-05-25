import torch
from .utils import check_tensor


class ImagePostProcessor:
    def __call__(self, image: torch.Tensor):
        check_tensor(image, ndim=4, min_value=-1, max_value=1)
        image = (image + 1) / 2
        return image.float()
