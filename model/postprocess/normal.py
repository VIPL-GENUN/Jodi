import torch
import torch.nn.functional as F
from .utils import check_tensor


class NormalPostProcessor:
    def __call__(self, image: torch.Tensor):
        check_tensor(image, ndim=4, min_value=-1, max_value=1)
        normalized_image = F.normalize(image.float(), p=2, dim=1)
        normalized_image = (normalized_image + 1) / 2
        return normalized_image
