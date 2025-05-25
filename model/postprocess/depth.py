import torch
import torchvision.transforms as T
from torchvision.transforms.functional import resize

from .utils import check_tensor


def batch_min_max_norm(image: torch.Tensor):
    check_tensor(image, ndim=3)  # (B, H, W)
    image_min = image.flatten(start_dim=1).min(dim=1)[0][:, None, None]
    image_max = image.flatten(start_dim=1).max(dim=1)[0][:, None, None]
    image = (image - image_min) / (image_max - image_min)  # (B, H, W), [0, 1]
    return image


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: T.InterpolationMode = T.InterpolationMode.BILINEAR,
):
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, [new_height, new_width], resample_method, antialias=True)
    return resized_img


class DepthPostProcessor:
    def __call__(self, image: torch.Tensor):
        check_tensor(image, ndim=4, min_value=-1, max_value=1)
        image = torch.mean(image.float(), dim=1)  # (B, H, W)
        image = batch_min_max_norm(image)  # (B, H, W), range [0, 1]
        image = image.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)
        return image
