# This file is modified from https://github.com/NVlabs/Sana

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn as nn

__all__ = ["LayerNorm2d", "build_norm", "RMSNorm"]


class LayerNorm2d(nn.LayerNorm):
    rmsnorm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x if LayerNorm2d.rmsnorm else x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, rmsnorm={self.rmsnorm}"


# register normalization function here
#   name: module, kwargs with default values
REGISTERED_NORMALIZATION_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "bn2d": (nn.BatchNorm2d, {"num_features": None, "eps": 1e-5, "momentum": 0.1, "affine": True}),
    "syncbn": (nn.SyncBatchNorm, {"num_features": None, "eps": 1e-5, "momentum": 0.1, "affine": True}),
    "ln": (nn.LayerNorm, {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True}),
    "ln2d": (LayerNorm2d, {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True}),
}


def build_norm(name="bn2d", num_features=None, affine=True, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
        kwargs["elementwise_affine"] = affine
    else:
        kwargs["num_features"] = num_features
        kwargs["affine"] = affine
    if name in REGISTERED_NORMALIZATION_DICT:
        norm_cls, default_args = copy.deepcopy(REGISTERED_NORMALIZATION_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return norm_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError("do not support: %s" % name)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
            Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return (self.weight * self._norm(x.float())).type_as(x)
