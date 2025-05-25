import torch
from torch import Tensor


def check_tensor_ndim(tensor: Tensor, ndim: int):
    if tensor.ndim != ndim:
        raise ValueError(f"Expected tensor with {ndim} dimensions, but got {tensor.ndim} dimensions.")


def check_tensor_dtype(tensor: Tensor, dtype: torch.dtype):
    if tensor.dtype != dtype:
        raise TypeError(f"Expected tensor with dtype {dtype}, but got {tensor.dtype}.")


def check_tensor_shape(tensor: Tensor, shape: tuple):
    if tensor.shape != shape:
        raise ValueError(f"Expected tensor with shape {shape}, but got {tensor.shape}.")


def check_tensor_range(tensor: Tensor, min_value: float = None, max_value: float = None):
    if min_value is not None and (tensor < min_value).any():
        raise ValueError(f"Tensor contains values less than {min_value}.")
    if max_value is not None and (tensor > max_value).any():
        raise ValueError(f"Tensor contains values greater than {max_value}.")


def check_tensor(
        tensor: Tensor,
        ndim: int = None,
        dtype: torch.dtype = None,
        shape: tuple = None,
        min_value: float = None,
        max_value: float = None,
):
    if ndim is not None:
        check_tensor_ndim(tensor, ndim)
    if dtype is not None:
        check_tensor_dtype(tensor, dtype)
    if shape is not None:
        check_tensor_shape(tensor, shape)
    if min_value is not None or max_value is not None:
        check_tensor_range(tensor, min_value, max_value)
