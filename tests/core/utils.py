"""Test utilities."""

import torch
import cupy as cp
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility in CuPy and PyTorch.

    Args:
        seed: Seed value applied to both CuPy and PyTorch RNGs.
    """
    cp.random.seed(seed)
    torch.manual_seed(seed)


def generate_random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """Generate a random PyTorch tensor with the given shape.

    Args:
        shape: Desired tensor shape.

    Returns:
        torch.Tensor: Tensor of random floats.
    """
    return torch.randn(shape, dtype=torch.float32)


def generate_random_labels(shape: tuple[int, ...], low: int, high: int) -> torch.Tensor:
    """Generate random integer labels within a specified range.

    Args:
        shape: Desired label tensor shape.
        low: Inclusive lower bound for label values.
        high: Exclusive upper bound for label values.

    Returns:
        torch.Tensor: Tensor of random integer labels.
    """
    return torch.randint(low, high, shape)


def generate_random_ndarray(shape: tuple[int, ...]) -> cp.ndarray:
    """Generate a random CuPy ndarray with the given shape.

    Args:
        shape: Desired array shape.

    Returns:
        cp.ndarray: CuPy array of random floats.
    """
    return cp.random.randn(*shape, dtype=cp.float32)


def tensor_to_cupy(tensor: torch.Tensor) -> cp.ndarray:
    """Convert a PyTorch tensor to a CuPy array.

    Args:
        tensor: Input PyTorch tensor to convert.

    Returns:
        cp.ndarray: CuPy array containing the tensor data.
    """
    return cp.asarray(tensor.numpy())


def cupy_to_tensor(array: cp.ndarray) -> torch.Tensor:
    """Convert a CuPy array to a PyTorch tensor.

    Args:
        array: Input CuPy array to convert.

    Returns:
        torch.Tensor: PyTorch tensor containing the array data.
    """
    return torch.from_numpy(cp.asnumpy(array))


def cupy_to_numpy(array: cp.ndarray) -> np.ndarray:
    """Convert a CuPy array to a NumPy array.

    Args:
        array: Input CuPy array to convert.

    Returns:
        np.ndarray: NumPy array containing the data copied from CuPy.
    """
    return cp.asnumpy(array)

