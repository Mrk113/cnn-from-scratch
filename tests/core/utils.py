import torch
import cupy as cp
import numpy as np

def set_seed(seed: int) -> None:
    #Set seed for reproducibility.
    cp.random.seed(seed)
    torch.manual_seed(seed)

def generate_random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    #Generate a random tensor with the given shape.
    return torch.randn(shape)

def generate_random_labels(shape: tuple[int, ...], low: int, high: int) -> torch.Tensor:
    #Generate random integer labels within the specified range.
    return torch.randint(low, high, shape)

def generate_random_ndarray(shape: tuple[int, ...]) -> cp.ndarray:
    #Generate a random CuPy ndarray with the given shape.
    return cp.random.randn(*shape)

def tensor_to_cupy(tensor: torch.Tensor) -> cp.ndarray:
    #Convert a PyTorch tensor to a CuPy array.
    return cp.asarray(tensor.numpy())

def cupy_to_tensor(array: cp.ndarray) -> torch.Tensor:
    #Convert a CuPy array to a PyTorch tensor.
    return torch.from_numpy(cp.asnumpy(array))

def cupy_to_numpy(array: cp.ndarray) -> np.ndarray:
    #Convert a CuPy array to a NumPy array.
    return cp.asnumpy(array)

