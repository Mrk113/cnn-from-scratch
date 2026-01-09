import cupy as cp
from core.datasets import CIFAR10

def test_cifar10_shape():
    dataset = CIFAR10(root="data", train=True, download=True)
    assert dataset.data.shape == (50000, 3, 32, 32)
    assert dataset.targets.shape == (50000,)

    test_dataset = CIFAR10(root="data", train=False, download=True)
    assert test_dataset.data.shape == (10000, 3, 32, 32)
    assert test_dataset.targets.shape == (10000,)

def test_cifar10_item():
    dataset = CIFAR10(root="data", train=True, download=True)
    img, target = dataset[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(target, cp.ndarray)
    assert target.ndim == 0