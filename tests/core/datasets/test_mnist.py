import cupy as cp
from core.datasets import MNIST

def test_mnist_shape():
    dataset = MNIST(root="data", train=True, download=True)
    assert dataset.data.shape == (60000, 28, 28)
    assert dataset.targets.shape == (60000,)

    test_dataset = MNIST(root="data", train=False, download=True)
    assert test_dataset.data.shape == (10000, 28, 28)
    assert test_dataset.targets.shape == (10000,)

def test_mnist_item():
    dataset = MNIST(root="data", train=True, download=True)
    img, target = dataset[0]
    assert img.shape == (28, 28)
    assert isinstance(target, cp.ndarray)
    assert target.ndim == 0