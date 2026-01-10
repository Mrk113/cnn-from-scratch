import torch
import cupy as cp
from torchvision import transforms as T

from core.transforms import RandomCrop
from ..utils import set_seed, generate_random_tensor, tensor_to_cupy

def assert_random_crop(
    shape: tuple[int, ...],
    crop_size: tuple[int, int],
    padding: int | None = None,
    fill: int = 0,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor = generate_random_tensor(shape)
    cupy = tensor_to_cupy(tensor)

    random_crop_t = T.RandomCrop(size=crop_size, padding=padding, fill=fill)
    random_crop_cp = RandomCrop(size=crop_size,
                                padding=padding, 
                                fill=fill, 
                                rand_algo=torch.randint
                                )

    set_seed(11)
    out_t = random_crop_t(tensor) 
    set_seed(11)
    out_cp = random_crop_cp(cupy)
    out_t = tensor_to_cupy(out_t)

    assert cp.allclose(out_t, out_cp, atol=atol)

def test_random_crop_img():
    assert_random_crop(
        shape=(3, 64, 64),
        crop_size=(32, 32),
        padding=None,
        fill=0
    )

def test_random_crop_gray():
    assert_random_crop(
        shape=(1, 28, 28),
        crop_size=(20, 20),
        padding=4,
        fill=128
    )

def test_random_crop_batch():
    assert_random_crop(
        shape=(16, 3, 32, 32),
        crop_size=(24, 24),
        padding=2,
        fill=255
    )

def test_random_crop_padding():
    assert_random_crop(
        shape=(3, 50, 50),
        crop_size=(40, 40),
        padding=10,
        fill=0
    )

def test_random_crop_fill():
    assert_random_crop(
        shape=(3, 100, 100),
        crop_size=(80, 80),
        padding=5,
        fill=200
    )