import cupy as cp
import torch

from torchvision import transforms as T

from core.transforms import RandomHorizontalFlip
from ..utils import set_seed, generate_random_tensor, tensor_to_cupy


def assert_random_hflip(
    shape: tuple[int, ...],
    p: float,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor = generate_random_tensor(shape)
    cupy = tensor_to_cupy(tensor)

    random_hflip_t = T.RandomHorizontalFlip(p=p)
    random_hflip_cp = RandomHorizontalFlip(p=p, rand_algo=torch.rand)

    set_seed(11)
    out_t = random_hflip_t(tensor)
    set_seed(11)
    out_cp = random_hflip_cp(cupy)
    out_t = tensor_to_cupy(out_t)

    assert cp.allclose(out_t, out_cp, atol=atol)

def test_hflip_img():
    assert_random_hflip(
        shape=(3, 64, 64),
        p=0.5
    )

def test_hflip_gray():
    assert_random_hflip(
        shape=(1, 28, 28),
        p=0.5
    )

def test_hflip_batch():
    assert_random_hflip(
        shape=(16, 3, 32, 32),
        p=0.5
    )