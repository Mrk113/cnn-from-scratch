import cupy as cp
from core.transforms import Scale

from ..utils import (
    generate_random_ndarray, 
    set_seed
)

def asser_scale(
    shape: tuple,
    factor: float,
) -> None:
    set_seed(42)
    cupy = generate_random_ndarray(shape)
    scale_cp = Scale(factor=factor)
    out_t = scale_cp(cupy)

    assert cp.all(out_t >= -1) and cp.all(out_t <= 1)

def test_scale_img():
    asser_scale(
        shape=(3, 64, 64),
        factor=255.0
    )

def test_scale_gray():
    asser_scale(
        shape=(1, 28, 28),
        factor=255.0
    )

def test_scale_batch():
    asser_scale(
        shape=(16, 3, 32, 32),
        factor=255.0
    )