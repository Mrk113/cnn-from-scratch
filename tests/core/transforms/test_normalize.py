import cupy as cp
from torchvision import transforms as T
from core.transforms import Normalize

from ..utils import generate_random_tensor, tensor_to_cupy, set_seed

def assert_normalize(
    shape: tuple[int, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    axis: int | None = None,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor = generate_random_tensor(shape)
    cupy = tensor_to_cupy(tensor)

    normalize_t = T.Normalize(mean=mean, std=std)
    normalize_cp = Normalize(mean=mean, std=std, axis=axis)

    out_t = normalize_t(tensor)
    out_cp = normalize_cp(cupy)
    out_t = tensor_to_cupy(out_t)

    assert cp.allclose(out_t, out_cp, atol=atol)

def test_normalize_img():
    assert_normalize(
        shape=(3, 64, 64),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        axis=0
    )

def test_normalize_gray():
    assert_normalize(
        shape=(1, 28, 28),
        mean=(0.5,),
        std=(0.5,),
        axis=0
    )

def test_normalize_batch():
    assert_normalize(
        shape=(16, 3, 32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        axis=1
    )