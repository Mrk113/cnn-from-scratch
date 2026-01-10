import cupy as cp
from torch.nn import functional as F
from core.transforms import OneHot

from ..utils import set_seed, generate_random_labels, tensor_to_cupy

def assert_one_hot(
    shape: tuple[int, ...],
    num_classes: int,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor = generate_random_labels(shape, low=0, high=num_classes)
    cupy = tensor_to_cupy(tensor)

    one_hot_cp = OneHot(num_classes=num_classes)

    out_t = F.one_hot(tensor, num_classes=num_classes)
    out_cp = one_hot_cp(cupy)
    out_t = tensor_to_cupy(out_t)

    assert cp.allclose(out_t, out_cp, atol=atol)

def test_one_hot():
    assert_one_hot(
        shape=(16,),
        num_classes=10
    )