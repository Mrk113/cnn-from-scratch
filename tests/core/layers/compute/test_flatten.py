import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers.compute.flatten import Flatten
from tests.core.utils import set_seed, generate_random_tensor, tensor_to_cupy


def assert_flatten(shape: tuple[int, ...],
                   *,
                   atol: float = 1e-6
                  ) -> None:
    set_seed(11)
    in_t = generate_random_tensor(shape)
    in_t.requires_grad_(True)
    in_cp = tensor_to_cupy(in_t.detach())

    flatten = Flatten()

    out_cp = flatten.forward(in_cp)
    out_t = in_t.reshape(in_t.shape[0], -1)

    assert_allclose(out_cp, tensor_to_cupy(out_t.detach()), rtol=1e-6, atol=atol)

    grad_cp = cp.ones_like(out_cp)
    grad_t = torch.ones_like(out_t)

    down_cp = flatten.backward(grad_cp)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_cp, tensor_to_cupy(down_t), rtol=1e-6, atol=atol)


def test_flatten():
    assert_flatten((4, 3, 5, 7))
