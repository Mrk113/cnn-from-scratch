import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers import ReLU
from ...utils import set_seed, generate_random_tensor, tensor_to_cupy


def assert_relu(shape: tuple[int, ...],
                *,
                atol: float = 1e-5
               ) -> None:
    set_seed(11)
    in_t = generate_random_tensor(shape)
    in_t.requires_grad_(True)
    in_cp = tensor_to_cupy(in_t.detach())

    relu_cp = ReLU()
    relu_t = torch.nn.ReLU()

    out_cp = relu_cp.forward(in_cp)
    out_t = relu_t.forward(in_t)

    assert_allclose(out_cp, out_t.detach(), rtol=1e-5, atol=atol)

    grad_cp = cp.ones_like(in_cp)
    grad_t = torch.ones_like(in_t)

    down_cp = relu_cp.backward(grad_cp)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_cp, down_t, rtol=1e-5, atol=atol)


def test_relu():
    assert_relu((4, 3, 32, 32))