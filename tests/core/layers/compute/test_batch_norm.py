import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers import BatchNorm
from ...utils import set_seed, generate_random_tensor, tensor_to_cupy


def torch_batch_norm(shape: tuple[int, ...]):
    if len(shape) in (2, 3):
        return torch.nn.BatchNorm1d(shape[1], affine=True, track_running_stats=False, eps=1e-5)
    if len(shape) == 4:
        return torch.nn.BatchNorm2d(shape[1], affine=True, track_running_stats=False, eps=1e-5)


def assert_batch_norm(shape: tuple[int, ...], *, atol: float = 1e-5) -> None:
    set_seed(11)
    x_t = generate_random_tensor(shape)
    x_t.requires_grad_(True)
    x_cp = tensor_to_cupy(x_t.detach())

    bn_t = torch_batch_norm(shape)
    bn_t.weight.data = torch.ones_like(bn_t.weight)
    bn_t.bias.data = torch.zeros_like(bn_t.bias)

    bn_cp = BatchNorm()

    out_cp = bn_cp.forward(x_cp)
    out_t = bn_t.forward(x_t)

    assert_allclose(out_cp, tensor_to_cupy(out_t.detach()), rtol=1e-5, atol=atol)

    grad_cp = cp.ones_like(out_cp)
    grad_t = torch.ones_like(out_t)

    down_cp = bn_cp.backward(grad_cp, lr=0.0)
    out_t.backward(grad_t)
    down_t = x_t.grad

    assert_allclose(down_cp, tensor_to_cupy(down_t), rtol=1e-5, atol=atol)


def test_batch_norm_1d_matrix():
    assert_batch_norm((4, 6))


def test_batch_norm_1d_sequence():
    assert_batch_norm((4, 6, 10))


def test_batch_norm_2d():
    assert_batch_norm((4, 6, 8, 8))