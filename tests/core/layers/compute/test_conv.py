import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers import Conv
from ...utils import set_seed, generate_random_tensor, tensor_to_cupy, cupy_to_tensor


def assert_conv(batch_size: int,
                in_channels: int,
                out_channels: int,
                img_size: int = 8,
                *,
                kernel_size: int,
                stride: int = 1,
                padding: int = 1,
                atol: float = 1e-5
               ) -> None:
    set_seed(11)
    in_t = generate_random_tensor((batch_size, in_channels, img_size, img_size))
    in_t.requires_grad_(True)
    in_cp = tensor_to_cupy(in_t.detach())

    conv_cp = Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    conv_t = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dtype=torch.float32)

    # Copy params for consistent comparisons
    conv_t.weight.data = cupy_to_tensor(conv_cp.kernels)
    conv_t.bias.data = cupy_to_tensor(conv_cp.biases)

    out_cp = conv_cp.forward(in_cp)
    out_t = conv_t.forward(in_t)

    assert_allclose(out_cp, out_t.detach(), rtol=1e-5, atol=atol)
    grad_cp = cp.ones_like(out_cp)
    grad_t = torch.ones_like(out_t)

    down_cp = conv_cp.backward(grad_cp, lr=0.01)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_cp, down_t.detach(), rtol=1e-5, atol=atol)
    assert_allclose(conv_cp.kernels_grad, conv_t.weight.grad, rtol=1e-5, atol=atol)
    assert_allclose(conv_cp.biases_grad, conv_t.bias.grad, rtol=1e-5, atol=atol)


def test_conv():
    assert_conv(batch_size=2, in_channels=3, out_channels=4, kernel_size=3)

def test_conv_stride():
    assert_conv(batch_size=2, in_channels=3, out_channels=4, kernel_size=3, stride=2)

def test_conv_padding():
    assert_conv(batch_size=2, in_channels=3, out_channels=4, kernel_size=3, padding=0)