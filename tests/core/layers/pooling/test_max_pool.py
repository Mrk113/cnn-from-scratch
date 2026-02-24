import torch
import torch.nn.functional as F
import cupy as cp
from cupy.testing import assert_allclose

from core.layers import MaxPool
from ...utils import set_seed, generate_random_tensor, tensor_to_cupy


def assert_max_pool(batch_size: int,
                    channels: int,
                    kernel_size: int,
                    img_size: int = 8,
                    *,
                    stride: int = None,
                    padding: int = 0,
                    atol: float = 1e-5
                   ) -> None:
    set_seed(11)
    in_t = generate_random_tensor((batch_size, channels, img_size, img_size))
    in_t.requires_grad_(True)
    in_cp = tensor_to_cupy(in_t.detach())

    stride = stride if stride is not None else kernel_size

    pool_cp = MaxPool(kernel_size, stride=stride, padding=padding)

    # Torch reference: pad with zeros explicitly (torch MaxPool uses -inf).
    pad_tuple = (padding, padding, padding, padding)
    in_t_ref = F.pad(in_t, pad_tuple, mode="constant", value=0.0) if padding > 0 else in_t
    pool_t = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=0)

    out_cp = pool_cp.forward(in_cp)
    out_t = pool_t.forward(in_t_ref)

    assert_allclose(out_cp, out_t.detach(), rtol=1e-5, atol=atol)
    grad_cp = cp.ones_like(out_cp)
    grad_t = torch.ones_like(out_t)

    down_cp = pool_cp.backward(grad_cp)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_cp, down_t.detach(), rtol=1e-5, atol=atol) 

def test_max_pool():
    assert_max_pool(batch_size=2, channels=3, kernel_size=2)

def test_max_pool_stride():
    assert_max_pool(batch_size=2, channels=3, kernel_size=2, img_size=7, stride=2)

def test_max_pool_padding():
    assert_max_pool(batch_size=2, channels=3, kernel_size=2, img_size=7, padding=1)