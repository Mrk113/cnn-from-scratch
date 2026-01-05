import torch
import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from core.layers.pooling.max_pool import MaxPool

def test_MaxPool():
    # Settings
    torch.manual_seed(11) 
    batch_size = 2
    channels = 3
    input_shape = (batch_size, channels, 8, 8)
    kernel_size = 2

    in_t = torch.randn(input_shape, requires_grad=True)
    in_i = cp.array(in_t.detach().numpy())

    # Test forward 
    op_i = MaxPool(kernel_size)
    op_t = torch.nn.MaxPool2d(kernel_size)

    out_i = op_i.forward(in_i)
    out_t = op_t.forward(in_t)

    assert_allclose(out_i, out_t.detach().numpy(), rtol=1e-5, err_msg="MaxPool forward pass missmatch")

    # Test backward
    grad_i = cp.ones_like(out_i)
    grad_t = torch.ones_like(out_t) 

    down_i = op_i.backward(grad_i)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_i, down_t.detach().numpy(), rtol=1e-5, err_msg="MaxPool backward pass missmatch")