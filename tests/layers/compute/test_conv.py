import torch
import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from core.layers.compute.conv import Conv

def test_Conv():
    # Settings
    torch.manual_seed(11) 
    batch_size = 2
    in_channels = 3
    out_channels = 4
    input_shape = (batch_size, in_channels, 8, 8)
    kernel_size = 3

    in_t = torch.randn(input_shape, requires_grad=True)
    in_i = cp.array(in_t.detach().numpy())

    # Test forward 
    op_i = Conv(in_channels, out_channels, kernel_size)
    op_t = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    # Copy weights and biases from op_i to op_t for consistency
    op_t.weight.data = torch.tensor(np.array(op_i.kernels.get()), dtype=torch.float32)
    op_t.bias.data = torch.tensor(np.array(op_i.biases.get()), dtype=torch.float32)

    out_i = op_i.forward(in_i)
    out_t = op_t.forward(in_t)

    assert_allclose(out_i, out_t.detach().numpy(), rtol=1e-5, atol=1e-5, err_msg="Conv forward pass missmatch")

    # Test backward pass
    grad_i = cp.ones_like(out_i)
    grad_t = torch.ones_like(out_t)

    down_i = op_i.backward(grad_i, learning_rate=0.01)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_i, down_t.numpy(), rtol=1e-5, atol=1e-5, err_msg="Conv backward pass missmatch")
    
    # Test kernels and biases gradients
    assert_allclose(op_i.kernels_grad.get(), op_t.weight.grad.numpy(), rtol=1e-5, atol=1e-5, err_msg="Conv kernels gradient missmatch")
    assert_allclose(op_i.biases_grad.get(), op_t.bias.grad.numpy(), rtol=1e-5, atol=1e-5, err_msg="Conv biases gradient missmatch")