import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers.activation.softmax import Softmax

def test_Softmax():
    # Settings
    torch.manual_seed(11) 
    batch_size = 4
    in_shape = (batch_size, 3, 32, 32)

    in_t = torch.randn(in_shape, requires_grad=True)
    in_i = cp.array(in_t.detach().numpy())

    # Test forward pass
    op_i = Softmax()
    op_t = torch.nn.Softmax(dim=1)

    out_i = op_i.forward(in_i)
    out_t = op_t.forward(in_t)

    assert_allclose(out_i, out_t.detach(), rtol=1e-5, err_msg="Softmax forward pass missmatch")

    # Test backward pass
    grad_i = cp.ones_like(in_i)
    grad_t = torch.ones_like(in_t)

    down_i = op_i.backward(grad_i)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_i, down_t, rtol=1e-5, atol=1e-5, err_msg="Softmax backward pass missmatch")