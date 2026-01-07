import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers.compute.flatten import Flatten


def test_Flatten():
    torch.manual_seed(11)

    batch_size = 4
    in_t = torch.randn((batch_size, 3, 5, 7), requires_grad=True)
    in_i = cp.array(in_t.detach().numpy())

    op_i = Flatten()

    out_i = op_i.forward(in_i)
    out_t = in_t.reshape(batch_size, -1)

    assert_allclose(out_i, out_t.detach().numpy(), rtol=1e-6, atol=1e-6, err_msg="Flatten forward mismatch")

    grad_i = cp.ones_like(out_i)
    grad_t = torch.ones_like(out_t)

    down_i = op_i.backward(grad_i)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_i, down_t.numpy(), rtol=1e-6, atol=1e-6, err_msg="Flatten backward mismatch")
