import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.layers import FC
from ...utils import set_seed, generate_random_tensor, tensor_to_cupy, cupy_to_tensor


def assert_fc(batch_size: int,
              input_size: int,
              output_size: int,
              *,
              atol: float = 1e-5
             ) -> None:
    set_seed(11)
    in_t = generate_random_tensor((batch_size, input_size))
    in_t.requires_grad_(True)
    in_cp = tensor_to_cupy(in_t.detach())

    fc_cp = FC(input_size, output_size)
    fc_t = torch.nn.Linear(input_size, output_size)

    # Align weights/biases so forward/backward can be compared directly
    fc_t.weight.data = cupy_to_tensor(fc_cp.weights)
    fc_t.bias.data = cupy_to_tensor(fc_cp.biases)

    out_cp = fc_cp.forward(in_cp)
    out_t = fc_t.forward(in_t)

    assert_allclose(out_cp, out_t.detach(), rtol=1e-5, atol=atol)

    grad_cp = cp.ones_like(out_cp)
    grad_t = torch.ones_like(out_t)

    down_cp = fc_cp.backward(grad_cp, lr=0.01)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_cp, down_t.detach(), rtol=1e-5, atol=atol)
    assert_allclose(fc_cp.weights_grad, fc_t.weight.grad, rtol=1e-5, atol=atol)
    assert_allclose(fc_cp.biases_grad, fc_t.bias.grad, rtol=1e-5, atol=atol)


def test_fc():
    assert_fc(batch_size=4, input_size=10, output_size=5)