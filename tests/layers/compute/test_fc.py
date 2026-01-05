import torch
import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from core.layers.compute.fc import FC

def test_FC():
    # Settings
    torch.manual_seed(11) 
    batch_size = 4
    input_size = 10
    output_size = 5

    in_t = torch.randn((batch_size, input_size), requires_grad=True)
    in_i = cp.array(in_t.detach().numpy())

    # Test forward 
    op_i = FC(input_size, output_size)
    op_t = torch.nn.Linear(input_size, output_size)

    # Copy weights and biases from op_i to op_t for consistency
    op_t.weight.data = torch.tensor(np.array(op_i.weights.get().T), dtype=torch.float32)
    op_t.bias.data = torch.tensor(np.array(op_i.biases.get().flatten()), dtype=torch.float32)

    out_i = op_i.forward(in_i)
    out_t = op_t.forward(in_t)

    assert_allclose(out_i, out_t.detach().numpy(), rtol=1e-5, atol=1e-5, err_msg="FC forward pass missmatch")

    # Test backward pass
    grad_i = cp.ones_like(out_i)
    grad_t = torch.ones_like(out_t)

    down_i = op_i.backward(grad_i, learning_rate=0.01)
    out_t.backward(grad_t)
    down_t = in_t.grad

    assert_allclose(down_i, down_t.numpy(), rtol=1e-5, atol=1e-5, err_msg="FC backward pass missmatch")
    
    # Test weights and biases gradients
    assert_allclose(op_i.weights_grad.get(), op_t.weight.grad.numpy().T, rtol=1e-5, atol=1e-5, err_msg="FC weights gradient missmatch")
    assert_allclose(op_i.biases_grad.get(), op_t.bias.grad.numpy().reshape(1, -1), rtol=1e-5, atol=1e-5, err_msg="FC biases gradient missmatch")