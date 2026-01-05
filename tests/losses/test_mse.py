import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.losses.mse import MSE

def test_MSE():
    # Settings
    torch.manual_seed(11) 
    batch_size = 4 
    in_shape = (batch_size, 10)

    in_t = torch.randn(in_shape, requires_grad=True)
    targets_t = torch.randint(0, 2, in_shape).float()

    in_i = cp.array(in_t.detach().numpy())
    targets_i = cp.array(targets_t.numpy())

    # Test forward pass
    loss_i = MSE()
    loss_t = torch.nn.MSELoss()

    out_i = loss_i.compute(in_i, targets_i)
    out_t = loss_t(in_t, targets_t)

    assert_allclose(out_i, out_t.item(), rtol=1e-5, err_msg="MSE forward pass missmatch")

    # Test backward pass
    down_i = loss_i.gradient(in_i, targets_i)
    out_t.backward()
    down_t = in_t.grad

    assert_allclose(down_i, down_t, rtol=1e-5, err_msg="MSE backward pass missmatch")