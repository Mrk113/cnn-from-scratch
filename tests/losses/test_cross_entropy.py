import torch
import cupy as cp
from cupy.testing import assert_allclose

from core.losses.cross_entropy import CrossEntropy

def test_CrossEntropy():
    # Settings
    torch.manual_seed(11) 
    batch_size = 4 
    in_shape = (batch_size, 10)

    in_t = torch.randn(in_shape, requires_grad=True)
    targets_t = torch.randint(0, 10, (batch_size,))

    in_i = cp.array(in_t.detach().numpy())
    targets_i = cp.array(targets_t.numpy())

    # Test forward pass
    loss_i = CrossEntropy()
    loss_t = torch.nn.CrossEntropyLoss()

    out_i = loss_i.compute(in_i, targets_i)
    out_t = loss_t(in_t, targets_t)

    assert_allclose(out_i, out_t.item(), rtol=1e-5, err_msg="CrossEntropy forward pass missmatch")

    # Test backward pass
    down_i = loss_i.gradient(in_i, targets_i)
    out_t.backward()
    down_t = in_t.grad

    assert_allclose(down_i, down_t, rtol=1e-5, err_msg="CrossEntropy backward pass missmatch")