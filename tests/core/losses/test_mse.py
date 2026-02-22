from torch import nn as T
import cupy as cp
from cupy.testing import assert_allclose

from core.losses import MSE
from ..utils import (
    set_seed,
    generate_random_tensor,
    generate_random_labels,
    tensor_to_cupy,
)

def assert_mse_forward(
    batch: int,
    num_classes: int,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor_logits = generate_random_tensor((batch, num_classes))
    tensor_labels = generate_random_labels((batch, num_classes), low=0, high=1).float()

    cupy_logits = tensor_to_cupy(tensor_logits.detach())
    cupy_labels = tensor_to_cupy(tensor_labels.detach())

    mse_t = T.MSELoss()
    mse_cp = MSE()

    loss_t = mse_t(tensor_logits, tensor_labels)
    loss_cp = mse_cp.forward(cupy_logits, cupy_labels)

    assert_allclose(loss_cp, loss_t.item(), atol=atol)

def assert_mse_backward(
    batch: int,
    num_classes: int,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor_logits = generate_random_tensor((batch, num_classes)).requires_grad_()
    tensor_labels = generate_random_labels((batch, num_classes), low=0, high=1).float()

    cupy_logits = tensor_to_cupy(tensor_logits.detach())
    cupy_labels = tensor_to_cupy(tensor_labels.detach())

    mse_t = T.MSELoss()
    mse_cp = MSE()

    loss_t = mse_t(tensor_logits, tensor_labels)
    loss_t.backward()
    grad_t = tensor_to_cupy(tensor_logits.grad)

    grad_cp = mse_cp.backward(cupy_logits, cupy_labels)

    assert_allclose(grad_cp, grad_t, atol=atol)

def test_mse_batch():
    assert_mse_forward(batch=16, num_classes=10)
    assert_mse_backward(batch=16, num_classes=10)

def test_mse_single():
    assert_mse_forward(batch=1, num_classes=10)
    assert_mse_backward(batch=1, num_classes=10)