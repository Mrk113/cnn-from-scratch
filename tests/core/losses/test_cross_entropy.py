from torch import nn as T
from cupy.testing import assert_allclose

from core.losses import CrossEntropy
from ..utils import (
    set_seed,
    generate_random_tensor,
    generate_random_labels,
    tensor_to_cupy,
)

def assert_cross_entropy_forward(
    batch: int,
    num_classes: int,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor_logits = generate_random_tensor((batch, num_classes))
    tensor_labels = generate_random_labels((batch,), low=0, high=num_classes)

    cupy_logits = tensor_to_cupy(tensor_logits.detach())
    cupy_labels = tensor_to_cupy(tensor_labels)

    ce_t = T.CrossEntropyLoss()
    ce_cp = CrossEntropy()

    loss_t = ce_t(tensor_logits, tensor_labels)
    loss_cp = ce_cp.forward(cupy_logits, cupy_labels)

    assert_allclose(loss_cp, loss_t.item(), atol=atol)

def assert_cross_entropy_backward(
    batch: int,
    num_classes: int,
    *,
    atol: float = 1e-6
) -> None:
    set_seed(42)
    tensor_logits = generate_random_tensor((batch, num_classes)).requires_grad_()
    tensor_labels = generate_random_labels((batch,), low=0, high=num_classes)

    cupy_logits = tensor_to_cupy(tensor_logits.detach())
    cupy_labels = tensor_to_cupy(tensor_labels)

    ce_t = T.CrossEntropyLoss()
    ce_cp = CrossEntropy()

    loss_t = ce_t(tensor_logits, tensor_labels)
    loss_t.backward()
    grad_t = tensor_to_cupy(tensor_logits.grad)

    grad_cp = ce_cp.backward(cupy_logits, cupy_labels)

    assert_allclose(grad_cp, grad_t, atol=atol)

def test_cross_entropy_batch():
    assert_cross_entropy_forward(batch=16, num_classes=10)
    assert_cross_entropy_backward(batch=16, num_classes=10)

def test_cross_entropy_single_sample():
    assert_cross_entropy_forward(batch=1, num_classes=10)
    assert_cross_entropy_backward(batch=1, num_classes=10)