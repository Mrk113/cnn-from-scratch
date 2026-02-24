import torch
from torch import nn as T
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR as TorchStepLR

from core.lr_scheduler import StepLR
from cupy.testing import assert_allclose


def assert_step_lr(
    lr: float,
    step_size: int,
    gamma: float,
    epochs: int,
    *,
    atol: float = 1e-8,
) -> None:
    model = T.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler_torch = TorchStepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
    scheduler_custom = StepLR(
        lr=lr,
        step_size=step_size,
        gamma=gamma,
    )

    expected = []
    actual = []
    for epoch in range(epochs):
        optimizer.step()
        scheduler_torch.step(epoch)
        expected.append(scheduler_torch.get_last_lr()[0])
        actual.append(scheduler_custom(epoch))

    assert_allclose(
        torch.tensor(actual),
        torch.tensor(expected),
        atol=atol,
    )


def test_step_lr():
    assert_step_lr(
        lr=0.1,
        step_size=3,
        gamma=0.5,
        epochs=10,
    )
