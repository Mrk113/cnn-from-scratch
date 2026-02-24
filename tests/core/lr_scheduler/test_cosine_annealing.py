import torch
from torch import nn as T
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.lr_scheduler import CosineAnnealing
from cupy.testing import assert_allclose


def assert_cosine_annealing(
    lr_max: float,
    lr_min: float,
    T_max: int,
    *,
    atol: float = 1e-8,
) -> None:
    model = T.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=lr_max)
    scheduler_torch = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=lr_min,
    )
    scheduler_custom = CosineAnnealing(
        lr_max=lr_max,
        lr_min=lr_min,
        T_max=T_max,
    )

    expected = []
    actual = []
    for epoch in range(T_max + 1):
        optimizer.step()
        scheduler_torch.step(epoch)
        expected.append(scheduler_torch.get_last_lr()[0])
        actual.append(scheduler_custom(epoch))

    assert_allclose(
        torch.tensor(actual),
        torch.tensor(expected),
        atol=atol,
    )


def test_cosine_annealing():
    assert_cosine_annealing(
        lr_max=0.1,
        lr_min=0.001,
        T_max=10,
    )