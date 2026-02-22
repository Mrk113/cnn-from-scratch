"""Cosine annealing learning rate scheduler."""

import cupy as cp

from .scheduler import Scheduler


class CosineAnnealing(Scheduler):
    """Compute learning rates following a cosine annealing schedule."""

    def __init__(self, lr_max: float, lr_min: float, T_max: int) -> None:
        """Initialize scheduler parameters.

        Args:
            lr_max: Peak learning rate at the start of the schedule.
            lr_min: Minimum learning rate at the end of the schedule.
            T_max: Total number of epochs for one annealing cycle.
        """
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
    
    def __call__(self, epoch: int) -> float:
        """Compute the learning rate for a given epoch.

        Args:
            epoch: Current epoch index within the cycle.

        Returns:
            float: Learning rate for the provided epoch.
        """
        cosine = 1 + cp.cos(cp.pi * epoch / self.T_max)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * cosine
        return lr.item()