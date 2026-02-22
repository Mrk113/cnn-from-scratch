"""Public export of loss functions."""
from .cross_entropy import CrossEntropy
from .mse import MSE

__all__ = ["CrossEntropy", "MSE"]