"""Mean squared error loss implementation."""

import cupy as cp

from .loss import Loss


class MSE(Loss):
    """Compute mean squared error between predictions and targets."""

    def forward(self, predicted: cp.ndarray, actual: cp.ndarray) -> cp.ndarray:
        """Calculate the mean squared error.

        Args:
            predicted: Model outputs as a CuPy array.
            actual: Ground-truth targets as a CuPy array.

        Returns:
            cp.ndarray: Scalar loss value as a CuPy array.
        """
        return cp.mean((predicted - actual) ** 2)

    def backward(self, predicted: cp.ndarray, actual: cp.ndarray) -> cp.ndarray:
        """Compute the gradient of MSE with respect to predictions.

        Args:
            predicted: Model outputs as a CuPy array.
            actual: Ground-truth targets as a CuPy array.

        Returns:
            cp.ndarray: Gradient array matching the shape of predicted.
        """
        return 2.0 * (predicted - actual) / predicted.size