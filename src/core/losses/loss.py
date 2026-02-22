"""Base class for loss functions."""

import cupy as cp


class Loss:
    """Base class for loss computations"""

    def forward(self, predicted: cp.ndarray, actual: cp.ndarray) -> cp.ndarray:
        """Compute the forward loss value.

        Args:
            predicted: Model outputs.
            actual: Ground-truth targets.

        Returns:
            cp.ndarray: Scalar loss value represented as a CuPy array.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
    
    def backward(self, predicted: cp.ndarray, actual: cp.ndarray) -> cp.ndarray:
        """Compute the gradient of the loss with respect to predictions.

        Args:
            predicted: Model outputs.
            actual: Ground-truth targets.

        Returns:
            Any: Gradient of the loss with respect to predictions.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError