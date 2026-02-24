"""ReLU activation layer implementation."""

from typing import Optional

import cupy as cp

from ..layer import Layer


class ReLU(Layer):
    """Apply the ReLU activation."""
    def __init__(self):
        super().__init__()

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute ReLU activation for the input tensor.

        Args:
            x: Input tensor as a CuPy array.

        Returns:
            cp.ndarray: Output tensor with negative values clamped to zero.
        """
        self.input = x
        return cp.maximum(0, x)

    def backward(self,
                 grad: cp.ndarray,
                 lr: Optional[float] = None
                ) -> cp.ndarray:
        """Propagate gradients through the ReLU activation.

        Args:
            grad: Upstream gradient tensor matching the input shape.
            lr: Unused parameter present for interface consistency.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input tensor.
        """
        return grad * (self.input > 0)
