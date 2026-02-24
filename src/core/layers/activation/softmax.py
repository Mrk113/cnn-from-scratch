"""Softmax activation layer implementation."""

from typing import Optional

import cupy as cp

from ..layer import Layer


class Softmax(Layer):
    """Apply softmax activation across the class dimension."""
    def __init__(self):
        super().__init__()

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute softmax outputs.

        Args:
            x: Input logits as a CuPy array with shape (N, C).

        Returns:
            cp.ndarray: Softmax probabilities with the same shape as input.
        """
        shift = x - cp.max(x, axis=1, keepdims=True)
        exps = cp.exp(shift)
        self.output = exps / cp.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self,
                 grad: cp.ndarray,
                 lr: Optional[float] = None
                ) -> cp.ndarray:
        """Backpropagate gradients through the softmax activation.

        Args:
            grad: Upstream gradient with the same shape as input.
            lr: Unused parameter present for interface consistency.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input logits.
        """
        dot = cp.sum(grad * self.output, axis=1, keepdims=True)
        return self.output * (grad - dot)