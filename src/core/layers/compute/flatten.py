"""Flatten layer to convert spatial tensors into vectors."""

import cupy as cp

from ..layer import Layer


class Flatten(Layer):
    """Flatten spatial dimensions while preserving batch size."""

    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Flatten input to (batch_size, -1).

        Args:
            x: Input tensor with batch as first dimension.

        Returns:
            cp.ndarray: Reshaped tensor with batch dimension preserved.
        """
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad: cp.ndarray, lr: float = None) -> cp.ndarray:
        """Restore gradient to the original input shape.

        Args:
            grad: Gradient tensor from the next layer, flattened per sample.
            lr: Unused learning rate parameter for interface consistency.

        Returns:
            cp.ndarray: Gradient reshaped to the original input dimensions.
        """
        return grad.reshape(self.input_shape)