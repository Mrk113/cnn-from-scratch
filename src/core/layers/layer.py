"""Base class for neural network layers."""

import cupy as cp


class Layer:
    """Abstract class for neural network layers."""

    def __init__(self) -> None:
        """Initialize layer state placeholders."""
        self.input: cp.ndarray = None
        self.output: cp.ndarray = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute the forward pass for the layer.

        Args:
            x: Input tensor or data structure consumed by the layer.

        Returns:
            cp.ndarray: Output produced by the layer.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad: cp.ndarray, lr: float) -> cp.ndarray:
        """Compute the backward pass for the layer.

        Args:
            grad: Gradient of the loss with respect to the layer output.
            lr: Learning rate used for parameter updates.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the layer input.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Backward method not implemented.")