"""Fully connected layer implementation."""

import cupy as cp

from ..layer import Layer


class FC(Layer):
    """Apply an affine transformation to inputs."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize weights and biases.

        Args:
            input_size: Dimensionality of the input features.
            output_size: Dimensionality of the output features.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        scale = cp.sqrt(2.0 / input_size, dtype=cp.float32)
        self.weights = cp.random.randn(output_size, input_size, dtype=cp.float32) * scale
        self.biases = cp.zeros((1, output_size), dtype=cp.float32)
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute the affine forward pass.

        Args:
            x: Input tensor of shape (N, input_size).

        Returns:
            cp.ndarray: Output tensor of shape (N, output_size).
        """
        self.input = x
        self.output = cp.dot(self.input, self.weights.T) + self.biases
        return self.output

    def backward(self, grad: cp.ndarray, lr: float) -> cp.ndarray:
        """Backpropagate gradients and update parameters.

        Args:
            grad: Upstream gradient of shape (N, output_size).
            lr: Learning rate for parameter updates.

        Returns:
            cp.ndarray: Gradient with respect to the input, shape (N, input_size).
        """
        in_grad = cp.dot(grad, self.weights)
        self.weights_grad = cp.dot(grad.T, self.input)
        self.biases_grad = cp.sum(grad, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= lr * self.weights_grad
        self.biases -= lr * self.biases_grad

        return in_grad