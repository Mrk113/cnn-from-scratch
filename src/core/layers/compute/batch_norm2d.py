"""Batch normalization layer for 4D inputs (N, C, H, W)."""

import cupy as cp

from ..layer import Layer


class BatchNorm2d(Layer):
    """Apply batch normalization to convolutional feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma = None
        self.beta = None
        self.x_norm = None
        self.inv_std = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Normalize input across batch and spatial dimensions.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            cp.ndarray: Batch-normalized output with the same shape as input.
        """
        axes = (0, 2, 3)  # Default axes for 4D input (N, C, H, W)
        param_shape = (1, x.shape[1], 1, 1)

        if self.gamma is None:
            self.gamma = cp.ones(param_shape, dtype=cp.float32)
        if self.beta is None:
            self.beta = cp.zeros(param_shape, dtype=cp.float32)

        mean = cp.mean(x, axis=axes, keepdims=True)
        x_centered = x - mean
        var = cp.mean(x_centered * x_centered, axis=axes, keepdims=True)
        inv_std = cp.float32(1.0) / cp.sqrt(var + cp.float32(1e-5))
        x_norm = x_centered * inv_std

        self.inv_std = inv_std
        self.x_norm = x_norm

        out = x_norm * self.gamma + self.beta
        return out
    
    def backward(self, grad: cp.ndarray, lr: float) -> cp.ndarray:
        """Backpropagate gradients through batch normalization.

        Args:
            grad: Upstream gradient of shape (N, C, H, W).
            lr: Learning rate used to update gamma and beta.

        Returns:
            cp.ndarray: Gradient with respect to the input tensor.
        """
        x_norm = self.x_norm
        inv_std = self.inv_std
        axes = (0, 2, 3)  # Default axes for 4D input (N, C, H, W)

        # Number of elements per channel
        m = grad.size // self.gamma.size

        dbeta = cp.sum(grad, axis=axes, keepdims=True)
        dgamma = cp.sum(grad * x_norm, axis=axes, keepdims=True)

        dx_norm = grad * self.gamma
        dx_norm_sum = cp.sum(dx_norm, axis=axes, keepdims=True)
        dx_norm_xnorm_sum = cp.sum(dx_norm * x_norm, axis=axes, keepdims=True)

        inv_m = cp.float32(1.0) / cp.float32(m)
        dx = inv_m * inv_std * (
            cp.float32(m) * dx_norm - dx_norm_sum - x_norm * dx_norm_xnorm_sum
        )

        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        return dx