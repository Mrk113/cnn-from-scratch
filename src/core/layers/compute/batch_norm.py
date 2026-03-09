"""Batch normalization layer for channel-first inputs."""

import cupy as cp

from ..layer import Layer


class BatchNorm(Layer):
    """Apply batch normalization across the channel dimension for N-D inputs.

    Supports 2D, 3D, and 4D tensors with channel-first layouts. Parameters are
    initialized lazily to match the incoming shape.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        """Initialize batch norm with epsilon for numerical stability.

        Args:
            eps: Small constant added to variance to avoid division by zero.
        """
        super().__init__()
        self.eps = cp.float32(eps)
        self.gamma = None
        self.beta = None
        self.x_norm = None
        self.inv_std = None
        self.axes: tuple[int, ...] | None = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Normalize input across batch and spatial dimensions.

        Args:
            x: Input tensor with shape (N, C), (N, C, L), or (N, C, H, W).

        Returns:
            cp.ndarray: Batch-normalized tensor with the same shape as ``x``.

        Raises:
            ValueError: If the input dimensionality is not 2D, 3D, or 4D.
        """
        if x.ndim == 2:  # (N, C)
            axes = (0,)
            param_shape = (1, x.shape[1])
        elif x.ndim == 3:  # (N, C, L)
            axes = (0, 2)
            param_shape = (1, x.shape[1], 1)
        elif x.ndim == 4:  # (N, C, H, W)
            axes = (0, 2, 3)
            param_shape = (1, x.shape[1], 1, 1)
        else:
            raise ValueError("BatchNorm supports only 2D, 3D, or 4D inputs")

        if self.gamma is None or self.gamma.shape != param_shape:
            self.gamma = cp.ones(param_shape, dtype=cp.float32)
        if self.beta is None or self.beta.shape != param_shape:
            self.beta = cp.zeros(param_shape, dtype=cp.float32)

        mean = cp.mean(x, axis=axes, keepdims=True)
        x_centered = x - mean
        var = cp.mean(x_centered * x_centered, axis=axes, keepdims=True)
        inv_std = cp.float32(1.0) / cp.sqrt(var + self.eps)
        x_norm = x_centered * inv_std

        self.inv_std = inv_std
        self.x_norm = x_norm
        self.axes = axes

        return x_norm * self.gamma + self.beta

    def backward(self, grad: cp.ndarray, lr: float) -> cp.ndarray:
        """Backpropagate gradients through batch normalization.

        Args:
            grad: Upstream gradient tensor with the same shape as the forward output.
            lr: Learning rate used to update ``gamma`` and ``beta``.

        Returns:
            cp.ndarray: Gradient of the loss with respect to the input tensor.
        """
        x_norm = self.x_norm
        inv_std = self.inv_std
        axes = self.axes

        # Number of elements per channel (batch * spatial)
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