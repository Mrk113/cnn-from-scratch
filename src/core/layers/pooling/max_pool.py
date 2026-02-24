"""Max pooling layer implemented with im2col for efficiency."""

from typing import Optional

import cupy as cp

from core.layers.layer import Layer
from ...utils import pad, im2col, col2im


class MaxPool(Layer):
    """Apply max pooling over input feature maps."""

    def __init__(self,
                 kernel_size: int,
                 stride: Optional[int] = None,
                 padding: int = 0
                ) -> None:
        """Initialize pooling parameters.

        Args:
            kernel_size: Size of the (square) pooling window.
            stride: Stride of the pooling operation; defaults to kernel_size if None.
            padding: Zero-padding applied to input before pooling.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        self.argmax = None
        self.padded_shape = None
        self.im2col_indices: Optional[tuple[cp.ndarray, cp.ndarray, cp.ndarray]] = None
        self.im2col_key: Optional[tuple[int, int, int, int, int, int]] = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute the forward max-pooling pass.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            cp.ndarray: Output tensor after max pooling with shape (N, C, H_out, W_out).
        """
        self.input = x
        k = self.kernel_size
        s = self.stride
        p = self.padding

        x = pad(x, p, fill=0)
        self.padded_shape = x.shape
        N, C, H_p, W_p = x.shape

        H_out = (H_p - k) // s + 1
        W_out = (W_p - k) // s + 1

        # Cache im2col indices for the current spatial configuration to avoid
        # rebuilding them on every forward/backward step.
        im2col_key = (*x.shape, k, s)
        if self.im2col_key == im2col_key and self.im2col_indices is not None:
            cols = im2col(x, k, k, s, indices=self.im2col_indices)
        else:
            cols, self.im2col_indices = im2col(
                x, k, k, s, return_indices=True
            )
            self.im2col_key = im2col_key

        cols = cols.reshape(C, k * k, -1)  # (C, k*k, N*H_out*W_out)
        out = cp.max(cols, axis=1)  # (C, N*H_out*W_out)
        self.argmax = cp.argmax(cols, axis=1)  # Cache for backward pass
        out = out.reshape(C, N, H_out, W_out) # (C, N, H_out, W_out)
        out = out.transpose(1, 0, 2, 3)  # (N, C, H_out, W_out)

        return out

    def backward(self, grad: cp.ndarray, lr: Optional[float] = None) -> cp.ndarray:
        """Backpropagate gradients through the max-pooling operation.

        Args:
            grad: Upstream gradient of shape (N, C, H_out, W_out).
            lr: Unused learning rate parameter included for interface consistency.

        Returns:
            cp.ndarray: Gradient with respect to the input, shape (N, C, H, W).
        """
        N, C, H_out, W_out = grad.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding

        grad = grad.transpose(1, 0, 2, 3).reshape(C, -1)  # (C, N*H_out*W_out)
        dmax = cp.zeros((C, k * k, N*H_out*W_out), dtype=cp.float32)  # (C, k*k, N*H_out*W_out)
        c_idx = cp.arange(C).reshape(-1, 1)  # (C, 1)
        nh_idx = cp.arange(N*H_out*W_out).reshape(1, -1)  # (1, N*H_out*W_out)
        dmax[c_idx, self.argmax, nh_idx] = grad  # (C, k*k, N*H_out*W_out)

        dx = col2im(dmax, self.padded_shape, k, k, s, indices=self.im2col_indices)  # (N, C, H_p, W_p)

        if p > 0:
            dx = dx[:, :, p:-p, p:-p]

        return dx