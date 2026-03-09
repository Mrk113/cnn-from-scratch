"""2D convolution layer implemented with im2col for efficiency."""

from typing import Optional, Tuple

import cupy as cp

from ...utils import pad, im2col, col2im
from ..layer import Layer


class Conv(Layer):
    """Perform a 2D convolution (cross-correlation) over input feature maps."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0
                ) -> None:
        """Initialize convolution parameters and weights.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            kernel_size: Size of the (square) convolution kernel.
            stride: Stride of the convolution.
            padding: Zero-padding applied to both spatial dimensions.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = cp.sqrt(
            2.0 / (self.in_channels * self.kernel_size * self.kernel_size),
            dtype=cp.float32,
        )
        
        # Initialze kernels (C_out, C_in, k, k)
        self.kernels = (
            cp.random.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                dtype=cp.float32
            )
            * scale
        )
        # Initialize biases (C_out,)
        self.biases = cp.zeros((self.out_channels,), dtype=cp.float32)

        self.padded_shape = None
        self.w_row = None
        self.x_col = None
        self.im2col_indices = None
        self.im2col_key = None  # (N, C, H_p, W_p, k, s)

        self.kernels_grad = None
        self.biases_grad = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute the forward convolution pass.

        Args:
            x: Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            cp.ndarray: Output tensor of shape (N, C_out, H_out, W_out).
        """
        self.input = x
        w = self.kernels
        b = self.biases
        p = self.padding
        s = self.stride

        N, C, H_in, W_in = x.shape
        C_out, _, kH, kW = w.shape

        # Compute output spatial dimensions
        H_out = (H_in + 2 * p - kH) // s + 1
        W_out = (W_in + 2 * p - kW) // s + 1

        # Apply padding to the input
        x = pad(x, p, fill=0)
        self.padded_shape = x.shape  # Cache for backward pass

        # Cache im2col indices for the current spatial configuration to avoid
        # rebuilding them on every forward/backward step.
        im2col_key = (*x.shape, kH, s)
        if self.im2col_key == im2col_key and self.im2col_indices is not None:
            x_col = im2col(x, kH, kW, s, indices=self.im2col_indices)
        else:
            x_col, self.im2col_indices = im2col(
                x, kH, kW, s, return_indices=True
            )
            self.im2col_key = im2col_key

        # Perform cross-correlation via im2col for efficiency
        w_row = w.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
        self.w_row = w_row  # Cache for backward pass
        self.x_col = x_col  # Cache for backward pass
        out = w_row.dot(x_col) + b.reshape(-1, 1) # (C_out, N*H_out*W_out)
        out = out.reshape(C_out, N, H_out, W_out) # (C_out, N, H_out, W_out)
        return out.transpose(1, 0, 2, 3)  # (N, C_out, H_out, W_out)

    def backward(self, grad: cp.ndarray, lr: float) -> cp.ndarray:
        """Backpropagate gradients and update convolution parameters.

        Args:
            grad: Upstream gradient of shape (N, C_out, H_out, W_out).
            lr: Learning rate for parameter updates.

        Returns:
            cp.ndarray: Gradient with respect to the input, shape (N, C_in, H, W).
        """
        x = self.input
        w = self.kernels
        p = self.padding
        s = self.stride

        C_out, _, kH, kW = w.shape

        grad_2d = grad.transpose(1, 0, 2, 3).reshape(self.out_channels, -1) # (C_out, N*H_out*W_out)
        dw = grad_2d.dot(self.x_col.T)  # (C_out, C_in*kH*kW)
        dw = dw.reshape(w.shape)  # (C_out, C_in, kH, kW)
        db = cp.sum(grad_2d, axis=1)  # (C_out,)

        self.kernels_grad = dw
        self.biases_grad = db

        dx = self.w_row.T.dot(grad_2d)  # (C_in*kH*kW, N*H_out*W_out)
        dx = col2im(dx, self.padded_shape, kH, kW, s, indices=self.im2col_indices)

        if p > 0:
            dx = dx[:, :, p:-p, p:-p]

        # Update weights and biases
        self.kernels -= lr * dw
        self.biases -= lr * db

        return dx


