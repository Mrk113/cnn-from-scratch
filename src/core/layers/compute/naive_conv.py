"""2D convolution layer implemented with explicit loops (naive baseline)."""

from typing import Optional

import cupy as cp

from ...utils import pad
from ..layer import Layer


class NaiveConv(Layer):
    """Perform a 2D convolution (cross-correlation) over input feature maps."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0
                ) -> None:
        """Initialize convolution parameters and weights using plain loops.

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

        # Initialize kernels (C_out, C_in, k, k) and biases (C_out,)
        self.kernels = (
            cp.random.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                dtype=cp.float32,
            )
            * scale
        )
        self.biases = cp.zeros((self.out_channels,), dtype=cp.float32)

        self.padded_input = None
        self.kernels_grad = None
        self.biases_grad = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Compute the forward pass using nested Python loops.

        Args:
            x: Input tensor of shape (N, C_in, H, W).

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

        H_out = (H_in + 2 * p - kH) // s + 1
        W_out = (W_in + 2 * p - kW) // s + 1

        x_padded = pad(x, p, fill=0)
        self.padded_input = x_padded

        out = cp.zeros((N, C_out, H_out, W_out), dtype=cp.float32)

        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    h_start = h * s
                    for w_out in range(W_out):
                        w_start = w_out * s
                        window = x_padded[n, :, h_start:h_start + kH, w_start:w_start + kW]
                        out[n, c_out, h, w_out] = cp.sum(window * w[c_out]) + b[c_out]

        return out

    def backward(self, grad: cp.ndarray, lr: float) -> cp.ndarray:
        """Backpropagate gradients with explicit loops and update parameters.

        Args:
            grad: Upstream gradient of shape (N, C_out, H_out, W_out).
            lr: Learning rate for parameter updates.

        Returns:
            cp.ndarray: Gradient with respect to the input tensor, shape (N, C_in, H, W).
        """

        x_padded = self.padded_input
        w = self.kernels
        p = self.padding
        s = self.stride

        N, C, H_p, W_p = x_padded.shape
        C_out, _, kH, kW = w.shape
        _, _, H_out, W_out = grad.shape

        kernels_grad = cp.zeros_like(w)
        biases_grad = cp.zeros_like(self.biases)
        dx_padded = cp.zeros_like(x_padded)

        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    h_start = h * s
                    for w_out in range(W_out):
                        w_start = w_out * s
                        g = grad[n, c_out, h, w_out]
                        window = x_padded[n, :, h_start:h_start + kH, w_start:w_start + kW]

                        # Accumulate parameter gradients
                        kernels_grad[c_out] += g * window
                        biases_grad[c_out] += g

                        # Accumulate input gradients
                        dx_padded[n, :, h_start:h_start + kH, w_start:w_start + kW] += g * w[c_out]

        self.kernels_grad = kernels_grad
        self.biases_grad = biases_grad

        self.kernels -= lr * kernels_grad
        self.biases -= lr * biases_grad

        if p == 0:
            return dx_padded
        return dx_padded[:, :, p:-p, p:-p]

