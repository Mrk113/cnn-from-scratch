from core.layers.layer import Layer
import cupy as cp


class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, dtype=cp.float32):
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.batch_size = None

        # Enforce float32 everywhere for performance and consistency.
        # Keep the dtype parameter for backwards compatibility but ignore it.
        self.dtype = cp.float32

        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be >= 1")
        if self.kernel_size % 2 != 1:
            raise ValueError("Conv requires an odd kernel_size to match PyTorch padding=kernel_size//2.")

        # PyTorch-like "same" padding for odd kernels.
        self._pad = self.kernel_size // 2

        # He init (roughly) to keep activations stable.
        scale = cp.sqrt(cp.float32(2.0) / cp.float32(self.in_channels * self.kernel_size * self.kernel_size))
        self.kernels = (
            cp.random.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ).astype(cp.float32)
            * scale.astype(cp.float32)
        )
        self.biases = cp.zeros((self.out_channels,), dtype=cp.float32)

        self.kernels_grad = None
        self.biases_grad = None

        # Saved for backward
        self.input = None
        self._cols = None
        self._x_padded_shape = None
        self._h_out = None
        self._w_out = None

        # Cached indices for col2im scatter-add.
        self._col2im_cache_key = None
        self._col2im_c = None
        self._col2im_i = None
        self._col2im_j = None

    def _im2col(self, x_padded: cp.ndarray) -> cp.ndarray:
        """Return 2D matrix of shape (N*H_out*W_out, C*k*k)."""
        N, C, H_p, W_p = x_padded.shape
        k = self.kernel_size

        windows = cp.lib.stride_tricks.sliding_window_view(x_padded, (k, k), axis=(2, 3))
        # (N, C, H_out, W_out, k, k) -> (N, H_out, W_out, C, k, k)
        windows = windows.transpose(0, 2, 3, 1, 4, 5)
        cols = windows.reshape(N * windows.shape[1] * windows.shape[2], C * k * k)
        return cols

    def _ensure_col2im_indices(self, C: int, H_padded: int, W_padded: int) -> None:
        k = self.kernel_size
        key = (int(C), int(H_padded), int(W_padded), int(k))
        if self._col2im_cache_key == key:
            return

        H_out = H_padded - k + 1
        W_out = W_padded - k + 1

        i0 = cp.repeat(cp.arange(k, dtype=cp.int32), k)
        i0 = cp.tile(i0, C)
        j0 = cp.tile(cp.arange(k, dtype=cp.int32), k)
        j0 = cp.tile(j0, C)

        i1 = cp.repeat(cp.arange(H_out, dtype=cp.int32), W_out)
        j1 = cp.tile(cp.arange(W_out, dtype=cp.int32), H_out)

        self._col2im_i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        self._col2im_j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        self._col2im_c = cp.repeat(cp.arange(C, dtype=cp.int32), k * k).reshape(-1, 1)
        self._col2im_cache_key = key

    def _col2im(self, cols_2d: cp.ndarray, x_padded_shape: tuple[int, int, int, int]) -> cp.ndarray:
        """Inverse of im2col: cols_2d shape (N*H_out*W_out, C*k*k)."""
        N, C, H_p, W_p = map(int, x_padded_shape)
        k = self.kernel_size
        H_out = H_p - k + 1
        W_out = W_p - k + 1

        self._ensure_col2im_indices(C=C, H_padded=H_p, W_padded=W_p)

        # (N*H_out*W_out, C*k*k) -> (N, C*k*k, H_out*W_out)
        cols = cols_2d.reshape(N, H_out * W_out, C * k * k).transpose(0, 2, 1)

        x_padded = cp.zeros((N, C, H_p, W_p), dtype=cp.float32)
        cp.add.at(x_padded, (slice(None), self._col2im_c, self._col2im_i, self._col2im_j), cols)
        return x_padded

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        if input.ndim != 4:
            raise ValueError("Conv input must be 4D (N, C, H, W)")

        N, C, H, W = input.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        x = cp.ascontiguousarray(input.astype(cp.float32, copy=False))
        self.input = x
        self.batch_size = int(N)
        self._h_out = int(H)
        self._w_out = int(W)

        pad = int(self._pad)
        if pad > 0:
            x_padded = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        else:
            x_padded = x
        self._x_padded_shape = tuple(x_padded.shape)

        cols = self._im2col(x_padded)  # (N*H*W, C*k*k)
        self._cols = cols

        w_col = self.kernels.reshape(self.out_channels, -1)  # (F, C*k*k)
        out_2d = cols @ w_col.T  # (N*H*W, F)
        out_2d += self.biases.reshape(1, -1)

        out = out_2d.reshape(N, H, W, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, output_gradient: cp.ndarray, learning_rate: float) -> cp.ndarray:
        if output_gradient.ndim != 4:
            raise ValueError("Conv output gradient must be 4D (N, F, H, W)")
        if self.input is None or self._cols is None or self._x_padded_shape is None:
            raise ValueError("Conv.backward called before forward")

        dout = cp.ascontiguousarray(output_gradient.astype(cp.float32, copy=False))
        N, F, H_out, W_out = dout.shape
        if F != self.out_channels:
            raise ValueError(f"Expected {self.out_channels} output channels, got {F}")
        if (int(H_out), int(W_out)) != (self._h_out, self._w_out):
            raise ValueError("Output gradient spatial dims do not match forward output")

        dout_2d = dout.transpose(0, 2, 3, 1).reshape(-1, F)  # (N*H*W, F)
        cols = self._cols  # (N*H*W, C*k*k)

        dW = (dout_2d.T @ cols).reshape(self.kernels.shape)
        db = cp.sum(dout_2d, axis=0)
        self.kernels_grad = dW
        self.biases_grad = db

        w_col = self.kernels.reshape(F, -1)
        dcols = dout_2d @ w_col  # (N*H*W, C*k*k)

        dx_padded = self._col2im(dcols, self._x_padded_shape)
        pad = int(self._pad)
        if pad > 0:
            dx = dx_padded[:, :, pad:-pad, pad:-pad]
        else:
            dx = dx_padded

        self.kernels -= (learning_rate * dW).astype(cp.float32, copy=False)
        self.biases -= (learning_rate * db).astype(cp.float32, copy=False)

        return dx


