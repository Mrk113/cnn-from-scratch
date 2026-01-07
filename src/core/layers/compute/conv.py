from core.layers.layer import Layer
import cupy as cp


class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

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

    def _im2col(self, x_padded: cp.ndarray, H_out: int, W_out: int) -> cp.ndarray:
        """Return 2D matrix of shape (N*H_out*W_out, C*k*k)."""

        N, C, _, _ = x_padded.shape
        k = self.kernel_size
        s = self.stride

        stN, stC, stH, stW = x_padded.strides
        shape = (N, C, k, k, H_out, W_out)
        strides = (stN, stC, stH, stW, stH * s, stW * s)

        x_cols = cp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        cols = x_cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        return cols

    def _col2im(self, cols_2d: cp.ndarray, x_padded_shape: tuple[int, int, int, int], H_out: int, W_out: int) -> cp.ndarray:
        """Inverse of im2col: cols_2d shape (N*H_out*W_out, C*k*k)."""
        N, C, H_p, W_p = map(int, x_padded_shape)
        k = self.kernel_size
        s = self.stride

        i0 = cp.repeat(cp.arange(k, dtype=cp.int32), k)
        i0 = cp.tile(i0, C)
        j0 = cp.tile(cp.arange(k, dtype=cp.int32), k)
        j0 = cp.tile(j0, C)

        i1 = cp.repeat(cp.arange(H_out, dtype=cp.int32) * s, W_out)
        j1 = cp.tile(cp.arange(W_out, dtype=cp.int32) * s, H_out)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        c = cp.repeat(cp.arange(C, dtype=cp.int32), k * k).reshape(-1, 1)

        # (N*H_out*W_out, C*k*k) -> (N, C*k*k, H_out*W_out)
        cols = cols_2d.reshape(N, H_out * W_out, C * k * k).transpose(0, 2, 1)

        x_padded = cp.zeros((N, C, H_p, W_p), dtype=cp.float32)
        cp.add.at(x_padded, (slice(None), c, i, j), cols)
        return x_padded

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        N, C, H, W = input.shape

        x = cp.ascontiguousarray(input.astype(cp.float32, copy=False))
        self.input = x

        k = self.kernel_size
        p = self.padding
        s = self.stride

        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1
        self._h_out = int(H_out)
        self._w_out = int(W_out)

        if p > 0:
            x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        else:
            x_padded = x

        self._x_padded_shape = tuple(x_padded.shape)

        cols = self._im2col(x_padded, H_out=H_out, W_out=W_out)  # (N*H_out*W_out, C*k*k)
        self._cols = cols

        w_col = self.kernels.reshape(self.out_channels, -1)  # (F, C*k*k)
        out_2d = cols @ w_col.T  # (N*H*W, F)
        out_2d += self.biases.reshape(1, -1)

        out = out_2d.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, output_gradient: cp.ndarray, learning_rate: float) -> cp.ndarray:
        dout = cp.ascontiguousarray(output_gradient.astype(cp.float32, copy=False))
        N, F, H_out, W_out = dout.shape
        
        p = self.padding

        dout_2d = dout.transpose(0, 2, 3, 1).reshape(-1, F)  # (N*H*W, F)
        cols = self._cols  # (N*H*W, C*k*k)

        dW = (dout_2d.T @ cols).reshape(self.kernels.shape)
        db = cp.sum(dout_2d, axis=0)
        self.kernels_grad = dW
        self.biases_grad = db

        w_col = self.kernels.reshape(F, -1)
        dcols = dout_2d @ w_col  # (N*H*W, C*k*k)

        dx_padded = self._col2im(dcols, self._x_padded_shape, H_out=H_out, W_out=W_out)
        if p > 0:
            dx = dx_padded[:, :, p:-p, p:-p]
        else:
            dx = dx_padded

        self.kernels -= (learning_rate * dW).astype(cp.float32, copy=False)
        self.biases -= (learning_rate * db).astype(cp.float32, copy=False)

        return dx


