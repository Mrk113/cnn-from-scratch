from core.layers.layer import Layer
import cupy as cp

class MaxPool(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.argmax = None
        self._cache_key = None
        self._base_h = None
        self._base_w = None
        self._b_idx = None
        self._c_idx = None

    def forward(self, input):
        if input.ndim != 4:
            raise ValueError("MaxPool input must be 4D (N, C, H, W)")

        x = input.astype(cp.float32, copy=False)
        self.input = x
        batch_size, channels, height, width = x.shape
        k = int(self.kernel_size)
        if k <= 0:
            raise ValueError("kernel_size must be >= 1")
        if height % k != 0 or width % k != 0:
            raise ValueError("Input height/width must be divisible by kernel_size")

        h_out = height // k
        w_out = width // k

        windows = x.reshape(batch_size, channels, h_out, k, w_out, k)
        windows = windows.transpose(0, 1, 2, 4, 3, 5)  # (N, C, H_out, W_out, k, k)

        # Forward output
        out = windows.max(axis=(4, 5)).astype(cp.float32, copy=False)

        # Track argmax positions for backward (row-major within each kxk window)
        flat = windows.reshape(batch_size, channels, h_out, w_out, k * k)
        argmax = flat.argmax(axis=-1).astype(cp.int32, copy=False)
        self.argmax = argmax

        cache_key = (int(batch_size), int(channels), int(h_out), int(w_out), int(k))
        if self._cache_key != cache_key:
            # Base coordinates for each pooling window.
            self._base_h = (cp.arange(h_out, dtype=cp.int32) * cp.int32(k)).reshape(1, 1, h_out, 1)
            self._base_w = (cp.arange(w_out, dtype=cp.int32) * cp.int32(k)).reshape(1, 1, 1, w_out)

            b_idx = cp.arange(batch_size, dtype=cp.int32).reshape(batch_size, 1, 1, 1)
            c_idx = cp.arange(channels, dtype=cp.int32).reshape(1, channels, 1, 1)
            self._b_idx = cp.broadcast_to(b_idx, (batch_size, channels, h_out, w_out))
            self._c_idx = cp.broadcast_to(c_idx, (batch_size, channels, h_out, w_out))
            self._cache_key = cache_key

        self.output = out
        return out

    def backward(self, output_gradient, learning_rate=None):
        if self.input is None or self.argmax is None:
            raise ValueError("MaxPool.backward called before forward")

        grad = output_gradient.astype(cp.float32, copy=False)
        dx = cp.zeros_like(self.input, dtype=cp.float32)
        batch_size, channels, h_out, w_out = grad.shape

        k = int(self.kernel_size)
        local_h = (self.argmax // k).astype(cp.int32, copy=False)
        local_w = (self.argmax - local_h * k).astype(cp.int32, copy=False)

        max_h = (self._base_h + local_h).astype(cp.int32, copy=False)
        max_w = (self._base_w + local_w).astype(cp.int32, copy=False)

        # Pooling windows are non-overlapping (stride == kernel_size), so each input
        # location belongs to exactly one output window. That means no atomic add is
        # required here; a simple indexed write is correct and faster.
        dx[self._b_idx, self._c_idx, max_h, max_w] = grad
        self.input_gradient = dx
        return dx