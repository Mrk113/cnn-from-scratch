from core.layers.layer import Layer
import cupy as cp

class MaxPool(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)

        self.argmax = None
        self.input = None
        self._x_pad_shape = None
        self._h_out = None
        self._w_out = None

    def forward(self, input):
        x = input.astype(cp.float32, copy=False)
        self.input = x
        N, C, H, W = x.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            x_pad = cp.pad(
                x,
                ((0, 0), (0, 0), (p, p), (p, p)),
            )
        else:
            x_pad = x

        H_p, W_p = int(x_pad.shape[2]), int(x_pad.shape[3])
        H_out = (H_p - k) // s + 1
        W_out = (W_p - k) // s + 1
        self._h_out = int(H_out)
        self._w_out = int(W_out)
        self._x_pad_shape = tuple(x_pad.shape)

        stN, stC, stH, stW = x_pad.strides
        shape = (N, C, H_out, W_out, k, k)
        strides = (stN, stC, stH * s, stW * s, stH, stW)
        windows = cp.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)

        # Forward output
        out = windows.max(axis=(4, 5)).astype(cp.float32, copy=False)

        # Track argmax positions for backward (row-major within each kxk window)
        flat = windows.reshape(N, C, H_out, W_out, k * k)
        argmax = flat.argmax(axis=-1).astype(cp.int32, copy=False)
        self.argmax = argmax

        self.output = out
        return out

    def backward(self, output_gradient, learning_rate=None):
        if self.input is None or self.argmax is None:
            raise ValueError("MaxPool.backward called before forward")

        grad = output_gradient.astype(cp.float32, copy=False)
        N, C, H_out, W_out = grad.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding

        local_h = (self.argmax // k).astype(cp.int32, copy=False)
        local_w = (self.argmax - local_h * k).astype(cp.int32, copy=False)

        base_h = (cp.arange(H_out, dtype=cp.int32) * cp.int32(s)).reshape(1, 1, H_out, 1)
        base_w = (cp.arange(W_out, dtype=cp.int32) * cp.int32(s)).reshape(1, 1, 1, W_out)
        max_h = base_h + local_h
        max_w = base_w + local_w

        dx_pad = cp.zeros(self._x_pad_shape, dtype=cp.float32)
        n_idx = cp.arange(N, dtype=cp.int32).reshape(N, 1, 1, 1)
        c_idx = cp.arange(C, dtype=cp.int32).reshape(1, C, 1, 1)
        n_idx = cp.broadcast_to(n_idx, (N, C, H_out, W_out))
        c_idx = cp.broadcast_to(c_idx, (N, C, H_out, W_out))

        # With stride < kernel_size, pooling windows overlap, so accumulation is required.
        cp.add.at(dx_pad, (n_idx, c_idx, max_h, max_w), grad)

        if p > 0:
            dx = dx_pad[:, :, p:-p, p:-p]
        else:
            dx = dx_pad

        self.input_gradient = dx
        return dx