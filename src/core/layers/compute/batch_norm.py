import cupy as cp
from core.layers.layer import Layer

class BatchNorm(Layer):
    def __init__(self):
        self.gamma = None
        self.beta = None
        self.x_norm = None
        self.inv_std = None
        self.axes = None

    def forward(self, input):
        x = input.astype(cp.float32, copy=False)

        if len(x.shape) == 4:
            axes = (0, 2, 3)
            param_shape = (1, x.shape[1], 1, 1)
        else:
            axes = (0,)
            param_shape = (1, x.shape[1])

        if self.gamma is None:
            self.gamma = cp.ones(param_shape, dtype=cp.float32)
        if self.beta is None:
            self.beta = cp.zeros(param_shape, dtype=cp.float32)

        mean = cp.mean(x, axis=axes, keepdims=True)
        x_centered = x - mean
        var = cp.mean(x_centered * x_centered, axis=axes, keepdims=True)
        inv_std = cp.reciprocal(cp.sqrt(var + cp.float32(1e-5)))
        x_norm = x_centered * inv_std

        self.inv_std = inv_std
        self.x_norm = x_norm
        self.axes = axes

        out = x_norm * self.gamma + self.beta
        self.output = out
        return out
    
    def backward(self, output_gradient, learning_rate):
        dout = output_gradient.astype(cp.float32, copy=False)
        x_norm = self.x_norm
        inv_std = self.inv_std
        axes = self.axes
        
        # Calculate N: product of dimensions being reduced
        N = dout.size // self.gamma.size

        dbeta = cp.sum(dout, axis=axes, keepdims=True)
        dgamma = cp.sum(dout * x_norm, axis=axes, keepdims=True)

        dx_norm = dout * self.gamma
        sum_dx_norm = cp.sum(dx_norm, axis=axes, keepdims=True)
        sum_dx_norm_xnorm = cp.sum(dx_norm * x_norm, axis=axes, keepdims=True)
        
        dx = (cp.float32(1.0) / cp.float32(N)) * inv_std * (
            cp.float32(N) * dx_norm - sum_dx_norm - x_norm * sum_dx_norm_xnorm
        )

        # Use smaller learning rate for gamma/beta
        param_lr = learning_rate * 0.1
        self.gamma -= param_lr * dgamma
        self.beta -= param_lr * dbeta

        return dx