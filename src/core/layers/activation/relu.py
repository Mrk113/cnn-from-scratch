from core.layers.layer import Layer
import cupy as cp

class ReLU(Layer):
    """ReLU activation"""

    def forward(self, input):
        x = input.astype(cp.float32, copy=False)
        self.input = x
        return cp.maximum(0, x)

    def backward(self, output_gradient, learning_rate=None):
        grad = output_gradient.astype(cp.float32, copy=False)
        return grad * (self.input > 0)
