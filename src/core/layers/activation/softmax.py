from core.layers.layer import Layer
import cupy as cp

class Softmax(Layer):
    def forward(self, input):
        x = input.astype(cp.float32, copy=False)
        shift = x - cp.max(x, axis=1, keepdims=True)
        exps = cp.exp(shift)
        self.output = exps / cp.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        grad = output_gradient.astype(cp.float32, copy=False)
        dot = cp.sum(grad * self.output, axis=1, keepdims=True)
        return self.output * (grad - dot)