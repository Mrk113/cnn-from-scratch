from core.layers.layer import Layer
import cupy as cp

class Softmax(Layer):
    def forward(self, input):
        shift = input - cp.max(input, axis=1, keepdims=True)
        exps = cp.exp(shift)
        self.output = exps / cp.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        dot = cp.sum(output_gradient * self.output, axis=1, keepdims=True)
        input_gradient = self.output * (output_gradient - dot)
        return input_gradient