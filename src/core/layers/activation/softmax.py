from layer import Layer
import cupy as cp

class Softmax(Layer):
    def forward(self, input):
        shift = input - cp.max(input, axis=1, keepdims=True)
        exps = cp.exp(shift)
        self.output = exps / cp.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return cp.dot((cp.identity(self.output.shape[1]) - self.output.T) * self.output, output_gradient)