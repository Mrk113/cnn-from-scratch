from layer import Layer
import cupy as cp

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return cp.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * (self.input > 0)
        return input_gradient
