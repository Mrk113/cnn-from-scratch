from core.layers.layer import Layer
import cupy as cp

class FC(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        scale = cp.sqrt(cp.asarray(2.0 / input_size, dtype=cp.float32))
        self.weights = (cp.random.randn(input_size, output_size).astype(cp.float32) * scale.astype(cp.float32))
        self.biases = cp.zeros((1, output_size), dtype=cp.float32)
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input):
        self.input = input.astype(cp.float32, copy=False)
        self.output = cp.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        grad = output_gradient.astype(cp.float32, copy=False)
        input_gradient = cp.dot(grad, self.weights.T)
        self.weights_grad = cp.dot(self.input.T, grad)
        self.biases_grad = cp.sum(grad, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= (learning_rate * self.weights_grad).astype(cp.float32, copy=False)
        self.biases -= (learning_rate * self.biases_grad).astype(cp.float32, copy=False)

        return input_gradient