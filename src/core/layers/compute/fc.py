from core.layers.layer import Layer
import cupy as cp

class FC(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = cp.random.randn(input_size, output_size) * cp.sqrt(2. / input_size)
        self.biases = cp.zeros((1, output_size))
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input):
        self.input = input
        self.output = cp.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = cp.dot(output_gradient, self.weights.T)
        self.weights_grad = cp.dot(self.input.T, output_gradient)
        self.biases_grad = cp.sum(output_gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad

        return input_gradient