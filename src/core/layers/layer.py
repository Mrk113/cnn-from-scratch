"""Base class for neural network layers."""

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Backward method not implemented.")