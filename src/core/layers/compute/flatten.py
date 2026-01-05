from core.layers.layer import Layer
import cupy as cp

class Flatten(Layer):
    def __init__(self):
        self.input_shape = None    

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input_shape = input.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, grad_output: cp.ndarray, learning_rate: float = None) -> cp.ndarray:
        return grad_output.reshape(self.input_shape)