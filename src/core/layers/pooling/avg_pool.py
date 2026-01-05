from core.layers.layer import Layer
import cupy as cp

class AvgPool(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape
        self.output = cp.zeros(
            (
                batch_size,
                channels,
                height // self.kernel_size,
                width // self.kernel_size
            )
        )

        for b in range(batch_size):
            for c in range(channels):
                for m in range(0, height, self.kernel_size):
                    for n in range(0, width, self.kernel_size):
                        h_end = m + self.kernel_size
                        w_end = n + self.kernel_size
                        pool_region = input[b, c, m:h_end, n:w_end]
                        
                        out_h = m // self.kernel_size
                        out_w = n // self.kernel_size
                        
                        self.output[b, c, out_h, out_w] = cp.mean(pool_region)

        return self.output

    def backward(self, output_gradient, learning_rate=None):
        self.input_gradient = cp.zeros_like(self.input)
        batch_size, channels, height, width = self.output.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for m in range(0, height):
                    for n in range(0, width):
                        grad = output_gradient[b, c, m, n] / (self.kernel_size * self.kernel_size)
                        for i in range(self.kernel_size):
                            for j in range(self.kernel_size):
                                self.input_gradient[
                                    b,
                                    c,
                                    m * self.kernel_size + i,
                                    n * self.kernel_size + j
                                ] += grad

        return self.input_gradient