from core.layers.layer import Layer
import cupy as cp

class MaxPool(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.max_indices = None

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
        self.max_indices = cp.zeros(
            (
                batch_size,
                channels,
                height // self.kernel_size,
                width // self.kernel_size,
                2,
            ),
            dtype=cp.int64,
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
                        
                        self.output[b, c, out_h, out_w] = cp.max(pool_region)

                        max_flat = int(cp.argmax(pool_region).item())
                        local_h = max_flat // self.kernel_size
                        local_w = max_flat % self.kernel_size
                        self.max_indices[b, c, out_h, out_w, 0] = m + local_h
                        self.max_indices[b, c, out_h, out_w, 1] = n + local_w
                        

        return self.output

    def backward(self, output_gradient, learning_rate=None):
        self.input_gradient = cp.zeros_like(self.input)
        batch_size, channels, height, width = self.output.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for m in range(0, height):
                    for n in range(0, width):
                        mH, mW = self.max_indices[b, c, m, n]
                        mH = int(mH)
                        mW = int(mW)
                        self.input_gradient[b, c, mH, mW] = output_gradient[b, c, m, n]

        return self.input_gradient