from layer import Layer
import cupy as cp

class Conv(Layer):
    def __init__(self, out_channels, kernel_size):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = None
        self.kernels = None 
        self.biases = None

    def forward(self, input):
        batch_size, in_channels, input_height, input_width = input.shape
        if input_height < self.kernel_size or input_width < self.kernel_size:
            raise ValueError("Input size must be greater than or equal to kernel size")

        self.input = input
        self.in_channels = in_channels
        self.kernels = cp.random.randn(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        self.biases = cp.random.randn(self.out_channels, 1, 1)

        self.output = cp.zeros((
            batch_size,
            self.out_channels,
            input_height - self.kernel_size + 1,
            input_width - self.kernel_size + 1
            ))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    self.output[b, oc] += self._corrolate2d(
                        input[b, ic],
                        self.kernels[oc, ic],
                    )
                self.output[b, oc] += self.biases[oc]

        return self.output


    def backward(self, output_gradient, learning_rate):
        # TODO: ...
        pass

    def _corrolate2d(self, input, kernel):
        H, W = input.shape
        kH, kW = kernel.shape
        H_out = H - kH + 1
        W_out = W - kW + 1

        output = cp.empty((H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                output[i, j] = cp.sum(
                    input[i:i+kH, j:j+kW] * kernel
                )
        return output