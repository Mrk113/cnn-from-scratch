from core.layers.layer import Layer
import cupy as cp

class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.batch_size = None
        self.kernels = cp.random.randn(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ) 
        self.biases = cp.random.randn(self.out_channels,)
        self.kernels_grad = None
        self.biases_grad = None

    def forward(self, input):
        batch_size, _, input_height, input_width = input.shape
        if input_height < self.kernel_size or input_width < self.kernel_size:
            raise ValueError("Input size must be greater than or equal to kernel size")

        self.input = input
        self.batch_size = batch_size

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
        kernel_gradient = cp.zeros_like(self.kernels)
        input_gradient = cp.zeros_like(self.input)
        bias_gradient = cp.zeros_like(self.biases)

        for b in range(self.batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    kernel_gradient[oc, ic] += self._corrolate2d(
                        self.input[b, ic],
                        output_gradient[b, oc]
                    )
                    input_gradient[b, ic] += self._convolve2d(
                        output_gradient[b, oc],
                        cp.flip(self.kernels[oc, ic], axis=(0, 1))
                    )
                bias_gradient[oc] += cp.sum(output_gradient[b, oc])

            self.kernels_grad = kernel_gradient
            self.biases_grad = bias_gradient

        self.kernels -= learning_rate * kernel_gradient / self.batch_size
        self.biases -= learning_rate * bias_gradient / self.batch_size
        return input_gradient
                
    def _convolve2d(self, input, kernel):
        H, W = input.shape
        kH, kW = kernel.shape
        H_out = H + kH - 1
        W_out = W + kW - 1

        padded_input = cp.pad(input, ((kH - 1, kH - 1), (kW - 1, kW - 1)), mode='constant')
        output = cp.empty((H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                output[i, j] = cp.sum(
                    padded_input[i:i+kH, j:j+kW] * kernel
                )
        return output

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