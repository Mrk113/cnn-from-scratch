from loss import Loss
import cupy as cp

class MSE(Loss):
    def compute(predicted, actual):
        return cp.mean((predicted - actual) ** 2)

    def gradient(predicted, actual):
        return 2 * (predicted - actual) / actual.shape[0]