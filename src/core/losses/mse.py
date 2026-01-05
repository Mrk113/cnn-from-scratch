from core.losses.loss import Loss
import cupy as cp

class MSE(Loss):
    def compute(self, predicted, actual):
        return cp.mean((predicted - actual) ** 2)

    def gradient(self, predicted, actual):
        return 2 * (predicted - actual) / predicted.size