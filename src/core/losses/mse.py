from core.losses.loss import Loss
import cupy as cp

class MSE(Loss):
    def compute(self, predicted, actual):
        p = predicted.astype(cp.float32, copy=False)
        a = actual.astype(cp.float32, copy=False)
        return cp.mean((p - a) ** 2, dtype=cp.float32)

    def gradient(self, predicted, actual):
        p = predicted.astype(cp.float32, copy=False)
        a = actual.astype(cp.float32, copy=False)
        return (cp.float32(2.0) * (p - a)) / cp.float32(p.size)