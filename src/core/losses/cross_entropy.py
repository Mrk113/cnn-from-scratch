from loss import Loss
import cupy as cp

class CrossEntropy(Loss):
    def compute(predicted, actual):
        if cp.any(predicted < 0) or cp.any(predicted > 1):
            raise ValueError("CrossEntropy: predicted values must be in [0, 1].")

        row_sums = cp.sum(predicted, axis=1)
        if not cp.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                "CrossEntropy: predicted values must sum to 1 along each row "
                "(softmax probabilities)."
            )

        # avoid log(0) by clipping values
        eps = 1e-12
        predicted = cp.clip(predicted, eps, 1. - eps)
        ce = -cp.sum(actual * cp.log(predicted), axis=1)
        return cp.mean(ce)

    def gradient(predicted, actual):
        if cp.any(predicted < 0) or cp.any(predicted > 1):
            raise ValueError("CrossEntropy: predicted values must be in [0, 1].")

        row_sums = cp.sum(predicted, axis=1)
        if not cp.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                "CrossEntropy: predicted rows must sum to 1 "
                "(softmax probabilities)."
            )
    
        # avoid division by zero by clipping values
        eps = 1e-12
        predicted = cp.clip(predicted, eps, 1. - eps)

        grad = -(actual/predicted)
        return grad / actual.shape[0]