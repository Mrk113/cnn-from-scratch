from core.losses.loss import Loss
import cupy as cp

class CrossEntropy(Loss):
    def compute(self, predicted, actual):
        shift = predicted - cp.max(predicted, axis=1, keepdims=True)
        logsumexp = cp.log(cp.sum(cp.exp(shift), axis=1, keepdims=True))
        log_probs = shift - logsumexp 

        if actual.ndim == 1:
            # actual is class indices
            loss = -log_probs[cp.arange(predicted.shape[0]), actual]
        else:
            # actual is one-hot
            loss = -cp.sum(actual * log_probs, axis=1)

        return cp.mean(loss)

    def gradient(self, predicted, actual):
        shift = predicted - cp.max(predicted, axis=1, keepdims=True)
        exps = cp.exp(shift)
        probs = exps / cp.sum(exps, axis=1, keepdims=True)

        if actual.ndim == 1:
            y = cp.zeros_like(probs)
            y[cp.arange(predicted.shape[0]), actual] = 1
        else:
            y = actual

        return (probs - y) / predicted.shape[0]