"""Cross-entropy loss implementation."""

import cupy as cp

from .loss import Loss


class CrossEntropy(Loss):
    """Compute cross-entropy loss for classification outputs."""

    def forward(self, predicted: cp.ndarray, actual: cp.ndarray) -> cp.ndarray:
        """Calculate cross-entropy loss.

        Args:
            predicted: Raw model logits as a CuPy array of shape (N, C).
            actual: Class indices (shape (N,)) or one-hot labels (shape (N, C)).

        Returns:
            cp.ndarray: Scalar loss value as a CuPy array.
        """
        # Shift logits by row-wise max for numerical stability before exp.
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

    def backward(self, predicted: cp.ndarray, actual: cp.ndarray) -> cp.ndarray:
        """Compute gradient of cross-entropy with respect to predictions.

        Args:
            predicted: Raw model logits as a CuPy array of shape (N, C).
            actual: Class indices (shape (N,)) or one-hot labels (shape (N, C)).

        Returns:
            cp.ndarray: Gradient array matching predicted shape.
        """
        shift = predicted - cp.max(predicted, axis=1, keepdims=True)
        exps = cp.exp(shift)
        probs = exps / cp.sum(exps, axis=1, keepdims=True)

        if actual.ndim == 1:
            # actual is class indices
            y = cp.zeros_like(probs)
            y[cp.arange(predicted.shape[0]), actual] = 1
        else:
            # actual is already one-hot
            y = actual

        return (probs - y) / predicted.shape[0]