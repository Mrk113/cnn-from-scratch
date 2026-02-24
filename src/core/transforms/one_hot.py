"""One-hot encoding transform."""

import cupy as cp

from .transform import Transform


class OneHot(Transform):
    """Convert integer labels into one-hot encoded vectors."""

    def __init__(self, num_classes: int) -> None:
        """Store the number of classes for encoding.

        Args:
            num_classes: Total number of distinct classes.
        """
        self.num_classes = num_classes

    def __call__(self, labels: cp.ndarray) -> cp.ndarray:
        """One-hot encode input labels.

        Args:
            labels: CuPy array of integer class indices.

        Returns:
            cp.ndarray: One-hot encoded labels with shape (N, num_classes).
        """
        labels = labels.astype(cp.int32, copy=False)
        # Allocate all-zero matrix then set class positions to 1.0.
        one_hot = cp.zeros((labels.size, self.num_classes), dtype=cp.float32)
        one_hot[cp.arange(labels.size), labels] = 1.0
        return one_hot