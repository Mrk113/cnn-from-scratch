import cupy as cp

from .transform import Transform

class OneHot(Transform):
    # One-hot encode the given label.
    # Usefull for MSE loss
    # Num_classes: Total number of classes for one-hot encoding.
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, labels: cp.ndarray) -> cp.ndarray:
        labels = labels.astype(cp.int32, copy=False)
        one_hot = cp.zeros((labels.size, self.num_classes), dtype=cp.float32)
        one_hot[cp.arange(labels.size), labels] = 1.0
        return one_hot