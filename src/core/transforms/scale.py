import cupy as cp

from .transform import Transform

class Scale(Transform):
    # Scale image pixel values from [0, 255] to [0, 1]
    def __init__(self, factor: float = 255.0) -> None:
        self.factor = factor

    def __call__(self, img) -> cp.ndarray:

        if self.factor == 0:
            raise ValueError("Scale factor cannot be zero.")

        return img / self.factor