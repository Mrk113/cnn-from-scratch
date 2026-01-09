import cupy as cp

from core.transforms.compose import Transform

class RandomHorizontalFlip(Transform):
    # Randomly flip the image horizontally with probability p.
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        # Apply horizontal flip with probability p
        if cp.random.rand() < self.p:
            return cp.flip(img, axis=-1)
        return img
