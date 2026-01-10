import cupy as cp
from typing import Callable

from core.transforms.compose import Transform
from ..utils import hflip

class RandomHorizontalFlip(Transform):
    # Randomly flip the image horizontally with probability p.
    def __init__(self, p: float, *, rand_algo: Callable = cp.random.rand) -> None:
        self.p = p
        self.rand_algo = rand_algo

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        # Apply horizontal flip with probability p
        if self.rand_algo(1) < self.p:
            return hflip(img, axis=-1)
        return img
