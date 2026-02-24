"""Random horizontal flip transform."""

import cupy as cp
from typing import Callable

from .transform import Transform
from ..utils import hflip


class RandomHorizontalFlip(Transform):
    """Flip images horizontally with a given probability."""

    def __init__(self, p: float, *, rand_algo: Callable = cp.random.rand) -> None:
        """Initialize flip probability and random generator.

        Args:
            p: Probability of applying the horizontal flip.
            rand_algo: Callable producing random numbers; defaults to CuPy rand.
                       Useful when testing against PyTorch implementations.
        """
        self.p = p
        self.rand_algo = rand_algo

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        """Apply horizontal flip based on probability p.

        Args:
            img: Input image as a CuPy array.

        Returns:
            cp.ndarray: Flipped image if triggered; otherwise the original image.
        """
        if self.rand_algo(1) < self.p:
            return hflip(img, axis=-1)
        return img
