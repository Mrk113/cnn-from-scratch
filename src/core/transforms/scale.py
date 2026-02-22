"""Scaling transform."""

import cupy as cp
from typing import Any

from .transform import Transform


class Scale(Transform):
    """Scale image pixel values by a constant factor."""

    def __init__(self, factor: float = 255.0) -> None:
        """Initialize the scaling factor.

        Args:
            factor: Divisor used to scale pixel values.
        """
        self.factor = factor

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        """Scale input image values.

        Args:
            img: Input image as a CuPy array.

        Returns:
            cp.ndarray: Image divided by the configured factor.

        Raises:
            ValueError: If the scaling factor is zero.
        """
        if self.factor == 0:
            raise ValueError("Scale factor cannot be zero.")

        return img / self.factor