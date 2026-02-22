"""Random crop transform."""

import cupy as cp
from typing import Callable, Optional, Tuple

from .transform import Transform
from ..utils import get_dims, pad, crop


class RandomCrop(Transform):
    """Randomly crop an image to a target size."""

    def __init__(self, 
                 size: Tuple[int, int], 
                 padding: Optional[int] = None, 
                 fill: int = 0,
                 *,
                 rand_algo: Callable = cp.random.randint
                ) -> None:
        """Initialize crop configuration.

        Args:
            size: Desired output size as (height, width).
            padding: Optional border padding applied before cropping.
            fill: Fill value used when padding is applied.
            rand_algo: Callable to generate random integers; defaults to CuPy randint.
                       Useful when testing against PyTorch implementations.
        """
        self.padding = padding
        self.size = size
        self.fill = fill
        self.rand_algo = rand_algo

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        """Apply optional padding and perform a random crop.

        Args:
            img: Input image as a CuPy array.

        Returns:
            cp.ndarray: Cropped image with the configured output size.
        """
        if self.padding is not None:
            img = pad(img, self.padding, self.fill)

        i, j, h, w = self.get_params(img, self.size)
        return crop(img, i, j, h, w)
    
    def get_params(self, 
                   img: cp.ndarray, 
                   output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:
        """Compute top-left corner and size for a random crop.

        Args:
            img: Input image as a CuPy array.
            output_size: Target crop size as (height, width).

        Returns:
            Tuple[int, int, int, int]: (top, left, height, width) crop parameters.

        Raises:
            ValueError: If requested crop is larger than the image dimensions.
        """
        _, h, w = get_dims(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError("Requested crop size is bigger than image size")
        
        if w == tw and h == th:
            return 0, 0, h, w
        
        i = self.rand_algo(0, h - th + 1, size=(1,)).item()
        j = self.rand_algo(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw


