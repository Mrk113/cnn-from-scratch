"""Normalization transform."""

import cupy as cp
from typing import Optional, Sequence

from .transform import Transform


class Normalize(Transform):
    """Normalize tensors using per-channel mean and standard deviation.

    Inputs are expected to be in floating-point format (typically scaled to
    the [0, 1] range). When an axis is provided, mean and std are broadcast
    along that channel dimension.
    """

    def __init__(self,
                 mean: Sequence[float] | cp.ndarray,
                 std: Sequence[float] | cp.ndarray,
                 axis: Optional[int] = None
                 ) -> None:
        """Store normalization parameters.

        Args:
            mean: Mean values used for centering; one value per channel.
            std: Standard deviation values used for scaling; one value per channel.
            axis: Channel axis along which to apply normalization; if None, no
                channel axis is assumed and mean/std must match the input shape.
        """
        self.mean = cp.array(mean)
        self.std = cp.array(std)
        self.axis = axis
 
    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        """Apply normalization to an input image tensor.

        Args:
            img: Input image as a CuPy array.

        Returns:
            cp.ndarray: Normalized image tensor.

        Raises:
            ValueError: If any standard deviation entry is zero.
        """
        if cp.any(self.std == 0):
            raise ValueError("Standard deviation cannot be zero for normalization.")
        
        if self.axis is None:
            return (img - self.mean) / self.std
        
        axis = self.axis if self.axis >= 0 else img.ndim + self.axis
        # Reshape mean/std for broadcasting along the specified channel axis.
        shape = [1] * img.ndim
        shape[axis] = -1
        mean = self.mean.reshape(shape)
        std = self.std.reshape(shape)

        return (img - mean) / std