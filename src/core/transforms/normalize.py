import cupy as cp

from .transform import Transform

class Normalize(Transform):
        # Normalize the image with mean and std
        # Note: img is expected to be in float format [0, 1]
        # most datasets provide mean and std in [0, 1] range
        # Axis: along which channel axis to normalize. 
        # If None, no channel axis.
    def __init__(self, mean, std, axis=None) -> None:
        self.mean = cp.array(mean)
        self.std = cp.array(std)
        self.axis = axis
 
    def __call__(self, img: cp.ndarray) -> cp.ndarray:

        if cp.any(self.std == 0):
            raise ValueError("Standard deviation cannot be zero for normalization.")
        
        if self.axis is None:
            return (img - self.mean) / self.std
        
        axis = self.axis if self.axis >= 0 else img.ndim + self.axis
        shape = [1] * img.ndim
        shape[axis] = -1
        mean = self.mean.reshape(shape)
        std = self.std.reshape(shape)

        return (img - mean) / std 