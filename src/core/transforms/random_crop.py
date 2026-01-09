import cupy as cp
from core.transforms.compose import Transform

from ..utils import get_dims, pad, crop

class RandomCrop(Transform):
    # Randomly crop the given image.
    # Padding: Optional padding on each border of the image. With a given fill.
    # Size: Desired output size of the crop (height, width).
    def __init__(self, 
                 size: tuple[int, int], 
                 padding: int = None, 
                 fill: int = 0
                ) -> None:
        self.padding = padding
        self.size = size
        self.fill = fill

    def __call__(self, img: cp.ndarray) -> cp.ndarray:
        # Apply padding if specified (pad returns a new array)
        if self.padding is not None:
            img = pad(img, self.padding, self.fill)

        # Get random crop parameters
        i, j, h, w = self.get_params(img, self.size)
        return crop(img, i, j, h, w)
    
    def get_params(self, 
                   img: cp.ndarray, 
                   output_size: tuple[int, int]
                   ) -> tuple[int, int, int, int]:
        # Get parameters for a random crop.
        _, h, w = get_dims(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError("Requested crop size is bigger than image size")
        
        if w == tw and h == th:
            return 0, 0, h, w
        
        i = cp.random.randint(0, h - th + 1)
        j = cp.random.randint(0, w - tw + 1)
        return i, j, th, tw 


