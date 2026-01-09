from typing import Sequence, Callable

from .transform import Transform

class Compose(Transform):
    """
    Composes several transforms together. So the data is transformed by
    each call of __getitem__ in Dataset.
    """
    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x