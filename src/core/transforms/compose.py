"""Compose transform utils."""

from typing import Any, Callable, Sequence


class Compose():
    """Chain multiple transforms in order.

    Each invocation applies the configured transforms sequentially to the
    provided input, allowing reusable application of multiple transforms.
    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        """Initialize composed transforms.

        Args:
            transforms: Ordered sequence of callables to apply to the input.
        """
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        """Apply all transforms sequentially to the input.

        Args:
            x: Input object to transform.

        Returns:
            Any: Result after all transforms have been applied.
        """
        for transform in self.transforms:
            x = transform(x)
        return x