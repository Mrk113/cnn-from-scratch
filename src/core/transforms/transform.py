"""Transform base class."""

from typing import Any


class Transform:
    """Base class to implement transforms."""

    def __call__(self, x: Any) -> Any:
        """Apply the transform to the given input.

        Args:
            x: Input object to transform.

        Returns:
            Any: Transformed output.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
