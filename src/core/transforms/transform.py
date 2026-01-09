from typing import Any

class Transform:
    """Base class to implement transforms."""
    def __call__(self, x: Any) -> Any:
        raise NotImplementedError
