"""Logging base class."""

from typing import Any


class Logger:
    """Base callable interface for logging data."""

    def __call__(self, data: dict) -> None:
        """Log provided data.

        Args:
            data: Arbitrary data to be logged.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError