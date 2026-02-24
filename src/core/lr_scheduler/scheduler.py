"""Scheduler base class."""


class Scheduler:
    """Base class for learning rate schedulers."""

    def __init__(self) -> None:
        """Initialize scheduler state."""
        raise NotImplementedError
    
    def __call__(self, epoch: int) -> float:
        """Compute schedule value for the given epoch.

        Args:
            epoch: Epoch index to compute the schedule.

        Returns:
            float: Scheduled value (learning rate).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError