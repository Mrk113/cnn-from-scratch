"""Step learning rate scheduler."""

from .scheduler import Scheduler


class StepLR(Scheduler):
    """Decay the learning rate by gamma every step_size epochs."""

    def __init__(self, lr: float, step_size: int = 1, gamma: float = 1.0) -> None:
        """Initialize step scheduler parameters.

        Args:
            lr: Initial learning rate.
            step_size: Interval of epochs between decays.
            gamma: Multiplicative factor for learning rate decay.
        """
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
    
    def __call__(self, epoch: int) -> float:
        """Compute the learning rate for a given epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            float: Updated learning rate after applying step decay.
        """
        if epoch > 0 and epoch % self.step_size == 0:
            self.lr *= self.gamma
        return self.lr