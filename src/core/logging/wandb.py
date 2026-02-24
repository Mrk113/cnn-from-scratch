"""Weights & Biases logger implementation."""

from typing import Any, Optional
import wandb as wb

from .logger import Logger


class Wandb(Logger):
    """Log metrics and artifacts to Weights & Biases."""

    def __init__(self, project_name: str, run_name: Optional[str] = None) -> None:
        """Initialize a W&B run.

        Args:
            project_name: Name of the W&B project.
            run_name: Optional run name to use; if None, W&B auto-generates one.
        """
        self.run = wb.init(project=project_name, name=run_name)

    def __call__(self, data: dict) -> None:
        """Log a dictionary of metrics to W&B.

        Args:
            data: Mapping of metric names to values.
        """
        self.run.log(data)