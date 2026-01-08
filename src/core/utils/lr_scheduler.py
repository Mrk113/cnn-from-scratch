from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Protocol, Type


@dataclass(frozen=True)
class TrainState:
    epoch: int
    batch: int
    global_step: int
    epochs: int
    steps_per_epoch: int


class LRScheduler(Protocol):
    """Simple LR scheduler interface.

    The scheduler is initialized via `setup` and then queried via `lr(state)`.
    """

    def setup(self, *, base_lr: float, epochs: int, steps_per_epoch: int) -> None:
        ...

    def lr(self, state: TrainState) -> float:
        ...


class ConstantLR:
    def __init__(self) -> None:
        self._base_lr: float | None = None

    def setup(self, *, base_lr: float, epochs: int, steps_per_epoch: int) -> None:
        self._base_lr = float(base_lr)

    def lr(self, state: TrainState) -> float:
        if self._base_lr is None:
            raise RuntimeError("ConstantLR.setup() must be called before lr().")
        return self._base_lr


class StepLR:
    """Decay LR by `gamma` every `step_size` epochs."""

    def __init__(self, *, step_size: int, gamma: float = 0.1) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self._base_lr: float | None = None

    def setup(self, *, base_lr: float, epochs: int, steps_per_epoch: int) -> None:
        self._base_lr = float(base_lr)

    def lr(self, state: TrainState) -> float:
        if self._base_lr is None:
            raise RuntimeError("StepLR.setup() must be called before lr().")
        k = state.epoch // self.step_size
        return self._base_lr * (self.gamma ** k)


class CosineAnnealingLR:
    """Cosine annealing from base_lr down to min_lr over all steps."""

    def __init__(self, *, min_lr: float = 0.0) -> None:
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")
        self.min_lr = float(min_lr)
        self._base_lr: float | None = None
        self._total_steps: int | None = None

    def setup(self, *, base_lr: float, epochs: int, steps_per_epoch: int) -> None:
        self._base_lr = float(base_lr)
        self._total_steps = int(epochs) * int(steps_per_epoch)
        if self._total_steps <= 0:
            raise ValueError("epochs * steps_per_epoch must be > 0")

    def lr(self, state: TrainState) -> float:
        if self._base_lr is None or self._total_steps is None:
            raise RuntimeError("CosineAnnealingLR.setup() must be called before lr().")
        t = min(max(state.global_step, 0), self._total_steps)
        # If t == total_steps, return min_lr.
        cos_inner = math.pi * (t / self._total_steps)
        return self.min_lr + 0.5 * (self._base_lr - self.min_lr) * (1.0 + math.cos(cos_inner))


_SCHEDULER_REGISTRY: Dict[str, Callable[..., Any]] = {
    "constant": ConstantLR,
    "step": StepLR,
    "cosine": CosineAnnealingLR,
}


def create_lr_scheduler(spec: Mapping[str, Any] | None) -> Any:
    """Create a scheduler from a dict spec.

    Example:
        {"name": "step", "step_size": 10, "gamma": 0.5}
        {"name": "cosine", "min_lr": 1e-5}
    """

    if spec is None:
        return ConstantLR()

    name = str(spec.get("name", "constant")).lower()
    factory = _SCHEDULER_REGISTRY.get(name)
    if factory is None:
        known = ", ".join(sorted(_SCHEDULER_REGISTRY.keys()))
        raise ValueError(f"Unknown lr_scheduler '{name}'. Known: {known}")

    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return factory(**kwargs)
