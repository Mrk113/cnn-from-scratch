"""Training utilities for running epochs and evaluations."""

import time
from typing import Any, Iterable, Optional, Tuple

import cupy as cp
from tqdm import tqdm

from core.datasets import DataSet


class Trainer:
    """Training and evaluation loops for a model."""

    def __init__(self, model: Any, loss: Any, lr_sched: Any, logger: Optional[Any] = None) -> None:
        """Initialize trainer with model, loss, scheduler, and optional logger.

        Args:
            model: Object exposing ``forward`` and ``backward`` methods.
            loss: Loss object exposing ``forward`` and ``backward`` methods.
            lr_sched: Scheduler callable providing learning rates per epoch.
            logger: Optional callable for logging metrics.
        """
        self.model = model
        self.loss = loss
        self.lr_sched = lr_sched
        self.logger = logger
        self.logs: dict[str, Any] = {}
    
    def fit(self,
            train_data: DataSet,
            test_data: Optional[DataSet] = None,
            *,
            epochs: int,
            batch_size: int,
            test_interval: int = 1,
            test_batch_size: Optional[int] = None
          ) -> None:
        """Run training for the specified number of epochs.

        Args:
            train_data: Dataset object providing (data, target) batches.
            test_data: Optional dataset object for evaluation.
            epochs: Total number of training epochs.
            batch_size: Training batch size.
            test_interval: Evaluate every N epochs when test_data is provided.
            test_batch_size: Optional evaluation batch size; defaults to training batch size.
        """
        num_batches = (len(train_data) + batch_size - 1) // batch_size

        for e in range(epochs):
            epoch_start = time.perf_counter()

            self.logs = {}
            loss = 0.0
            lr = self.lr_sched(e)
 
            for b in tqdm(range(0, len(train_data), batch_size), 
                          desc=f"Epoch {e+1}/{epochs}"
                ):
                data, targets = train_data[b : b + batch_size]

                # Forward pass
                pred = self.model.forward(data)
                loss += self.loss.forward(pred, targets)

                # Backward pass
                grad_loss = self.loss.backward(pred, targets)
                self.model.backward(grad_loss, lr)
            
            loss /= num_batches

            epoch_time = time.perf_counter() - epoch_start

            # Logging
            summary = f"Epoch {e+1}/{epochs}: Lr: {lr:.4f} - Train Loss: {loss:.4f}"
            self.logs["epoch"] = e + 1
            self.logs["lr"] = lr
            self.logs["train_loss"] = float(loss)
            self.logs["epoch_time(s)"] = epoch_time

            # Evaluation loop
            if test_data is not None and (e + 1) % test_interval == 0:
                tb = test_batch_size or batch_size
                test_loss, test_acc = self.evaluate(test_data, batch_size=tb)
                # Logging
                summary += f" - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}"
                self.logs["test_loss"] = test_loss
                self.logs["test_acc"] = test_acc
            
            if self.logger is not None:
                self.logger(self.logs)

            tqdm.write(summary)

    def evaluate(self, data: DataSet, batch_size: int) -> Tuple[float, float]:
        """Evaluate the model on provided data.

        Args:
            data: Dataset object providing (data, target) batches.
            batch_size: Batch size used during evaluation.

        Returns:
            Tuple[float, float]: Average loss and accuracy over the dataset.
        """
        eval_start = time.perf_counter()

        total_loss = 0.0
        correct = 0
        total = 0

        for b in tqdm(range(0, len(data), batch_size),
                      desc="Evaluating",
                      leave=False
            ):
            x, y = data[b : b + batch_size]

            pred = self.model.forward(x)
            loss = self.loss.forward(pred, y)

            total_loss += cp.sum(loss).item() * x.shape[0]
            pred_labels = cp.argmax(pred, axis=1)
            if y.ndim > 1:
                # One Hot encoding case
                y_labels = cp.argmax(y, axis=1)
            else:
                # Regular label case
                y_labels = y
            correct += cp.sum(pred_labels == y_labels).item()
            total += y_labels.size

        avg_loss = total_loss / total
        accuracy = correct / total

        eval_time = time.perf_counter() - eval_start
        self.logs["eval_time(s)"] = eval_time

        return avg_loss, accuracy