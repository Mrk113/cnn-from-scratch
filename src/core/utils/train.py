from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import cupy as cp
from tqdm import tqdm

from core.utils.lr_scheduler import ConstantLR, TrainState, create_lr_scheduler

@dataclass
class Data:
    X: cp.ndarray
    Y: cp.ndarray

@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    verbose: bool
    lr_scheduler: Any | Mapping[str, Any] | None = None


def forward(network, X):
    output = X
    for layer in network:
        output = layer.forward(output)
    return output


def evaluate(network, loss, data: Data, batch_size: int):

    total_loss = cp.float32(0.0) 
    correct = cp.int32(0)
    dataset_size = int(data.X.shape[0])

    for i in range(0, dataset_size, batch_size):
        X_batch = data.X[i : i + batch_size]
        Y_batch = data.Y[i : i + batch_size]

        output = forward(network, X_batch)

        batch_loss = loss.compute(output, Y_batch)
        total_loss += batch_loss * X_batch.shape[0]

        pred = cp.argmax(output, axis=1)
        if Y_batch.ndim == 1:
            target = Y_batch
        else:
            target = cp.argmax(Y_batch, axis=1)
        correct += cp.equal(pred, target).sum()

    mean_loss = total_loss / dataset_size
    accuracy = 100.0 * correct / dataset_size

    return float(mean_loss.get()), float(accuracy.get())


def train(
    network,
    loss,
    train_data: Data,
    val_data: Data,
    config: TrainConfig,
    transform=None,
    shuffle: bool = True,
):
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": [],
    }

    print("Starting training...")

    dataset_size = int(train_data.X.shape[0])

    indices = cp.arange(dataset_size, dtype=cp.int32)

    steps_per_epoch = (dataset_size + config.batch_size - 1) // config.batch_size

    if config.lr_scheduler is None:
        lr_scheduler = ConstantLR()
    elif isinstance(config.lr_scheduler, Mapping):
        lr_scheduler = create_lr_scheduler(config.lr_scheduler)
    else:
        lr_scheduler = config.lr_scheduler

    lr_scheduler.setup(
        base_lr=float(config.learning_rate),
        epochs=int(config.epochs),
        steps_per_epoch=int(steps_per_epoch),
    )

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()

        total_loss = 0.0
        seen = 0

        # Shuffle indices in-place to reuse the same allocation.
        if shuffle:
            cp.random.shuffle(indices)

        for b in tqdm(
            range(0, dataset_size, config.batch_size),
            disable=not config.verbose,
            desc=f"Epoch {epoch+1}/{config.epochs}",
        ):
            batch_idx = indices[b : b + config.batch_size]
            X_batch = train_data.X[batch_idx]
            Y_batch = train_data.Y[batch_idx]
            batch_size = int(X_batch.shape[0])

            global_step = epoch * steps_per_epoch + (b // config.batch_size)
            state = TrainState(
                epoch=epoch,
                batch=(b // config.batch_size),
                global_step=int(global_step),
                epochs=int(config.epochs),
                steps_per_epoch=int(steps_per_epoch),
            )
            current_lr = float(lr_scheduler.lr(state))

            # Transform
            if transform is not None:
                for t in transform:
                    X_batch = t.apply(X_batch)

            # Forward pass
            output = forward(network, X_batch)

            # Compute loss
            batch_error = loss.compute(output, Y_batch)
            total_loss += batch_error * batch_size
            seen += batch_size

            # Backward pass
            grad = loss.gradient(output, Y_batch)
            for layer in reversed(network):
                grad = layer.backward(grad, current_lr)

        train_loss = total_loss / seen
        train_loss = float(train_loss.get())
        history["train_loss"].append(train_loss)

        val_loss, val_acc = evaluate(network, loss, val_data, batch_size=config.batch_size)
        history["test_loss"].append(val_loss)
        history["test_acc"].append(val_acc)

        epoch_duration = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_duration)

        if config.verbose:
            print(
                f"Epoch {epoch+1}/{config.epochs}: Train Loss: {train_loss:.4f} - Test Loss: {val_loss:.4f} "
                f"- Test Acc: {val_acc:.1f} - Epoch Time: {epoch_duration:.2f}s"
            )

    return history