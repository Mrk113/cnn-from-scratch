import cupy as cp
from tqdm import tqdm
import time
from dataclasses import dataclass

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


def forward(network, X):
    output = X
    for layer in network:
        output = layer.forward(output)
    return output


def evaluate(network, loss, data: Data, batch_size: int):

    total_loss = cp.array(0.0, dtype=cp.float32)
    correct = cp.array(0, dtype=cp.int32)
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

            # Forward pass
            output = forward(network, X_batch)

            # Compute loss
            batch_error = loss.compute(output, Y_batch)
            total_loss += batch_error * batch_size
            seen += batch_size

            # Backward pass
            grad = loss.gradient(output, Y_batch)
            for layer in reversed(network):
                grad = layer.backward(grad, config.learning_rate)

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