import cupy as cp
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
    eval_every: int


def forward(network, X):
    output = X
    for layer in network:
        output = layer.forward(output)
    return output


def evaluate(network, loss, data: Data, batch_size: int):
    return 1, 1


def train(
    network,
    loss,
    train_data: Data,
    val_data: Data,
    config: TrainConfig,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(config.epochs):
        error = 0
        for i in range(0, train_data.X.shape[0], config.batch_size):
            X_batch = train_data.X[i : i + config.batch_size]
            Y_batch = train_data.Y[i : i + config.batch_size]

            # Forward pass
            output = forward(network, X_batch)

            # Compute loss
            batch_error = loss.compute(output, Y_batch)
            error += batch_error

            # Backward pass
            grad = loss.gradient(output, Y_batch)
            for layer in reversed(network):
                grad = layer.backward(grad, config.learning_rate)

        num_batches = (train_data.X.shape[0] + config.batch_size - 1) // config.batch_size
        error /= max(1, num_batches)

        train_loss = float(error.get())
        history["train_loss"].append(train_loss)

            
        val_loss, val_acc = evaluate(network, loss, val_data, batch_size=config.batch_size)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if config.verbose:
            print(f"Epoch: {epoch+1}/{config.epochs} - Train Loss: {train_loss:.6f}")

    return history