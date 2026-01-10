import cupy as cp
from tqdm import tqdm
from .history import History

class Model:
    # Model for training
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        # Forward pass through all layers
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad, lr):
        # Backward pass through all layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def config(self, *, loss, lr, lr_scheduler = None):
        # Configure the model with loss function and learning rate scheduler
        self.loss = loss
        self.lr = lr
        self.lr_scheduler = lr_scheduler
    
    def fit(self, 
            train_data, 
            val_data=None, 
            *, 
            epochs=10, 
            batch_size=32, 
            val_interval=1
            ) -> None:
        # Train the model with given training data and optional validation data

        if not hasattr(self, 'loss'):
            raise RuntimeError("Model loss function is not configured. Please call model.config() before training.")
        
        history = History(
            "epoch",
            "lr",
            "train_loss", 
            "val_loss", 
            "val_acc"
        )

        num_batches = (len(train_data) + batch_size - 1) // batch_size

        for e in range(epochs):

            loss = 0.0
            lr = self.lr

            for b in tqdm(range(0, len(train_data), batch_size), 
                          desc=f"Epoch {e+1}/{epochs}"
                ):
                data, targets = train_data[b : b + batch_size]

                # Forward pass
                pred = self.forward(data)
                loss += self.loss.forward(pred, targets)

                # Backward pass
                grad_loss = self.loss.backward(pred, targets)
                # LR Scheduler step
                if self.lr_scheduler is not None:
                    lr = self.lr_scheduler.step()
                self.backward(grad_loss, lr)

            loss /= num_batches
            # Logging
            log = f"Epoch {e+1}/{epochs}: Lr: {lr:.4f} - Train Loss: {loss:.4f}"
            history["epoch"].append(e + 1)
            history["lr"].append(lr)
            history["train_loss"].append(loss)

            # Validation loop
            if val_data is not None and (e + 1) % val_interval == 0:
                val_loss, val_acc = evaluate(self, val_data)
                # Logging
                log += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            tqdm.write(log)
        return history
    


                