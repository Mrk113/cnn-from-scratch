def train(network, loss, X, Y, epochs=300, learning_rate=0.01, batch_size=32, verbose=True, log=False):
    for epoch in range(epochs):
        error = 0
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]

            # Forward pass
            output = X_batch
            for layer in network:
                output = layer.forward(output)

            # Compute loss
            batch_error = loss.compute(output, Y_batch)
            error += batch_error

            # Backward pass
            grad = loss.gradient(output, Y_batch)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        num_batches = (X.shape[0] + batch_size - 1) // batch_size
        error /= num_batches

        if verbose:
            print(f"Epoch: {epoch+1}/{epochs} - Loss: {error}")