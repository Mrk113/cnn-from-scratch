"""Base class for loss functions."""

class Loss:
    def compute(predicted, actual):
        raise NotImplementedError("Compute method not implemented.")
    
    def gradient(predicted, actual):
        raise NotImplementedError("Gradient method not implemented.")