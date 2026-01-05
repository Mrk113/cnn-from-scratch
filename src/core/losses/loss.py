"""Base class for loss functions."""

class Loss:
    def compute(self, predicted, actual):
        raise NotImplementedError("Compute method not implemented.")
    
    def gradient(self, predicted, actual):
        raise NotImplementedError("Gradient method not implemented.")