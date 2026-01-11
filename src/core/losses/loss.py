"""Base class for loss functions."""

class Loss:
    def forward(self, predicted, actual):
        raise NotImplementedError
    
    def backward(self, predicted, actual):
        raise NotImplementedError