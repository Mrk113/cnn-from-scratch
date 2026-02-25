"""Simple sequential model wrapper."""

import cupy as cp
from typing import Iterable

from core.layers import Layer
from .utils import float32_contiguous

class Model:
    """Run forward and backward passes through a sequence of layers."""

    def __init__(self, layers: Iterable[Layer]) -> None:
        """Store an ordered collection of layers.

        Args:
            layers: Iterable of layer objects providing forward and backward.
        """
        self.layers = layers

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """Propagate inputs through all layers in order. Preprocess inputs
        to be contiguous float32 for more efficient GPU processing.

        Args:
            x: Input tensor by the first layer.

        Returns:
            cp.ndarray: Output produced by the final layer.
        """
        for layer in self.layers:
            # More efficent GPU processing.
            x = float32_contiguous(x)
            x = layer.forward(x)
        return x
    
    def backward(self, grad: cp.ndarray, lr: float) -> None:
        """Propagate gradients backward through all layers. Preprocess gradients
        to be contiguous float32 for more efficient GPU processing.

        Args:
            grad: Upstream gradient from the loss.
            lr: Learning rate passed to each layer's backward step.
        """
        for layer in reversed(self.layers):
            # More efficent GPU processing.
            grad = float32_contiguous(grad)
            grad = layer.backward(grad, lr)
                