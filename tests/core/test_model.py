import cupy as cp

from core.model import Model


class FakeLayer:
    """Minimal layer fake."""

    def __init__(self, forward_fn=None, backward_fn=None):
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn
        self.weight = cp.array([1.0])

    def forward(self, x):
        return self.forward_fn(x)

    def backward(self, grad, learning_rate):
        out = self.backward_fn(grad, learning_rate)
        self.weight -= learning_rate * cp.sum(grad)
        return out


def test_forward():
    x = cp.array([1.0, 2.0])

    layer1 = FakeLayer(forward_fn=lambda t: t + 1)
    layer2 = FakeLayer(forward_fn=lambda t: t * 3)

    model = Model([layer1, layer2])

    out = model.forward(x)

    assert cp.allclose(out, (x + 1) * 3)


def test_backward():
    grad_out = cp.array([1.0, 2.0])
    lr = 0.1

    layer1 = FakeLayer(backward_fn=lambda g, lr: g - 1)
    layer2 = FakeLayer(backward_fn=lambda g, lr: g * 2)

    model = Model([layer1, layer2])

    model.backward(grad_out, lr)

    assert cp.allclose(layer1.weight, 1.0 - lr * cp.sum(grad_out * 2))
    assert cp.allclose(layer2.weight, 1.0 - lr * cp.sum(grad_out))

