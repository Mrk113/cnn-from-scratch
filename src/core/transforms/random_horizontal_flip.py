import cupy as cp

from core.transforms.transfrom import Transform

class RandomHorizontalFlip(Transform):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def apply(self, X: cp.ndarray) -> cp.ndarray:
        x = X.astype(cp.float32, copy=False)
        p = float(self.prob)
        if p <= 0.0:
          return x
        if p >= 1.0:
          return cp.flip(x, axis=3)

        n = int(x.shape[0])
        mask = (cp.random.rand(n) < p)
        if bool(mask.all()):
          return cp.flip(x, axis=3)
        if not bool(mask.any()):
          return x

        out = x.copy()
        out[mask] = cp.flip(out[mask], axis=3)
        return out