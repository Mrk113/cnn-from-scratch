import cupy as cp
from core.transforms.transfrom import Transform

class RandomCrop(Transform):
    def __init__(self, padding: int):
        self.padding = padding

    def apply(self, X: cp.ndarray) -> cp.ndarray:
      x = X.astype(cp.float32, copy=False)
      n, c, h, w = x.shape

      p = int(self.padding)
      if p > 0:
        x_padded = cp.pad(
          x,
          ((0, 0), (0, 0), (p, p), (p, p)),
          mode="constant",
        )
        max_offset = 2 * p
        top = cp.random.randint(0, max_offset + 1, size=n, dtype=cp.int32)
        left = cp.random.randint(0, max_offset + 1, size=n, dtype=cp.int32)
      else:
        x_padded = x
        top = cp.zeros((n,), dtype=cp.int32)
        left = cp.zeros((n,), dtype=cp.int32)

      rows = top[:, None] + cp.arange(h, dtype=cp.int32)[None, :]
      cols = left[:, None] + cp.arange(w, dtype=cp.int32)[None, :]

      n_idx = cp.arange(n, dtype=cp.int32)[:, None, None, None]
      c_idx = cp.arange(c, dtype=cp.int32)[None, :, None, None]
      row_idx = rows[:, None, :, None]
      col_idx = cols[:, None, None, :]

      return x_padded[n_idx, c_idx, row_idx, col_idx]
