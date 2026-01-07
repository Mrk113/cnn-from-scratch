import cupy as cp

class Transform:
    def apply(self, X: cp.ndarray) -> cp.ndarray:
        raise NotImplementedError("Transform apply method not implemented.")