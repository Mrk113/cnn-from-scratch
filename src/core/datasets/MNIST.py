import cupy as cp
from core.datasets.dataset import DataSet

class MNIST(DataSet):
    def __init__(self,
                 *,
                 flatten: bool = True,
                 normalize: bool = True,
                 label_encoding: str = "one_hot"):
        super().__init__(
            base_url="https://storage.googleapis.com/cvdf-datasets/mnist/",
            files=(
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ),
            flatten=flatten,
            normalize=normalize,
            label_encoding=label_encoding,
        )    

    def load(self) -> None:
        # Load MNIST dataset from local files or online source
        # Here we use a placeholder for loading logic

        # Placeholder loading logic
        # In practice, load the actual MNIST data here
        X_train = cp.random.rand(60000, 1, 28, 28).astype(cp.float32)
        y_train = cp.random.randint(0, 10, size=(60000,)).astype(cp.int32)
        X_test = cp.random.rand(10000, 1, 28, 28).astype(cp.float32)
        y_test = cp.random.randint(0, 10, size=(10000,)).astype(cp.int32)

        self._data = (X_train, y_train, X_test, y_test)

    def normalize(self) -> None:
        # Normalize MNIST dataset
        # Data must be [0, 1] before normalization
        # due to magic numbers used

        if self._data is None:
            raise RuntimeError("Data not loaded. Call load() before normalize().")
        X_train, y_train, X_test, y_test = self._data

        if cp.max(X_train) <= 1.0 or cp.max(X_test) <= 1.0:
             raise RuntimeError("Data already normalized to [0, 1].")

        # Normalize to [0, 1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        self._data = (X_train, y_train, X_test, y_test)

        super().normalize(std=0.3081, mean=0.1307)
        