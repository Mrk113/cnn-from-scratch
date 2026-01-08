import os
import cupy as cp
from urllib.request import urlretrieve

class DataSet:
    """Base class for datasets."""
    def __init__(self,
                 base_url: str,
                 files: tuple[str, ...],
                 root_dir: str = "data/",
                 *, 
                 flatten: bool = False, 
                 normalize: bool = False, 
                 label_encoding: str = "index") -> None:
        # Provided by subclasses
        self.base_url = base_url
        self.files = files
        self.root_dir = root_dir

        # Options
        self.flatten = flatten
        self.normalize = normalize
        self.label_encoding = label_encoding
        self._data = None

    @property
    def data(self) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        # Return data for usage

        if self._data is None:
            raise RuntimeError("Data not loaded. Call load() before accessing data.")
        return self._data
    
    def normalize(self, std: float, mean: float) -> None:
        # Normalize dataset with given mean and std
        # Following standard formula: (X - mean) / std
        # Applies to both train and test sets not the labels

        # Magic number std (standard deviation) calculated
        # by sqrt(1 / N * H * W * sum over all (pixels - mean)^2))
        # Magic number mean (mean value) calculated
        # by 1 / N * H * W * sum over all pixels

        if self._data is None:
            raise RuntimeError("Data not loaded. Call load() before normalize().")

        X_train, y_train, X_test, y_test = self._data
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        self._data = (X_train, y_train, X_test, y_test)

    def flatten(self) -> None:
        # Flatten dataset samples to 1D arrays
        # Useful for fully connected networks

        if self._data is None:
            raise RuntimeError("Data not loaded. Call load() before flatten().")
        
        self._data = tuple(arr.reshape(arr.shape[0], -1) for arr in self._data)

    def download(self) -> None:
        # Download dataset files into dir_path
        dir_path = os.path.join(self.root_dir, self.name)
        os.makedirs(dir_path, exist_ok=True)

        for file_name in self.files:
            url = self.base_url + file_name
            dest_path = os.path.join(dir_path, file_name)
            # Check if file already exists and skip download
            if os.path.exists(dest_path):
                continue
            print(f"Downloading {url} -> {dest_path}")
            urlretrieve(url, dest_path)

    def load(self) -> None:
        # Load dataset into memory
        # Must be implemented by subclasses
        raise NotImplementedError