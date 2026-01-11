import os
import cupy as cp
from typing import Callable, Optional

from .dataset import DataSet
from ..utils import download_from_url, gzip_extract, read_idx_file

class MNIST(DataSet):
    """MNIST Dataset."""

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    resources = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    train_list = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    test_list = ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    mean = [0.1307]
    std = [0.3081]

    def __init__(self,
                 *,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True
                 ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        self.data, self.targets = self._load_data()

    def __len__(self):
        # Returns the total number of samples
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[cp.ndarray, cp.ndarray]:
        # Returns the image and target at the specified index with applied transforms
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target
    
    def _check_exists(self) -> bool:
        # Check if dataset files exist
        dir_path = os.path.join(self.root, "MNIST")
        if self.train:
            files = self.train_list
        else:
            files = self.test_list
        return all(
            os.path.exists(os.path.join(dir_path, file_name))
            for file_name in files
        ) 
    
    def _load_data(self) -> tuple[cp.ndarray, cp.ndarray]:
        # Load data from files
        dir_path = os.path.join(self.root, "MNIST")
        if self.train:
            img_file, target_file = self.train_list
        else:
            img_file, target_file = self.test_list

        img_path = os.path.join(dir_path, img_file)
        target_path = os.path.join(dir_path, target_file)

        # Load images
        images = read_idx_file(img_path)
        # Load targets
        targets = read_idx_file(target_path)

        return images, targets

    def download(self) -> None:
        # Download dataset files into dir_path
        if self._check_exists():
            return

        dir_path = os.path.join(self.root, "MNIST")
        download_from_url(self.base_url, self.resources, dir_path)
        gzip_extract(self.resources, dir_path) 