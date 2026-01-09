import os
import cupy as cp
from typing import Optional, Callable

from .dataset import DataSet
from ..utils import download_from_url, targz_extract, read_bin_file

class CIFAR10(DataSet):
    """CIFAR-10 Dataset."""

    base_url = "https://www.cs.toronto.edu/~kriz/"
    resource = "cifar-10-binary.tar.gz"
    train_list = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    ]
    test_list = [
        "test_batch.bin"
    ]

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
        dir_path = os.path.join(self.root, "CIFAR10")
        if self.train:
            files = self.train_list
        else:
            files = self.test_list
        return all(
            os.path.exists(os.path.join(dir_path, "cifar-10-batches-bin", file_name))
            for file_name in files
        )
    
    def _load_data(self) -> tuple[cp.ndarray, cp.ndarray]:
        # Load data from files
        dir_path = os.path.join(self.root, "CIFAR10")
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        data_list = []
        targets_list = []

        for file_name in file_list:
            file_path = os.path.join(dir_path, "cifar-10-batches-bin", file_name)
            images, labels = read_bin_file(file_path)
            data_list.append(images)
            targets_list.append(labels)

        data = cp.concatenate(data_list, axis=0)
        targets = cp.concatenate(targets_list, axis=0)

        return data, targets
    
    def download(self) -> None:
        # Download dataset files into dir_path
        if self._check_exists():
            return
        dir_path = os.path.join(self.root, "CIFAR10")
        download_from_url(self.base_url, [self.resource], dir_path)
        targz_extract(os.path.join(dir_path, self.resource), dir_path)