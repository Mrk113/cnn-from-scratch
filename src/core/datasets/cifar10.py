"""CIFAR-10 dataset.

This module exposes a CuPy CIFAR-10 dataset wrapper that can
download the binary archives, load them into memory, and apply optional
transforms to images and labels.
"""

import os
import cupy as cp
from typing import Optional, Callable

from .dataset import DataSet
from ..utils import download_from_url, targz_extract, read_bin_file

class CIFAR10(DataSet):
    """Provide access to the CIFAR-10 dataset.

    Instances manage train or test splits and load
    samples as CuPy arrays for GPU processing.
    """

    base_url = "https://www.cs.toronto.edu/~kriz/"
    resources = ["cifar-10-binary.tar.gz"]
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
    # Values derived from the training set (RGB channels)
    std = [0.2023, 0.1994, 0.2010]
    mean = [0.4914, 0.4822, 0.4465]

    def __init__(self,
                 *,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True
                 ) -> None:
        """Initialize dataset metadata and load samples.

        Args:
            root: Root directory where CIFAR-10 files are stored or downloaded.
            train: Whether to load the training split (True) or test split (False).
            transform: Optional function applied to each image.
            target_transform: Optional function applied to each target label.
            download: Whether to download the dataset if required files are missing.

        Raises:
            RuntimeError: If the dataset files are absent after an attempted download.
        """
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
        """Fetch a sample and apply any configured transforms.

        Args:
            index: Zero-based sample index to retrieve.

        Returns:
            tuple[cp.ndarray, cp.ndarray]: Image tensor and corresponding label,
            with applied transforms.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        """Return dataset size for the selected split.

        Returns:
            int: Number of samples available.
        """
        return len(self.data)
    
    def _check_exists(self) -> bool:
        """Verify that required CIFAR-10 binary files exist on disk.

        Returns:
            bool: True if all expected files for the split are present.
        """
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
        """Load CIFAR-10 images and labels from binary batches.

        Returns:
            tuple[cp.ndarray, cp.ndarray]: Concatenated image and label arrays
            for the chosen split.
        """
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
            # Collect batches before a single concat to avoid repeated reallocations.
            data_list.append(images)
            targets_list.append(labels)

        data = cp.concatenate(data_list, axis=0)
        targets = cp.concatenate(targets_list, axis=0)

        return data, targets
    
    def download(self) -> None:
        """Download and extract CIFAR-10 archives when missing.

        The download is skipped if all expected files already exist.
        """
        if self._check_exists():
            return
        dir_path = os.path.join(self.root, "CIFAR10")
        download_from_url(self.base_url, self.resources, dir_path)
        targz_extract(self.resources, dir_path)

