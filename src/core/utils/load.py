from __future__ import annotations

import gzip
import os
import shutil
import struct
import tarfile
from typing import Literal
from urllib.request import urlretrieve

import cupy as cp

LabelEncoding = Literal["one_hot", "index"]


class DatasetBase:
  def __init__(
    self,
    name: str,
    base_url: str,
    files: tuple[str, ...],
    root_dir: str = "data/",
  ):
    self.name = name
    self.base_url = base_url
    self.files = files
    self.root_dir = root_dir

  def one_hot(self, labels: cp.ndarray, num_classes: int, dtype=cp.float32) -> cp.ndarray:
    labels = labels.astype(cp.int64, copy=False)
    out = cp.zeros((labels.shape[0], num_classes), dtype=dtype)
    out[cp.arange(labels.shape[0]), labels] = 1
    return out

  def dir_path(self) -> str:
    return os.path.join(self.root_dir, self.name)

  def download(self) -> None:
    os.makedirs(self.dir_path(), exist_ok=True)
    for file_name in self.files:
      url = self.base_url + file_name
      dest_path = os.path.join(self.dir_path(), file_name)
      if os.path.exists(dest_path):
        continue
      print(f"Downloading {url} -> {dest_path}")
      urlretrieve(url, dest_path)

  def extract(self) -> None:
    raise NotImplementedError

  def dataload(
    self,
    split: Literal["train", "test"],
    flatten: bool = True,
    normalize: bool = True,
    label_encoding: LabelEncoding = "one_hot",
  ):
    raise NotImplementedError


class MNISTDataset(DatasetBase):
  def __init__(self):
    super().__init__(
      name="mnist",
      base_url="https://storage.googleapis.com/cvdf-datasets/mnist/",
      files=(
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
      ),
    )

  def extract(self) -> None:
    for file_name in self.files:
      if not file_name.endswith(".gz"):
        continue
      gz_path = os.path.join(self.dir_path(), file_name)
      out_path = gz_path[:-3]
      if os.path.exists(out_path):
        continue
      print(f"Extracting {gz_path} -> {out_path}")
      with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
          shutil.copyfileobj(f_in, f_out)

  def _load_idx(self, file_path: str):
    with open(file_path, 'rb') as f:
      magic = f.read(4)
      _, _, dtype_code, ndim = struct.unpack('>BBBB', magic)

      dtype_map = {
        0x08: cp.uint8,
        0x09: cp.int8,
        0x0B: cp.int16,
        0x0C: cp.int32,
        0x0D: cp.float32,
        0x0E: cp.float64
      }

      dtype = dtype_map.get(dtype_code)
      if dtype is None:
        raise ValueError(f"Unsupported data type code: {dtype_code}")

      shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
      data = cp.frombuffer(f.read(), dtype=dtype).reshape(shape)
      return data

  def dataload(
    self,
    split: Literal["train", "test"],
    flatten: bool,
    normalize: bool,
    label_encoding: LabelEncoding,
  ):
    base = self.dir_path()
    if split == "train":
      images_path = os.path.join(base, "train-images-idx3-ubyte")
      labels_path = os.path.join(base, "train-labels-idx1-ubyte")
    elif split == "test":
      images_path = os.path.join(base, "t10k-images-idx3-ubyte")
      labels_path = os.path.join(base, "t10k-labels-idx1-ubyte")
    else:
      raise ValueError(f"Unknown split: {split}")

    X = self._load_idx(images_path).astype(cp.float32, copy=False)
    y = self._load_idx(labels_path)

    if normalize:
      if X.dtype != cp.float32:
        X = X.astype(cp.float32, copy=False)
      X /= cp.float32(255.0)
    if flatten:
      X = X.reshape(X.shape[0], -1)

    if label_encoding == "one_hot":
      y = self.one_hot(y, 10, dtype=cp.float32)
    elif label_encoding == "index":
      y = y.astype(cp.int64, copy=False)
    else:
      raise ValueError(f"Unknown label encoding: {label_encoding}")

    return X, y


class CIFAR10Dataset(DatasetBase):
  def __init__(self):
    super().__init__(
      name="cifar10",
      base_url="https://www.cs.toronto.edu/~kriz/",
      files=("cifar-10-binary.tar.gz",),
    )
  def extract(self) -> None:
    archive = os.path.join(self.dir_path(), "cifar-10-binary.tar.gz")
    if os.path.exists(self.dir_path() + "/cifar-10-batches-bin/"):
      return
    print(f"Extracting {archive} -> {self.dir_path()}")
    with tarfile.open(archive, "r:gz") as tar:
      tar.extractall(path=self.dir_path())

  def _read_bin(self, path: str):
    with open(path, "rb") as f:
      raw = f.read()
    arr = cp.frombuffer(raw, dtype=cp.uint8)
    record_size = 1 + 3072
    if arr.size % record_size != 0:
      raise ValueError(f"Unexpected CIFAR-10 file size: {path}")
    records = arr.reshape(-1, record_size)
    labels = records[:, 0].astype(cp.int64)
    images = records[:, 1:].reshape(-1, 3, 32, 32)
    return images, labels

  def dataload(
    self,
    split: Literal["train", "test"],
    flatten: bool,
    normalize: bool,
    label_encoding: LabelEncoding,
  ):
    base = self.dir_path() + "/cifar-10-batches-bin/"
    if split == "train":
      X, y = None, None
      for i in range(1, 6):
        images_path = os.path.join(base, f"data_batch_{i}.bin")
        X_batch, y_batch = self._read_bin(images_path)
        if i == 1:
          X = X_batch
          y = y_batch
        else:
          X = cp.concatenate((X, X_batch), axis=0)
          y = cp.concatenate((y, y_batch), axis=0)
    elif split == "test":
      images_path = os.path.join(base, "test_batch.bin")
      X, y = self._read_bin(images_path) 
    else:
      raise ValueError(f"Unknown split: {split}")
    
    X = X.astype(cp.float32, copy=False)

    if normalize:
      mean = cp.mean(X, axis=(0, 2, 3), keepdims=True)
      std = cp.std(X, axis=(0, 2, 3), keepdims=True)
      X = (X - mean) / (std)
    if flatten:
      X = X.reshape(X.shape[0], -1)

    if label_encoding == "one_hot":
      y = self.one_hot(y, 10, dtype=cp.float32)
    elif label_encoding == "index":
      y = y.astype(cp.int64, copy=False)
    else:
      raise ValueError(f"Unknown label encoding: {label_encoding}")
    
    return X, y

data_sets = {
  "mnist": MNISTDataset(),
  "cifar10": CIFAR10Dataset(),
}

def load(
  data_set: str,
  flatten: bool = True,
  normalize: bool = True,
  label_encoding: LabelEncoding = "one_hot",
):
  """Convenience loader: returns (X_train, y_train, X_test, y_test)."""

  if data_set not in data_sets:
    raise ValueError(f"Unknown dataset: {data_set}")

  ds = data_sets[data_set] 

  ds.download()
  ds.extract()

  X_train, y_train = ds.dataload(
    "train", flatten=flatten, normalize=normalize, label_encoding=label_encoding
  )
  X_test, y_test = ds.dataload(
    "test", flatten=flatten, normalize=normalize, label_encoding=label_encoding
  )
  return X_train, y_train, X_test, y_test