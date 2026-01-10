import os
import gzip
import shutil
import struct
import tarfile
import cupy as cp
from urllib.request import urlretrieve

def download_from_url(url: str, files: list[str], dir_path: str) -> None:
    # Download files 
    os.makedirs(dir_path, exist_ok=True)
    for file_name in files:
      file_url = url + file_name
      dest_path = os.path.join(dir_path, file_name)
      if os.path.exists(dest_path):
        continue
      print(f"Downloading {file_url} -> {dest_path}")
      urlretrieve(file_url, dest_path)

def gzip_extract(files: list[str], dir_path: str) -> None:
    # Extract .gz files 
    for file_name in files:
      if not file_name.endswith(".gz"):
        continue
      gz_path = os.path.join(dir_path, file_name)
      out_path = gz_path[:-3]
      if os.path.exists(out_path):
        continue
      with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
          shutil.copyfileobj(f_in, f_out)
      print(f"Extracting {gz_path} -> {out_path}")

def read_idx_file(file_path: str) -> cp.ndarray:
    # Read IDX file format
    # Structure: 
    # Header: | Prefix 0x00 (2 bytes) | Data type (1 byte) | Number of dimensions (1 byte) | Dimension sizes (4 bytes each) |
    # Data: | Actual data bytes |
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        _, _, dtype_code, ndim = struct.unpack('>BBBB', magic)

        dtype_map = {
            0x08: cp.uint8,
            0x09: cp.int8,
            0x0B: cp.int16,
            0x0C: cp.int32,
            0x0D: cp.float32,
            0x0E: cp.float64,
        }

        if dtype_code not in dtype_map:
            raise ValueError(f"Unsupported data type code: {dtype_code}")

        dtype = dtype_map[dtype_code]
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndim))
        data = cp.frombuffer(f.read(), dtype=dtype).reshape(shape)

    return data

def targz_extract(file: str, dir_path: str) -> None:
    # Extract .tar.gz files
    file_path = os.path.join(dir_path, file)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dir_path)
    print(f"Extracting {file_path} -> {dir_path}")

def read_bin_file(file_path: str) -> tuple[cp.ndarray, cp.ndarray]:
    # Read CIFAR-10 binary file format
    # Structure:
    # | 1 byte label | 3072 bytes image (32x32x3) |
    with open(file_path, 'rb') as f:
        num_images = os.path.getsize(file_path) // 3073
        data = cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num_images, 3073)
        images = data[:, 1:].reshape(num_images, 3, 32, 32)
        labels = data[:, 0]
    return images, labels

def get_dims(x: cp.ndarray) -> tuple[int, int, int]:
    # Get dimensions of the image
    channels = 1 if x.ndim == 2 else x.shape[-3]
    height, width = x.shape[-2:]
    return channels, height, width

def crop(x: cp.ndarray, i: int, j: int, h: int, w: int) -> cp.ndarray:
    # Crop the image
    return x[..., i:i+h, j:j+w]

def pad(x: cp.ndarray, p: int, fill: int) -> cp.ndarray: 
    # Constant padding for the image
    if p == 0:
        return x
    
    out = cp.empty(x.shape[:-2] + (x.shape[-2] + 2 * p, x.shape[-1] + 2 * p), dtype=x.dtype)
    out.fill(fill)
    out[..., p:p + x.shape[-2], p:p + x.shape[-1]] = x
    return out

def hflip(x: cp.ndarray, axis: int = -1) -> cp.ndarray:
    # Horizontal flip for the image
    return cp.flip(x, axis=axis)