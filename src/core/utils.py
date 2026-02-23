"""Utility helpers for dataset handling and array operations."""

import gzip
import os
import shutil
import struct
import tarfile
from urllib.request import urlretrieve

import cupy as cp


def download_from_url(url: str, files: list[str], dir_path: str) -> None:
    """Download one or more files from a base URL into a directory.

    Args:
        url: Base URL where files are hosted.
        files: Filenames to download from the base URL.
        dir_path: Local directory to store downloaded files.
    """
    os.makedirs(dir_path, exist_ok=True)
    for file_name in files:
        file_url = url + file_name
        dest_path = os.path.join(dir_path, file_name)
        if os.path.exists(dest_path):
            continue
        print(f"Downloading {file_url} -> {dest_path}")
        urlretrieve(file_url, dest_path)


def gzip_extract(files: list[str], dir_path: str) -> None:
    """Extract a list of .gz files if not already extracted.

    Args:
        files: Filenames (expected to end with .gz) to extract.
        dir_path: Directory containing the compressed files and extraction target.
    """
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
    """Read data from an IDX file into a CuPy array.

    The IDX format stores a header describing data type and dimensions followed 
    by raw data bytes.

    Args:
        file_path: Path to the IDX file.

    Returns:
        cp.ndarray: Array containing the file contents with the appropriate shape and dtype.

    Raises:
        ValueError: If the file encodes an unsupported data type.
    """
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
    """Extract a .tar.gz archive into the given directory.

    Args:
        file: Archive filename located within dir_path.
        dir_path: Destination directory for extraction.
    """
    file_path = os.path.join(dir_path, file)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dir_path)
    print(f"Extracting {file_path} -> {dir_path}")


def read_bin_file(file_path: str) -> tuple[cp.ndarray, cp.ndarray]:
    """Read CIFAR-10 binary file into image and label arrays.

    The binary layout is 1 label byte followed by 3072 image bytes (32x32x3).

    Args:
        file_path: Path to a CIFAR-10 binary batch file.

    Returns:
        tuple[cp.ndarray, cp.ndarray]: Images shaped (N, 3, 32, 32) and labels shaped (N,).
    """
    with open(file_path, 'rb') as f:
        num_images = os.path.getsize(file_path) // 3073
        data = cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num_images, 3073)
        images = data[:, 1:].reshape(num_images, 3, 32, 32)
        labels = data[:, 0]
    return images, labels


def get_dims(x: cp.ndarray) -> tuple[int, int, int]:
    """Return channel, height, and width dimensions for an image tensor.

    Args:
        x: Image tensor with shape (..., H, W) or (..., C, H, W).

    Returns:
        tuple[int, int, int]: Channels, height, and width extracted from the input.
    """
    channels = 1 if x.ndim == 2 else x.shape[-3]
    height, width = x.shape[-2:]
    return channels, height, width


def crop(x: cp.ndarray, i: int, j: int, h: int, w: int) -> cp.ndarray:
    """Crop a subregion from an image tensor.

    Args:
        x: Input image tensor.
        i: Starting row index of the crop (inclusive).
        j: Starting column index of the crop (inclusive).
        h: Height of the cropped region.
        w: Width of the cropped region.

    Returns:
        cp.ndarray: Cropped view of the input tensor.
    """
    return x[..., i:i+h, j:j+w]


def pad(x: cp.ndarray, p: int, fill: int) -> cp.ndarray: 
    """Apply constant padding around spatial dimensions of an image tensor.

    Args:
        x: Input image tensor.
        p: Padding width applied to all spatial borders.
        fill: Constant value used to fill the padded area.

    Returns:
        cp.ndarray: New tensor with padding applied.
    """
    if p == 0:
        return x
    
    out = cp.empty(x.shape[:-2] + (x.shape[-2] + 2 * p, x.shape[-1] + 2 * p), dtype=x.dtype)
    out.fill(fill)
    out[..., p:p + x.shape[-2], p:p + x.shape[-1]] = x
    return out


def hflip(x: cp.ndarray, axis: int = -1) -> cp.ndarray:
    """Horizontally flip an image tensor along the specified axis.

    Args:
        x: Input image tensor.
        axis: Axis along which to flip; defaults to the last dimension.

    Returns:
        cp.ndarray: Flipped tensor.
    """
    return cp.flip(x, axis=axis)
