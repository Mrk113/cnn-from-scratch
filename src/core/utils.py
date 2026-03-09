"""Utility helpers for dataset handling and array operations."""

import gzip
import time
import os
import shutil
import struct
import tarfile
from tqdm import tqdm
from urllib.request import urlretrieve

import cupy as cp

def float32_contiguous(x: cp.ndarray) -> cp.ndarray:
    """Ensure the input array is float32 and C-contiguous.

    Args:
        x: Input array to be converted.

    Returns:
        cp.ndarray: Converted array with float32 dtype and C-contiguous memory layout.
        If needed.
    """
    if x.dtype != cp.float32:
        x = x.astype(cp.float32, copy=False)
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
    return x


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


def targz_extract(files: list[str], dir_path: str) -> None:
    """Extract a list of .tar.gz archive into the given directory.

    Args:
        files: List of archive filenames located within dir_path.
        dir_path: Destination directory for extraction.
    """
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
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


def im2col(im: cp.ndarray,
           kH: int,
           kW: int,
           s: int,
           *,
           indices: tuple[int, int, int, int] | None = None,
           return_indices: bool = False
          ) -> cp.ndarray | tuple[cp.ndarray, tuple[int, int, int, int]]:
    """Convert input tensor into columns for convolution.

    Args:
        im: Padded input tensor of shape (N, C, H_padded, W_padded).
        kH: Kernel height.
        kW: Kernel width.
        s: Stride for the convolution operation.
        indices: Optional precomputed (i, j, d) index tuple from ``get_indices``.
        return_indices: When True, return both the column matrix and the indices
            used to build it. This allows callers to reuse the same indices in
            the backward pass instead of recomputing them each time.

    Returns:
        Either the column matrix with shape (C*kH*kW, N*H_out*W_out) or a tuple
        of (cols, indices) when return_indices is True.
    """
    if indices is None:
        indices = get_indices(im.shape, kH, kW, s)

    i, j, d = indices

    # Advanced indexing gathers all sliding windows in one go. The transpose
    # reshapes to the expected (C*kH*kW, N*H_out*W_out) layout without the
    # extra concat used previously.
    cols = im[:, d, i, j].transpose(1, 0, 2).reshape(d.size, -1)

    if return_indices:
        return cols, indices
    return cols

def col2im(cols: cp.ndarray,
         x_shape: tuple[int, int, int, int],
         kH: int,
         kW: int,
         s: int,
         *,
         indices: tuple[cp.ndarray, cp.ndarray, cp.ndarray] | None = None
        ) -> cp.ndarray:
    """Convert columns back to the original image shape after convolution.

    Args:
        cols: 2D array of shape (C*kH*kW, N*H_out*W_out) containing columnized data.
        x_shape: Original shape of the input tensor (N, C, H_padded, W_padded).
        kH: Kernel height.
        kW: Kernel width.
        s: Stride for the convolution operation.
    """
    N, C, H_p, W_p = x_shape
    if indices is None:
        i, j, d = get_indices(x_shape, kH, kW, s)
    else:
        i, j, d = indices

    cols = cols.reshape(C * kH * kW, N, -1).transpose(1, 0, 2)  # (N, C*kH*kW, H_out*W_out)
    im = cp.zeros(x_shape, dtype=cols.dtype)
    # Use scatter_add to accumulate values into the correct positions in the output image
    # slice(None) is used to select all elements along the batch dimension
    cp.add.at(im, (slice(None), d, i, j), cols)  # (N, C, H_padded, W_padded)

    return im


def get_indices(x_shape: tuple[int, int, int, int],
                kH: int,
                kW: int,
                s: int
               ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Generate indices for im2col operation.

    Args:
        x_shape: Shape of the input tensor (N, C, H, W).
        kH: Kernel height.
        kW: Kernel width.
        s: Stride for the convolution operation.

    Returns:
        tuple[cp.ndarray, cp.ndarray, cp.ndarray]: Indices for rows, columns, and channels.
    """
    N, C, H_p, W_p = x_shape
    H_out = (H_p - kH) // s + 1
    W_out = (W_p - kW) // s + 1

    # Offsets within the kernel window
    i0 = cp.repeat(cp.arange(kH), kW)              # (kH*kW,)
    i0 = cp.tile(i0, C)                            # (C * kH * kW,)
    j0 = cp.tile(cp.arange(kW), kH)                # (kH*kW,)
    j0 = cp.tile(j0, C)                            # (C * kH * kW,)

    # Top-left corners of each sliding window (output positions)
    i1 = s * cp.repeat(cp.arange(H_out), W_out)    # (H_out * W_out,)
    j1 = s * cp.tile(cp.arange(W_out), H_out)      # (H_out * W_out,)

    # Combine to get full indices 
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)      # (C * kH * kW, H_out * W_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)      # (C * kH * kW, H_out * W_out)

    # Channel index for each row 
    d = cp.repeat(cp.arange(C), kH * kW).reshape(-1, 1)    # (C * kH * kW, 1)

    return i, j, d


def generate_random_ndarray(shape: tuple[int, ...]) -> cp.ndarray:
    """Generate a random CuPy ndarray with the given shape.

    Args:
        shape: Desired array shape.

    Returns:
        cp.ndarray: CuPy array of random floats.
    """
    return cp.random.randn(*shape, dtype=cp.float32)


def time_layer(layer, x, grad, *, lr: float) -> tuple[float, float]:
    """Measure forward and backward latency for a single pass.

    Args:
        layer: Layer object exposing ``forward`` and ``backward`` methods.
        x: Input tensor for the forward pass.
        grad: Upstream gradient tensor for the backward pass.
        lr: Learning rate forwarded to ``backward`` for layers that update parameters.

    Returns:
        tuple[float, float]: Forward time in seconds, backward time in seconds.
    """

    start = time.perf_counter()
    layer.forward(x)
    end = time.perf_counter()

    forward_s = end - start

    start = time.perf_counter()
    layer.backward(grad, lr)
    end = time.perf_counter()

    backward_s = end - start

    return forward_s, backward_s

def benchmark_layer(
    layer,
    *,
    step: int,
    max_batch: int,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    logger,
    lr: float = 0.01,
) -> None:
    """Benchmark forward/backward runtime as batch size increases.

    Args:
        layer: Layer object exposing ``forward`` and ``backward`` methods.
        step: Increment to grow the batch size between iterations.
        max_batch: Maximum batch size to benchmark (inclusive upper bound).
        input_shape: Shape of a single input sample (C, H, W).
        output_shape: Shape of a single output/gradient sample (C, H, W).
        logger: Callable that accepts a dict of metrics per batch size.
        lr: Learning rate forwarded to ``backward`` for layers that update parameters.
    """

    for batch in tqdm(range(1, max_batch+ 1, step), desc="Benchmarking"):
        x_shape = (batch, *input_shape)
        grad_shape = (batch, *output_shape)
        x = generate_random_ndarray(x_shape)
        grad = generate_random_ndarray(grad_shape)

        forward_t, backward_t = time_layer(layer, x, grad, lr=lr)
        log = {
            "batch": batch,
            "forward_time(s)": forward_t,
            "backward_time(s)": backward_t,
        }
        logger(log)
