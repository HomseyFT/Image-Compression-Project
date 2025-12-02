"""JPEG-like image compression for grayscale images using 8x8 DCT and quantization.

This module provides a minimal, self-contained implementation of the core
JPEG-style pipeline for black-and-white (grayscale) images:

- Load/save images via Pillow
- Convert to 8-bit grayscale
- Pad to a multiple of 8x8 blocks
- Level shift samples by 128
- Apply 2D DCT on 8x8 blocks (implemented with pure NumPy math)
- Quantize coefficients with a JPEG-like luminance quantization matrix
- Store quantized coefficients in a custom NPZ-based format
- Decompress by reversing the above steps

Usage (CLI):

    # Compress a grayscale image to a custom .npz format
    python compression.py compress input.png output.npz --quality 50

    # Decompress from .npz back to a PNG image
    python compression.py decompress input.npz output.png

You can also use the `compress_array` and `decompress_to_array` functions
programmatically for in-memory use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


BLOCK_SIZE = 8


# Standard JPEG luminance quantization matrix (ISO/IEC 10918-1 Annex K.1)
STANDARD_LUMA_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


@dataclass
class CompressedImage:
    """In-memory representation of a compressed grayscale image.

    Attributes
    ----------
    coeffs:
        Quantized DCT coefficients as a 4D array of shape
        (num_blocks_y, num_blocks_x, 8, 8), dtype=int16.
    orig_shape:
        Original image shape (height, width).
    padded_shape:
        Padded image shape used internally (height, width).
    quality:
        JPEG-style quality factor in [1, 100].
    """

    coeffs: np.ndarray
    orig_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    quality: int


# Cache for DCT bases keyed by block size
_DCT_CACHE: dict[int, np.ndarray] = {}


def _get_dct_matrix(n: int = BLOCK_SIZE) -> np.ndarray:
    """Return the orthonormal 1D DCT (type-II) basis matrix of size n×n.

    This implements the same transform class JPEG uses (up to overall scaling).
    The matrix is cached so it is only computed once per block size.
    """

    if n in _DCT_CACHE:
        return _DCT_CACHE[n]

    # Create DCT-II matrix with "ortho" normalization
    D = np.zeros((n, n), dtype=np.float64)
    factor = np.pi / (2.0 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)

    for u in range(n):
        alpha = scale0 if u == 0 else scale
        for x in range(n):
            D[u, x] = alpha * np.cos((2 * x + 1) * u * factor)

    _DCT_CACHE[n] = D
    return D


def _pad_to_block_size(
    img: np.ndarray, block_size: int = BLOCK_SIZE
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad a 2D image array so both dimensions are multiples of block_size.

    Padding is done by extending the edge values (mode="edge").
    Returns the padded image and its new shape.
    """

    if img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image array")

    h, w = img.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h == 0 and pad_w == 0:
        return img.copy(), (h, w)

    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, padded.shape


def _quality_to_scale(quality: int) -> float:
    """Map JPEG-like quality [1, 100] to a scaling factor for Q.

    This follows the common approximation used in many JPEG encoders.
    """

    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    return scale


def _build_quant_matrix(quality: int) -> np.ndarray:
    """Return an 8x8 quantization matrix scaled for the given quality.

    Values are clipped to [1, 255] as in JPEG.
    """

    scale = _quality_to_scale(quality)
    q = np.floor((STANDARD_LUMA_Q * scale + 50.0) / 100.0)
    q[q < 1] = 1
    q[q > 255] = 255
    return q.astype(np.float32)


def compress_array(image: np.ndarray, quality: int = 50) -> CompressedImage:
    """Compress a 2D grayscale image array using JPEG-like DCT + quantization.

    Parameters
    ----------
    image:
        2D array of shape (H, W), values in [0, 255]. Other dtypes will be
        converted to float32 internally.
    quality:
        JPEG-like quality factor in [1, 100]. Higher means better quality and
        less compression.
    """

    if image.ndim != 2:
        raise ValueError("compress_array expects a 2D grayscale image")

    orig_shape = image.shape
    img = image.astype(np.float32, copy=False)

    # Pad to full blocks and level shift
    padded, padded_shape = _pad_to_block_size(img, BLOCK_SIZE)
    padded = padded - 128.0

    h_p, w_p = padded_shape
    by = h_p // BLOCK_SIZE
    bx = w_p // BLOCK_SIZE

    D = _get_dct_matrix(BLOCK_SIZE)
    Q = _build_quant_matrix(quality)

    coeffs = np.empty((by, bx, BLOCK_SIZE, BLOCK_SIZE), dtype=np.int16)

    for j in range(by):
        for i in range(bx):
            block = padded[
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
            ]
            dct_block = D @ block @ D.T
            q_block = np.round(dct_block / Q).astype(np.int16)
            coeffs[j, i] = q_block

    return CompressedImage(
        coeffs=coeffs,
        orig_shape=orig_shape,
        padded_shape=padded_shape,
        quality=int(quality),
    )


def decompress_to_array(comp: CompressedImage) -> np.ndarray:
    """Decompress a :class:`CompressedImage` back to a uint8 grayscale array."""

    by, bx, _, _ = comp.coeffs.shape
    h_p, w_p = comp.padded_shape
    h, w = comp.orig_shape

    if h_p != by * BLOCK_SIZE or w_p != bx * BLOCK_SIZE:
        raise ValueError("Inconsistent block/padded shapes in compressed data")

    D = _get_dct_matrix(BLOCK_SIZE)
    Q = _build_quant_matrix(comp.quality)

    padded = np.empty((h_p, w_p), dtype=np.float32)

    for j in range(by):
        for i in range(bx):
            q_block = comp.coeffs[j, i].astype(np.float32)
            dct_block = q_block * Q
            block = D.T @ dct_block @ D
            block = block + 128.0
            padded[
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
            ] = block

    # Crop back to original shape and clip to valid range
    img = padded[:h, :w]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def compress_image_file(input_path: str, output_path: str, quality: int = 50) -> None:
    """Compress a grayscale image from disk and save a custom .npz file.

    The resulting NPZ contains:
        - coeffs: int16 quantized DCT coefficients
        - orig_shape: int32 array of length 2
        - padded_shape: int32 array of length 2
        - quality: scalar int32
    """

    with Image.open(input_path) as im:
        im = im.convert("L")  # ensure grayscale
        arr = np.array(im, dtype=np.uint8)

    comp = compress_array(arr, quality=quality)

    np.savez_compressed(
        output_path,
        coeffs=comp.coeffs,
        orig_shape=np.array(comp.orig_shape, dtype=np.int32),
        padded_shape=np.array(comp.padded_shape, dtype=np.int32),
        quality=np.int32(comp.quality),
    )


def decompress_image_file(input_path: str, output_path: str) -> None:
    """Decompress from a custom .npz file and save a grayscale PNG image."""

    data = np.load(input_path)

    coeffs = data["coeffs"].astype(np.int16)
    orig_shape = tuple(int(x) for x in data["orig_shape"])
    padded_shape = tuple(int(x) for x in data["padded_shape"])
    quality = int(data["quality"])

    comp = CompressedImage(
        coeffs=coeffs,
        orig_shape=orig_shape,  # type: ignore[arg-type]
        padded_shape=padded_shape,  # type: ignore[arg-type]
        quality=quality,
    )

    img_arr = decompress_to_array(comp)
    im = Image.fromarray(img_arr, mode="L")
    im.save(output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JPEG-like DCT-based compressor for grayscale images.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # compress
    p_compress = subparsers.add_parser("compress", help="Compress an image to .npz")
    p_compress.add_argument("input", help="Input image path (any format Pillow supports)")
    p_compress.add_argument("output", help="Output .npz file path")
    p_compress.add_argument(
        "--quality",
        type=int,
        default=50,
        help="JPEG-like quality factor [1-100] (default: 50)",
    )

    # decompress
    p_decompress = subparsers.add_parser(
        "decompress", help="Decompress from .npz to a grayscale PNG"
    )
    p_decompress.add_argument("input", help="Input .npz file path")
    p_decompress.add_argument("output", help="Output image path (e.g., .png)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "compress":
        compress_image_file(args.input, args.output, quality=args.quality)
    elif args.command == "decompress":
        decompress_image_file(args.input, args.output)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
