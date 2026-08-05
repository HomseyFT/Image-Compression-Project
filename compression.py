"""JPEG-like image compression for grayscale images using 8x8 DCT and quantization.

This module provides a minimal, self-contained implementation of the core
JPEG-style pipeline for black-and-white (grayscale) images:

- Load/save images via Pillow
- Convert to 8-bit grayscale
- Pad to a multiple of 8x8 blocks
- Level shift samples by 128
- Apply 2D DCT on 8x8 blocks (implemented with pure NumPy math)
- Quantize coefficients with a JPEG-like luminance quantization matrix
- Entropy code with JPEG-style DC/AC symbols plus per-file Huffman tables
- Pack into a custom ICJ2 container
- Decompress by reversing the above steps

Usage (CLI):

    # Compress a grayscale image to the custom ICJ2 container
    python compression.py compress input.png output.icj --quality 50

    # Decompress from ICJ2 back to a PNG image
    python compression.py decompress input.icj output.png

You can also use the `compress_array` and `decompress_to_array` functions
programmatically for in-memory use.
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


# Container magic. ICJ2 supersedes the original ICJ1 layout; no backwards
# compatibility is provided, as ICJ1 files could encode an unclamped quality
# byte that decoded against the wrong quantization matrix.
MAGIC = b"ICJ2"

# Maximum Huffman code length the container can represent. The decoder scans
# lengths 1..MAX_CODE_LENGTH.
MAX_CODE_LENGTH = 32

# Trellis (rate-distortion optimized quantization) tuning. The multiplier is
# expressed relative to the mean squared quantizer step so that it tracks
# --quality automatically.
#
# Fitted by sweeping the multiplier and measuring rate at EQUAL PSNR against
# the baseline quality-swept curve -- never "bytes saved", which any lossy
# knob achieves by simply moving down the curve.
#
# 0.030 maximises both the mean and the worst-case gain across photographic
# crops (detail 19.1%, background 9.0%, whole image 10.5%, noise 3.9%).
#
# An earlier fit appeared to show smooth gradients regressing past ~0.020 and
# the constant was set low to avoid it. That was a measurement artifact: such
# images produce degenerate rate-distortion curves with flat spots (quality
# 50 -> 55 gains +0.00 dB while the rate grows), so interpolating rate at a
# given PSNR is unstable. The tell was that the "regression" persisted as
# lambda -> 0, where the DP provably reproduces the baseline bit-for-bit.
# Only content with a monotonically rising curve can score this.
TRELLIS_LAMBDA_SCALE = 0.030
TRELLIS_ITERATIONS = 2


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


# Zigzag scan order (raster index = row * 8 + col)
ZIGZAG_ORDER = np.array(
    [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
        32,
        25,
        18,
        11,
        4,
        5,
        12,
        19,
        26,
        33,
        40,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        23,
        30,
        37,
        44,
        51,
        58,
        59,
        52,
        45,
        38,
        31,
        39,
        46,
        53,
        60,
        61,
        54,
        47,
        55,
        62,
        63,
    ],
    dtype=np.int32,
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


# Working precision for the transform. float32 is ~2.7x faster than float64
# here and perturbs only ~0.007% of quantized coefficients by +/-1 (rounding
# ties), which is far below the quantization step and invisible in the
# reconstruction. See SPEC.md phase 3.
DCT_DTYPE = np.float32

# Cache for DCT bases keyed by block size, holding (D, D.T).
_DCT_CACHE: dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _get_dct_matrices(n: int = BLOCK_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(D, D.T)``, the orthonormal 1D DCT-II basis and its transpose.

    This implements the same transform class JPEG uses (up to overall
    scaling). Both matrices are cached as contiguous arrays so that every
    forward/inverse call is a pair of batched GEMMs with no per-call
    transposition or dtype promotion.
    """

    cached = _DCT_CACHE.get(n)
    if cached is not None:
        return cached

    # DCT-II with "ortho" normalization, built as an outer product.
    u = np.arange(n).reshape(n, 1)
    x = np.arange(n).reshape(1, n)
    D = np.cos((2 * x + 1) * u * np.pi / (2.0 * n))
    D *= np.sqrt(2.0 / n)
    D[0, :] = np.sqrt(1.0 / n)

    D = np.ascontiguousarray(D, dtype=DCT_DTYPE)
    DT = np.ascontiguousarray(D.T)

    _DCT_CACHE[n] = (D, DT)
    return D, DT


def _to_blocks(img: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """View a padded (H, W) image as (blocks_y, blocks_x, block, block)."""

    h, w = img.shape
    return img.reshape(
        h // block_size, block_size, w // block_size, block_size
    ).transpose(0, 2, 1, 3)


def _from_blocks(blocks: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Inverse of :func:`_to_blocks`: reassemble blocks into a 2D image."""

    by, bx = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(by * block_size, bx * block_size)


def _forward_dct_2d(blocks: np.ndarray) -> np.ndarray:
    """Apply the separable 2D DCT to an array of shape ``(..., 8, 8)``.

    ``matmul`` broadcasts over all leading dimensions, so an entire image is
    transformed in a single batched call.
    """

    D, DT = _get_dct_matrices(blocks.shape[-1])
    return D @ blocks @ DT


def _inverse_dct_2d(blocks: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_forward_dct_2d`."""

    D, DT = _get_dct_matrices(blocks.shape[-1])
    return DT @ blocks @ D


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


def _clamp_quality(quality: int) -> int:
    """Validate and clamp a quality factor into [1, 100].

    This is the single point at which quality is normalised. Every public
    entry point must funnel through it *before* the value is stored, so that
    the quality recorded in a :class:`CompressedImage` (and written to the
    container header) is always the same value the encoder actually used.

    Clamping only inside :func:`_quality_to_scale` is not sufficient: the
    unclamped value would still be persisted, and a value outside [0, 255]
    would alias when packed into the single-byte header field. A quality of
    -5, for example, encodes with the q=1 matrix but stores byte 251, which
    the decoder then clamps to 100 -- reconstructing against a completely
    different matrix and silently producing garbage.
    """

    if isinstance(quality, bool) or not isinstance(quality, (int, np.integer)):
        raise TypeError(f"quality must be an integer, got {type(quality).__name__}")

    return max(1, min(100, int(quality)))


def _quality_to_scale(quality: int) -> float:
    """Map JPEG-like quality [1, 100] to a scaling factor for Q.

    This follows the common approximation used in many JPEG encoders.
    Callers are expected to have passed ``quality`` through
    :func:`_clamp_quality` already; the clamp here is defensive only.
    """

    quality = _clamp_quality(quality)

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


def _trellis_quantize(
    dct: np.ndarray, Q: np.ndarray, ac_bits: np.ndarray, lam: float
) -> np.ndarray:
    """Rate-distortion optimized quantization (trellis) over the zigzag scan.

    For every block this runs a Lagrangian dynamic program that chooses, per
    coefficient, between zeroing it, keeping the nearest level, or dropping
    one magnitude step -- minimising ``D + lambda * R`` where ``R`` is the
    *actual* Huffman cost including run-length structure, and the end-of-block
    position is itself a free decision.

    A blind deadzone cannot do this: the win comes from knowing that zeroing
    a coefficient may collapse a run symbol or move the EOB earlier, paying
    for itself in bits that a fixed bias cannot see.

    Because the DCT is orthonormal, squared error in the coefficient domain
    equals squared error in pixels, so distortion is scored locally without
    an inverse transform per candidate.

    The DP state is the pending zero-run (0..15). Transitions into a state
    ``s > 0`` can only come from ``s - 1`` via a zero, so only the state-0
    decisions need recording for backtracking.

    Returns quantized levels of shape ``(by, bx, 8, 8)``, int16. This is an
    ordinary coefficient array: the decoder and container are unchanged.
    """

    by, bx = dct.shape[:2]
    n_blocks = by * bx

    q_zz = Q.reshape(-1)[ZIGZAG_ORDER].astype(np.float64)
    t = (dct.reshape(n_blocks, 64) / Q.reshape(-1))[:, ZIGZAG_ORDER].astype(np.float64)

    base = np.round(t).astype(np.int64)
    cands = np.stack([base, base - np.sign(base)])          # (2, B, 64)

    d_zero = (q_zz * t) ** 2
    suffix = np.zeros((n_blocks, 65))
    suffix[:, 1:64] = np.cumsum(d_zero[:, ::-1], axis=1)[:, ::-1][:, 1:64]

    eob = float(ac_bits[0x00])
    zrl = float(ac_bits[0xF0])
    runs = np.arange(16)
    bi = np.arange(n_blocks)

    dpJ = np.full((n_blocks, 16), np.inf)
    dpJ[:, 0] = 0.0

    act0 = np.zeros((64, n_blocks), dtype=np.int8)     # -1 ZRL, else candidate
    prev0 = np.zeros((64, n_blocks), dtype=np.int8)

    # Baseline option: every AC coefficient zero, so the block is a bare EOB.
    best = suffix[:, 1] + lam * eob
    end_k = np.zeros(n_blocks, dtype=np.int16)
    end_prev = np.zeros(n_blocks, dtype=np.int8)
    end_cand = np.zeros(n_blocks, dtype=np.int8)

    for k in range(1, 64):
        zJ = dpJ + d_zero[:, k][:, None]

        newJ = np.full((n_blocks, 16), np.inf)
        newJ[:, 1:] = zJ[:, :15]

        # Run of 16 zeros forces a ZRL and resets the run.
        best0 = zJ[:, 15] + lam * zrl
        best0_act = np.full(n_blocks, -1, dtype=np.int8)
        best0_prev = np.full(n_blocks, 15, dtype=np.int8)

        for ci in range(2):
            Lk = cands[ci, :, k]
            nz = Lk != 0
            if not nz.any():
                continue

            size = _value_category_vec(Lk)
            rbits = ac_bits[(runs[None, :] << 4) | np.clip(size, 0, 15)[:, None]]
            rbits = rbits + size[:, None]

            dL = (q_zz[k] * (t[:, k] - Lk)) ** 2
            J = np.where(nz[:, None], dpJ + dL[:, None] + lam * rbits, np.inf)

            r = np.argmin(J, axis=1)
            Jm = J[bi, r]

            better = Jm < best0
            best0 = np.where(better, Jm, best0)
            best0_act = np.where(better, ci, best0_act).astype(np.int8)
            best0_prev = np.where(better, r, best0_prev).astype(np.int8)

            # Option: make this the last coded coefficient of the block.
            endJ = Jm + suffix[:, k + 1] + lam * (eob if k < 63 else 0.0)
            be = endJ < best
            best = np.where(be, endJ, best)
            end_k = np.where(be, k, end_k).astype(np.int16)
            end_prev = np.where(be, r, end_prev).astype(np.int8)
            end_cand = np.where(be, ci, end_cand).astype(np.int8)

        newJ[:, 0] = best0
        act0[k] = best0_act
        prev0[k] = best0_prev
        dpJ = newJ

    # --- Backtrack, vectorized across blocks ---
    levels = np.zeros((n_blocks, 64), dtype=np.int64)
    levels[:, 0] = base[:, 0]                 # DC is left to the plain quantizer
    cur_s = np.zeros(n_blocks, dtype=np.int64)

    for k in range(63, 0, -1):
        at_end = end_k == k
        if at_end.any():
            levels[at_end, k] = cands[end_cand[at_end], bi[at_end], k]
            cur_s[at_end] = end_prev[at_end]

        inside = k < end_k
        if not inside.any():
            continue

        s = cur_s.copy()
        pending = inside & (s > 0)
        cur_s[pending] = s[pending] - 1

        at0 = inside & (s == 0)
        if at0.any():
            a = act0[k]
            is_zrl = at0 & (a == -1)
            cur_s[is_zrl] = 15
            is_cand = at0 & (a >= 0)
            if is_cand.any():
                levels[is_cand, k] = cands[a[is_cand], bi[is_cand], k]
                cur_s[is_cand] = prev0[k][is_cand]

    out = np.zeros((n_blocks, 64), dtype=np.int64)
    out[:, ZIGZAG_ORDER] = levels
    return out.reshape(by, bx, BLOCK_SIZE, BLOCK_SIZE).astype(np.int16)


def _trellis_lambda(Q: np.ndarray) -> float:
    """Lagrange multiplier for a given quantization matrix.

    Rate-distortion theory puts the optimum at ``lambda = -dD/dR``; for a
    uniform quantizer of step ``D`` the high-rate approximation is
    proportional to the step squared. Tying lambda to the quantizer this way
    keeps ``--quality`` meaning what it always did -- each quality level
    simply becomes cheaper in bits -- rather than introducing a second knob
    that silently slides along the rate-distortion curve.

    The constant is fitted empirically; see SPEC.md phase 4.2.
    """

    return TRELLIS_LAMBDA_SCALE * float(np.mean(Q.astype(np.float64) ** 2))


def _load_grayscale(path: str) -> np.ndarray:
    """Load any Pillow-readable image as an 8-bit grayscale array.

    Palette images carrying transparency are routed through RGBA first.
    Converting those straight to "L" makes Pillow warn, because the
    transparency index would be composited against an undefined background;
    alpha is irrelevant to this codec, so it is dropped explicitly.
    """

    with Image.open(path) as im:
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
        return np.array(im.convert("L"), dtype=np.uint8)


def _ac_bit_costs(ac_table: dict[int, tuple[int, int]]) -> np.ndarray:
    """Dense AC symbol -> code length lookup for the trellis rate model.

    Symbols absent from the table are unreachable in the current stream;
    they are priced prohibitively so the DP never selects one.
    """

    bits = np.full(256, 1e6)
    for sym, (_code, n) in ac_table.items():
        bits[sym] = n
    return bits


def compress_array(
    image: np.ndarray, quality: int = 50, trellis: bool = True
) -> CompressedImage:
    """Compress a 2D grayscale image array using JPEG-like DCT + quantization.

    Parameters
    ----------
    image:
        2D array of shape (H, W), values in [0, 255]. Other dtypes will be
        converted to float32 internally.
    quality:
        JPEG-like quality factor in [1, 100]. Higher means better quality and
        less compression.
    trellis:
        Enable rate-distortion optimized quantization. Encoder-side only --
        the output is an ordinary coefficient array, so the decoder and the
        container format are unaffected.
    """

    if image.ndim != 2:
        raise ValueError("compress_array expects a 2D grayscale image")

    quality = _clamp_quality(quality)

    orig_shape = image.shape
    img = image.astype(DCT_DTYPE, copy=False)

    # Pad to full blocks and level shift
    padded, padded_shape = _pad_to_block_size(img, BLOCK_SIZE)
    padded = padded - DCT_DTYPE(128.0)

    Q = _build_quant_matrix(quality).astype(DCT_DTYPE)

    blocks = _to_blocks(padded, BLOCK_SIZE)
    dct = _forward_dct_2d(blocks)
    coeffs = np.round(dct / Q).astype(np.int16)

    if trellis:
        lam = _trellis_lambda(Q)
        # The rate model needs Huffman costs, which come from the symbol
        # distribution -- which trellis then changes. Re-deriving the tables
        # and re-running closes that loop; iterating is bounded and keeps the
        # best result seen, since convergence is not guaranteed.
        for _ in range(TRELLIS_ITERATIONS):
            _, _, ac_table = _encode_blocks_huffman(coeffs)
            ac_bits = _ac_bit_costs(ac_table)
            candidate = _trellis_quantize(dct.astype(np.float64), Q, ac_bits, lam)
            if np.array_equal(candidate, coeffs):
                break
            coeffs = candidate

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

    Q = _build_quant_matrix(comp.quality).astype(DCT_DTYPE)

    blocks = _inverse_dct_2d(comp.coeffs.astype(DCT_DTYPE) * Q)
    padded = _from_blocks(blocks, BLOCK_SIZE) + DCT_DTYPE(128.0)

    # Crop back to original shape and clip to valid range
    img = padded[:h, :w]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# === Huffman-based entropy coding (custom container) ========================


class _BitWriter:
    """Simple big-endian bit writer.

    Bits are packed most-significant-first and the final partial byte is
    zero-padded by :meth:`flush`.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._acc = 0
        self._n_bits = 0

    @property
    def bytes(self) -> bytes:
        return bytes(self._buffer)

    def _write_byte(self, b: int) -> None:
        self._buffer.append(b & 0xFF)

    def write_bits(self, value: int, n_bits: int) -> None:
        if n_bits == 0:
            return
        for i in range(n_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self._acc = (self._acc << 1) | bit
            self._n_bits += 1
            if self._n_bits == 8:
                self._write_byte(self._acc)
                self._acc = 0
                self._n_bits = 0

    def flush(self) -> None:
        if self._n_bits:
            self._acc <<= 8 - self._n_bits
            self._write_byte(self._acc)
            self._acc = 0
            self._n_bits = 0


class _BitReader:
    """Bit reader matching :class:`_BitWriter` format."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._bit_buf = 0
        self._bits_left = 0

    def _read_byte(self) -> int:
        if self._pos >= len(self._data):
            raise EOFError("Unexpected end of bitstream")
        b = self._data[self._pos]
        self._pos += 1
        return b

    def read_bit(self) -> int:
        if self._bits_left == 0:
            self._bit_buf = self._read_byte()
            self._bits_left = 8
        self._bits_left -= 1
        return (self._bit_buf >> self._bits_left) & 1

    def read_bits(self, n_bits: int) -> int:
        v = 0
        for _ in range(n_bits):
            v = (v << 1) | self.read_bit()
        return v


def _value_category(v: int) -> int:
    """Return JPEG-like magnitude category for an integer value."""

    if v == 0:
        return 0
    v = abs(v)
    cat = 0
    while v:
        v >>= 1
        cat += 1
    return cat


def _value_to_bits(v: int, cat: int) -> int:
    """Map signed value to JPEG-style additional bits for given category."""

    if cat == 0:
        return 0
    if v >= 0:
        return v
    # For negative values, invert bits within the category width.
    return (1 << cat) - 1 + v  # v is negative


def _bits_to_value(bits: int, cat: int) -> int:
    """Inverse of :func:`_value_to_bits`. Reconstruct signed value."""

    if cat == 0:
        return 0
    threshold = 1 << (cat - 1)
    if bits >= threshold:
        return bits
    return bits - ((1 << cat) - 1)


def _build_huffman_table(freqs: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Build canonical Huffman codes from symbol -> frequency.

    Returns a mapping ``symbol -> (code, n_bits)``.
    Symbols with zero frequency are omitted.
    """

    # Filter out unused symbols
    items = [(sym, f) for sym, f in freqs.items() if f > 0]
    if not items:
        raise ValueError("Cannot build Huffman table with no symbols")

    if len(items) == 1:
        # Degenerate table: single symbol gets code '0'.
        sym = items[0][0]
        return {sym: (0, 1)}

    # Build Huffman tree using a simple priority queue.
    counter = 0
    heap: list[tuple[int, int, object]] = []
    for sym, f in items:
        heapq.heappush(heap, (f, counter, (sym, None, None)))
        counter += 1

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = (None, n1, n2)
        heapq.heappush(heap, (f1 + f2, counter, merged))
        counter += 1

    [(_, _, root)] = heap

    lengths: dict[int, int] = {}

    def walk(node: object, depth: int) -> None:
        sym, left, right = node  # type: ignore[misc]
        if sym is not None:
            lengths[sym] = max(depth, 1)
        else:
            walk(left, depth + 1)  # type: ignore[arg-type]
            walk(right, depth + 1)  # type: ignore[arg-type]

    walk(root, 0)

    # The decoder only scans lengths 1..MAX_CODE_LENGTH, so anything longer
    # would produce an undecodable stream. Natural images stay far below this
    # bound, but fail loudly rather than emit a corrupt file.
    max_len = max(lengths.values())
    if max_len > MAX_CODE_LENGTH:
        raise ValueError(
            f"Huffman code length {max_len} exceeds the {MAX_CODE_LENGTH}-bit "
            "limit supported by the container format"
        )

    return _assign_canonical_codes(lengths)


def _assign_canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Assign canonical Huffman codes from symbol -> code length.

    Sorting by ``(length, symbol)`` makes the assignment a pure function of
    the lengths, which is why the container stores lengths only and rebuilds
    the codes on load.
    """

    table: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= length - prev_len
        table[sym] = (code, length)
        code += 1
        prev_len = length

    return table


def _build_decode_table(huff_table: dict[int, tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Build (length, code) -> symbol map for decoding."""

    decode: dict[tuple[int, int], int] = {}
    for sym, (code, n_bits) in huff_table.items():
        decode[(n_bits, code)] = sym
    return decode


def _value_category_vec(v: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_value_category` over an integer array."""

    a = np.abs(v).astype(np.int64)
    return np.where(a == 0, 0, np.floor(np.log2(np.maximum(a, 1))).astype(np.int64) + 1)


def _value_to_bits_vec(v: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_value_to_bits`."""

    return np.where(v >= 0, v, (np.int64(1) << cat) - 1 + v)


@dataclass
class _SymbolStream:
    """JPEG-style symbol stream for a whole image, in emission order.

    Splitting extraction from Huffman coding lets the same scan feed both the
    frequency pass and the bit-emission pass, and gives the rate model a
    single source of truth about how many symbols a coefficient array costs.
    """

    dc_cats: np.ndarray      # (B,) DC magnitude category per block
    dc_diffs: np.ndarray     # (B,) DC differential per block
    ac_block: np.ndarray     # (M,) block index, sorted in emission order
    ac_symbols: np.ndarray   # (M,) run/size byte (0x00 EOB, 0xF0 ZRL)
    ac_values: np.ndarray    # (M,) coefficient value (0 for EOB/ZRL)
    ac_sizes: np.ndarray     # (M,) mantissa bit count
    n_blocks: int


def _scan_symbols(coeffs: np.ndarray) -> _SymbolStream:
    """Extract the DC/AC symbol stream for every block, fully vectorized.

    Reproduces the sequential JPEG scan exactly: DC differentials in raster
    block order, then per block the AC run/size symbols in zigzag order with
    ZRL every 16 zeros and a trailing EOB unless the block runs to position
    63.
    """

    n_blocks = coeffs.shape[0] * coeffs.shape[1]
    zz = coeffs.reshape(n_blocks, 64)[:, ZIGZAG_ORDER].astype(np.int64)

    # --- DC: differential across blocks in raster order ---
    dc = zz[:, 0]
    dc_diffs = dc - np.concatenate(([0], dc[:-1]))
    dc_cats = _value_category_vec(dc_diffs)

    # --- AC: run-length over the 63 remaining zigzag positions ---
    ac = zz[:, 1:]
    blk, pos = np.nonzero(ac)          # C order => block-major, then position
    vals = ac[blk, pos]
    sizes = _value_category_vec(vals)

    # Zeros since the previous non-zero in the same block.
    same_block = np.empty(len(blk), dtype=bool)
    if len(blk):
        same_block[0] = False
        same_block[1:] = blk[1:] == blk[:-1]
    prev_pos = np.where(same_block, np.concatenate(([0], pos[:-1])), -1)
    runs = pos - prev_pos - 1

    # A run of 16+ zeros needs that many ZRL symbols ahead of the coefficient.
    n_zrl = runs // 16
    rem = runs % 16

    rep = n_zrl + 1
    src = np.repeat(np.arange(len(blk)), rep)
    group_start = np.cumsum(rep) - rep
    sub = np.arange(len(src)) - group_start[src]
    is_zrl = sub < n_zrl[src]

    e_block = blk[src]
    e_pos = pos[src]
    e_sub = sub
    e_sym = np.where(is_zrl, 0xF0, (rem[src] << 4) | sizes[src])
    e_val = np.where(is_zrl, 0, vals[src])
    e_size = np.where(is_zrl, 0, sizes[src])

    # --- EOB for every block that does not run to the final position ---
    last_pos = np.full(n_blocks, -1, dtype=np.int64)
    if len(blk):
        last_pos[blk] = pos       # ascending order => final write is the max
    eob_blocks = np.nonzero(last_pos < 62)[0]

    n_eob = len(eob_blocks)
    e_block = np.concatenate([e_block, eob_blocks])
    e_pos = np.concatenate([e_pos, np.full(n_eob, 63, dtype=np.int64)])
    e_sub = np.concatenate([e_sub, np.zeros(n_eob, dtype=np.int64)])
    e_sym = np.concatenate([e_sym, np.zeros(n_eob, dtype=np.int64)])
    e_val = np.concatenate([e_val, np.zeros(n_eob, dtype=np.int64)])
    e_size = np.concatenate([e_size, np.zeros(n_eob, dtype=np.int64)])

    # Emission order: block, then zigzag position, then ZRLs before the value.
    order = np.lexsort((e_sub, e_pos, e_block))

    return _SymbolStream(
        dc_cats=dc_cats,
        dc_diffs=dc_diffs,
        ac_block=e_block[order],
        ac_symbols=e_sym[order],
        ac_values=e_val[order],
        ac_sizes=e_size[order],
        n_blocks=n_blocks,
    )


def _pack_bits(codes: np.ndarray, lengths: np.ndarray) -> bytes:
    """Pack (code, length) pairs MSB-first into a byte string.

    Equivalent to feeding every pair through :class:`_BitWriter`, including
    the zero padding of the final partial byte.
    """

    total = int(lengths.sum())
    if total == 0:
        return b""

    offsets = np.cumsum(lengths) - lengths
    sym = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    shift = lengths[sym] - 1 - (np.arange(total, dtype=np.int64) - offsets[sym])
    bits = ((codes[sym] >> shift) & 1).astype(np.uint8)
    return np.packbits(bits).tobytes()


def _encode_blocks_huffman(coeffs: np.ndarray) -> tuple[bytes, dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Run JPEG-like DC/AC coding + Huffman on quantized blocks.

    Parameters
    ----------
    coeffs:
        Quantized DCT coefficients, shape (blocks_y, blocks_x, 8, 8), int16.

    Returns
    -------
    bitstream:
        Huffman-coded bitstream.
    dc_table, ac_table:
        Huffman tables used for DC categories (0-11) and AC symbols (0-255).
    """

    stream = _scan_symbols(coeffs)

    # First pass: symbol frequencies -> optimal per-image Huffman tables.
    dc_counts = np.bincount(stream.dc_cats, minlength=12)
    ac_counts = np.bincount(stream.ac_symbols, minlength=256)
    dc_table = _build_huffman_table({i: int(n) for i, n in enumerate(dc_counts)})
    ac_table = _build_huffman_table({i: int(n) for i, n in enumerate(ac_counts)})

    # Second pass: emit. Per block the layout is
    #   [DC code][DC mantissa]  then  [AC symbol][AC mantissa] * n_ac
    dc_code, dc_len = _table_to_arrays(dc_table, 12)
    ac_code, ac_len = _table_to_arrays(ac_table, 256)

    n_ac = np.bincount(stream.ac_block, minlength=stream.n_blocks)
    per_block = 2 + 2 * n_ac
    block_start = np.concatenate(([0], np.cumsum(per_block)[:-1]))

    codes = np.zeros(int(per_block.sum()), dtype=np.int64)
    lengths = np.zeros_like(codes)

    codes[block_start] = dc_code[stream.dc_cats]
    lengths[block_start] = dc_len[stream.dc_cats]
    codes[block_start + 1] = _value_to_bits_vec(stream.dc_diffs, stream.dc_cats)
    lengths[block_start + 1] = stream.dc_cats

    ac_start = np.concatenate(([0], np.cumsum(n_ac)[:-1]))
    within = np.arange(len(stream.ac_block)) - ac_start[stream.ac_block]
    idx = block_start[stream.ac_block] + 2 + 2 * within

    codes[idx] = ac_code[stream.ac_symbols]
    lengths[idx] = ac_len[stream.ac_symbols]
    codes[idx + 1] = _value_to_bits_vec(stream.ac_values, stream.ac_sizes)
    lengths[idx + 1] = stream.ac_sizes

    return _pack_bits(codes, lengths), dc_table, ac_table


def _table_to_arrays(table: dict[int, tuple[int, int]], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense (code, length) lookup arrays for a Huffman table."""

    code = np.zeros(size, dtype=np.int64)
    length = np.zeros(size, dtype=np.int64)
    for sym, (cd, n) in table.items():
        code[sym] = cd
        length[sym] = n
    return code, length


def _decode_blocks_huffman(
    by: int,
    bx: int,
    bitstream: bytes,
    dc_table: dict[int, tuple[int, int]],
    ac_table: dict[int, tuple[int, int]],
) -> np.ndarray:
    """Inverse of :func:`_encode_blocks_huffman`.

    Returns an array of shape (by, bx, 8, 8) with quantized coefficients.
    """

    reader = _BitReader(bitstream)
    dc_decode = _build_decode_table(dc_table)
    ac_decode = _build_decode_table(ac_table)

    coeffs = np.zeros((by, bx, BLOCK_SIZE, BLOCK_SIZE), dtype=np.int16)
    prev_dc = 0

    for j in range(by):
        for i in range(bx):
            zz = np.zeros(64, dtype=np.int16)

            # Decode DC
            code = 0
            for length in range(1, 33):
                code = (code << 1) | reader.read_bit()
                key = (length, code)
                if key in dc_decode:
                    cat = dc_decode[key]
                    break
            else:
                raise ValueError("Failed to decode DC coefficient")

            if cat:
                bits = reader.read_bits(cat)
                diff = _bits_to_value(bits, cat)
            else:
                diff = 0
            dc_val = prev_dc + diff
            prev_dc = dc_val
            zz[0] = dc_val

            # Decode AC
            k = 1
            while k < 64:
                code = 0
                symbol = None
                for length in range(1, 33):
                    code = (code << 1) | reader.read_bit()
                    key = (length, code)
                    if key in ac_decode:
                        symbol = ac_decode[key]
                        break
                if symbol is None:
                    raise ValueError("Failed to decode AC coefficient")

                if symbol == 0x00:  # EOB
                    break
                if symbol == 0xF0:  # ZRL
                    k += 16
                    if k > 64:
                        raise ValueError("ZRL went past end of block")
                    continue

                run = symbol >> 4
                size = symbol & 0x0F
                k += run
                if k >= 64:
                    raise ValueError("Run-length went past end of block")
                bits = reader.read_bits(size)
                zz[k] = _bits_to_value(bits, size)
                k += 1

            # Map zigzag back to 8x8 block
            flat = np.zeros(64, dtype=np.int16)
            flat[ZIGZAG_ORDER] = zz
            coeffs[j, i] = flat.reshape(BLOCK_SIZE, BLOCK_SIZE)

    return coeffs


def _read_exactly(f, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise, so truncated files fail loudly."""

    data = f.read(n)
    if len(data) != n:
        raise ValueError(
            f"Truncated container: expected {n} bytes, got {len(data)}"
        )
    return data


def _serialize_huffman_table(table: dict[int, tuple[int, int]]) -> bytes:
    """Serialize a canonical Huffman table as ``(symbol, length)`` pairs.

    The codes themselves are *not* stored: canonical assignment is a pure
    function of the lengths (see :func:`_assign_canonical_codes`), so 4 bytes
    of code per symbol were redundant. This costs 2 bytes per symbol instead
    of 6.
    """

    buf = bytearray()
    buf.extend(len(table).to_bytes(2, "big"))  # count of used symbols
    for sym, (_code, length) in sorted(table.items()):
        buf.append(sym & 0xFF)
        buf.append(length & 0xFF)
    return bytes(buf)


def _deserialize_huffman_table(data: bytes) -> dict[int, tuple[int, int]]:
    if len(data) < 2:
        raise ValueError("Truncated Huffman table header")

    count = int.from_bytes(data[0:2], "big")

    expected = 2 + count * 2
    if len(data) < expected:
        raise ValueError(
            f"Truncated Huffman table: need {expected} bytes for {count} "
            f"symbols, got {len(data)}"
        )

    lengths: dict[int, int] = {}
    for i in range(count):
        sym = data[2 + 2 * i]
        length = data[3 + 2 * i]
        if not 1 <= length <= MAX_CODE_LENGTH:
            raise ValueError(f"Invalid Huffman code length {length} for symbol {sym}")
        lengths[sym] = length

    if not lengths:
        raise ValueError("Huffman table contains no symbols")

    return _assign_canonical_codes(lengths)


def compress_huffman_file(input_path: str, output_path: str, quality: int = 50) -> None:
    """Compress an image using DCT + quantization + Huffman into a custom binary.

    The ICJ2 container format is::

        magic:      4 bytes   ASCII 'ICJ2'
        height:     4 bytes   unsigned big-endian
        width:      4 bytes   unsigned big-endian
        quality:    1 byte    1-100 (always clamped before writing)
        blocks_y:   2 bytes   unsigned big-endian
        blocks_x:   2 bytes   unsigned big-endian
        dc_len:     2 bytes   byte length of the DC table that follows
        dc_table:   dc_len bytes
        ac_len:     2 bytes   byte length of the AC table that follows
        ac_table:   ac_len bytes
        bit_len:    4 bytes   length of following bitstream in bytes
        bitstream:  bit_len bytes, big-endian bit packing, zero-padded

    Each Huffman table is serialised as a 2-byte count of used symbols
    followed by that many 2-byte ``(symbol, code length)`` pairs. Canonical
    codes are rebuilt from the lengths on load.
    """

    arr = _load_grayscale(input_path)
    comp = compress_array(arr, quality=quality)
    coeffs = comp.coeffs
    bitstream, dc_table, ac_table = _encode_blocks_huffman(coeffs)

    h, w = comp.orig_shape
    h_p, w_p = comp.padded_shape
    by = h_p // BLOCK_SIZE
    bx = w_p // BLOCK_SIZE

    header = bytearray()
    header.extend(MAGIC)
    header.extend(int(h).to_bytes(4, "big"))
    header.extend(int(w).to_bytes(4, "big"))
    header.append(int(comp.quality) & 0xFF)
    header.extend(int(by).to_bytes(2, "big"))
    header.extend(int(bx).to_bytes(2, "big"))

    # Write tables with their byte lengths prefixed
    dc_bytes = _serialize_huffman_table(dc_table)
    ac_bytes = _serialize_huffman_table(ac_table)
    header.extend(len(dc_bytes).to_bytes(2, "big"))
    header.extend(dc_bytes)
    header.extend(len(ac_bytes).to_bytes(2, "big"))
    header.extend(ac_bytes)

    header.extend(len(bitstream).to_bytes(4, "big"))

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(bitstream)


def decompress_huffman_file(input_path: str, output_path: str) -> None:
    """Inverse of :func:`compress_huffman_file`.

    Reads the custom container, decodes Huffman-coded coefficients, performs
    inverse DCT + dequantization, and writes a grayscale image.
    """

    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(
                f"Not an {MAGIC.decode()} file (bad magic: {magic!r})"
            )
        h = int.from_bytes(_read_exactly(f, 4), "big")
        w = int.from_bytes(_read_exactly(f, 4), "big")
        quality = _read_exactly(f, 1)[0]
        by = int.from_bytes(_read_exactly(f, 2), "big")
        bx = int.from_bytes(_read_exactly(f, 2), "big")

        dc_len = int.from_bytes(_read_exactly(f, 2), "big")
        dc_bytes = _read_exactly(f, dc_len)
        ac_len = int.from_bytes(_read_exactly(f, 2), "big")
        ac_bytes = _read_exactly(f, ac_len)
        dc_table = _deserialize_huffman_table(dc_bytes)
        ac_table = _deserialize_huffman_table(ac_bytes)

        bit_len = int.from_bytes(_read_exactly(f, 4), "big")
        bitstream = _read_exactly(f, bit_len)

    coeffs = _decode_blocks_huffman(by, bx, bitstream, dc_table, ac_table)

    comp = CompressedImage(
        coeffs=coeffs,
        orig_shape=(h, w),
        padded_shape=(by * BLOCK_SIZE, bx * BLOCK_SIZE),
        quality=int(quality),
    )

    img_arr = decompress_to_array(comp)
    im = Image.fromarray(img_arr, mode="L")
    im.save(output_path)


def _quality_arg(value: str) -> int:
    """argparse type for --quality.

    The library API clamps silently, but at the CLI an out-of-range value is
    far more likely to be a typo than an intent, so reject it outright rather
    than quietly encoding at a different quality than was asked for.
    """

    try:
        q = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"quality must be an integer, got {value!r}")

    if not 1 <= q <= 100:
        raise argparse.ArgumentTypeError(f"quality must be in [1, 100], got {q}")

    return q


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JPEG-like DCT-based compressor for grayscale images.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_compress = subparsers.add_parser(
        "compress",
        help="Compress an image with DCT + quantization + Huffman into ICJ2",
    )
    p_compress.add_argument("input", help="Input image path (any format Pillow supports)")
    p_compress.add_argument("output", help="Output binary path (e.g., .icj)")
    p_compress.add_argument(
        "--quality",
        type=_quality_arg,
        default=50,
        help="JPEG-like quality factor [1-100] (default: 50)",
    )

    p_decompress = subparsers.add_parser(
        "decompress",
        help="Decompress an ICJ2 file back to a grayscale PNG",
    )
    p_decompress.add_argument("input", help="Input .icj file path")
    p_decompress.add_argument("output", help="Output image path (e.g., .png)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "compress":
        compress_huffman_file(args.input, args.output, quality=args.quality)
    elif args.command == "decompress":
        decompress_huffman_file(args.input, args.output)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
