"""JPEG-like image compression for grayscale images using 8x8 DCT and quantization.

This module provides a minimal, self-contained implementation of the core
JPEG-style pipeline for black-and-white (grayscale) images:

- Load/save images via Pillow
- Convert to 8-bit grayscale
- Pad to a multiple of 8x8 blocks
- Level shift samples by 128
- Apply 2D DCT on 8x8 blocks (implemented with pure NumPy math)
- Quantize coefficients with a JPEG-like luminance quantization matrix
- Entropy code with JPEG-style DC/AC symbols plus per-file Huffman tables,
  with AC symbols split across per-frequency-band context tables
- Pack into a custom ICJ4 container
- Decompress by reversing the above steps

Usage (CLI):

    # Compress an image to the custom ICJ4 container
    python compression.py compress input.png output.icj --quality 50

    # Decompress from ICJ4 back to a PNG image
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


# Container magic. ICJ4 adds colour (YCbCr, planar, per-class table sets).
# ICJ3 split AC coding across per-frequency-band context tables; see
# AC_LAYOUT_EDGES. ICJ2 superseded ICJ1, whose unclamped quality byte could
# decode against the wrong quantization matrix. No backwards compatibility is
# provided for any of them.
MAGIC = b"ICJ4"

# Maximum Huffman code length the container can represent. The decoder scans
# lengths 1..MAX_CODE_LENGTH.
MAX_CODE_LENGTH = 32


# --- AC context modeling ----------------------------------------------------
#
# The AC symbol stream is coded with one Huffman table per frequency band
# rather than a single table for the whole image. A symbol's context is the
# zigzag position at which it is emitted -- that is, where its zero run
# *starts*, not where the coefficient lands. This matters: the decoder knows
# the former before reading the symbol (it is the next position to fill) and
# cannot know the latter until it has decoded the run. Context selection is
# therefore causal on both sides, and no side information is transmitted.
#
# Bands are logarithmic, fine at low frequencies where the distribution moves
# fastest and coarse at high frequencies where symbols are sparse and an extra
# table would cost more than it saves.
#
# Measured on dog.png against a single AC table, net of real table cost:
#
#   context variable      q30     q50     q80
#   ordinal index (16)   +8.9%   +8.1%   +5.6%
#   zigzag position (16) +9.5%   +8.4%   +6.1%
#   zigzag log-band (15) +9.8%   +9.4%   +7.3%   <- chosen
#
# Conditioning on the *previous symbol* was also measured and rejected: its
# idealized 14.2% conditional-entropy gain collapses to under 2% once table
# cost is paid, because a 256-context split is unaffordable. See SPEC.md 7.1.
#
# --- Why the layout is chosen per image, not fixed ---
#
# Splitting is only worth it when there are enough symbols to amortise a table.
# The 15-band layout is a clear win on a 2500x2500 photo (+21% on AC at q50)
# and an outright *loss* on a 256x256 one at low quality (-7.1% at q10), where
# 15 tables cost more than the entire AC payload. Fixing any single layout
# therefore regresses one end of the size range:
#
#   AC-side gain vs. a single table, by layout
#                    1        3        5        8       15
#   256px  q10   +0.00%   -0.35%   -0.69%   -4.03%   -7.08%
#   256px  q50   +0.00%   +3.46%   +4.67%   +3.35%   +0.28%
#   256px  q90   +0.00%   +6.45%   +8.67%   +8.57%   +7.65%
#   1000px q10   +0.00%   +8.81%  +12.19%  +12.77%  +13.55%
#   1000px q50   +0.00%  +15.44%  +19.43%  +19.95%  +21.08%
#
# So the encoder prices every layout and writes the winner's id in the
# container. Layout 0 is a single table -- exactly ICJ2's behaviour -- which
# makes ICJ3 provably never worse than its predecessor on any input.
AC_LAYOUT_EDGES = (
    (1, 64),                                                    # 0: 1 context
    (1, 4, 12, 64),                                             # 1: 3
    (1, 3, 6, 12, 24, 64),                                      # 2: 5
    (1, 2, 3, 5, 7, 10, 16, 28, 64),                            # 3: 8
    (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 19, 24, 32, 44, 64),   # 4: 15
)


def _build_ac_band_table(edges: tuple[int, ...]) -> np.ndarray:
    """Map zigzag position 0..63 -> AC context index for one layout."""

    band = np.zeros(64, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        band[lo:hi] = i
    return band


AC_LAYOUTS = tuple(_build_ac_band_table(e) for e in AC_LAYOUT_EDGES)
AC_LAYOUT_SIZES = tuple(len(e) - 1 for e in AC_LAYOUT_EDGES)
MAX_AC_CONTEXTS = max(AC_LAYOUT_SIZES)

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

# The same constant refitted for chroma. **It does not transfer from luma**,
# and the failure is not subtle: running chroma at the luma-derived lambda
# costs 4.9 percentage points of BD-rate, consistently across all ten Kodak
# images. Measured against libjpeg at 4:2:0:
#
#   chroma lambda multiplier   0.00    0.05    0.15    0.30    0.60    1.00
#   mean BD-rate             -16.96  -17.18  -17.28  -16.85  -15.05  -12.36
#
# Note that 1.00 -- the luma formula applied unchanged -- is *worse than
# switching trellis off entirely* (-16.96%). The cause is that lambda is tied
# to mean(Q^2), and the chroma table is mostly 99s, so mean(Q^2) is 8125
# against luma's 4500 at q50. The formula therefore hands chroma a **larger**
# lambda (243.75 vs 135.01) precisely where it needs a much smaller one:
# chroma is already coarsely quantized, so there is little redundant precision
# left to trade for bits, and aggressive zeroing destroys what little chroma
# signal survives.
#
# 0.15 of the luma scale is the optimum; the curve is flat between 0.05 and
# 0.30, so the exact value is not delicate. Fitting beats simply disabling
# trellis on chroma by 0.32 pp, which is why chroma keeps it rather than
# opting out.
TRELLIS_LAMBDA_SCALE_CHROMA = TRELLIS_LAMBDA_SCALE * 0.15

# Default trellis refinement passes. Each pass re-derives the Huffman rate
# model from the previous pass's output and re-runs the DP.
#
# Measured across the 11-image corpus at 512 px, BD-rate vs. libjpeg and total
# sweep time:
#
#   passes    mean BD-rate    time     gain over previous
#   1            -13.78%      4.7 s    --
#   2            -13.94%      8.5 s    -0.163 pp for 1.80x
#   3            -14.06%     11.9 s    -0.119 pp for 1.40x
#
# Held at 2 because SPEC.md locks "compression ratio over speed": dropping to
# 1 trades 1.2% of the total compression gain for a 2x encode speedup. The
# gain is small but consistent -- 10 of 11 images improve -- so it is a real
# effect, not noise. An earlier single-image measurement put it at 0.07 pp and
# suggested pass 2 was nearly free; that understated it by 2.3x, which is what
# the multi-image corpus exists to catch.
#
# Callers that would rather have the encode time can pass
# ``trellis_iterations=1``; this is likely to matter for colour, where three
# planes multiply the cost.
TRELLIS_ITERATIONS = 2


BLOCK_SIZE = 8


# --- Colour ------------------------------------------------------------------
#
# Chroma sampling schemes. The factors are (horizontal, vertical) decimation
# applied to Cb and Cr; luma is never subsampled.
SAMPLING_444 = 0
SAMPLING_422 = 1
SAMPLING_420 = 2

SAMPLING_FACTORS = {
    SAMPLING_444: (1, 1),
    SAMPLING_422: (2, 1),
    SAMPLING_420: (2, 2),
}

SAMPLING_NAMES = {
    SAMPLING_444: "4:4:4",
    SAMPLING_422: "4:2:2",
    SAMPLING_420: "4:2:0",
}

# Component classes: Cb and Cr share a table set, as in JPEG.
CLASS_LUMA = 0
CLASS_CHROMA = 1
COMPONENT_CLASSES = (CLASS_LUMA, CLASS_CHROMA, CLASS_CHROMA)


# Standard JPEG chrominance quantization matrix (ISO/IEC 10918-1 Annex K.2).
# Far coarser than the luma table beyond the lowest frequencies -- the flat
# 99s encode the fact that the eye resolves colour detail poorly.
STANDARD_CHROMA_Q = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


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
    """In-memory representation of a compressed image, grayscale or colour.

    Attributes
    ----------
    planes:
        One quantized coefficient array per component, each of shape
        (num_blocks_y, num_blocks_x, 8, 8), dtype=int16. Length 1 for
        grayscale, 3 for YCbCr.
    orig_shape:
        Original image shape: ``(H, W)`` or ``(H, W, 3)``.
    padded_shapes:
        Per-plane padded pixel dimensions used internally.
    quality:
        JPEG-style quality factor in [1, 100].
    sampling:
        Chroma sampling scheme; see :data:`SAMPLING_FACTORS`. Meaningless for
        grayscale, where it is recorded as :data:`SAMPLING_444`.

    **No MCUs.** Because the scan is planar rather than interleaved, each
    plane is an independent array of 8x8 blocks and pads to a multiple of 8 on
    its own. Interleaved JPEG must instead pad luma to the 16x16 MCU so that
    subsampled chroma lands on block boundaries -- the classic place to get
    colour support wrong. Planar coding removes that coupling outright, which
    is a second argument for it beyond the ~0.6% DC-prediction gain that
    motivated the choice.
    """

    planes: list[np.ndarray]
    orig_shape: Tuple[int, ...]
    padded_shapes: list[Tuple[int, int]]
    quality: int
    sampling: int = SAMPLING_444

    @property
    def coeffs(self) -> np.ndarray:
        """Luma coefficients. Grayscale-era alias for ``planes[0]``."""

        return self.planes[0]

    @property
    def padded_shape(self) -> Tuple[int, int]:
        """Luma padded shape. Grayscale-era alias for ``padded_shapes[0]``."""

        return self.padded_shapes[0]

    @property
    def n_components(self) -> int:
        return len(self.planes)

    @property
    def is_color(self) -> bool:
        return len(self.planes) == 3


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


def _build_quant_matrix(quality: int, base: np.ndarray | None = None) -> np.ndarray:
    """Return an 8x8 quantization matrix scaled for the given quality.

    ``base`` defaults to the luminance table; pass :data:`STANDARD_CHROMA_Q`
    for chroma components. Values are clipped to [1, 255] as in JPEG.
    """

    if base is None:
        base = STANDARD_LUMA_Q
    scale = _quality_to_scale(quality)
    q = np.floor((base * scale + 50.0) / 100.0)
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


def _trellis_lambda(Q: np.ndarray, scale: float = TRELLIS_LAMBDA_SCALE) -> float:
    """Lagrange multiplier for a given quantization matrix.

    Rate-distortion theory puts the optimum at ``lambda = -dD/dR``; for a
    uniform quantizer of step ``D`` the high-rate approximation is
    proportional to the step squared. Tying lambda to the quantizer this way
    keeps ``--quality`` meaning what it always did -- each quality level
    simply becomes cheaper in bits -- rather than introducing a second knob
    that silently slides along the rate-distortion curve.

    ``scale`` is fitted empirically and is **per component class**: see
    SPEC.md phase 4.2 for luma and the note at
    :data:`TRELLIS_LAMBDA_SCALE_CHROMA` for why chroma needs its own. The
    ``mean(Q^2)`` proportionality holds within a class but does not carry
    across them.
    """

    return scale * float(np.mean(Q.astype(np.float64) ** 2))


# BT.601 full-range, the JPEG/JFIF convention. Rows map RGB -> Y, Cb, Cr;
# chroma is offset by 128 so all three planes share the [0, 255] domain the
# rest of the pipeline assumes.
_RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float32,
)

_YCBCR_TO_RGB = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float32,
)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) RGB array to float32 YCbCr, chroma offset by 128."""

    out = rgb.astype(DCT_DTYPE, copy=False) @ _RGB_TO_YCBCR.T
    out[..., 1:] += DCT_DTYPE(128.0)
    return out


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_rgb_to_ycbcr`, clipped to uint8."""

    shifted = ycbcr.astype(DCT_DTYPE, copy=False).copy()
    shifted[..., 1:] -= DCT_DTYPE(128.0)
    rgb = shifted @ _YCBCR_TO_RGB.T
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _subsample(plane: np.ndarray, hs: int, vs: int) -> np.ndarray:
    """Decimate a plane by ``(hs, vs)`` using box averaging.

    Averaging rather than dropping samples: a box filter is the matched
    decimation prefilter, and point-sampling chroma aliases badly on saturated
    edges. Odd dimensions are edge-padded first so the box always has full
    support.
    """

    if hs == 1 and vs == 1:
        return plane

    h, w = plane.shape
    pad_h = (vs - h % vs) % vs
    pad_w = (hs - w % hs) % hs
    if pad_h or pad_w:
        plane = np.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge")

    h, w = plane.shape
    return plane.reshape(h // vs, vs, w // hs, hs).mean(axis=(1, 3))


def _upsample(plane: np.ndarray, hs: int, vs: int, shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`_subsample`, to exactly ``shape``.

    Bilinear, with half-sample-offset centres so reconstructed chroma sits
    where the box filter took it from rather than shifted by half a chroma
    sample. libjpeg's "fancy upsampling" is the same idea; the triangle filter
    here is within noise of it and far simpler. Nearest-neighbour was measured
    and is visibly worse on colour edges.
    """

    if hs == 1 and vs == 1:
        return plane[: shape[0], : shape[1]]

    h, w = plane.shape
    target_h, target_w = shape

    # Sample positions of the output grid in input coordinates.
    ys = (np.arange(target_h, dtype=np.float32) + 0.5) / vs - 0.5
    xs = (np.arange(target_w, dtype=np.float32) + 0.5) / hs - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    top = plane[y0][:, x0] * (1 - wx) + plane[y0][:, x1] * wx
    bottom = plane[y1][:, x0] * (1 - wx) + plane[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def _plane_shapes(
    h: int, w: int, sampling: int, n_components: int
) -> list[Tuple[int, int]]:
    """Unpadded pixel dimensions of each component plane."""

    if n_components == 1:
        return [(h, w)]

    hs, vs = SAMPLING_FACTORS[sampling]
    ch = (h + vs - 1) // vs
    cw = (w + hs - 1) // hs
    return [(h, w), (ch, cw), (ch, cw)]


def _load_image(path: str) -> np.ndarray:
    """Load an image as uint8: (H, W) if greyscale on disk, else (H, W, 3).

    Mode ``P`` with transparency is routed through RGBA first, as Pillow warns
    when compositing an undefined background straight to another mode; alpha
    is irrelevant to this codec and is dropped explicitly.
    """

    with Image.open(path) as im:
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
        if im.mode in ("L", "1", "I;16", "I"):
            return np.array(im.convert("L"), dtype=np.uint8)
        return np.array(im.convert("RGB"), dtype=np.uint8)


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


def _marginal_ac_table(coeffs: np.ndarray) -> dict[int, tuple[int, int]]:
    """A single order-0 AC table over all contexts, for the trellis rate model.

    The bitstream is coded with per-context tables, but the trellis DP prices
    a symbol *before* knowing which context it will land in: zeroing a
    coefficient shifts every later symbol's zigzag position, and therefore its
    band. Pricing against the marginal distribution sidesteps that circularity.

    This under-prices the real gain slightly -- context tables are cheaper than
    the marginal -- so the DP is conservative about zeroing rather than
    over-eager, which is the safe direction for a rate model to err in. A
    context-aware variant was measured; see SPEC.md 7.6.
    """

    stream = _scan_symbols(coeffs)
    counts = np.bincount(stream.ac_symbols, minlength=256)
    return _build_huffman_table({i: int(n) for i, n in enumerate(counts)})


def _ac_bit_costs(ac_table: dict[int, tuple[int, int]]) -> np.ndarray:
    """Dense AC symbol -> code length lookup for the trellis rate model.

    Symbols absent from the table are unreachable in the current stream;
    they are priced prohibitively so the DP never selects one.
    """

    bits = np.full(256, 1e6)
    for sym, (_code, n) in ac_table.items():
        bits[sym] = n
    return bits


def _compress_plane(
    plane: np.ndarray,
    Q: np.ndarray,
    trellis: bool,
    iterations: int,
    lambda_scale: float = TRELLIS_LAMBDA_SCALE,
) -> tuple[np.ndarray, Tuple[int, int]]:
    """Transform, quantize and optionally trellis-optimize a single plane."""

    padded, padded_shape = _pad_to_block_size(plane, BLOCK_SIZE)
    padded = padded - DCT_DTYPE(128.0)

    blocks = _to_blocks(padded, BLOCK_SIZE)
    dct = _forward_dct_2d(blocks)
    coeffs = np.round(dct / Q).astype(np.int16)

    if trellis:
        lam = _trellis_lambda(Q, lambda_scale)
        # The rate model needs Huffman costs, which come from the symbol
        # distribution -- which trellis then changes. Re-deriving the tables
        # and re-running closes that loop; iterating is bounded and keeps the
        # best result seen, since convergence is not guaranteed.
        for _ in range(iterations):
            ac_bits = _ac_bit_costs(_marginal_ac_table(coeffs))
            candidate = _trellis_quantize(dct.astype(np.float64), Q, ac_bits, lam)
            if np.array_equal(candidate, coeffs):
                break
            coeffs = candidate

    return coeffs, padded_shape


def _decompress_plane(
    coeffs: np.ndarray, Q: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Dequantize and inverse-transform one plane, cropped to ``shape``."""

    blocks = _inverse_dct_2d(coeffs.astype(DCT_DTYPE) * Q)
    padded = _from_blocks(blocks, BLOCK_SIZE) + DCT_DTYPE(128.0)
    return padded[: shape[0], : shape[1]]


def _quant_matrix_for(component_class: int, quality: int) -> np.ndarray:
    """Quantization matrix for a component class, scaled for ``quality``."""

    base = STANDARD_LUMA_Q if component_class == CLASS_LUMA else STANDARD_CHROMA_Q
    return _build_quant_matrix(quality, base).astype(DCT_DTYPE)


def compress_array(
    image: np.ndarray,
    quality: int = 50,
    trellis: bool = True,
    trellis_iterations: int | None = None,
    sampling: int = SAMPLING_420,
) -> CompressedImage:
    """Compress a grayscale or RGB image using JPEG-like DCT + quantization.

    Parameters
    ----------
    image:
        ``(H, W)`` grayscale or ``(H, W, 3)`` RGB, values in [0, 255]. Other
        dtypes are converted to float32 internally.
    quality:
        JPEG-like quality factor in [1, 100]. Higher means better quality and
        less compression.
    trellis:
        Enable rate-distortion optimized quantization. Encoder-side only --
        the output is an ordinary coefficient array, so the decoder and the
        container format are unaffected.
    trellis_iterations:
        Refinement passes, defaulting to :data:`TRELLIS_ITERATIONS`. Each pass
        roughly doubles encode time for a diminishing rate gain; see the
        measurements at that constant. Ignored when ``trellis`` is False.
    sampling:
        Chroma subsampling for colour input: :data:`SAMPLING_420` (default),
        :data:`SAMPLING_422` or :data:`SAMPLING_444`. Ignored for grayscale.

    The grayscale path is bit-identical to the pre-colour codec; colour is
    strictly additive.
    """

    quality = _clamp_quality(quality)

    if trellis:
        iterations = (
            TRELLIS_ITERATIONS if trellis_iterations is None else int(trellis_iterations)
        )
        if iterations < 0:
            raise ValueError(f"trellis_iterations must be >= 0, got {iterations}")
    else:
        iterations = 0

    if image.ndim == 2:
        pixel_planes = [image.astype(DCT_DTYPE, copy=False)]
        sampling = SAMPLING_444
    elif image.ndim == 3 and image.shape[2] == 3:
        if sampling not in SAMPLING_FACTORS:
            raise ValueError(
                f"sampling must be one of {sorted(SAMPLING_FACTORS)}, got {sampling}"
            )
        ycbcr = _rgb_to_ycbcr(image)
        hs, vs = SAMPLING_FACTORS[sampling]
        pixel_planes = [
            ycbcr[..., 0],
            _subsample(ycbcr[..., 1], hs, vs),
            _subsample(ycbcr[..., 2], hs, vs),
        ]
    else:
        raise ValueError(
            "compress_array expects (H, W) grayscale or (H, W, 3) RGB, got "
            f"shape {image.shape}"
        )

    n_components = len(pixel_planes)
    planes, padded_shapes = [], []
    for index, plane in enumerate(pixel_planes):
        cls = COMPONENT_CLASSES[index] if n_components == 3 else CLASS_LUMA
        scale = (
            TRELLIS_LAMBDA_SCALE
            if cls == CLASS_LUMA
            else TRELLIS_LAMBDA_SCALE_CHROMA
        )
        coeffs, padded_shape = _compress_plane(
            plane, _quant_matrix_for(cls, quality), trellis, iterations, scale
        )
        planes.append(coeffs)
        padded_shapes.append(padded_shape)

    return CompressedImage(
        planes=planes,
        orig_shape=image.shape,
        padded_shapes=padded_shapes,
        quality=int(quality),
        sampling=sampling,
    )


def decompress_to_array(comp: CompressedImage) -> np.ndarray:
    """Decompress a :class:`CompressedImage` to a uint8 array.

    Returns ``(H, W)`` for grayscale, ``(H, W, 3)`` RGB for colour.
    """

    h, w = comp.orig_shape[0], comp.orig_shape[1]
    shapes = _plane_shapes(h, w, comp.sampling, comp.n_components)

    for index, (coeffs, padded_shape) in enumerate(
        zip(comp.planes, comp.padded_shapes)
    ):
        by, bx = coeffs.shape[:2]
        if padded_shape != (by * BLOCK_SIZE, bx * BLOCK_SIZE):
            raise ValueError(
                f"Inconsistent block/padded shapes for component {index}"
            )

    decoded = []
    for index, coeffs in enumerate(comp.planes):
        cls = COMPONENT_CLASSES[index] if comp.n_components == 3 else CLASS_LUMA
        decoded.append(
            _decompress_plane(coeffs, _quant_matrix_for(cls, comp.quality), shapes[index])
        )

    if comp.n_components == 1:
        return np.clip(decoded[0], 0, 255).astype(np.uint8)

    hs, vs = SAMPLING_FACTORS[comp.sampling]
    ycbcr = np.stack(
        [
            decoded[0],
            _upsample(decoded[1], hs, vs, (h, w)),
            _upsample(decoded[2], hs, vs, (h, w)),
        ],
        axis=-1,
    )
    return _ycbcr_to_rgb(ycbcr)


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
    ac_positions: np.ndarray  # (M,) zigzag position the symbol is emitted at
    n_blocks: int

    def contexts(self, layout: int) -> np.ndarray:
        """AC context per symbol under a given band layout."""

        return AC_LAYOUTS[layout][self.ac_positions]


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

    # Zigzag position at which each symbol is emitted: where its zero run
    # starts. The run for this group begins one past the previous non-zero
    # (``prev_pos + 1`` zero-based, hence ``+ 2`` in 1-based zigzag terms), and
    # each preceding ZRL advances 16. Deliberately *not* the position the
    # coefficient lands at -- the decoder cannot know that until it has
    # decoded the run, so it would not be a causal context.
    e_kpos = (prev_pos[src] + 2) + 16 * sub

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

    # An EOB sits one past the block's last non-zero; an all-zero block emits
    # it at position 1, which ``last_pos == -1`` yields for free.
    e_kpos = np.concatenate([e_kpos, last_pos[eob_blocks] + 2])

    # Emission order: block, then zigzag position, then ZRLs before the value.
    order = np.lexsort((e_sub, e_pos, e_block))

    return _SymbolStream(
        dc_cats=dc_cats,
        dc_diffs=dc_diffs,
        ac_block=e_block[order],
        ac_symbols=e_sym[order],
        ac_values=e_val[order],
        ac_sizes=e_size[order],
        ac_positions=np.clip(e_kpos[order], 1, 63),
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


def _build_ac_context_tables(
    symbols: np.ndarray, contexts: np.ndarray, n_contexts: int
) -> list[dict[int, tuple[int, int]]]:
    """One Huffman table per AC context, from the symbol stream.

    A context that never occurs gets an empty table. That is not an error:
    high-frequency bands go unused on smooth images and at low quality, where
    every block ends long before the tail of the zigzag. Unused contexts are
    omitted from the container entirely.
    """

    tables: list[dict[int, tuple[int, int]]] = []
    for ctx in range(n_contexts):
        counts = np.bincount(symbols[contexts == ctx], minlength=256)
        if not counts.any():
            tables.append({})
            continue
        tables.append(_build_huffman_table({i: int(n) for i, n in enumerate(counts)}))
    return tables


def _ac_layout_cost(
    symbols: np.ndarray, positions: np.ndarray, layout: int
) -> tuple[float, list[dict[int, tuple[int, int]]]]:
    """Total bytes -- code bits plus serialized tables -- for one band layout.

    Table cost is counted because it is the whole reason coarser layouts win
    on small images. Scoring code bits alone would always prefer the finest
    split, which is exactly the mistake that makes a 256px image at q10 come
    out 7% *larger*.

    Takes raw ``(symbols, positions)`` rather than a stream so that a whole
    component class -- Cb and Cr together -- can be priced as one table set.
    """

    contexts = AC_LAYOUTS[layout][positions]
    tables = _build_ac_context_tables(symbols, contexts, AC_LAYOUT_SIZES[layout])

    bits = 0
    table_bytes = 0
    for ctx, table in enumerate(tables):
        if not table:
            continue
        counts = np.bincount(symbols[contexts == ctx], minlength=256)
        bits += sum(int(counts[sym]) * n for sym, (_cd, n) in table.items())
        table_bytes += len(_serialize_huffman_table(table))

    return bits / 8.0 + table_bytes, tables


def _choose_ac_layout(
    symbols: np.ndarray, positions: np.ndarray
) -> tuple[int, list[dict[int, tuple[int, int]]]]:
    """Pick the band layout that costs the fewest bytes for this symbol set.

    Ties go to the coarser layout, which keeps the container smaller and the
    decoder's table build cheaper for no rate cost.
    """

    best_layout = 0
    best_cost, best_tables = _ac_layout_cost(symbols, positions, 0)
    for layout in range(1, len(AC_LAYOUTS)):
        cost, tables = _ac_layout_cost(symbols, positions, layout)
        if cost < best_cost:
            best_layout, best_cost, best_tables = layout, cost, tables
    return best_layout, best_tables


def _ac_tables_to_arrays(
    tables: list[dict[int, tuple[int, int]]]
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(context, symbol) -> code/length`` lookups for all AC tables."""

    code = np.zeros((len(tables), 256), dtype=np.int64)
    length = np.zeros((len(tables), 256), dtype=np.int64)
    for ctx, table in enumerate(tables):
        for sym, (cd, n) in table.items():
            code[ctx, sym] = cd
            length[ctx, sym] = n
    return code, length


@dataclass
class _ClassTables:
    """The Huffman table set shared by every component of one class."""

    dc_table: dict[int, tuple[int, int]]
    layout: int
    ac_tables: list[dict[int, tuple[int, int]]]


def _build_class_tables(streams: list[_SymbolStream]) -> _ClassTables:
    """Derive one table set from the pooled symbols of a component class.

    Cb and Cr are pooled rather than given a table each, as in JPEG: their
    statistics are near-identical, so two table sets would roughly double
    table cost to buy almost nothing.
    """

    dc_cats = np.concatenate([s.dc_cats for s in streams])
    ac_symbols = np.concatenate([s.ac_symbols for s in streams])
    ac_positions = np.concatenate([s.ac_positions for s in streams])

    dc_counts = np.bincount(dc_cats, minlength=12)
    dc_table = _build_huffman_table({i: int(n) for i, n in enumerate(dc_counts)})
    layout, ac_tables = _choose_ac_layout(ac_symbols, ac_positions)
    return _ClassTables(dc_table=dc_table, layout=layout, ac_tables=ac_tables)


def _emit_plane(
    stream: _SymbolStream, tables: _ClassTables
) -> tuple[np.ndarray, np.ndarray]:
    """Codes and lengths for one plane, in emission order.

    Per block the layout is
    ``[DC code][DC mantissa]`` then ``[AC symbol][AC mantissa] * n_ac``.
    """

    dc_code, dc_len = _table_to_arrays(tables.dc_table, 12)
    ac_code, ac_len = _ac_tables_to_arrays(tables.ac_tables)
    ac_contexts = stream.contexts(tables.layout)

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

    codes[idx] = ac_code[ac_contexts, stream.ac_symbols]
    lengths[idx] = ac_len[ac_contexts, stream.ac_symbols]
    codes[idx + 1] = _value_to_bits_vec(stream.ac_values, stream.ac_sizes)
    lengths[idx + 1] = stream.ac_sizes

    return codes, lengths


def _encode_planes(
    planes: list[np.ndarray], classes: tuple[int, ...]
) -> tuple[bytes, list[_ClassTables]]:
    """Entropy-code every plane into one bitstream, planar order.

    Planes are emitted back to back -- all of Y, then all of Cb, then all of
    Cr -- rather than interleaved into MCUs. Each plane restarts its own DC
    predictor, which is what :func:`_scan_symbols` already does per call.
    """

    streams = [_scan_symbols(p) for p in planes]

    n_classes = max(classes) + 1
    table_sets = [
        _build_class_tables([streams[i] for i, c in enumerate(classes) if c == cls])
        for cls in range(n_classes)
    ]

    codes, lengths = [], []
    for stream, cls in zip(streams, classes):
        plane_codes, plane_lengths = _emit_plane(stream, table_sets[cls])
        codes.append(plane_codes)
        lengths.append(plane_lengths)

    return (
        _pack_bits(np.concatenate(codes), np.concatenate(lengths)),
        table_sets,
    )


def _encode_blocks_huffman(
    coeffs: np.ndarray,
) -> tuple[bytes, dict[int, tuple[int, int]], list[dict[int, tuple[int, int]]], int]:
    """Single-plane entropy coding. Thin wrapper over :func:`_encode_planes`.

    Returns ``(bitstream, dc_table, ac_tables, layout)``.
    """

    bitstream, (tables,) = _encode_planes([coeffs], (CLASS_LUMA,))
    return bitstream, tables.dc_table, tables.ac_tables, tables.layout


def _table_to_arrays(table: dict[int, tuple[int, int]], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense (code, length) lookup arrays for a Huffman table."""

    code = np.zeros(size, dtype=np.int64)
    length = np.zeros(size, dtype=np.int64)
    for sym, (cd, n) in table.items():
        code[sym] = cd
        length[sym] = n
    return code, length


# Width of the direct-indexed Huffman decode window. A code no longer than
# this is resolved in a single lookup; longer ones fall back to a bit-at-a-time
# scan. 12 bits costs 4096 entries per table and covers essentially every code
# natural images produce, while staying cheap enough to rebuild per file.
DECODE_LUT_BITS = 12

# Zero bytes the reader will invent past the end of the bitstream. The encoder
# pads the final code to a byte boundary, so a correct decode overruns by at
# most 7 bits; anything beyond this slack is a corrupt stream running away,
# and must raise rather than decode padding into plausible garbage.
_EOF_SLACK_BYTES = 8


def _build_decode_lut(table: dict[int, tuple[int, int]]):
    """Build ``(symbols, lengths, long_codes)`` for fast decoding.

    ``symbols`` and ``lengths`` are Python lists indexed by the next
    :data:`DECODE_LUT_BITS` bits of the stream; a symbol of ``-1`` means the
    code is longer than the window and must be resolved via ``long_codes``,
    keyed by ``(length, code)``.

    Lists rather than arrays: this is indexed once per symbol from Python, and
    a list index is several times cheaper than a NumPy scalar index.

    Returns ``None`` for an empty table, i.e. an AC context that never occurs.
    """

    if not table:
        return None

    size = 1 << DECODE_LUT_BITS
    symbols = [-1] * size
    lengths = [0] * size
    long_codes: dict[tuple[int, int], int] = {}

    for sym, (code, n) in table.items():
        if n <= DECODE_LUT_BITS:
            shift = DECODE_LUT_BITS - n
            start = code << shift
            for i in range(start, start + (1 << shift)):
                symbols[i] = sym
                lengths[i] = n
        else:
            long_codes[(n, code)] = sym

    return symbols, lengths, long_codes


def _decode_long_code(read_bit, long_codes: dict[tuple[int, int], int], what: str) -> int:
    """Resolve a code longer than the LUT window, bit at a time.

    Off the hot path: natural images produce codes well under
    :data:`DECODE_LUT_BITS`, so this runs rarely enough not to matter.
    """

    code = 0
    for length in range(1, MAX_CODE_LENGTH + 1):
        code = (code << 1) | read_bit()
        found = long_codes.get((length, code))
        if found is not None:
            return found
    raise ValueError(f"Failed to decode {what}")


def _decode_planes(
    block_shapes: list[Tuple[int, int]],
    classes: tuple[int, ...],
    bitstream: bytes,
    table_sets: list[_ClassTables],
) -> list[np.ndarray]:
    """Inverse of :func:`_encode_planes`.

    Returns one ``(by, bx, 8, 8)`` array per plane.

    The AC table is selected per symbol by ``AC_LAYOUTS[layout]`` indexed with
    ``k``, the position about to be filled. ``k`` is known before the symbol is
    read, which is what makes the context causal and free of side information.

    **On the style of this function.** Huffman decoding is inherently
    sequential -- it cannot be vectorized the way the encoder was in phase 4.5
    -- so the only lever is the constant factor, and in CPython that means
    function calls and attribute lookups. Profiling a factored version (a bit
    reader object exposing ``peek``/``skip``/``read_bits``) showed 902k calls
    for 126k symbols, with call and ``self.`` overhead accounting for roughly
    two thirds of decode time; inlining the accumulator into locals cut that
    to 106k calls.

    Net effect, 2500x2500 at q50: **0.68 s -> 0.41 s (1.6x)**. To be honest
    about the ceiling, that is most of what this approach can give: the
    remainder is ~680 ns per symbol of irreducible CPython loop work plus
    0.09 s materializing the output array, and going meaningfully faster
    wants a C extension rather than more micro-optimization. Four conversion
    strategies were measured for that last step and ``np.array`` on the
    nested list won, so it is not the thing to optimize next either.

    ``tests/test_context.py`` carries an independent, deliberately naive
    reference decoder and asserts the two agree bit for bit.
    """

    # One decode structure per class, built once and reused across the planes
    # that share it.
    class_luts = []
    for tables in table_sets:
        dc_lut = _build_decode_lut(tables.dc_table)
        if dc_lut is None:
            raise ValueError("Container has an empty DC table")
        class_luts.append(
            (
                dc_lut,
                [_build_decode_lut(t) for t in tables.ac_tables],
                AC_LAYOUTS[tables.layout].tolist(),
            )
        )

    zigzag = ZIGZAG_ORDER.tolist()

    data = bitstream
    limit = len(data)
    hard_end = limit + _EOF_SLACK_BYTES
    window_bits = DECODE_LUT_BITS
    window_mask = (1 << window_bits) - 1

    acc = 0        # bit accumulator, MSB-first
    nb = 0         # valid bits held in `acc`
    pos = 0        # next byte of `data` to consume

    def _read_bit() -> int:
        """Slow-path bit read, sharing the accumulator via nonlocal state."""
        nonlocal acc, nb, pos
        if nb == 0:
            if pos < limit:
                byte = data[pos]
            elif pos < hard_end:
                byte = 0
            else:
                raise EOFError("Unexpected end of bitstream")
            pos += 1
            acc = byte
            nb = 8
        nb -= 1
        bit = (acc >> nb) & 1
        acc &= (1 << nb) - 1
        return bit

    out_planes: list[np.ndarray] = []

    for (by, bx), cls in zip(block_shapes, classes):
        (dc_sym, dc_len, dc_long), luts, band = class_luts[cls]

        rows: list[list[int]] = []
        # Each plane restarts its own DC predictor, matching the encoder's
        # per-plane scan.
        prev_dc = 0

        for _ in range(by * bx):
            flat = [0] * 64

            # --- DC ---
            while nb < window_bits:
                if pos < limit:
                    byte = data[pos]
                elif pos < hard_end:
                    byte = 0
                else:
                    raise EOFError("Unexpected end of bitstream")
                pos += 1
                acc = (acc << 8) | byte
                nb += 8

            w = (acc >> (nb - window_bits)) & window_mask
            cat = dc_sym[w]
            if cat >= 0:
                nb -= dc_len[w]
                acc &= (1 << nb) - 1
            else:
                cat = _decode_long_code(_read_bit, dc_long, "DC coefficient")

            if cat:
                while nb < cat:
                    if pos < limit:
                        byte = data[pos]
                    elif pos < hard_end:
                        byte = 0
                    else:
                        raise EOFError("Unexpected end of bitstream")
                    pos += 1
                    acc = (acc << 8) | byte
                    nb += 8
                nb -= cat
                bits = (acc >> nb) & ((1 << cat) - 1)
                acc &= (1 << nb) - 1
                prev_dc += bits if bits >= (1 << (cat - 1)) else bits - ((1 << cat) - 1)

            flat[zigzag[0]] = prev_dc

            # --- AC ---
            k = 1
            while k < 64:
                lut = luts[band[k]]
                if lut is None:
                    raise ValueError("AC stream references an unused context")
                ac_sym, ac_len, ac_long = lut

                while nb < window_bits:
                    if pos < limit:
                        byte = data[pos]
                    elif pos < hard_end:
                        byte = 0
                    else:
                        raise EOFError("Unexpected end of bitstream")
                    pos += 1
                    acc = (acc << 8) | byte
                    nb += 8

                w = (acc >> (nb - window_bits)) & window_mask
                symbol = ac_sym[w]
                if symbol >= 0:
                    nb -= ac_len[w]
                    acc &= (1 << nb) - 1
                else:
                    symbol = _decode_long_code(_read_bit, ac_long, "AC coefficient")

                if symbol == 0x00:          # EOB
                    break
                if symbol == 0xF0:          # ZRL
                    k += 16
                    if k > 64:
                        raise ValueError("ZRL went past end of block")
                    continue

                k += symbol >> 4
                if k >= 64:
                    raise ValueError("Run-length went past end of block")

                size = symbol & 0x0F
                if size:
                    while nb < size:
                        if pos < limit:
                            byte = data[pos]
                        elif pos < hard_end:
                            byte = 0
                        else:
                            raise EOFError("Unexpected end of bitstream")
                        pos += 1
                        acc = (acc << 8) | byte
                        nb += 8
                    nb -= size
                    bits = (acc >> nb) & ((1 << size) - 1)
                    acc &= (1 << nb) - 1
                    flat[zigzag[k]] = (
                        bits if bits >= (1 << (size - 1)) else bits - ((1 << size) - 1)
                    )
                k += 1

            rows.append(flat)

        out_planes.append(
            np.array(rows, dtype=np.int16).reshape(by, bx, BLOCK_SIZE, BLOCK_SIZE)
        )

    return out_planes


def _decode_blocks_huffman(
    by: int,
    bx: int,
    bitstream: bytes,
    dc_table: dict[int, tuple[int, int]],
    ac_tables: list[dict[int, tuple[int, int]]],
    layout: int,
) -> np.ndarray:
    """Single-plane decode. Thin wrapper over :func:`_decode_planes`."""

    tables = _ClassTables(dc_table=dc_table, layout=layout, ac_tables=ac_tables)
    return _decode_planes([(by, bx)], (CLASS_LUMA,), bitstream, [tables])[0]


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


def _deserialize_huffman_table(
    data: bytes, allow_empty: bool = False
) -> dict[int, tuple[int, int]]:
    """Rebuild a canonical Huffman table from its serialized lengths.

    ``allow_empty`` permits a zero-symbol table, which is legitimate only for
    an unused AC context (see :func:`_build_ac_context_tables`). The DC table
    is always populated, so it keeps the strict check.
    """

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
        if allow_empty:
            return {}
        raise ValueError("Huffman table contains no symbols")

    return _assign_canonical_codes(lengths)


def _read_huffman_table(f) -> dict[int, tuple[int, int]]:
    """Read one self-delimiting Huffman table from an open container.

    The table opens with its own 2-byte symbol count, so the reader knows how
    much to consume without a separate length field.
    """

    count_bytes = _read_exactly(f, 2)
    count = int.from_bytes(count_bytes, "big")
    return _deserialize_huffman_table(
        count_bytes + _read_exactly(f, 2 * count), allow_empty=True
    )


def _block_shapes(padded_shapes: list[Tuple[int, int]]) -> list[Tuple[int, int]]:
    """Block dimensions implied by padded pixel dimensions."""

    return [(h // BLOCK_SIZE, w // BLOCK_SIZE) for h, w in padded_shapes]


def _padded_shapes_for(
    h: int, w: int, sampling: int, n_components: int
) -> list[Tuple[int, int]]:
    """Padded plane dimensions, derived rather than stored.

    ICJ3 wrote ``blocks_y``/``blocks_x`` into the header even though both are
    a pure function of the image dimensions. ICJ4 drops them, which is what
    lets a colour-capable container be *smaller* than its grayscale-only
    predecessor rather than larger.
    """

    def pad(n: int) -> int:
        return ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    return [
        (pad(ph), pad(pw))
        for ph, pw in _plane_shapes(h, w, sampling, n_components)
    ]


def _write_class_tables(header: bytearray, tables: _ClassTables) -> None:
    """Append one class's table set to a container header."""

    # Tables are self-delimiting -- each opens with its own symbol count -- so
    # no byte-length prefix is stored.
    header.extend(_serialize_huffman_table(tables.dc_table))

    # Only contexts that actually occur are written, identified by a bitmap.
    # Serialising every context unconditionally would spend 2 bytes apiece on
    # empty tables, which is pure overhead on an image that uses few bands --
    # the regime phase 4.1 was written to protect.
    header.append(tables.layout & 0xFF)
    present = 0
    for ctx, table in enumerate(tables.ac_tables):
        if table:
            present |= 1 << ctx
    header.extend(present.to_bytes(2, "big"))
    for table in tables.ac_tables:
        if table:
            header.extend(_serialize_huffman_table(table))


def _read_class_tables(f) -> _ClassTables:
    """Inverse of :func:`_write_class_tables`."""

    dc_table = _read_huffman_table(f)

    layout = _read_exactly(f, 1)[0]
    if layout >= len(AC_LAYOUTS):
        raise ValueError(
            f"Container declares AC band layout {layout}, but this build "
            f"defines only {len(AC_LAYOUTS)}. The layout table is part of "
            "the format; a file written against a different "
            "AC_LAYOUT_EDGES cannot be decoded."
        )

    present = int.from_bytes(_read_exactly(f, 2), "big")
    ac_tables = [
        _read_huffman_table(f) if present & (1 << ctx) else {}
        for ctx in range(AC_LAYOUT_SIZES[layout])
    ]
    return _ClassTables(dc_table=dc_table, layout=layout, ac_tables=ac_tables)


def compress_huffman_file(
    input_path: str,
    output_path: str,
    quality: int = 50,
    sampling: int = SAMPLING_420,
) -> None:
    """Compress an image using DCT + quantization + Huffman into a custom binary.

    The ICJ4 container format is::

        magic:      4 bytes   ASCII 'ICJ4'
        height:     4 bytes   unsigned big-endian
        width:      4 bytes   unsigned big-endian
        quality:    1 byte    1-100 (always clamped before writing)
        format:     1 byte    high nibble = component count (1 or 3)
                              low nibble  = chroma sampling scheme
        per class:  table set (1 class for grayscale, 2 for colour)
            dc_table:   self-delimiting (see below)
            layout:     1 byte   index into AC_LAYOUT_EDGES
            present:    2 bytes  bitmap, bit i set if context i has a table
            ac_tables:  one self-delimiting table per set bit
        bit_len:    4 bytes   length of following bitstream in bytes
        bitstream:  bit_len bytes, big-endian bit packing, zero-padded

    Each Huffman table is serialised as a 2-byte count of used symbols
    followed by that many 2-byte ``(symbol, code length)`` pairs. Canonical
    codes are rebuilt from the lengths on load. Tables are therefore
    self-delimiting and carry no byte-length prefix.

    Contexts that never occur are omitted entirely rather than written as
    empty tables, which would otherwise be pure overhead on an image that
    uses few bands.

    Block dimensions are **not** stored: they follow from the image size and
    sampling scheme. Dropping those redundant fields is what makes ICJ4
    smaller than ICJ3 on grayscale despite gaining a format byte, so colour
    support costs the grayscale path nothing.

    The scan is **planar**: all of Y, then all of Cb, then all of Cr, each
    restarting its own DC predictor. Cb and Cr share one table set, as in
    JPEG. See :class:`CompressedImage` on why planar removes the MCU-padding
    coupling entirely.
    """

    arr = _load_image(input_path)
    comp = compress_array(arr, quality=quality, sampling=sampling)

    classes = (
        COMPONENT_CLASSES if comp.n_components == 3 else (CLASS_LUMA,)
    )
    bitstream, table_sets = _encode_planes(comp.planes, classes)

    h, w = comp.orig_shape[0], comp.orig_shape[1]

    header = bytearray()
    header.extend(MAGIC)
    header.extend(int(h).to_bytes(4, "big"))
    header.extend(int(w).to_bytes(4, "big"))
    header.append(int(comp.quality) & 0xFF)
    header.append(((comp.n_components & 0x0F) << 4) | (comp.sampling & 0x0F))

    for tables in table_sets:
        _write_class_tables(header, tables)

    header.extend(len(bitstream).to_bytes(4, "big"))

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(bitstream)


def decompress_huffman_file(input_path: str, output_path: str) -> None:
    """Inverse of :func:`compress_huffman_file`.

    Reads the custom container, decodes Huffman-coded coefficients, performs
    inverse DCT + dequantization, and writes a grayscale or RGB image.
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

        fmt = _read_exactly(f, 1)[0]
        n_components = fmt >> 4
        sampling = fmt & 0x0F
        if n_components not in (1, 3):
            raise ValueError(
                f"Container declares {n_components} components; only 1 "
                "(grayscale) and 3 (YCbCr) are defined"
            )
        if sampling not in SAMPLING_FACTORS:
            raise ValueError(
                f"Container declares chroma sampling {sampling}, which is not "
                f"one of {sorted(SAMPLING_FACTORS)}"
            )

        n_classes = 1 if n_components == 1 else 2
        table_sets = [_read_class_tables(f) for _ in range(n_classes)]

        bit_len = int.from_bytes(_read_exactly(f, 4), "big")
        bitstream = _read_exactly(f, bit_len)

    padded_shapes = _padded_shapes_for(h, w, sampling, n_components)
    classes = COMPONENT_CLASSES if n_components == 3 else (CLASS_LUMA,)
    planes = _decode_planes(
        _block_shapes(padded_shapes), classes, bitstream, table_sets
    )

    comp = CompressedImage(
        planes=planes,
        orig_shape=(h, w, 3) if n_components == 3 else (h, w),
        padded_shapes=padded_shapes,
        quality=int(quality),
        sampling=sampling,
    )

    img_arr = decompress_to_array(comp)
    im = Image.fromarray(img_arr, mode="RGB" if n_components == 3 else "L")
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
