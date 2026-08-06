"""Codec adapters that turn an image into a rate-distortion curve.

Every adapter reports **whole-file bytes**, not payload bytes, so the
comparison against libjpeg is apples to apples: container headers and Huffman
tables are part of what a format costs, and they are the entire story on small
images (phase 4.1 moved a 3x5 test file's tables from 208 B to 45 B against a
121 B total).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

import compression as c

from .bdrate import Curve

# Fixed-size fields of the ICJ3 container: magic(4) + h(4) + w(4) + quality(1)
# + blocks_y(2) + blocks_x(2) + layout(1) + present(2) + bit_len(4). Tables are
# self-delimiting and unused AC contexts are omitted.
# `test_icj_size_matches_the_real_container` pins this against a written file
# so it cannot drift away from `compress_huffman_file`.
ICJ3_FIXED_HEADER_BYTES = 24

DEFAULT_SWEEP = (20, 30, 40, 50, 60, 70, 80, 90)


def psnr(recon: np.ndarray, ref: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB, capped for identical inputs."""

    mse = float(np.mean((recon.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0**2 / max(mse, 1e-12))


def icj_size(img: np.ndarray, quality: int, trellis: bool = True) -> tuple[int, np.ndarray]:
    """Encoded ICJ3 size in bytes, and the reconstruction, without touching disk."""

    comp = c.compress_array(img, quality=quality, trellis=trellis)
    bitstream, dc_table, ac_tables, _layout = c._encode_blocks_huffman(comp.coeffs)
    size = (
        ICJ3_FIXED_HEADER_BYTES
        + len(c._serialize_huffman_table(dc_table))
        + sum(len(c._serialize_huffman_table(t)) for t in ac_tables if t)
        + len(bitstream)
    )
    return size, c.decompress_to_array(comp)


def icj_curve(
    img: np.ndarray,
    qualities=DEFAULT_SWEEP,
    trellis: bool = True,
    name: str | None = None,
) -> Curve:
    """Rate-distortion curve for this codec."""

    if name is None:
        name = "icj" if trellis else "icj-notrellis"
    points = []
    for q in qualities:
        size, recon = icj_size(img, q, trellis=trellis)
        points.append((size, psnr(recon, img)))
    return Curve.from_points(name, points)


def libjpeg_curve(
    img: np.ndarray, qualities=DEFAULT_SWEEP, name: str = "libjpeg"
) -> Curve:
    """Rate-distortion curve for libjpeg-turbo via Pillow.

    ``optimize=True`` builds per-image Huffman tables, which is the fair
    comparison -- this codec does the same, and leaving it off would flatter
    us by roughly the table-optimization gain rather than by anything we did.
    """

    points = []
    for q in qualities:
        buf = io.BytesIO()
        Image.fromarray(img, mode="L").save(
            buf, format="JPEG", quality=int(q), optimize=True
        )
        data = buf.getvalue()
        with Image.open(io.BytesIO(data)) as im:
            recon = np.array(im.convert("L"))
        points.append((len(data), psnr(recon, img)))
    return Curve.from_points(name, points)
