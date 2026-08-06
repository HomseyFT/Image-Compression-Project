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

# Fixed-size fields of the ICJ4 container: magic(4) + h(4) + w(4) + quality(1)
# + format(1) + bit_len(4). Block dimensions are derived, not stored. Each
# component class then adds layout(1) + present(2) plus its self-delimiting
# tables; unused AC contexts are omitted.
# `test_icj_size_matches_the_real_container` pins this against a written file
# so it cannot drift away from `compress_huffman_file`.
ICJ4_FIXED_HEADER_BYTES = 18
ICJ4_PER_CLASS_BYTES = 3

DEFAULT_SWEEP = (20, 30, 40, 50, 60, 70, 80, 90)


def psnr(recon: np.ndarray, ref: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB, capped for identical inputs."""

    mse = float(np.mean((recon.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0**2 / max(mse, 1e-12))


def icj_size(
    img: np.ndarray,
    quality: int,
    trellis: bool = True,
    sampling: int = c.SAMPLING_420,
) -> tuple[int, np.ndarray]:
    """Encoded ICJ4 size in bytes, and the reconstruction, without touching disk."""

    comp = c.compress_array(img, quality=quality, trellis=trellis, sampling=sampling)
    classes = c.COMPONENT_CLASSES if comp.n_components == 3 else (c.CLASS_LUMA,)
    bitstream, table_sets = c._encode_planes(comp.planes, classes)

    size = ICJ4_FIXED_HEADER_BYTES + len(bitstream)
    for tables in table_sets:
        size += ICJ4_PER_CLASS_BYTES
        size += len(c._serialize_huffman_table(tables.dc_table))
        size += sum(
            len(c._serialize_huffman_table(t)) for t in tables.ac_tables if t
        )
    return size, c.decompress_to_array(comp)


def icj_curve(
    img: np.ndarray,
    qualities=DEFAULT_SWEEP,
    trellis: bool = True,
    name: str | None = None,
    sampling: int = c.SAMPLING_420,
) -> Curve:
    """Rate-distortion curve for this codec, grayscale or colour."""

    if name is None:
        name = "icj" if trellis else "icj-notrellis"
    points = []
    for q in qualities:
        size, recon = icj_size(img, q, trellis=trellis, sampling=sampling)
        points.append((size, psnr(recon, img)))
    return Curve.from_points(name, points)


def libjpeg_curve(
    img: np.ndarray,
    qualities=DEFAULT_SWEEP,
    name: str = "libjpeg",
    subsampling: int = 2,
) -> Curve:
    """Rate-distortion curve for libjpeg-turbo via Pillow.

    ``optimize=True`` builds per-image Huffman tables, which is the fair
    comparison -- this codec does the same, and leaving it off would flatter
    us by roughly the table-optimization gain rather than by anything we did.

    ``subsampling`` follows Pillow's convention (0 = 4:4:4, 1 = 4:2:2,
    2 = 4:2:0) and is ignored for grayscale input. It defaults to 4:2:0, which
    is what libjpeg itself defaults to, so the comparison is like for like.
    """

    color = img.ndim == 3
    mode = "RGB" if color else "L"
    points = []
    for q in qualities:
        buf = io.BytesIO()
        kwargs = dict(format="JPEG", quality=int(q), optimize=True)
        if color:
            kwargs["subsampling"] = subsampling
        Image.fromarray(img, mode=mode).save(buf, **kwargs)
        data = buf.getvalue()
        with Image.open(io.BytesIO(data)) as im:
            recon = np.array(im.convert(mode))
        points.append((len(data), psnr(recon, img)))
    return Curve.from_points(name, points)


def plane_psnr(recon: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Per-plane PSNR in YCbCr, for judging chroma decisions honestly.

    Chroma subsampling is close to perceptually free but visibly hurts chroma
    PSNR, so a single RGB number **understates** 4:2:0. Any comparison across
    sampling schemes has to say which metric it used; this exists so that the
    chroma cost is visible rather than averaged away.
    """

    if recon.ndim != 3:
        return {"Y": psnr(recon, ref)}

    a = c._rgb_to_ycbcr(ref)
    b = c._rgb_to_ycbcr(recon)
    return {
        name: 10.0 * np.log10(255.0**2 / max(float(np.mean((a[..., i] - b[..., i]) ** 2)), 1e-12))
        for i, name in enumerate(("Y", "Cb", "Cr"))
    }
