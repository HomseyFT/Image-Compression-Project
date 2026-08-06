"""Shared fixtures: a procedural image corpus plus the real-photo sample.

The synthetic images are generated from a seeded RNG rather than committed as
binaries, so the corpus is reproducible and adds nothing to the repository.
Each entry isolates a different failure mode of the codec:

``flat``
    Constant field. All AC coefficients vanish, so every block is a bare
    EOB -- exercises the degenerate single-symbol Huffman table.
``gradient``
    Smooth ramp. Energy concentrated in the lowest frequencies; the case
    block-DCT handles best.
``texture``
    Band-limited noise. Broad, dense coefficient spectrum.
``edges``
    Hard rectangles. Sharp discontinuities produce large high-frequency
    coefficients and the widest magnitude categories.
``checkerboard``
    Per-pixel alternation. Drives the maximum-magnitude AC coefficient and
    is the worst case for the category/nibble packing.

These fixtures test *correctness*. Do not use them to score rate-distortion
changes: ``flat``, ``gradient`` and ``edges`` produce degenerate R-D curves
(near-empty files, and flat spots where raising quality gains +0.00 dB while
the rate grows), which makes "rate at a given PSNR" numerically unstable and
manufactures phantom gains and losses of 5% or more. An early trellis lambda
fit was skewed by exactly this. Score R-D only on content whose curve rises
monotonically -- in practice, photographs. See ``real_images`` and
``test_trellis.test_rd_scoring_requires_a_monotone_curve``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image

import compression

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "images"
DOG = IMAGE_DIR / "dog.png"

# Downscale for the fixtures. The corpus is 11 photographs and trellis encode
# is ~0.4 s per megapixel, so scoring them at native resolution would take the
# suite from seconds to minutes.
FIXTURE_MAX_SIDE = 384


def _flat(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 128, dtype=np.uint8)


def _gradient(h: int, w: int) -> np.ndarray:
    ramp = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(ramp, (h, 1)).astype(np.uint8)


def _texture(h: int, w: int) -> np.ndarray:
    rng = np.random.RandomState(20260804)
    return (rng.rand(h, w) * 255).astype(np.uint8)


def _edges(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    img[: h // 8, :] = 200
    return img


def _checkerboard(h: int, w: int) -> np.ndarray:
    return ((np.indices((h, w)).sum(axis=0) % 2) * 255).astype(np.uint8)


SYNTHETIC = {
    "flat": _flat,
    "gradient": _gradient,
    "texture": _texture,
    "edges": _edges,
    "checkerboard": _checkerboard,
}


@pytest.fixture(scope="session")
def corpus() -> dict[str, np.ndarray]:
    """Synthetic images at a non-multiple-of-8 size, to also cover padding."""

    return {name: fn(72, 56) for name, fn in SYNTHETIC.items()}


@pytest.fixture(scope="session")
def photo() -> np.ndarray:
    """The reference photo sample, downscaled to keep the suite fast.

    Missing this **fails**, it does not skip. Golden pins and every BD-rate
    gate hang off this fixture, so a silent skip would take 53 tests out of
    the suite while it still reported all-green -- which is precisely what
    happened when ``dog.png`` was moved into ``images/``. A corpus that has
    gone missing is a broken checkout, not a reason to test less.
    """

    if not DOG.exists():
        raise FileNotFoundError(
            f"{DOG} is missing -- golden pins and BD-rate gates cannot run "
            "without it. Restore it, or run `python -m bench.fetch_corpus` "
            "for the Kodak set."
        )
    full = Image.fromarray(compression._load_grayscale(str(DOG)), mode="L")
    return np.array(full.resize((256, 256), Image.LANCZOS), dtype=np.uint8)


@pytest.fixture(scope="session")
def real_images() -> dict[str, np.ndarray]:
    """Real photographs, for rate-distortion gating.

    Every image in ``images/`` is picked up automatically -- the Kodak set is
    the conventional choice. Tuning quantization, trellis or entropy
    parameters against a single image risks overfitting, so prefer this
    fixture over ``corpus`` when measuring compression gains.

    Images are downscaled to :data:`FIXTURE_MAX_SIDE`; see the note there.
    """

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"{IMAGE_DIR} is missing. Run `python -m bench.fetch_corpus`."
        )

    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images: dict[str, np.ndarray] = {}
    for path in sorted(IMAGE_DIR.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        arr = compression._load_grayscale(str(path))
        h, w = arr.shape
        if max(h, w) > FIXTURE_MAX_SIDE:
            scale = FIXTURE_MAX_SIDE / max(h, w)
            arr = np.array(
                Image.fromarray(arr, mode="L").resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
                ),
                dtype=np.uint8,
            )
        images[path.stem] = arr

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}.")
    return images
