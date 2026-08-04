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

Synthetic images alone are a poor rate-distortion gate -- their statistics
are not those of natural images -- so ``photo`` (dog.png) is included and
should be preferred for any R-D comparison. See ``real_images``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_DIR = pathlib.Path(__file__).resolve().parent / "images"


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
    """The real-photo sample, downscaled to keep the suite fast."""

    path = REPO_ROOT / "dog.png"
    if not path.exists():
        pytest.skip("dog.png not available")
    with Image.open(path) as im:
        im = im.convert("L").resize((256, 256), Image.LANCZOS)
        return np.array(im, dtype=np.uint8)


@pytest.fixture(scope="session")
def real_images(photo: np.ndarray) -> dict[str, np.ndarray]:
    """Real photographs, for rate-distortion gating.

    Always contains ``dog``. Any additional images dropped into
    ``tests/images/`` are picked up automatically -- the Kodak set is the
    conventional choice. Tuning quantization or trellis parameters against a
    single image risks overfitting, so prefer this fixture over ``corpus``
    when measuring compression gains.
    """

    images = {"dog": photo}
    if IMAGE_DIR.is_dir():
        for path in sorted(IMAGE_DIR.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                with Image.open(path) as im:
                    images[path.stem] = np.array(im.convert("L"), dtype=np.uint8)
    return images
