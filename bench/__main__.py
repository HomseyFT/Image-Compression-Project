"""``python -m bench`` -- rate-distortion report as a markdown table.

Scores the codec against libjpeg-turbo and against its own no-trellis
baseline, over whatever photographs are present in ``tests/images/`` plus the
repository's ``dog.png``. Output is intended to be pasted into the README.

Synthetic corpus images are deliberately *not* included: their curves have
flat spots and :func:`bench.bdrate.check_monotone` will refuse them. That is
the guard working, not a bug.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

import compression as c

from .bdrate import NonMonotoneCurveError, bd_rate
from .codecs import DEFAULT_SWEEP, icj_curve, libjpeg_curve

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "images"
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_corpus(
    max_side: int | None = None, color: bool = False
) -> dict[str, np.ndarray]:
    """Every photograph in ``images/``, as grayscale or RGB."""

    images: dict[str, np.ndarray] = {}

    if IMAGE_DIR.is_dir():
        for path in sorted(IMAGE_DIR.iterdir()):
            if path.suffix.lower() not in SUFFIXES:
                continue
            img = c._load_image(str(path)) if color else c._load_grayscale(str(path))
            if color and img.ndim != 3:
                continue      # a greyscale source has no colour to score
            images[path.stem] = img

    if max_side:
        for name, img in list(images.items()):
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                size = (max(1, int(w * scale)), max(1, int(h * scale)))
                mode = "RGB" if img.ndim == 3 else "L"
                images[name] = np.array(
                    Image.fromarray(img, mode=mode).resize(size, Image.LANCZOS)
                )

    return images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="downscale images so the longest side is at most this (default: 512). "
        "Use 0 for full resolution -- slow, since trellis dominates encode time.",
    )
    parser.add_argument(
        "--qualities",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP),
        help="quality sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="score colour (YCbCr 4:2:0) instead of grayscale. libjpeg is run "
        "at the same subsampling, so the comparison isolates coding from the "
        "chroma choice.",
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=c.SAMPLING_420,
        choices=sorted(c.SAMPLING_FACTORS),
        help="chroma sampling with --color: 0=4:4:4, 1=4:2:2, 2=4:2:0 "
        "(default: 2)",
    )
    args = parser.parse_args(argv)

    images = load_corpus(max_side=args.max_side or None, color=args.color)
    if not images:
        print(
            f"No photographs found in {IMAGE_DIR} -- run "
            "`python -m bench.fetch_corpus`.",
            file=sys.stderr,
        )
        return 1

    rows: list[tuple[str, float, float]] = []
    failures: list[tuple[str, str]] = []

    # Pillow's subsampling codes happen to match ours (0/1/2), but spell the
    # mapping out rather than relying on the coincidence.
    pil_subsampling = {c.SAMPLING_444: 0, c.SAMPLING_422: 1, c.SAMPLING_420: 2}[
        args.sampling
    ]

    for name, img in images.items():
        try:
            ours = icj_curve(img, args.qualities, trellis=True, sampling=args.sampling)
            base = icj_curve(img, args.qualities, trellis=False, sampling=args.sampling)
            jpeg = libjpeg_curve(
                img, args.qualities, subsampling=pil_subsampling
            )
            rows.append(
                (name, bd_rate(jpeg, ours), bd_rate(base, ours))
            )
        except NonMonotoneCurveError as exc:
            failures.append((name, str(exc)))

    if rows:
        res = "full" if not args.max_side else f"max side {args.max_side}"
        kind = (
            f"colour {c.SAMPLING_NAMES[args.sampling]}" if args.color else "grayscale"
        )
        print(f"BD-rate, negative = fewer bits at equal quality ({kind}, {res}, "
              f"{len(images)} image(s), quality {min(args.qualities)}-{max(args.qualities)})\n")
        print("| image | vs. libjpeg | vs. no-trellis |")
        print("|---|---|---|")
        for name, vs_jpeg, vs_base in rows:
            print(f"| {name} | {vs_jpeg:+.2f}% | {vs_base:+.2f}% |")
        if len(rows) > 1:
            mj = float(np.mean([r[1] for r in rows]))
            mb = float(np.mean([r[2] for r in rows]))
            print(f"| **mean** | **{mj:+.2f}%** | **{mb:+.2f}%** |")

    for name, msg in failures:
        print(f"\nskipped {name}: {msg}", file=sys.stderr)

    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
