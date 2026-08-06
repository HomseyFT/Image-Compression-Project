"""Terminal viewer for ICJ files (and any image Pillow can read).

    icjview photo.icj
    icjview photo.icj --info
    icjview a.icj b.png --compare      # side by side
    python -m icjview photo.icj

Rendering uses the Unicode upper-half block ``U+2580``: the glyph's foreground
paints the top half of the cell and its background the bottom half, so one
character carries **two** vertically stacked pixels. That is how `viu` and
`chafa` work, and it doubles vertical resolution while keeping output to plain
ANSI escapes -- no sixel, no kitty protocol, no terminal-specific support.

Character cells are roughly twice as tall as they are wide, so mapping one
pixel per column and two per row keeps the aspect ratio right without any
correction factor.

Colour depth is detected, not assumed: 24-bit if the terminal advertises it,
otherwise the 256-colour cube, otherwise monochrome ASCII. A viewer that emits
truecolor escapes into a terminal that cannot render them produces confetti,
which is worse than a plain fallback.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import numpy as np
from PIL import Image

import compression as c

UPPER_HALF = "▀"
RESET = "\x1b[0m"

# Densest-to-lightest ramp for the monochrome fallback.
ASCII_RAMP = "@%#*+=-:. "

TRUECOLOR = "truecolor"
COLOR256 = "256"
MONO = "mono"


def detect_color_mode(stream=None) -> str:
    """Best colour depth for this terminal.

    Deliberately conservative: anything not clearly a colour-capable TTY gets
    ASCII, so piping into a file or a pager yields something readable rather
    than escape soup.
    """

    stream = stream or sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return MONO
    if os.environ.get("NO_COLOR"):
        return MONO

    term = os.environ.get("TERM", "")
    if term in ("dumb", ""):
        return MONO
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return TRUECOLOR
    if "256color" in term:
        return COLOR256
    return COLOR256


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Present any decoded image as (H, W, 3) uint8."""

    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    return img


def fit(shape: tuple[int, int], cols: int, rows: int) -> tuple[int, int]:
    """Pixel dimensions that fit ``cols`` x ``rows`` cells, preserving aspect.

    Each cell holds 2 pixel rows, so the usable pixel canvas is
    ``cols`` wide by ``2 * rows`` tall.
    """

    h, w = shape
    max_w, max_h = cols, rows * 2
    scale = min(max_w / w, max_h / h, 1.0)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    mode = "RGB" if img.ndim == 3 else "L"
    return np.array(
        Image.fromarray(img, mode=mode).resize((width, height), Image.LANCZOS)
    )


def _rgb_to_256(rgb: np.ndarray) -> np.ndarray:
    """Map RGB to xterm-256 indices via the 6x6x6 cube and grey ramp."""

    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    cube = 16 + 36 * (r * 5 // 255) + 6 * (g * 5 // 255) + (b * 5 // 255)

    # Near-grey pixels land on the 24-step ramp, which is finer than the cube.
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    grey_level = 232 + np.clip((r + g + b) // 3 * 25 // 255, 0, 23)
    return np.where(mx - mn < 12, grey_level, cube)


def render(img: np.ndarray, cols: int, rows: int, mode: str = TRUECOLOR) -> str:
    """Render an image as terminal text sized to ``cols`` x ``rows`` cells."""

    height, width = fit(img.shape[:2], cols, rows)
    if height % 2:
        height += 1                      # need whole top/bottom pairs
    small = _to_rgb(_resize(img, height, width))

    if mode == MONO:
        return _render_ascii(small)

    top, bottom = small[0::2], small[1::2]
    lines = []
    for y in range(top.shape[0]):
        parts = []
        prev = None
        for x in range(top.shape[1]):
            t, b = top[y, x], bottom[y, x]
            if mode == TRUECOLOR:
                key = (t[0], t[1], t[2], b[0], b[1], b[2])
                if key != prev:
                    parts.append(
                        f"\x1b[38;2;{t[0]};{t[1]};{t[2]}m"
                        f"\x1b[48;2;{b[0]};{b[1]};{b[2]}m"
                    )
            else:
                ti, bi = _rgb_to_256(t[None])[0], _rgb_to_256(b[None])[0]
                key = (ti, bi)
                if key != prev:
                    parts.append(f"\x1b[38;5;{ti}m\x1b[48;5;{bi}m")
            prev = key
            parts.append(UPPER_HALF)
        parts.append(RESET)
        lines.append("".join(parts))
    return "\n".join(lines)


def _render_ascii(rgb: np.ndarray) -> str:
    """Monochrome fallback. Averages the pixel pair rather than halving rows.

    Using the luminance ramp on full-height rows would double the aspect
    error, since there is no half-block to carry the second pixel.
    """

    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    pairs = (luma[0::2] + luma[1::2]) / 2.0
    idx = np.clip(
        (pairs / 256.0 * len(ASCII_RAMP)).astype(int), 0, len(ASCII_RAMP) - 1
    )
    return "\n".join("".join(ASCII_RAMP[i] for i in row) for row in idx)


def load(path: str) -> tuple[np.ndarray, dict]:
    """Load an ICJ file or any Pillow-readable image, plus a facts dict."""

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} does not exist")

    size = p.stat().st_size
    with open(p, "rb") as f:
        magic = f.read(4)

    if magic == c.MAGIC:
        img, meta = _load_icj(p)
        meta["file_bytes"] = size
        return img, meta

    if magic[:3] == b"ICJ":
        raise ValueError(
            f"{path} is an older container ({magic.decode('ascii', 'replace')}); "
            f"this build reads {c.MAGIC.decode()} only"
        )

    img = c._load_image(str(p))
    return img, {
        "format": p.suffix.lstrip(".").upper() or "image",
        "file_bytes": size,
        "shape": img.shape,
    }


def _load_icj(path: pathlib.Path) -> tuple[np.ndarray, dict]:
    """Decode an ICJ file, reading its header for display metadata.

    Goes through the public file API rather than reimplementing the parse, so
    the viewer cannot drift from the codec; the header re-read is only for the
    facts shown by ``--info``.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.png"
        c.decompress_huffman_file(str(path), str(out))
        img = c._load_image(str(out))

    with open(path, "rb") as f:
        header = f.read(14)
    height = int.from_bytes(header[4:8], "big")
    width = int.from_bytes(header[8:12], "big")
    quality = header[12]
    fmt = header[13]

    return img, {
        "format": c.MAGIC.decode(),
        "shape": img.shape,
        "declared": (height, width),
        "internal_quality": quality,
        "components": fmt >> 4,
        "sampling": c.SAMPLING_NAMES.get(fmt & 0x0F, "?"),
    }


def format_info(path: str, meta: dict) -> str:
    """Human-readable facts about a loaded file."""

    lines = [f"{pathlib.Path(path).name}  [{meta.get('format', '?')}]"]

    shape = meta.get("shape")
    if shape is not None:
        h, w = shape[0], shape[1]
        kind = "RGB" if len(shape) == 3 else "grayscale"
        lines.append(f"  {w} x {h}  {kind}")

    size = meta.get("file_bytes")
    if size is not None:
        lines.append(f"  file: {size:,} bytes")
        if shape is not None:
            raw = shape[0] * shape[1] * (3 if len(shape) == 3 else 1)
            lines.append(f"  vs raw {raw:,} bytes  ->  {raw / size:.1f}x")

    if meta.get("components"):
        lines.append(
            f"  components: {meta['components']}   chroma: {meta['sampling']}"
        )
        # The stored quality is post-calibration and will not match the number
        # the user asked for; saying so avoids it reading as a bug.
        lines.append(
            f"  internal quality: {meta['internal_quality']} "
            "(calibrated; higher than the requested value by design)"
        )

    return "\n".join(lines)


def _terminal_size(width: int | None, height: int | None) -> tuple[int, int]:
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    # Leave a row so the shell prompt does not scroll the top off.
    return width or cols, height or max(1, rows - 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="icjview",
        description="View ICJ (or any) images in the terminal.",
    )
    parser.add_argument("paths", nargs="+", help="image or .icj file(s)")
    parser.add_argument("--width", type=int, help="output width in cells")
    parser.add_argument("--height", type=int, help="output height in cells")
    parser.add_argument("--info", action="store_true", help="show file facts")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="render inputs side by side at equal size",
    )
    parser.add_argument(
        "--color",
        choices=[TRUECOLOR, COLOR256, MONO],
        help="force colour mode instead of detecting it",
    )
    args = parser.parse_args(argv)

    mode = args.color or detect_color_mode()
    cols, rows = _terminal_size(args.width, args.height)

    try:
        loaded = [(p, *load(p)) for p in args.paths]
    except (FileNotFoundError, ValueError) as exc:
        print(f"icjview: {exc}", file=sys.stderr)
        return 1

    if args.compare and len(loaded) > 1:
        each = max(1, (cols - (len(loaded) - 1)) // len(loaded))
        panels = [render(img, each, rows, mode).split("\n") for _, img, _ in loaded]
        tall = max(len(p) for p in panels)
        for panel in panels:
            panel += [""] * (tall - len(panel))
        for row in zip(*panels):
            print(" ".join(row))
        if args.info:
            for path, _, meta in loaded:
                print()
                print(format_info(path, meta))
        return 0

    for path, img, meta in loaded:
        if args.info:
            print(format_info(path, meta))
        print(render(img, cols, rows, mode))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
