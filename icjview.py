"""Terminal viewer for ICJ files (and any image Pillow can read).

    icjview photo.icj
    icjview photo.icj --info
    icjview a.icj b.png --compare      # side by side
    icjview photo.icj --render blocks  # force a rendering path
    python -m icjview photo.icj

**Rendering is detected, not assumed**, on two independent axes.

*Transmission.* Terminals that speak a graphics protocol get real pixels:
kitty (also Ghostty and WezTerm, which implement the same protocol) and
iTerm2. There the viewer is limited by the window, not by the character grid,
which is roughly a 10x resolution gain -- an 80x24 terminal offers a 160x48
sub-cell canvas but well over 1000x700 actual pixels.

Everything else falls back to sub-cell glyph rendering: quadrants (2x2 pixels
per cell) by default, sextants (2x3, Unicode 13) on request, half-blocks
(1x2) for older fonts, then ASCII. Quadrants rather than sextants is a font
coverage decision -- a missing glyph renders worse than a coarser one.

*Colour depth.* 24-bit if the terminal advertises it, else the 256-colour
cube, else monochrome ASCII. A viewer that emits truecolor escapes into a
terminal that cannot render them produces confetti, which is worse than a
plain fallback. Piping to a non-TTY and ``NO_COLOR`` both force ASCII, ahead
of everything above, so redirected output never gets escape soup injected.

Quality is measured, not asserted: ``bench/viewer_quality.py`` scores a render
in CIEDE2000 against a linear-light reference, and every choice here had to
beat the alternative on the 11-image corpus. One did not survive that -- see
the note above :func:`_nearest_palette` on why the colour paths do not dither.

Two details that are easy to get wrong and are handled here (SPEC 10.0):
the xterm-256 colour cube is **not** linear -- its levels are
0/95/135/175/215/255 -- and resampling must happen in **linear light**, since
averaging gamma-encoded values darkens exactly the detail-dense regions a
large downscale is made of.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import functools
import io
import os
import pathlib
import shutil
import struct
import sys
import termios
import zlib

import numpy as np
from PIL import Image

import compression as c

UPPER_HALF = "▀"
FULL_BLOCK = "█"
RESET = "\x1b[0m"

# Densest-to-lightest ramp for the monochrome fallback.
ASCII_RAMP = "@%#*+=-:. "

# Colour depth.
TRUECOLOR = "truecolor"
COLOR256 = "256"
MONO = "mono"

# Transmission / rendering path.
KITTY = "kitty"
ITERM = "iterm"
SEXTANT = "sextant"
QUADRANT = "quadrant"
BLOCKS = "blocks"
ASCII = "ascii"
AUTO = "auto"

RENDER_KINDS = (AUTO, KITTY, ITERM, SEXTANT, QUADRANT, BLOCKS, ASCII)

# Pixels per cell for each glyph path: (horizontal, vertical).
SUBDIVISIONS = {
    BLOCKS: (1, 2),
    QUADRANT: (2, 2),
    SEXTANT: (2, 3),
    ASCII: (1, 2),
}

# Fallback when the terminal will not report its pixel geometry.
DEFAULT_CELL_ASPECT = 2.0
DEFAULT_CELL_PIXELS = (8, 16)


# --- terminal capabilities ---------------------------------------------------


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


def _supports_kitty() -> bool:
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    if "kitty" in os.environ.get("TERM", ""):
        return True
    # Ghostty and WezTerm implement the same protocol.
    return os.environ.get("TERM_PROGRAM", "").lower() in ("ghostty", "wezterm")


def _supports_iterm() -> bool:
    if os.environ.get("TERM_PROGRAM", "") == "iTerm.app":
        return True
    return os.environ.get("LC_TERMINAL", "") == "iTerm2"


def detect_render_kind(stream=None, color_mode: str | None = None) -> str:
    """Best transmission path for this terminal.

    Environment variables only -- no terminal query, no raw mode, no blocking
    read. SPEC 10.2.1 cut the Device Attributes probe (and sixel with it):
    it was the one piece of this viewer that could hang on terminal I/O, and
    it existed solely to detect a protocol no target terminal needs.
    """

    stream = stream or sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return ASCII
    if (color_mode or detect_color_mode(stream)) == MONO:
        return ASCII
    if _supports_kitty():
        return KITTY
    if _supports_iterm():
        return ITERM
    return QUADRANT


def _winsize(stream=None) -> tuple[int, int, int, int] | None:
    """``(rows, cols, xpixel, ypixel)`` from the TTY, or None."""

    for candidate in (stream, sys.stdout, sys.stderr):
        if candidate is None or not hasattr(candidate, "fileno"):
            continue
        try:
            packed = fcntl.ioctl(candidate.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        except (OSError, ValueError, io.UnsupportedOperation):
            continue
        rows, cols, xpixel, ypixel = struct.unpack("HHHH", packed)
        if rows and cols:
            return rows, cols, xpixel, ypixel
    return None


def cell_pixels(stream=None) -> tuple[int, int]:
    """Pixel size of one character cell as ``(width, height)``.

    Many terminals leave ``ws_xpixel``/``ws_ypixel`` zeroed, so the result is
    a guess often enough that callers must not treat it as exact.
    """

    size = _winsize(stream)
    if size is None:
        return DEFAULT_CELL_PIXELS
    rows, cols, xpixel, ypixel = size
    if not xpixel or not ypixel:
        return DEFAULT_CELL_PIXELS
    return max(1, xpixel // cols), max(1, ypixel // rows)


def cell_aspect(stream=None) -> float:
    """Cell height divided by cell width.

    Assuming exactly 2.0 stretches every image slightly; real cells run
    roughly 2.1-2.4. The true value is available whenever the terminal fills
    in its pixel geometry, so the constant is a fallback, not a model.
    """

    size = _winsize(stream)
    if size is None:
        return DEFAULT_CELL_ASPECT
    rows, cols, xpixel, ypixel = size
    if not xpixel or not ypixel:
        return DEFAULT_CELL_ASPECT
    aspect = (ypixel / rows) / (xpixel / cols)
    # Guard against nonsense from terminals that report partial geometry.
    return aspect if 1.0 <= aspect <= 4.0 else DEFAULT_CELL_ASPECT


# --- colour ------------------------------------------------------------------


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB transfer, inverted. Input and output in [0, 1].

    The piecewise standard curve, not a 2.2 power approximation: the linear
    segment matters near black, which is where a heavy downscale spends much
    of its time.
    """

    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# xterm's 6x6x6 cube is not linear in 0..255. Assuming it is (the old
# ``r * 5 // 255``) costs 40% of the achievable colour accuracy; see SPEC 10.0.
CUBE_LEVELS = np.array([0, 95, 135, 175, 215, 255], dtype=np.int16)


@functools.lru_cache(maxsize=1)
def _xterm_palette() -> tuple[np.ndarray, np.ndarray]:
    """The 240 addressable xterm colours as ``(rgb, index)``.

    The 16 ANSI colours are excluded deliberately: they are theme-dependent,
    so a renderer that used them would produce a different picture in every
    colour scheme.
    """

    r, g, b = np.meshgrid(CUBE_LEVELS, CUBE_LEVELS, CUBE_LEVELS, indexing="ij")
    cube = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)

    greys = (8 + 10 * np.arange(24, dtype=np.int16))
    grey = np.repeat(greys[:, None], 3, axis=1)

    rgb = np.concatenate([cube, grey], axis=0).astype(np.uint8)
    idx = np.concatenate([np.arange(16, 232), np.arange(232, 256)]).astype(np.int32)
    return rgb, idx


@functools.lru_cache(maxsize=1)
def _palette_linear() -> np.ndarray:
    return srgb_to_linear(_xterm_palette()[0].astype(np.float32) / 255.0)


def _nearest_palette(rgb_linear: np.ndarray) -> np.ndarray:
    """Index into the 240-colour palette, nearest in linear light.

    ``rgb_linear`` is ``(N, 3)``. The canvas is a few thousand pixels at most,
    so the direct N x 240 search beats a lookup table on both accuracy and
    total time -- the table's build cost is never amortized.
    """

    pal = _palette_linear()
    d = ((rgb_linear[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def _rgb_to_256(rgb: np.ndarray) -> np.ndarray:
    """Map uint8 RGB ``(..., 3)`` to xterm-256 indices, nearest match.

    Greys are ordinary candidates in the same search rather than a special
    case; the old ``max - min < 12`` heuristic decided *which palette* to use
    before checking whether it was closer.
    """

    arr = np.asarray(rgb)
    flat = arr.reshape(-1, 3).astype(np.float32) / 255.0
    nearest = _nearest_palette(srgb_to_linear(flat))
    return _xterm_palette()[1][nearest].reshape(arr.shape[:-1])


# Error diffusion is deliberately absent from the colour cell paths, and that
# is a measured decision rather than an omission (SPEC 10.1.4).
#
# Floyd-Steinberg was implemented here first, run over the per-cell foreground
# and background colours. It made the picture visibly worse: heavy speckle,
# and 11 of 11 corpus images scored worse than plain nearest-palette. The
# reason is that ``fg`` and ``bg`` are not images. They are two colour lists
# that the glyph mask interleaves back together spatially, so diffusing error
# through each plane independently produces two incoherent error fields that
# recombine as noise.
#
# Dithering across cells is not needed anyway: the cell mask already carries
# spatial modulation at sub-cell resolution, which is the same job. The ASCII
# path *does* dither, because there the canvas really is an image.


@functools.lru_cache(maxsize=1)
def _palette_rgb_by_code() -> np.ndarray:
    """256-entry uint8 table so a palette index can be painted back to RGB."""

    table = np.zeros((256, 3), dtype=np.uint8)
    rgb, idx = _xterm_palette()
    table[idx] = rgb
    return table


# --- geometry ----------------------------------------------------------------


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Present any decoded image as (H, W, 3) uint8."""

    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    return img


def fit(
    shape: tuple[int, int],
    cols: int,
    rows: int,
    sub_x: int = 1,
    sub_y: int = 2,
    aspect: float = DEFAULT_CELL_ASPECT,
) -> tuple[int, int]:
    """Pixel dimensions that fit ``cols`` x ``rows`` cells, preserving aspect.

    Each cell carries ``sub_x`` by ``sub_y`` pixels, so the usable canvas is
    ``cols * sub_x`` by ``rows * sub_y``. One canvas pixel is ``aspect *
    sub_x / sub_y`` times taller than it is wide; scaling the two axes by the
    same factor would then distort everything but the half-block case, which
    is why the vertical factor is divided by that ratio.
    """

    h, w = shape
    pixel_aspect = aspect * sub_x / sub_y
    max_w, max_h = cols * sub_x, rows * sub_y

    scale = min(max_w / w, max_h * pixel_aspect / h, 1.0)
    return max(1, int(round(h * scale / pixel_aspect))), max(1, int(round(w * scale)))


def _resize(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resample in linear light.

    Pillow filters average the values it is given; handing it gamma-encoded
    sRGB makes every downscale darker than the truth, worst where detail is
    densest. Decoding first costs one float conversion and fixes it.
    """

    if img.shape[0] == height and img.shape[1] == width:
        return img

    rgb = _to_rgb(img)
    linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)

    # Pillow resamples 'F' one band at a time; three passes keeps the
    # arithmetic in float and avoids a round trip through uint8.
    bands = [
        np.array(
            Image.fromarray(linear[..., k], mode="F").resize(
                (width, height), Image.LANCZOS
            )
        )
        for k in range(3)
    ]
    out = linear_to_srgb(np.stack(bands, axis=-1))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _canvas(
    img: np.ndarray, cols: int, rows: int, kind: str, aspect: float
) -> np.ndarray:
    """Resized RGB canvas, padded to whole cells for ``kind``."""

    sub_x, sub_y = SUBDIVISIONS[kind]
    height, width = fit(img.shape[:2], cols, rows, sub_x, sub_y, aspect)

    height += (-height) % sub_y
    width += (-width) % sub_x
    return _to_rgb(_resize(img, height, width))


# --- glyph tables ------------------------------------------------------------

# Quadrant bit order: 1=TL, 2=TR, 4=BL, 8=BR.
QUADRANT_GLYPHS = (
    " ", "▘", "▝", "▀", "▖", "▌", "▞", "▛",
    "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█",
)


@functools.lru_cache(maxsize=1)
def _sextant_glyphs() -> tuple[str, ...]:
    """2x3 sextants, indexed by a 6-bit mask (bit 0 = top-left, row-major).

    U+1FB00..U+1FB3B covers masks 1..62 *except* 21 and 42, the two that
    coincide with the pre-existing half-width blocks U+258C and U+2590.
    """

    glyphs, offset = [], 0
    for mask in range(64):
        if mask == 0:
            glyphs.append(" ")
        elif mask == 63:
            glyphs.append(FULL_BLOCK)
        elif mask == 21:
            glyphs.append("▌")
            offset += 1
        elif mask == 42:
            glyphs.append("▐")
            offset += 1
        else:
            glyphs.append(chr(0x1FB00 + mask - 1 - offset))
    return tuple(glyphs)


# --- sub-cell decomposition --------------------------------------------------


def _split_cells(canvas: np.ndarray, sub_x: int, sub_y: int) -> np.ndarray:
    """Reshape a canvas into ``(cell_rows, cell_cols, sub_y * sub_x, 3)``."""

    h, w = canvas.shape[:2]
    return (
        canvas.reshape(h // sub_y, sub_y, w // sub_x, sub_x, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(h // sub_y, w // sub_x, sub_y * sub_x, 3)
    )


def _two_colour_split(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each cell's pixels into a foreground/background pair.

    A cell can show exactly two colours, so the question is which bipartition
    of its pixels loses least. Thresholding on luma is the obvious answer and
    the wrong one: it cannot separate two colours of equal brightness, which
    is precisely the case where a wrong split is most visible.

    Instead, one power iteration on the cell's colour covariance, seeded from
    luma. That converges to the principal axis for the strongly bimodal cells
    that matter, degrades to the luma answer when colour variance is
    isotropic, and stays a handful of vectorized ops.

    Returns ``(mask, fg, bg)`` with ``mask`` a bitfield over sub-pixels.
    """

    pix = srgb_to_linear(cells.astype(np.float32) / 255.0)
    mean = pix.mean(axis=2, keepdims=True)
    centered = pix - mean

    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    seed = centered @ luma
    axis = np.einsum("ijkc,ijk->ijc", centered, seed)

    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(norm > 1e-8, axis / np.maximum(norm, 1e-8), luma)

    proj = np.einsum("ijkc,ijc->ijk", centered, axis)
    hot = proj >= 0

    # A cell whose pixels are all one colour has an arbitrary split; forcing
    # it to all-background keeps the glyph a space and the colour exact.
    flat = np.ptp(proj, axis=2) < 1e-6
    hot &= ~flat[..., None]

    counts = hot.sum(axis=2)
    n = hot.shape[2]
    fg = np.where(
        counts[..., None] > 0,
        (pix * hot[..., None]).sum(axis=2) / np.maximum(counts[..., None], 1),
        mean[:, :, 0],
    )
    cold = n - counts
    bg = np.where(
        cold[..., None] > 0,
        (pix * ~hot[..., None]).sum(axis=2) / np.maximum(cold[..., None], 1),
        mean[:, :, 0],
    )

    bits = (1 << np.arange(n, dtype=np.int32))
    mask = (hot * bits).sum(axis=2)

    to8 = lambda a: np.clip(np.round(linear_to_srgb(a) * 255.0), 0, 255).astype(np.uint8)
    return mask, to8(fg), to8(bg)


# --- escape emission ---------------------------------------------------------


def _paint(fg: np.ndarray, bg: np.ndarray, glyphs, mode: str) -> str:
    """Emit coloured glyphs, re-stating colour only when it changes."""

    if mode == COLOR256:
        fg_idx = _rgb_to_256(fg)
        bg_idx = _rgb_to_256(bg)

    lines = []
    for y in range(fg.shape[0]):
        parts, prev = [], None
        for x in range(fg.shape[1]):
            if mode == TRUECOLOR:
                t, b = fg[y, x], bg[y, x]
                key = (t[0], t[1], t[2], b[0], b[1], b[2])
                escape = (
                    f"\x1b[38;2;{t[0]};{t[1]};{t[2]}m"
                    f"\x1b[48;2;{b[0]};{b[1]};{b[2]}m"
                )
            else:
                key = (fg_idx[y, x], bg_idx[y, x])
                escape = f"\x1b[38;5;{key[0]}m\x1b[48;5;{key[1]}m"

            if key != prev:
                parts.append(escape)
            prev = key
            parts.append(glyphs[y][x])
        parts.append(RESET)
        lines.append("".join(parts))
    return "\n".join(lines)


def _render_cells(img, cols, rows, mode, kind, aspect) -> str:
    canvas = _canvas(img, cols, rows, kind, aspect)
    sub_x, sub_y = SUBDIVISIONS[kind]

    cells = _split_cells(canvas, sub_x, sub_y)
    mask, fg, bg = _two_colour_split(cells)

    table = QUADRANT_GLYPHS if kind == QUADRANT else _sextant_glyphs()
    if kind == BLOCKS:
        # 1x2 cells: bit 0 is the top pixel, so the mask is already the
        # half-block selector.
        table = (" ", UPPER_HALF, "▄", FULL_BLOCK)

    glyphs = [[table[m] for m in row] for row in mask]
    return _paint(fg, bg, glyphs, mode)


def _render_ascii(img, cols, rows, aspect) -> str:
    """Monochrome fallback, dithered.

    Averages each pixel pair rather than halving rows: with no half-block to
    carry the second pixel, using full-height rows would double the aspect
    error.
    """

    canvas = _canvas(img, cols, rows, ASCII, aspect)
    linear = srgb_to_linear(canvas.astype(np.float32) / 255.0)
    luma = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    work = (luma[0::2] + luma[1::2]) / 2.0

    n = len(ASCII_RAMP)
    levels = srgb_to_linear(np.arange(n, dtype=np.float32) / (n - 1))

    h, w = work.shape
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            old = work[y, x]
            i = int(np.abs(levels - old).argmin())
            row.append(ASCII_RAMP[i])
            err = old - levels[i]

            if x + 1 < w:
                work[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x:
                    work[y + 1, x - 1] += err * (3 / 16)
                work[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1 / 16)
        out.append("".join(row))
    return "\n".join(out)


# --- graphics protocols ------------------------------------------------------


def _graphics_canvas(img, cols, rows, aspect, cell) -> tuple[np.ndarray, int, int]:
    """Downscale to the cell box's true pixel size, and the box it occupies.

    Scaling to the viewport rather than shipping the original keeps the
    payload bounded while giving up nothing: the terminal cannot show more
    pixels than the box has.
    """

    cell_w, cell_h = cell
    px_cols, px_rows = cols * cell_w, rows * cell_h

    h, w = img.shape[:2]
    scale = min(px_cols / w, px_rows / h, 1.0)
    height, width = max(1, round(h * scale)), max(1, round(w * scale))

    small = _resize(_to_rgb(img), height, width)
    box_cols = max(1, min(cols, -(-width // cell_w)))
    box_rows = max(1, min(rows, -(-height // cell_h)))
    return small, box_cols, box_rows


def render_kitty(img, cols, rows, aspect=DEFAULT_CELL_ASPECT, cell=None) -> str:
    """Kitty graphics protocol: transmit RGB pixels and display in place.

    ``C=1`` keeps the terminal from moving the cursor, so the caller decides
    how far to advance instead of depending on placement semantics that
    differ between the terminals implementing this protocol.
    """

    cell = cell or DEFAULT_CELL_PIXELS
    small, box_cols, box_rows = _graphics_canvas(img, cols, rows, aspect, cell)
    height, width = small.shape[:2]

    payload = base64.b64encode(zlib.compress(small.tobytes())).decode("ascii")
    control = (
        f"a=T,f=24,t=d,o=z,q=2,C=1,"
        f"s={width},v={height},c={box_cols},r={box_rows}"
    )

    chunks = [payload[i : i + 4096] for i in range(0, len(payload), 4096)] or [""]
    out = []
    for n, chunk in enumerate(chunks):
        more = 1 if n < len(chunks) - 1 else 0
        head = f"{control},m={more}" if n == 0 else f"m={more}"
        out.append(f"\x1b_G{head};{chunk}\x1b\\")

    return "".join(out) + "\n" * box_rows


def render_iterm(img, cols, rows, aspect=DEFAULT_CELL_ASPECT, cell=None) -> str:
    """iTerm2 inline image protocol: a base64 PNG in an OSC 1337 sequence."""

    cell = cell or DEFAULT_CELL_PIXELS
    small, box_cols, box_rows = _graphics_canvas(img, cols, rows, aspect, cell)

    buf = io.BytesIO()
    Image.fromarray(small, mode="RGB").save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")

    return (
        f"\x1b]1337;File=inline=1;size={len(buf.getvalue())};"
        f"width={box_cols};height={box_rows};preserveAspectRatio=1:"
        f"{payload}\x07" + "\n" * box_rows
    )


# --- public rendering --------------------------------------------------------


def render(
    img: np.ndarray,
    cols: int,
    rows: int,
    mode: str = TRUECOLOR,
    kind: str = BLOCKS,
    aspect: float = DEFAULT_CELL_ASPECT,
    cell: tuple[int, int] | None = None,
) -> str:
    """Render an image as terminal text sized to ``cols`` x ``rows`` cells."""

    if mode == MONO:
        return _render_ascii(img, cols, rows, aspect)
    if kind == KITTY:
        return render_kitty(img, cols, rows, aspect, cell)
    if kind == ITERM:
        return render_iterm(img, cols, rows, aspect, cell)
    if kind == ASCII:
        return _render_ascii(img, cols, rows, aspect)
    return _render_cells(img, cols, rows, mode, kind, aspect)


def rasterize(
    img: np.ndarray,
    cols: int,
    rows: int,
    mode: str = TRUECOLOR,
    kind: str = BLOCKS,
    aspect: float = DEFAULT_CELL_ASPECT,
) -> np.ndarray:
    """The pixel buffer a render would paint, at canvas resolution.

    This is what makes viewer quality measurable (SPEC 10.6): it reproduces
    the renderer's decisions -- resampling, two-colour-per-cell split, palette
    quantization, dithering -- and hands back something a colour metric can
    be run against. Kept beside the renderer, not in the benchmark, because a
    copy that drifted from the real code would measure nothing.
    """

    if kind in (KITTY, ITERM):
        cell = DEFAULT_CELL_PIXELS
        return _graphics_canvas(img, cols, rows, aspect, cell)[0]

    if mode == MONO or kind == ASCII:
        text = _render_ascii(img, cols, rows, aspect)
        lut = np.array(
            [ASCII_RAMP.index(ch) / (len(ASCII_RAMP) - 1) * 255 for ch in ASCII_RAMP]
        )
        rowsy = [[lut[ASCII_RAMP.index(ch)] for ch in line] for line in text.split("\n")]
        grey = np.array(rowsy, dtype=np.uint8)
        return np.repeat(np.repeat(grey, 2, axis=0)[..., None], 3, axis=2)

    canvas = _canvas(img, cols, rows, kind, aspect)
    sub_x, sub_y = SUBDIVISIONS[kind]
    cells = _split_cells(canvas, sub_x, sub_y)
    mask, fg, bg = _two_colour_split(cells)

    if mode == COLOR256:
        table = _palette_rgb_by_code()
        fg = table[_rgb_to_256(fg)]
        bg = table[_rgb_to_256(bg)]

    n = sub_x * sub_y
    bits = ((mask[..., None] >> np.arange(n)) & 1).astype(bool)
    painted = np.where(bits[..., None], fg[:, :, None, :], bg[:, :, None, :])

    ch, cw = mask.shape
    return (
        painted.reshape(ch, cw, sub_y, sub_x, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(ch * sub_y, cw * sub_x, 3)
    )


# --- loading -----------------------------------------------------------------


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

    Goes through the codec's own decode path rather than reimplementing the
    parse, so the viewer cannot drift from the codec; the header re-read is
    only for the facts shown by ``--info``.
    """

    with open(path, "rb") as f:
        header = f.read(14)
    height = int.from_bytes(header[4:8], "big")
    width = int.from_bytes(header[8:12], "big")
    quality = header[12]
    fmt = header[13]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.png"
        c.decompress_huffman_file(str(path), str(out))
        img = c._load_image(str(out))

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


# --- CLI ---------------------------------------------------------------------


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
        help="force colour depth instead of detecting it",
    )
    parser.add_argument(
        "--render",
        choices=list(RENDER_KINDS),
        default=AUTO,
        help="force a rendering path instead of detecting it",
    )
    parser.add_argument(
        "--cell-aspect",
        type=float,
        help="cell height / width, when the terminal will not report it",
    )
    args = parser.parse_args(argv)

    mode = args.color or detect_color_mode()
    kind = args.render if args.render != AUTO else detect_render_kind(color_mode=mode)
    aspect = args.cell_aspect or cell_aspect()
    cell = cell_pixels()
    cols, rows = _terminal_size(args.width, args.height)

    try:
        loaded = [(p, *load(p)) for p in args.paths]
    except (FileNotFoundError, ValueError) as exc:
        print(f"icjview: {exc}", file=sys.stderr)
        return 1

    if args.compare and len(loaded) > 1:
        # Graphics protocols paint a bitmap at the cursor; there are no text
        # rows to interleave, so side-by-side needs a cell-based path.
        if kind in (KITTY, ITERM):
            kind = QUADRANT

        each = max(1, (cols - (len(loaded) - 1)) // len(loaded))
        panels = [
            render(img, each, rows, mode, kind, aspect, cell).split("\n")
            for _, img, _ in loaded
        ]
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
        print(render(img, cols, rows, mode, kind, aspect, cell), end="")
        if kind not in (KITTY, ITERM):
            print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
