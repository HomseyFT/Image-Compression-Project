"""Fidelity metric for the terminal viewer (SPEC 10.6).

Viewer quality had no metric, which is why the defects in SPEC 10.0 survived:
the 23 viewer tests pin geometry and colour-mode detection, and nothing was
measuring whether the picture was *right*. This module supplies the missing
number so that 10.1.1, 10.1.2 and 10.3 can be gated rather than asserted.

The method: rasterize what the escape sequences would actually paint, then
compare it against a reference resample of the source at the same canvas
size. The reference is resampled in linear light, because that is the
ground truth a correct renderer should approach -- comparing against an
sRGB-space resample would bake in the very error 10.1.2 exists to remove.

Reported as mean CIEDE2000 (lower is better; a value near 1.0 is the
just-noticeable threshold) and PSNR. deltaE is the gate: PSNR is reported
because it is comparable with the rest of the project, but it is an RGB
metric and understates colour errors that are perceptually obvious.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import icjview as v
from icjview import srgb_to_linear


# --- colour science ----------------------------------------------------------

# D65, the sRGB reference white.
_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 sRGB (..., 3) -> CIE L*a*b* float64 (..., 3)."""

    linear = v.srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE

    # The CIE cube-root transfer, with its linear segment near black.
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(np.clip(xyz, 0, None)), (kappa * xyz + 16) / 116)

    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference, elementwise over the leading axes.

    Full formulation including the rotation term -- the simplified variants
    diverge exactly in the blue region, which is where 256-colour cube
    quantization does its worst work, so the shortcut would flatter the thing
    being measured.
    """

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7 + 1e-12)))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2

    hsum, hdiff = h1p + h2p, np.abs(h1p - h2p)
    hp_bar = np.where(
        C1p * C2p == 0,
        hsum,
        np.where(
            hdiff <= 180,
            hsum / 2,
            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )

    S_L = 1 + (0.015 * (Lp_bar - 50) ** 2) / np.sqrt(20 + (Lp_bar - 50) ** 2)
    S_C = 1 + 0.045 * Cp_bar
    S_H = 1 + 0.015 * Cp_bar * T

    R_T = (
        -2
        * np.sqrt(Cp_bar**7 / (Cp_bar**7 + 25.0**7 + 1e-12))
        * np.sin(np.radians(60 * np.exp(-(((hp_bar - 275) / 25) ** 2))))
    )

    return np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


# --- the metric --------------------------------------------------------------


# Cell size for scoring. 12x24 is divisible by every subdivision in use
# (1, 2 and 3), so each mode's canvas expands to the same physical grid by an
# exact integer factor and no resampling error is smuggled into the metric.
SCORING_CELL = (12, 24)


def reference(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Ground truth: the source resampled in linear light to a given size.

    Deliberately **not** ``v._resize``, despite computing the same thing.
    Routing the reference through the renderer's own resampler makes the
    ground truth move whenever the renderer is swapped out, so a variant is
    scored against a reference built with its own assumptions -- and every
    variant then looks self-consistently correct. That is how an early run of
    this metric "showed" gamma-naive resampling beating linear light across
    the corpus: both sides had silently agreed on what the truth was.

    It is the same failure recorded in the phase 4 fitting notes (scoring
    against a hull that moved with the thing under test), so it gets the same
    treatment: the reference is fixed, and the renderer is the only variable.
    """

    rgb = v._to_rgb(img)
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb

    linear = srgb_to_linear(rgb.astype(np.float32) / 255.0)
    bands = [
        np.array(
            Image.fromarray(linear[..., k], mode="F").resize(
                (width, height), Image.LANCZOS
            )
        )
        for k in range(3)
    ]
    out = v.linear_to_srgb(np.stack(bands, axis=-1))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def _physical(painted: np.ndarray, mode: str, kind: str, cell) -> np.ndarray:
    """Expand a canvas to the pixels the terminal would actually light up.

    A canvas pixel is not a point sample -- it is a solid rectangle covering
    its share of a character cell. Half-blocks paint 12x12 rectangles,
    sextants 6x8, and the graphics protocols paint single pixels. Comparing
    each mode against a reference at *its own* canvas size therefore measures
    the wrong thing: a two-colour cell holding two pixels reproduces itself
    exactly, so half-blocks score a perfect 0.0 while every finer mode is
    penalised for having more to get wrong. The metric would reward
    resolution loss.

    Expanding to the common physical grid first is what makes the modes
    comparable, and it is also what the eye sees.
    """

    if kind in (v.KITTY, v.ITERM):
        return painted

    cell_w, cell_h = cell
    sub_x, sub_y = v.SUBDIVISIONS[v.ASCII if mode == v.MONO else kind]
    return np.repeat(np.repeat(painted, cell_h // sub_y, axis=0), cell_w // sub_x, axis=1)


def score(
    img: np.ndarray, cols: int, rows: int, mode: str, kind: str, cell=SCORING_CELL
) -> dict:
    """Fidelity of one render, on the physical grid the terminal displays.

    Returns mean/95th-percentile CIEDE2000 and PSNR. The 95th percentile is
    reported because a mean hides banding, and ``block_delta_e`` because
    dithering deliberately trades per-pixel error for low-frequency accuracy
    -- judged pixelwise it looks worse than the banding it removes.
    """

    aspect = cell[1] / cell[0]
    painted = v.rasterize(img, cols, rows, mode=mode, kind=kind, aspect=aspect)
    shown = _physical(painted, mode, kind, cell)
    ref = reference(img, shown.shape[0], shown.shape[1])

    de = delta_e_2000(srgb_to_lab(shown), srgb_to_lab(ref))
    return {
        "delta_e": float(de.mean()),
        "delta_e_p95": float(np.percentile(de, 95)),
        "block_delta_e": block_delta_e(shown, ref),
        "psnr": psnr(shown, ref),
        "shape": shown.shape[:2],
    }


def canvas_score(
    img: np.ndarray, cols: int, rows: int, mode: str, kind: str, aspect: float = 2.0
) -> dict:
    """Fidelity at *canvas* resolution, for comparing components within a mode.

    Two instruments, two questions, and using the wrong one inverts the answer:

    * :func:`score` expands to the physical grid, so modes with different
      canvas sizes are comparable. Use it to choose **between** modes.
    * This one compares at the canvas the mode actually produces. Use it to
      judge a **component** -- resampler, palette, cell split -- with the mode
      held fixed.

    The physical grid carries a large blockiness term: a 12x-expanded canvas
    is being compared against a reference that still has all its detail, and
    that term dwarfs anything a resampler does. Measured that way,
    gamma-naive resampling appears to *beat* linear light on 11 of 11 corpus
    images. Measured at canvas resolution, where both sides produce the same
    number of samples, linear light wins 11 of 11 in every glyph mode. The
    first result is an artifact of asking a within-mode question with a
    cross-mode instrument.

    Note that a mode whose cells can represent themselves exactly -- half
    blocks, with two colours for two pixels -- scores 0.0 here. That is
    correct and is exactly why this instrument must not be used to rank modes.
    """

    painted = v.rasterize(img, cols, rows, mode=mode, kind=kind, aspect=aspect)
    ref = reference(img, painted.shape[0], painted.shape[1])

    de = delta_e_2000(srgb_to_lab(painted), srgb_to_lab(ref))
    return {
        "delta_e": float(de.mean()),
        "delta_e_p95": float(np.percentile(de, 95)),
        "block_delta_e": block_delta_e(painted, ref),
        "psnr": psnr(painted, ref),
        "shape": painted.shape[:2],
    }


def block_delta_e(a: np.ndarray, b: np.ndarray, block: int = 8) -> float:
    """Mean CIEDE2000 between block averages -- a crude model of the eye.

    Dithering raises pixelwise error on purpose: it scatters quantization
    error into high frequencies the viewer integrates away. Measured
    pixelwise it is a regression; measured over local means it is the win it
    actually is.
    """

    h = a.shape[0] // block * block
    w = a.shape[1] // block * block
    if h == 0 or w == 0:
        return float(delta_e_2000(srgb_to_lab(a), srgb_to_lab(b)).mean())

    def means(x):
        x = srgb_to_linear(x[:h, :w].astype(np.float64) / 255.0)
        x = x.reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3))
        return np.clip(np.round(v.linear_to_srgb(x) * 255.0), 0, 255).astype(np.uint8)

    return float(delta_e_2000(srgb_to_lab(means(a)), srgb_to_lab(means(b))).mean())


def compare(img: np.ndarray, cols: int, rows: int, variants: dict) -> dict:
    """Score several ``(mode, kind)`` variants of the same image."""

    return {name: score(img, cols, rows, m, k) for name, (m, k) in variants.items()}
