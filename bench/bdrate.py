"""Bjøntegaard delta-rate: average bitrate difference at equal quality.

The only honest way to score a lossy coding change is to compare bytes at
matched quality. Comparing bytes alone is meaningless, because any quality
knob makes files smaller -- that is what a quality knob *is*. This module is
the project's single implementation of that comparison; ad-hoc
``np.interp``-at-matched-PSNR scoring in scratch scripts is what produced the
phantom trellis-lambda cliff documented in SPEC.md.

BD-rate fits a cubic to ``log10(rate)`` as a function of PSNR for each curve,
integrates the difference over the PSNR interval the two curves share, and
divides by the width of that interval. The result is a percentage:

    negative  =  fewer bits for the same quality  =  better

:func:`check_monotone` is not optional decoration. See its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A curve needs at least this many points to fit a cubic.
MIN_POINTS = 4

# Minimum PSNR gain (dB) required between consecutive points of a curve for it
# to be considered scorable. See :func:`check_monotone`.
MIN_PSNR_STEP_DB = 0.05

# Minimum width (dB) of the shared PSNR interval. Integrating a cubic over a
# hairline overlap amplifies fit error without bound.
MIN_OVERLAP_DB = 0.5


class NonMonotoneCurveError(ValueError):
    """Raised when a rate-distortion curve cannot be meaningfully scored."""


class InsufficientOverlapError(ValueError):
    """Raised when two curves share too little PSNR range to compare."""


@dataclass(frozen=True)
class Curve:
    """A rate-distortion curve: bytes and PSNR at each quality setting.

    Points are stored sorted by rate ascending, which for a scorable curve is
    also PSNR ascending.
    """

    name: str
    rates: np.ndarray   # bytes, float
    psnrs: np.ndarray   # dB, float

    @classmethod
    def from_points(cls, name: str, points) -> "Curve":
        pts = np.asarray(list(points), dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be an iterable of (rate_bytes, psnr_db)")
        order = np.argsort(pts[:, 0])
        return cls(name=name, rates=pts[order, 0], psnrs=pts[order, 1])

    def __len__(self) -> int:
        return len(self.rates)


def check_monotone(
    curve: Curve, min_step_db: float = MIN_PSNR_STEP_DB
) -> None:
    """Raise unless ``curve`` rises monotonically enough to be scored.

    **This guard is the point of the module.** Smooth synthetic content
    produces curves with flat spots -- quality 50 -> 55 gains +0.00 dB while
    the rate grows. Interpolating "rate at a given PSNR" across a flat spot is
    numerically unstable and manufactures gains or losses of 5% or more out of
    nothing. An early trellis-lambda fit was skewed by exactly this, and the
    resulting constant was wrong for two commits.

    The failure is silent and the output looks entirely plausible, which is
    why this raises rather than warning or returning a sentinel. If you are
    scoring ``flat``, ``gradient``, ``edges`` or any other smooth synthetic
    image, you will land here -- that is correct. Score R-D on photographs.
    """

    if len(curve) < MIN_POINTS:
        raise NonMonotoneCurveError(
            f"curve {curve.name!r} has {len(curve)} points; "
            f"a cubic fit needs at least {MIN_POINTS}"
        )

    diffs = np.diff(curve.psnrs)
    if diffs.min() < min_step_db:
        worst = int(np.argmin(diffs))
        raise NonMonotoneCurveError(
            f"curve {curve.name!r} is not monotone enough to score: PSNR moves "
            f"{diffs[worst]:+.3f} dB (< {min_step_db} dB) while rate grows "
            f"{curve.rates[worst]:.0f} -> {curve.rates[worst + 1]:.0f} bytes. "
            "Interpolating rate at a given PSNR across a flat spot produces "
            "phantom gains; score R-D on photographic content instead. "
            "See SPEC.md 'Correction to the phase 4 fitting notes'."
        )


def bd_rate(reference: Curve, test: Curve, *, check: bool = True) -> float:
    """Average % bitrate change of ``test`` versus ``reference`` at equal PSNR.

    Negative means ``test`` needs fewer bits for the same quality.

    Both curves are validated by :func:`check_monotone` unless ``check`` is
    disabled -- which should only be done by tests that are deliberately
    exercising degenerate input.
    """

    if check:
        check_monotone(reference)
        check_monotone(test)

    lo = max(reference.psnrs.min(), test.psnrs.min())
    hi = min(reference.psnrs.max(), test.psnrs.max())
    if hi - lo < MIN_OVERLAP_DB:
        raise InsufficientOverlapError(
            f"curves {reference.name!r} and {test.name!r} share only "
            f"{max(hi - lo, 0.0):.2f} dB of PSNR range (need "
            f"{MIN_OVERLAP_DB} dB). Widen the quality sweep so the curves span "
            "a common quality range."
        )

    # Fit log10(rate) = f(PSNR) and average f over the shared interval.
    ref_fit = np.polyfit(reference.psnrs, np.log10(reference.rates), 3)
    test_fit = np.polyfit(test.psnrs, np.log10(test.rates), 3)

    ref_int = np.polyint(ref_fit)
    test_int = np.polyint(test_fit)

    ref_avg = (np.polyval(ref_int, hi) - np.polyval(ref_int, lo)) / (hi - lo)
    test_avg = (np.polyval(test_int, hi) - np.polyval(test_int, lo)) / (hi - lo)

    return float((10.0 ** (test_avg - ref_avg) - 1.0) * 100.0)
