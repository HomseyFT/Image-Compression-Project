"""Tests for the rate-distortion benchmark harness.

The harness exists to prevent a specific, historically-real failure: a
plausible-looking R-D number that is wrong. So these tests are less about
"does it compute" and more about "does it refuse when it should" -- the
monotonicity guard and the container-size accounting are the load-bearing
parts.

`test_bd_rate_is_zero_against_itself` and `test_bd_rate_detects_a_known_shift`
pin the arithmetic against cases with an analytically known answer, so a
refactor cannot quietly change what BD-rate means.
"""

from __future__ import annotations

import numpy as np
import pytest

import compression as c
from bench.bdrate import (
    MIN_OVERLAP_DB,
    Curve,
    InsufficientOverlapError,
    NonMonotoneCurveError,
    bd_rate,
    check_monotone,
)
from bench.codecs import icj_curve, icj_size, libjpeg_curve, psnr

SWEEP = (20, 35, 50, 65, 80)


# --- BD-rate arithmetic ------------------------------------------------------


def _synthetic_curve(name: str, scale: float = 1.0) -> Curve:
    """A well-behaved curve: rate grows, PSNR grows, no flat spots."""

    psnrs = np.array([28.0, 31.0, 34.0, 37.0, 40.0, 43.0])
    rates = scale * 10.0 ** (psnrs / 12.0)
    return Curve.from_points(name, zip(rates, psnrs))


def test_bd_rate_is_zero_against_itself():
    curve = _synthetic_curve("a")
    assert bd_rate(curve, curve) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("scale,expected", [(0.5, -50.0), (0.8, -20.0), (1.25, 25.0)])
def test_bd_rate_detects_a_known_shift(scale, expected):
    """Scaling every rate by k must read as exactly (k-1)*100% BD-rate.

    A uniform rate scaling is a vertical shift in log-rate, so the integral
    difference is exact regardless of curve shape -- the one case where the
    right answer is known in closed form.
    """

    ref = _synthetic_curve("ref")
    test = _synthetic_curve("test", scale=scale)
    assert bd_rate(ref, test) == pytest.approx(expected, abs=0.01)


def test_bd_rate_sign_convention_is_negative_for_better():
    """Fewer bits at equal quality must be negative. Guards the sign."""

    ref = _synthetic_curve("ref")
    better = _synthetic_curve("better", scale=0.7)
    assert bd_rate(ref, better) < 0
    assert bd_rate(better, ref) > 0


# --- The monotonicity guard --------------------------------------------------


def test_monotone_guard_accepts_a_photographic_curve(photo):
    check_monotone(icj_curve(photo, SWEEP))


@pytest.mark.parametrize("name", ["flat", "gradient"])
def test_monotone_guard_rejects_smooth_synthetic_content(corpus, name):
    """The whole point of the module.

    ``flat`` gains +0.000 dB across the entire sweep; ``gradient`` actually
    *inverts*, losing 1.24 dB between two quality steps while the rate grows.
    Scoring either manufactures phantom gains of 5%+, which is the error that
    produced a wrong trellis lambda for two commits. The harness must refuse,
    loudly, rather than return a plausible number.
    """

    curve = icj_curve(corpus[name], SWEEP)
    with pytest.raises(NonMonotoneCurveError, match="not monotone enough"):
        check_monotone(curve)


def test_edges_is_monotone_but_still_not_an_rd_instrument(corpus):
    """``edges`` passes the guard. That is correct, and it is not an invitation.

    conftest groups ``flat``, ``gradient`` and ``edges`` as unsuitable for R-D
    scoring, but for *two different reasons*: flat spots (flat, gradient) and
    near-empty files (edges). Only the first is something a monotonicity check
    can see -- edges rises a healthy +0.8 dB per step.

    Its files are 268-422 bytes, where the 25-byte header and ~100 bytes of
    Huffman tables dominate, so BD-rate on it measures table overhead rather
    than the quantizer. The guard cannot catch this and deliberately does not
    try; a size threshold would wrongly reject legitimately small real images.
    The defence is corpus choice -- ``python -m bench`` scores photographs
    only. This test exists so that nobody later "fixes" the guard to reject
    edges, or starts scoring it because it happens to pass.
    """

    curve = icj_curve(corpus["edges"], SWEEP)
    check_monotone(curve)                       # passes, by design
    assert curve.rates.max() < 1000             # and is far too small to score


def test_monotone_guard_rejects_a_flat_spot_specifically():
    """A single flat step is enough to reject, even if the rest is fine."""

    psnrs = [30.0, 33.0, 33.0 + 1e-4, 36.0, 39.0]   # one flat step
    rates = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    curve = Curve.from_points("flatspot", zip(rates, psnrs))
    with pytest.raises(NonMonotoneCurveError, match="flat spot|not monotone"):
        check_monotone(curve)


def test_monotone_guard_rejects_an_inverted_step():
    """PSNR falling as rate rises is the strongest form of unscorable."""

    curve = Curve.from_points(
        "inverted",
        zip([1000.0, 2000.0, 3000.0, 4000.0], [30.0, 33.0, 32.0, 36.0]),
    )
    with pytest.raises(NonMonotoneCurveError):
        check_monotone(curve)


def test_monotone_guard_requires_enough_points_for_a_cubic():
    curve = Curve.from_points("short", zip([1.0, 2.0, 3.0], [30.0, 33.0, 36.0]))
    with pytest.raises(NonMonotoneCurveError, match="at least"):
        check_monotone(curve)


def test_bd_rate_applies_the_guard_by_default(corpus):
    """The guard must be on unless explicitly disabled, or it protects nobody."""

    bad = icj_curve(corpus["flat"], SWEEP)
    good = _synthetic_curve("good")
    with pytest.raises(NonMonotoneCurveError):
        bd_rate(good, bad)
    with pytest.raises(NonMonotoneCurveError):
        bd_rate(bad, good)


def test_non_overlapping_curves_are_refused():
    low = Curve.from_points("low", zip([1e3, 2e3, 3e3, 4e3], [20.0, 22.0, 24.0, 26.0]))
    high = Curve.from_points("high", zip([1e3, 2e3, 3e3, 4e3], [40.0, 42.0, 44.0, 46.0]))
    with pytest.raises(InsufficientOverlapError, match="PSNR range"):
        bd_rate(low, high)


def test_hairline_overlap_is_refused():
    """Integrating a cubic over a sliver amplifies fit error without bound."""

    a = Curve.from_points("a", zip([1e3, 2e3, 3e3, 4e3], [30.0, 33.0, 36.0, 39.0]))
    b = Curve.from_points(
        "b",
        zip([1e3, 2e3, 3e3, 4e3],
            [39.0 - MIN_OVERLAP_DB / 2, 42.0, 45.0, 48.0]),
    )
    with pytest.raises(InsufficientOverlapError):
        bd_rate(a, b)


# --- Rate accounting ---------------------------------------------------------


@pytest.mark.parametrize("quality", [20, 50, 90])
def test_icj_size_matches_the_real_container(tmp_path, photo, quality):
    """The analytic size must equal what `compress_huffman_file` actually writes.

    `icj_size` reimplements the header layout to avoid disk I/O per data point.
    That duplication is only safe if pinned: if the container gains a field,
    this fails rather than silently under-reporting our own rate and flattering
    every benchmark in the repository.
    """

    src = tmp_path / "in.png"
    dst = tmp_path / "out.icj"
    from PIL import Image

    Image.fromarray(photo, mode="L").save(src)
    c.compress_huffman_file(str(src), str(dst), quality=quality)

    predicted, _ = icj_size(photo, quality)
    assert predicted == dst.stat().st_size


def test_curves_report_whole_file_bytes(photo):
    """Rate must include tables and header, not just payload."""

    curve = icj_curve(photo, SWEEP)
    for q, rate in zip(SWEEP, curve.rates):
        comp = c.compress_array(photo, quality=q)
        payload = len(c._encode_blocks_huffman(comp.coeffs)[0])
        assert rate > payload


def test_psnr_of_identical_images_is_capped():
    img = np.full((16, 16), 100, dtype=np.uint8)
    assert psnr(img, img) > 100


# --- End-to-end gate ---------------------------------------------------------


def test_codec_beats_libjpeg(photo):
    """The headline claim, as a regression gate.

    Pinned loosely: this guards against a phase 7/8 change silently destroying
    the trellis advantage, not against small drift. Measured ~-15 to -19% on
    the full-resolution sample; the 256x256 fixture scores lower because
    per-image Huffman tables are a larger share of a small file.
    """

    ours = icj_curve(photo, SWEEP)
    jpeg = libjpeg_curve(photo, SWEEP)
    score = bd_rate(jpeg, ours)
    assert score < -5.0, f"BD-rate vs libjpeg regressed to {score:+.2f}%"


def test_trellis_beats_the_no_trellis_baseline(photo):
    """Separates our gain from libjpeg's baseline, so a libjpeg version bump
    cannot be mistaken for a change in this codec."""

    ours = icj_curve(photo, SWEEP)
    base = icj_curve(photo, SWEEP, trellis=False)
    score = bd_rate(base, ours)
    assert score < -5.0, f"trellis BD-rate regressed to {score:+.2f}%"
