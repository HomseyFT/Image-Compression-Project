"""Trellis (rate-distortion optimized quantization) tests.

The load-bearing test here is `test_trellis_beats_the_rd_curve`. Everything
else can pass while trellis is worthless: a smaller file at lower PSNR proves
nothing, because lowering --quality achieves exactly that. Gains must be
measured at EQUAL quality against the baseline curve. A blind deadzone was
rejected during design for failing precisely this test.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import compression as c


def _rate(coeffs):
    return len(c._encode_blocks_huffman(coeffs)[0])


def _psnr(recon, ref):
    mse = float(np.mean((recon.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0**2 / max(mse, 1e-12))


def _rd_savings(img, probe_qualities, sweep=range(10, 96, 5)):
    """Rate saving (%) at equal PSNR, versus the baseline quality sweep.

    The only meaningful way to score a lossy coding change: compare bytes at
    matched quality, never bytes alone.
    """

    R, P = [], []
    for q in sweep:
        comp = c.compress_array(img, quality=q, trellis=False)
        R.append(_rate(comp.coeffs))
        P.append(_psnr(c.decompress_to_array(comp), img))
    R = np.array(R, dtype=float)
    P = np.array(P)
    order = np.argsort(P)
    P, R = P[order], R[order]

    out = []
    for q in probe_qualities:
        comp = c.compress_array(img, quality=q, trellis=True)
        rate = _rate(comp.coeffs)
        psnr = _psnr(c.decompress_to_array(comp), img)
        equivalent = float(np.interp(psnr, P, R))
        out.append((q, rate, psnr, 100.0 * (equivalent - rate) / equivalent))
    return out


def _plain_at_same_quantizer(img, quality):
    """Plain quantization using the SAME internal quantizer trellis will use.

    Since the quality dial was recalibrated, ``trellis=True`` and
    ``trellis=False`` at one requested quality deliberately use *different*
    quantizers -- both aim at the same JPEG-equivalent appearance, and only
    the trellis path needs correcting. Comparing the DP's output against
    plain rounding therefore has to pin the quantizer explicitly, or it
    compares two different encodes and proves nothing.
    """

    return c.compress_array(
        img, quality=c._calibrate_quality(quality, True), trellis=False
    ).coeffs


# --- Invariants -------------------------------------------------------------


@pytest.mark.parametrize("quality", [10, 30, 50, 75, 90])
def test_trellis_only_shrinks_magnitudes(photo, quality):
    """The DP may zero or step a level down, never up or across zero."""

    base = _plain_at_same_quantizer(photo, quality)
    tre = c.compress_array(photo, quality=quality, trellis=True).coeffs

    assert np.all(np.abs(tre) <= np.abs(base)), "a magnitude increased"
    moved = tre != 0
    assert np.all(np.sign(tre[moved]) == np.sign(base[moved])), "a sign flipped"


@pytest.mark.parametrize("quality", [10, 50, 90])
def test_trellis_leaves_dc_alone(photo, quality):
    base = _plain_at_same_quantizer(photo, quality)
    tre = c.compress_array(photo, quality=quality, trellis=True).coeffs
    np.testing.assert_array_equal(tre[:, :, 0, 0], base[:, :, 0, 0])


@pytest.mark.parametrize("quality", [10, 50, 90])
def test_trellis_output_is_losslessly_codable(photo, quality):
    """Trellis output must survive the entropy coder untouched."""

    coeffs = c.compress_array(photo, quality=quality, trellis=True).coeffs
    bitstream, dc, ac, layout = c._encode_blocks_huffman(coeffs)
    decoded = c._decode_blocks_huffman(
        coeffs.shape[0], coeffs.shape[1], bitstream, dc, ac, layout
    )
    np.testing.assert_array_equal(decoded, coeffs)


def test_trellis_disabled_matches_plain_quantization(photo):
    """trellis=False must be exactly the old round-to-nearest path."""

    padded, _ = c._pad_to_block_size(photo.astype(c.DCT_DTYPE))
    Q = c._build_quant_matrix(50).astype(c.DCT_DTYPE)
    expected = np.round(
        c._forward_dct_2d(c._to_blocks(padded - c.DCT_DTYPE(128.0))) / Q
    ).astype(np.int16)

    got = c.compress_array(photo, quality=50, trellis=False).coeffs
    np.testing.assert_array_equal(got, expected)


def test_trellis_reduces_rate(photo):
    for q in (30, 50, 75):
        base = _plain_at_same_quantizer(photo, q)
        tre = c.compress_array(photo, quality=q, trellis=True).coeffs
        assert _rate(tre) < _rate(base), f"trellis did not reduce rate at q={q}"


# --- The test that actually matters -----------------------------------------


def test_trellis_beats_the_rd_curve(photo):
    """Trellis must deliver rate savings at EQUAL PSNR, not just smaller files.

    Builds the baseline rate-distortion curve by sweeping quality, then checks
    that each trellis point sits strictly inside it -- fewer bytes than the
    baseline needs to reach the same PSNR. This is the check that
    distinguishes a genuine coding gain from sliding down the curve.
    """

    results = _rd_savings(photo, probe_qualities=(25, 40, 55, 70))

    for q, rate, psnr, saving in results:
        assert saving > 0, (
            f"q={q}: trellis used {rate} B for {psnr:.2f} dB, but the baseline "
            f"reaches that quality in {rate / (1 - saving / 100):.0f} B "
            "-- no coding gain"
        )

    mean = float(np.mean([s for *_, s in results]))
    assert mean > 3.0, (
        f"mean saving {mean:.1f}% is below the level that justified building "
        "this (expected ~5-6% on photographic content)"
    )


def test_trellis_holds_up_across_content(photo):
    """Gains must hold on varied natural content, not just one framing.

    Uses crops with different statistics (fine detail, flat background) so a
    lambda fitted to a single image cannot pass by overfitting.
    """

    h, w = photo.shape
    crops = {
        "whole": photo,
        "quadrant": photo[: h // 2, : w // 2],
        "centre": photo[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
    }

    for name, img in crops.items():
        results = _rd_savings(img, probe_qualities=(40, 70))
        for q, _r, psnr, saving in results:
            assert saving > 0, (
                f"{name} regressed at q={q}: {saving:+.1f}% rate at {psnr:.2f} dB"
            )


def test_trellis_iterations_is_a_knob_with_a_real_cost(photo):
    """More passes must not make things worse, and zero must be a no-op.

    The rate gain per pass is small (0.163 pp of BD-rate for pass 2, measured
    across the 11-image corpus) but consistent, so this checks the direction
    rather than a magnitude that would be noise at fixture scale.
    """

    plain = c.compress_array(photo, quality=50, trellis=False)
    none = c.compress_array(photo, quality=50, trellis_iterations=0)
    assert np.array_equal(none.coeffs, plain.coeffs), (
        "zero iterations must reproduce plain quantization exactly, including "
        "skipping the quality calibration"
    )

    base = _plain_at_same_quantizer(photo, 50)
    one = c.compress_array(photo, quality=50, trellis_iterations=1)
    two = c.compress_array(photo, quality=50, trellis_iterations=2)
    assert _rate(one.coeffs) < _rate(base)
    assert _rate(two.coeffs) <= _rate(one.coeffs)


def test_trellis_iterations_default_matches_the_constant(photo):
    """The explicit default must be the same as passing nothing."""

    implicit = c.compress_array(photo, quality=50)
    explicit = c.compress_array(
        photo, quality=50, trellis_iterations=c.TRELLIS_ITERATIONS
    )
    assert np.array_equal(implicit.coeffs, explicit.coeffs)


def test_trellis_iterations_rejects_negative(photo):
    with pytest.raises(ValueError, match="must be >= 0"):
        c.compress_array(photo, quality=50, trellis_iterations=-1)


def test_rd_scoring_requires_a_monotone_curve():
    """Degenerate content cannot be used to score rate-distortion changes.

    Smooth synthetic images produce curves with flat spots -- quality 50 -> 55
    can gain +0.00 dB while the rate grows -- which makes "rate at a given
    PSNR" numerically unstable and produces large phantom gains or losses.
    An earlier lambda fit was skewed by exactly this.

    The invariant that exposes it: as lambda -> 0 the DP must reproduce the
    baseline coefficients exactly, so any non-zero measured "saving" there is
    pure artifact.
    """

    ramp = np.tile(np.linspace(0, 255, 192, dtype=np.float32), (192, 1))
    smooth = (ramp + 20 * np.sin(np.arange(192) / 6)[:, None]).clip(0, 255).astype(np.uint8)

    original = c.TRELLIS_LAMBDA_SCALE
    c.TRELLIS_LAMBDA_SCALE = 1e-9
    try:
        # Pin the quantizer: the calibrated dial means trellis=True at q40
        # encodes at a finer internal quality than trellis=False would.
        base = _plain_at_same_quantizer(smooth, 40)
        tre = c.compress_array(smooth, quality=40, trellis=True).coeffs
    finally:
        c.TRELLIS_LAMBDA_SCALE = original

    np.testing.assert_array_equal(
        tre, base, "at lambda -> 0 trellis must collapse to plain quantization"
    )

    # And the curve really is non-monotone, which is why it cannot be scored.
    psnrs = []
    for q in range(10, 96, 5):
        comp = c.compress_array(smooth, quality=q, trellis=False)
        psnrs.append(_psnr(c.decompress_to_array(comp), smooth))
    assert np.min(np.diff(psnrs)) < 0.05, (
        "this fixture was expected to have a degenerate R-D curve; if it no "
        "longer does, it may have become valid for scoring"
    )


# --- Container is unaffected ------------------------------------------------


def test_trellis_needs_no_format_change(tmp_path, photo):
    """Trellis is encoder-side only: ICJ2 files stay ordinary ICJ2 files."""

    src, dst, out = tmp_path / "t.png", tmp_path / "t.icj", tmp_path / "o.png"
    Image.fromarray(photo, mode="L").save(src)

    c.compress_huffman_file(str(src), str(dst), quality=50)
    assert dst.read_bytes()[:4] == c.MAGIC

    c.decompress_huffman_file(str(dst), str(out))
    recon = np.array(Image.open(out), dtype=np.uint8)
    assert recon.shape == photo.shape
    assert _psnr(recon, photo) > 25


# --- Quality calibration -----------------------------------------------------


def test_calibration_is_monotone_and_bounded():
    """The dial must stay ordered and in range, or quality stops being a dial."""

    mapped = [c._calibrate_quality(q, True) for q in range(1, 101)]
    assert mapped == sorted(mapped), "calibration must be non-decreasing"
    assert min(mapped) >= 1 and max(mapped) <= 100
    assert c._calibrate_quality(100, True) == 100


def test_calibration_only_applies_when_trellis_runs():
    """Plain quantization already matches libjpeg's scale, so must not move.

    Correcting it would break the one case that was right before the fix.
    """

    for q in (1, 10, 50, 90, 100):
        assert c._calibrate_quality(q, False) == q


def test_calibration_raises_the_internal_quality():
    """Trellis costs quality at a fixed quantizer, so the fix must compensate up.

    A calibration that lowered the internal quality would be correcting in the
    wrong direction and would make the original bug worse.
    """

    for q in (20, 30, 50, 70):
        assert c._calibrate_quality(q, True) > q


def test_zero_iterations_skips_calibration(photo):
    """trellis_iterations=0 runs no DP, so it must not be compensated either."""

    plain = c.compress_array(photo, quality=50, trellis=False)
    none = c.compress_array(photo, quality=50, trellis_iterations=0)
    assert none.quality == plain.quality == 50


def test_calibrated_quality_tracks_libjpeg(photo):
    """The point of the whole exercise: q50 must *look like* libjpeg q50.

    Before calibration this codec's q50 measured 1.27 dB below libjpeg's on
    kodim01 and ~18 quality points below on the Kodak set. Every check in the
    suite was BD-rate, which integrates over the curve and is invariant to
    exactly this shift -- so nothing caught it for three phases. This test is
    the one that would have.
    """

    import io

    from PIL import Image

    for quality in (30, 50, 70):
        recon = c.decompress_to_array(c.compress_array(photo, quality=quality))
        ours = _psnr(recon, photo)

        buf = io.BytesIO()
        Image.fromarray(photo, mode="L").save(
            buf, format="JPEG", quality=quality, optimize=True
        )
        with Image.open(io.BytesIO(buf.getvalue())) as im:
            theirs = _psnr(np.array(im.convert("L")), photo)

        assert ours > theirs - 0.5, (
            f"q{quality}: {ours:.2f} dB vs libjpeg's {theirs:.2f} dB -- the "
            "quality dial has drifted below what its number promises"
        )
