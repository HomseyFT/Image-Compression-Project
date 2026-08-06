"""Colour support (phase 8): YCbCr, chroma subsampling, planar coding, ICJ4.

The load-bearing tests here guard two things that would otherwise rot quietly.

**Grayscale must not be taxed.** Colour is strictly additive: the grayscale
path has to produce identical coefficients and a container no larger than
before. `test_grayscale_is_unchanged_by_colour_support` and
`test_luma_plane_matches_the_grayscale_codec` pin that.

**The chroma lambda must stay refitted.** Running chroma at the luma-derived
lambda costs 4.9 pp of BD-rate -- worse than disabling trellis outright -- and
nothing about that failure is visible without measuring, because the output is
a perfectly valid image that is merely bigger than it should be.
`test_chroma_lambda_is_much_smaller_than_luma` and
`test_luma_lambda_on_chroma_is_worse` exist so a future tidy-up cannot
"simplify" the two scales back into one.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import compression as c

SAMPLINGS = (c.SAMPLING_444, c.SAMPLING_422, c.SAMPLING_420)


@pytest.fixture(scope="module")
def rgb() -> np.ndarray:
    """A real colour photograph, downscaled to keep the suite fast."""

    with Image.open("images/kodim01.png") as im:
        return np.array(im.convert("RGB").resize((192, 128), Image.LANCZOS))


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0**2 / max(mse, 1e-12))


# --- Colour transform --------------------------------------------------------


def test_rgb_ycbcr_round_trip_is_within_rounding(rgb):
    back = c._ycbcr_to_rgb(c._rgb_to_ycbcr(rgb))
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1


def test_neutral_grey_has_no_chroma():
    """R=G=B must give Cb=Cr=128 exactly, or greys acquire a colour cast."""

    grey = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    ycc = c._rgb_to_ycbcr(grey)
    assert np.abs(ycc[..., 1] - 128).max() < 1e-3
    assert np.abs(ycc[..., 2] - 128).max() < 1e-3
    # ...and Y must equal the original luminance, not be rescaled.
    assert np.abs(ycc[..., 0] - grey[..., 0]).max() < 1e-3


def test_saturated_primaries_survive_the_round_trip():
    prims = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
         [[255, 255, 0], [0, 255, 255], [255, 0, 255]]],
        dtype=np.uint8,
    )
    back = c._ycbcr_to_rgb(c._rgb_to_ycbcr(prims))
    assert np.abs(back.astype(int) - prims.astype(int)).max() <= 1


# --- Subsampling -------------------------------------------------------------


@pytest.mark.parametrize("sampling", SAMPLINGS)
@pytest.mark.parametrize("shape", [(64, 48), (65, 49), (7, 3), (1, 1), (16, 1)])
def test_subsample_upsample_restores_shape(sampling, shape):
    hs, vs = c.SAMPLING_FACTORS[sampling]
    plane = np.random.RandomState(0).rand(*shape).astype(np.float32) * 255
    out = c._upsample(c._subsample(plane, hs, vs), hs, vs, shape)
    assert out.shape == shape


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_constant_plane_survives_resampling_exactly(sampling):
    """A flat colour field must not acquire ringing or a DC shift."""

    hs, vs = c.SAMPLING_FACTORS[sampling]
    plane = np.full((32, 32), 77.0, dtype=np.float32)
    out = c._upsample(c._subsample(plane, hs, vs), hs, vs, (32, 32))
    assert np.abs(out - 77.0).max() < 1e-4


def test_subsampling_averages_rather_than_drops():
    """Box averaging, not point sampling -- the latter aliases on edges."""

    plane = np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float32)
    assert c._subsample(plane, 2, 2)[0, 0] == pytest.approx(50.0)


# --- Round trips -------------------------------------------------------------


@pytest.mark.parametrize("sampling", SAMPLINGS)
@pytest.mark.parametrize("quality", [10, 50, 90])
def test_colour_round_trip_shape_and_quality(rgb, sampling, quality):
    comp = c.compress_array(rgb, quality=quality, sampling=sampling)
    out = c.decompress_to_array(comp)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert _psnr(out, rgb) > 18


@pytest.mark.parametrize("sampling", SAMPLINGS)
@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 5, 3), (17, 8, 3), (65, 49, 3)])
def test_awkward_colour_shapes_round_trip(sampling, shape):
    img = np.random.RandomState(2).randint(0, 256, shape).astype(np.uint8)
    out = c.decompress_to_array(c.compress_array(img, quality=50, sampling=sampling))
    assert out.shape == shape


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_colour_container_matches_the_in_memory_path(tmp_path, rgb, sampling):
    """The file must decode to exactly what the array API produces."""

    src, icj, out = tmp_path / "i.png", tmp_path / "o.icj", tmp_path / "o.png"
    Image.fromarray(rgb, mode="RGB").save(src)
    c.compress_huffman_file(str(src), str(icj), quality=60, sampling=sampling)
    c.decompress_huffman_file(str(icj), str(out))

    from_file = np.array(Image.open(out).convert("RGB"))
    in_memory = c.decompress_to_array(
        c.compress_array(rgb, quality=60, sampling=sampling)
    )
    assert np.array_equal(from_file, in_memory)


def test_finer_sampling_keeps_more_chroma(rgb):
    """4:4:4 > 4:2:2 > 4:2:0 on chroma fidelity, and the reverse on size."""

    from bench.codecs import icj_size, plane_psnr

    results = {}
    for sampling in SAMPLINGS:
        size, recon = icj_size(rgb, 70, sampling=sampling)
        results[sampling] = (size, plane_psnr(recon, rgb))

    assert (
        results[c.SAMPLING_444][0]
        > results[c.SAMPLING_422][0]
        > results[c.SAMPLING_420][0]
    )
    for plane in ("Cb", "Cr"):
        assert (
            results[c.SAMPLING_444][1][plane]
            > results[c.SAMPLING_422][1][plane]
            > results[c.SAMPLING_420][1][plane]
        )


# --- Grayscale must not be taxed ---------------------------------------------


def test_grayscale_is_unchanged_by_colour_support(photo):
    """2D input must still take the grayscale path, with sampling ignored."""

    comp = c.compress_array(photo, quality=50)
    assert comp.n_components == 1
    assert not comp.is_color
    assert comp.sampling == c.SAMPLING_444
    # The sampling argument must not perturb a grayscale encode.
    other = c.compress_array(photo, quality=50, sampling=c.SAMPLING_444)
    assert np.array_equal(comp.coeffs, other.coeffs)


def test_luma_plane_matches_the_grayscale_codec(rgb):
    """The Y plane of a colour encode must be bit-identical to encoding Y alone.

    This is the invariant that makes planar coding worth its cost: each plane
    really is the existing grayscale path, so colour cannot silently change
    how luma is coded.
    """

    luma = c._rgb_to_ycbcr(rgb)[..., 0]
    colour = c.compress_array(rgb, quality=50, sampling=c.SAMPLING_420)
    grayscale = c.compress_array(luma, quality=50)
    assert np.array_equal(colour.planes[0], grayscale.coeffs)


def test_grayscale_container_declares_one_component(tmp_path, photo):
    src, icj = tmp_path / "g.png", tmp_path / "g.icj"
    Image.fromarray(photo, mode="L").save(src)
    c.compress_huffman_file(str(src), str(icj), quality=50)

    raw = icj.read_bytes()
    assert raw[:4] == b"ICJ4"
    assert raw[13] >> 4 == 1, "grayscale must record one component"


def test_planes_pad_to_eight_not_to_an_mcu():
    """Planar coding means no MCU: every plane pads to 8 independently.

    Interleaved JPEG must pad luma to 16x16 at 4:2:0 so subsampled chroma
    lands on block boundaries. Planar removes that coupling, and this pins it
    -- a 4:2:0 chroma plane of a 24-row image is 12 rows, which pads to 16,
    not to some MCU-derived multiple of the luma padding.
    """

    img = np.zeros((24, 24, 3), dtype=np.uint8)
    comp = c.compress_array(img, quality=50, sampling=c.SAMPLING_420)
    assert comp.padded_shapes[0] == (24, 24)      # luma already a multiple of 8
    assert comp.padded_shapes[1] == (16, 16)      # chroma 12x12 -> 16x16
    assert comp.padded_shapes[2] == (16, 16)


# --- The chroma lambda finding -----------------------------------------------


def test_chroma_lambda_is_much_smaller_than_luma():
    """Pins the refit. The luma formula gives chroma a *larger* lambda.

    mean(Q^2) is 8125 for chroma against 4500 for luma at q50, so tying lambda
    to it alone points the wrong way. Without the separate scale, chroma gets
    roughly 1.8x the luma lambda when it needs about a seventh of it.
    """

    luma_q = c._quant_matrix_for(c.CLASS_LUMA, 50)
    chroma_q = c._quant_matrix_for(c.CLASS_CHROMA, 50)

    # The naive formula really does point the wrong way -- that is the trap.
    assert c._trellis_lambda(chroma_q) > c._trellis_lambda(luma_q)

    # The fitted scales correct it.
    fitted_chroma = c._trellis_lambda(chroma_q, c.TRELLIS_LAMBDA_SCALE_CHROMA)
    fitted_luma = c._trellis_lambda(luma_q, c.TRELLIS_LAMBDA_SCALE)
    assert fitted_chroma < fitted_luma / 3


def test_luma_lambda_on_chroma_is_worse(rgb):
    """Regression guard: collapsing the two scales into one must cost rate.

    **Scored by BD-rate, not by PSNR at fixed quality.** The bad lambda
    over-zeroes chroma, so at a given quality setting it yields a file that is
    both smaller *and* worse -- two things moving at once, which is exactly
    the rate-distortion slide this project refuses to score. An earlier
    version of this test compared PSNR at q50 and read a 4.9 pp rate
    regression as a 0.49 dB blur, which is not the same claim.
    """

    from bench.bdrate import Curve, bd_rate
    from bench.codecs import icj_size, psnr

    sweep = (25, 40, 55, 70, 85)

    def curve(name: str) -> Curve:
        points = []
        for quality in sweep:
            size, recon = icj_size(rgb, quality)
            points.append((size, psnr(recon, rgb)))
        return Curve.from_points(name, points)

    fitted = curve("fitted")

    original = c.TRELLIS_LAMBDA_SCALE_CHROMA
    try:
        c.TRELLIS_LAMBDA_SCALE_CHROMA = c.TRELLIS_LAMBDA_SCALE
        naive = curve("luma-lambda-on-chroma")
    finally:
        c.TRELLIS_LAMBDA_SCALE_CHROMA = original

    score = bd_rate(naive, fitted)
    assert score < -1.0, (
        f"refitting the chroma lambda gained only {score:+.2f}% BD-rate; "
        "measured at -4.9 pp across the Kodak set"
    )
