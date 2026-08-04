"""Codec test suite.

The central invariant is that entropy coding is *lossless*: all quality loss
happens in quantization, so the Huffman round trip must reproduce the
quantized coefficient array exactly. Everything else builds on that.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import compression as c


QUALITIES = [1, 10, 50, 75, 100]
SHAPES = [(1, 1), (3, 5), (8, 8), (9, 17), (16, 8), (64, 64)]


# --- Coefficient round trip -------------------------------------------------


@pytest.mark.parametrize("name", ["flat", "gradient", "texture", "edges", "checkerboard"])
@pytest.mark.parametrize("quality", QUALITIES)
def test_huffman_roundtrip_is_lossless(corpus, name, quality):
    """Encode -> decode must reproduce quantized coefficients bit-exactly."""

    coeffs = c.compress_array(corpus[name], quality=quality).coeffs
    bitstream, dc, ac = c._encode_blocks_huffman(coeffs)
    decoded = c._decode_blocks_huffman(coeffs.shape[0], coeffs.shape[1], bitstream, dc, ac)

    assert decoded.dtype == coeffs.dtype
    np.testing.assert_array_equal(decoded, coeffs)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("quality", [1, 50, 100])
def test_roundtrip_random_shapes(shape, quality):
    rng = np.random.RandomState(abs(hash(shape)) % 2**31)
    img = (rng.rand(*shape) * 255).astype(np.uint8)

    coeffs = c.compress_array(img, quality=quality).coeffs
    bitstream, dc, ac = c._encode_blocks_huffman(coeffs)
    decoded = c._decode_blocks_huffman(coeffs.shape[0], coeffs.shape[1], bitstream, dc, ac)

    np.testing.assert_array_equal(decoded, coeffs)


@pytest.mark.parametrize("shape", SHAPES)
def test_shape_is_preserved_through_padding(shape):
    """Non-multiple-of-8 dimensions must crop back to the exact original."""

    rng = np.random.RandomState(7)
    img = (rng.rand(*shape) * 255).astype(np.uint8)

    comp = c.compress_array(img, quality=50)
    assert tuple(comp.orig_shape) == shape
    assert comp.padded_shape[0] % 8 == 0 and comp.padded_shape[1] % 8 == 0

    out = c.decompress_to_array(comp)
    assert out.shape == shape
    assert out.dtype == np.uint8


# --- Quality clamping (regression for the silent-corruption bug) ------------


@pytest.mark.parametrize("quality", [-5, 0, 101, 150, 10**9])
def test_out_of_range_quality_is_clamped_consistently(tmp_path, quality):
    """Encoder and decoder must agree on quality for out-of-range input.

    Regression: quality was clamped only inside _quality_to_scale while the
    raw value was stored, so `quality=-5` encoded at q=1 but persisted byte
    251, which decoded as q=100 -- silent garbage at MSE ~5377.
    """

    rng = np.random.RandomState(3)
    img = (rng.rand(64, 64) * 255).astype(np.uint8)
    src, dst, out = tmp_path / "a.png", tmp_path / "a.icj", tmp_path / "b.png"
    Image.fromarray(img, mode="L").save(src)

    c.compress_huffman_file(str(src), str(dst), quality=quality)

    stored = dst.read_bytes()[12]
    assert stored == c._clamp_quality(quality)
    assert 1 <= stored <= 100

    c.decompress_huffman_file(str(dst), str(out))
    recon = np.array(Image.open(out), dtype=np.float64)

    # Reconstruction must match what the *clamped* quality predicts.
    expected = c.decompress_to_array(c.compress_array(img, quality=quality))
    np.testing.assert_array_equal(recon.astype(np.uint8), expected)


@pytest.mark.parametrize("bad", ["50", 3.7, True, None, np.float64(50)])
def test_non_integer_quality_is_rejected(bad):
    with pytest.raises(TypeError):
        c._clamp_quality(bad)


def test_clamp_accepts_numpy_integers():
    assert c._clamp_quality(np.int32(75)) == 75


# --- Container robustness ---------------------------------------------------


@pytest.fixture()
def sample_icj(tmp_path):
    rng = np.random.RandomState(11)
    img = (rng.rand(32, 32) * 255).astype(np.uint8)
    src, dst = tmp_path / "s.png", tmp_path / "s.icj"
    Image.fromarray(img, mode="L").save(src)
    c.compress_huffman_file(str(src), str(dst), quality=50)
    return dst


def test_rejects_bad_magic(tmp_path, sample_icj):
    data = bytearray(sample_icj.read_bytes())
    data[0:4] = b"XXXX"
    bad = tmp_path / "bad.icj"
    bad.write_bytes(data)

    with pytest.raises(ValueError, match="magic"):
        c.decompress_huffman_file(str(bad), str(tmp_path / "o.png"))


def test_rejects_icj1_files(tmp_path, sample_icj):
    """The superseded ICJ1 magic must not be silently accepted."""

    data = bytearray(sample_icj.read_bytes())
    data[0:4] = b"ICJ1"
    old = tmp_path / "old.icj"
    old.write_bytes(data)

    with pytest.raises(ValueError):
        c.decompress_huffman_file(str(old), str(tmp_path / "o.png"))


@pytest.mark.parametrize("keep", [4, 10, 15, 25, 60])
def test_truncated_file_raises(tmp_path, sample_icj, keep):
    """Truncation must raise a defined error, never hang or emit garbage."""

    trunc = tmp_path / "t.icj"
    trunc.write_bytes(sample_icj.read_bytes()[:keep])

    with pytest.raises((ValueError, EOFError)):
        c.decompress_huffman_file(str(trunc), str(tmp_path / "o.png"))


def test_corrupt_table_length_raises(tmp_path, sample_icj):
    data = bytearray(sample_icj.read_bytes())
    data[17] = 0xFF  # implausible code length inside the DC table
    bad = tmp_path / "c.icj"
    bad.write_bytes(data)

    with pytest.raises(ValueError):
        c.decompress_huffman_file(str(bad), str(tmp_path / "o.png"))


# --- Bit I/O ----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_bitwriter_bitreader_roundtrip(seed):
    rng = np.random.RandomState(seed)
    items = [(int(rng.randint(0, 2**n)), n) for n in rng.randint(1, 17, size=64)]

    w = c._BitWriter()
    for value, n_bits in items:
        w.write_bits(value, n_bits)
    w.flush()

    r = c._BitReader(w.bytes)
    for value, n_bits in items:
        assert r.read_bits(n_bits) == value


def test_bitwriter_zero_length_write_is_noop():
    w = c._BitWriter()
    w.write_bits(0, 0)
    w.flush()
    assert w.bytes == b""


def test_bitreader_past_end_raises():
    r = c._BitReader(b"\x00")
    r.read_bits(8)
    with pytest.raises(EOFError):
        r.read_bit()


# --- Value / category coding ------------------------------------------------


@pytest.mark.parametrize("v", list(range(-2050, 2051, 7)))
def test_value_bits_roundtrip(v):
    cat = c._value_category(v)
    assert c._bits_to_value(c._value_to_bits(v, cat), cat) == v


def test_dc_category_stays_within_table():
    """Max DC differential must fit the 0..11 category table.

    Orthonormal DCT bounds the DC coefficient to [-1024, 1016], so the
    largest differential is 2040 -> category 11. This test pins that
    reasoning so a future transform change cannot silently overflow.
    """

    img = np.zeros((16, 8), dtype=np.uint8)
    img[:8, :] = 255
    coeffs = c.compress_array(img, quality=100).coeffs
    dcs = [int(coeffs[j, 0].reshape(-1)[c.ZIGZAG_ORDER][0]) for j in range(2)]
    assert c._value_category(dcs[0] - dcs[1]) <= 11


def test_ac_size_fits_nibble():
    """AC size must fit 4 bits, since symbol = (run << 4) | size."""

    board = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    coeffs = c.compress_array(board, quality=100).coeffs
    assert c._value_category(int(np.abs(coeffs).max())) <= 15


# --- Huffman table construction ---------------------------------------------


def test_single_symbol_table_is_usable():
    table = c._build_huffman_table({5: 100})
    assert table == {5: (0, 1)}


def test_empty_table_raises():
    with pytest.raises(ValueError):
        c._build_huffman_table({1: 0, 2: 0})


def test_canonical_codes_are_prefix_free():
    rng = np.random.RandomState(5)
    freqs = {i: int(rng.randint(1, 5000)) for i in range(64)}
    table = c._build_huffman_table(freqs)

    codes = sorted(((n, code) for code, n in table.values()))
    for i, (n1, c1) in enumerate(codes):
        for n2, c2 in codes[i + 1 :]:
            assert (c2 >> (n2 - n1)) != c1, "prefix collision"


def test_overlong_codes_are_rejected():
    """Codes beyond the container's 32-bit field must fail loudly."""

    freqs = {i: int(1.7 ** i) for i in range(1, 60)}
    try:
        table = c._build_huffman_table(freqs)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        assert max(n for _, n in table.values()) <= c.MAX_CODE_LENGTH


def test_table_serialization_roundtrip():
    rng = np.random.RandomState(2)
    freqs = {i: int(rng.randint(1, 1000)) for i in range(40)}
    table = c._build_huffman_table(freqs)
    assert c._deserialize_huffman_table(c._serialize_huffman_table(table)) == table


# --- End-to-end -------------------------------------------------------------


@pytest.mark.parametrize("quality", [10, 50, 90])
def test_file_roundtrip_end_to_end(tmp_path, photo, quality):
    src, dst, out = tmp_path / "p.png", tmp_path / "p.icj", tmp_path / "r.png"
    Image.fromarray(photo, mode="L").save(src)

    c.compress_huffman_file(str(src), str(dst), quality=quality)
    c.decompress_huffman_file(str(dst), str(out))

    recon = np.array(Image.open(out), dtype=np.uint8)
    assert recon.shape == photo.shape

    mse = np.mean((recon.astype(np.float64) - photo.astype(np.float64)) ** 2)
    assert mse < 400, f"unexpectedly poor reconstruction at q={quality}"
    assert dst.stat().st_size < photo.nbytes


def test_higher_quality_is_monotonically_better(photo):
    """PSNR must increase with quality; guards R-D changes in later phases."""

    prev = -np.inf
    for q in (10, 30, 50, 70, 90):
        recon = c.decompress_to_array(c.compress_array(photo, quality=q))
        mse = np.mean((recon.astype(np.float64) - photo.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(255**2 / max(mse, 1e-12))
        assert psnr > prev, f"PSNR regressed at q={q}"
        prev = psnr
