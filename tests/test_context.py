"""AC context modeling (phase 7) and the rewritten decoder.

Two things need pinning here, and they are different in kind.

**Causality.** The context is the zigzag position where a symbol's zero run
starts. The encoder computes that vectorized, in bulk, from the whole
coefficient array; the decoder computes it sequentially, one symbol at a time,
with no lookahead. If those two ever disagree the stream silently mis-decodes
into garbage rather than failing, so `test_contexts_match_a_sequential_decoder`
checks the vectorized assignment against a naive loop that only ever looks
backwards.

**The decoder rewrite.** `_decode_blocks_huffman` is written as one inlined
loop for speed, which makes it exactly the kind of code that harbours an
off-by-one. `_reference_decode` below is a deliberately slow, obviously-correct
implementation used to check it.
"""

from __future__ import annotations

import numpy as np
import pytest

import compression as c


# --- An independent reference decoder ---------------------------------------


class _RefBitReader:
    """Bit-at-a-time reader. Slow and obvious, on purpose."""

    def __init__(self, data: bytes) -> None:
        self._bits = [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]
        self._pos = 0

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            if self._pos >= len(self._bits):
                raise EOFError("reference reader ran off the end")
            v = (v << 1) | self._bits[self._pos]
            self._pos += 1
        return v


def _reference_decode(by, bx, bitstream, dc_table, ac_tables, layout):
    """Decode via linear search over (length, code) pairs. No LUTs, no inlining."""

    def lookup(reader, table, what):
        by_code = {(n, code): sym for sym, (code, n) in table.items()}
        code = 0
        for length in range(1, c.MAX_CODE_LENGTH + 1):
            code = (code << 1) | reader.read_bits(1)
            if (length, code) in by_code:
                return by_code[(length, code)]
        raise ValueError(f"Failed to decode {what}")

    reader = _RefBitReader(bitstream)
    coeffs = np.zeros((by * bx, 64), dtype=np.int16)
    prev_dc = 0

    for b in range(by * bx):
        zz = [0] * 64

        cat = lookup(reader, dc_table, "DC")
        if cat:
            prev_dc += c._bits_to_value(reader.read_bits(cat), cat)
        zz[0] = prev_dc

        k = 1
        while k < 64:
            ctx = int(c.AC_LAYOUTS[layout][k])
            symbol = lookup(reader, ac_tables[ctx], "AC")
            if symbol == 0x00:
                break
            if symbol == 0xF0:
                k += 16
                continue
            k += symbol >> 4
            size = symbol & 0x0F
            zz[k] = c._bits_to_value(reader.read_bits(size), size)
            k += 1

        flat = np.zeros(64, dtype=np.int16)
        flat[c.ZIGZAG_ORDER] = zz
        coeffs[b] = flat

    return coeffs.reshape(by, bx, c.BLOCK_SIZE, c.BLOCK_SIZE)


# --- Causality of the context ------------------------------------------------


def _positions_sequentially(symbols, blocks):
    """Emission positions as a decoder derives them: strictly backward-looking."""

    out = np.zeros(len(symbols), dtype=np.int64)
    k, current = 1, -1
    for i in range(len(symbols)):
        if blocks[i] != current:
            current, k = blocks[i], 1
        out[i] = k
        sym = symbols[i]
        if sym == 0x00:        # EOB ends the block
            pass
        elif sym == 0xF0:      # ZRL skips 16
            k += 16
        else:
            k += (sym >> 4) + 1
    return np.clip(out, 1, 63)


@pytest.mark.parametrize("quality", [1, 10, 50, 90, 100])
def test_contexts_match_a_sequential_decoder(photo, quality):
    """The vectorized position assignment must equal the decoder's view.

    A mismatch does not raise -- it silently decodes against the wrong table
    and produces garbage -- so this is the load-bearing test of phase 7.
    """

    comp = c.compress_array(photo, quality=quality)
    stream = c._scan_symbols(comp.coeffs)
    expected = _positions_sequentially(stream.ac_symbols, stream.ac_block)
    assert np.array_equal(stream.ac_positions, expected)


def test_contexts_match_across_the_synthetic_corpus(corpus):
    """Degenerate blocks -- all-zero, dense, maximum-magnitude -- included."""

    for name, img in corpus.items():
        comp = c.compress_array(img, quality=50)
        stream = c._scan_symbols(comp.coeffs)
        expected = _positions_sequentially(stream.ac_symbols, stream.ac_block)
        assert np.array_equal(stream.ac_positions, expected), name


def test_every_layout_agrees_with_the_sequential_view(photo):
    """Position -> context is a pure lookup, so every layout must agree too."""

    comp = c.compress_array(photo, quality=50)
    stream = c._scan_symbols(comp.coeffs)
    positions = _positions_sequentially(stream.ac_symbols, stream.ac_block)
    for layout in range(len(c.AC_LAYOUTS)):
        expected = c.AC_LAYOUTS[layout][positions]
        assert np.array_equal(stream.contexts(layout), expected), layout


def test_an_all_zero_block_emits_its_eob_in_the_first_band():
    """An empty block's EOB sits at position 1, not at the end of the scan."""

    coeffs = np.zeros((2, 2, 8, 8), dtype=np.int16)
    stream = c._scan_symbols(coeffs)
    assert list(stream.ac_symbols) == [0x00] * 4
    assert list(stream.ac_positions) == [1] * 4


@pytest.mark.parametrize("layout", range(len(c.AC_LAYOUT_EDGES)))
def test_band_tables_are_contiguous_and_complete(layout):
    band = c.AC_LAYOUTS[layout]
    size = c.AC_LAYOUT_SIZES[layout]
    assert band.shape == (64,)
    assert band.min() == 0 and band.max() == size - 1
    # Bands must be contiguous and non-decreasing across the scan.
    assert np.all(np.diff(band[1:]) >= 0)
    assert len(np.unique(band[1:])) == size


def test_layout_zero_is_a_single_table():
    """Layout 0 must reproduce ICJ2 behaviour, which is what bounds the format.

    Every other layout is only ever chosen when it prices cheaper, so this is
    what makes "ICJ3 is never worse than ICJ2" true rather than hoped for.
    """

    assert c.AC_LAYOUT_SIZES[0] == 1
    assert np.all(c.AC_LAYOUTS[0] == 0)


# --- Decoder equivalence -----------------------------------------------------


@pytest.mark.parametrize("quality", [10, 50, 90])
def test_decoder_matches_the_reference(photo, quality):
    small = photo[:64, :64]
    comp = c.compress_array(small, quality=quality)
    bitstream, dc_table, ac_tables, layout = c._encode_blocks_huffman(comp.coeffs)
    by, bx = comp.coeffs.shape[:2]

    fast = c._decode_blocks_huffman(by, bx, bitstream, dc_table, ac_tables, layout)
    slow = _reference_decode(by, bx, bitstream, dc_table, ac_tables, layout)

    assert np.array_equal(fast, slow)
    assert np.array_equal(fast, comp.coeffs)


def test_decoder_matches_the_reference_on_degenerate_content(corpus):
    for name, img in corpus.items():
        comp = c.compress_array(img, quality=50)
        bitstream, dc_table, ac_tables, layout = c._encode_blocks_huffman(comp.coeffs)
        by, bx = comp.coeffs.shape[:2]
        fast = c._decode_blocks_huffman(by, bx, bitstream, dc_table, ac_tables, layout)
        slow = _reference_decode(by, bx, bitstream, dc_table, ac_tables, layout)
        assert np.array_equal(fast, slow), name


def test_long_codes_fall_back_correctly():
    """Codes longer than the LUT window must still decode.

    The fast path resolves a code in one table index only if it fits in
    DECODE_LUT_BITS. Natural images stay well inside that, so the fallback
    would otherwise never be exercised -- and a broken rare path is worse than
    a broken common one, because nothing reveals it.
    """

    # A maximally skewed distribution drives one code to ~14 bits.
    freqs = {i: 1 for i in range(16)}
    freqs[0] = 1 << 20
    for i in range(1, 16):
        freqs[i] = 1 << (16 - i)
    table = c._build_huffman_table(freqs)
    assert max(n for _cd, n in table.values()) > c.DECODE_LUT_BITS

    lut = c._build_decode_lut(table)
    symbols, lengths, long_codes = lut
    assert long_codes, "expected at least one code past the LUT window"

    # Every symbol must round-trip through the LUT plus its fallback.
    for sym, (code, n) in table.items():
        writer = c._BitWriter()
        writer.write_bits(code, n)
        writer.write_bits(0, 24)      # padding so the reader can always fill
        writer.flush()
        data = writer.bytes
        if n <= c.DECODE_LUT_BITS:
            window = int.from_bytes(data[:2], "big") >> (16 - c.DECODE_LUT_BITS)
            assert symbols[window] == sym
        else:
            assert long_codes[(n, code)] == sym


# --- Container ---------------------------------------------------------------


def test_unused_contexts_are_omitted_not_serialized_empty(tmp_path):
    """A one-band image must not pay for 14 empty tables.

    Serialising all contexts unconditionally cost 4 bytes each: 56 bytes of
    overhead on a file whose entire payload is a couple of bytes. Phase 4.1
    fought for exactly these bytes; this guards the win.
    """

    from PIL import Image

    img = np.full((3, 5), 128, dtype=np.uint8)
    src, dst = tmp_path / "in.png", tmp_path / "out.icj"
    Image.fromarray(img, mode="L").save(src)
    c.compress_huffman_file(str(src), str(dst), quality=50)

    comp = c.compress_array(img, quality=50)
    _, _, ac_tables, layout = c._encode_blocks_huffman(comp.coeffs)
    assert sum(1 for t in ac_tables if t) == 1, "expected a single used band"
    assert dst.stat().st_size < 45


def test_container_rejects_an_unknown_band_layout(tmp_path, photo):
    """The layout id is range-checked, so a file from a build with other
    AC_LAYOUT_EDGES is rejected rather than silently mis-decoded."""

    from PIL import Image

    src, dst = tmp_path / "in.png", tmp_path / "out.icj"
    Image.fromarray(photo[:32, :32], mode="L").save(src)
    c.compress_huffman_file(str(src), str(dst), quality=50)

    # The layout byte follows the DC table, whose length is self-describing.
    raw = bytearray(dst.read_bytes())
    dc_count = int.from_bytes(raw[17:19], "big")
    layout_offset = 19 + 2 * dc_count
    assert raw[layout_offset] < len(c.AC_LAYOUTS)
    raw[layout_offset] = len(c.AC_LAYOUTS)
    dst.write_bytes(raw)

    with pytest.raises(ValueError, match="band layout|AC_LAYOUT_EDGES"):
        c.decompress_huffman_file(str(dst), str(tmp_path / "out.png"))


def test_icj3_magic_rejects_icj2(tmp_path, photo):
    from PIL import Image

    src, dst = tmp_path / "in.png", tmp_path / "out.icj"
    Image.fromarray(photo[:32, :32], mode="L").save(src)
    c.compress_huffman_file(str(src), str(dst), quality=50)

    raw = bytearray(dst.read_bytes())
    assert raw[:4] == b"ICJ3"
    raw[:4] = b"ICJ2"
    dst.write_bytes(raw)

    with pytest.raises(ValueError, match="bad magic"):
        c.decompress_huffman_file(str(dst), str(tmp_path / "out.png"))


# --- The gain ----------------------------------------------------------------


@pytest.mark.parametrize("quality", [1, 10, 30, 50, 80, 100])
def test_the_chosen_layout_is_never_worse_than_a_single_table(photo, quality):
    """The safety property that makes ICJ3 unconditionally >= ICJ2.

    This is not decoration. A fixed 15-band layout is an outright *loss* on
    small images at low quality -- measured at -7.1% on this 256px fixture at
    q10, because 15 tables cost more than the entire AC payload. Pricing every
    layout and keeping the best is what removes that regression, and layout 0
    being a single table is what bounds it.
    """

    comp = c.compress_array(photo, quality=quality)
    stream = c._scan_symbols(comp.coeffs)

    single_cost, _ = c._ac_layout_cost(stream, 0)
    layout, _tables = c._choose_ac_layout(stream)
    chosen_cost, _ = c._ac_layout_cost(stream, layout)

    assert chosen_cost <= single_cost, (
        f"q{quality}: chose layout {layout} costing {chosen_cost:.0f} B over "
        f"a single table at {single_cost:.0f} B"
    )


@pytest.mark.parametrize("quality", [30, 50, 80])
def test_context_tables_pay_for_themselves_on_a_photograph(photo, quality):
    """Above the smallest sizes, splitting must be a real win, not a wash.

    Scored on code bits *plus* serialized tables -- an over-split model shows
    up here as a loss, which is exactly how the 15-band layout was caught.
    """

    comp = c.compress_array(photo, quality=quality)
    stream = c._scan_symbols(comp.coeffs)

    single_cost, _ = c._ac_layout_cost(stream, 0)
    layout, _ = c._choose_ac_layout(stream)
    chosen_cost, _ = c._ac_layout_cost(stream, layout)

    gain = 1.0 - chosen_cost / single_cost
    assert gain > 0.02, f"context modeling gained only {gain:.1%} at q{quality}"


def test_finer_layouts_win_on_larger_images(photo):
    """Layout choice must track image size, not be incidental.

    The whole justification for making it adaptive is that the optimum moves.
    Tiling the fixture gives the same statistics with more symbols to amortise
    tables against, so the chosen layout must not get coarser.
    """

    small = c._scan_symbols(c.compress_array(photo[:64, :64], quality=50).coeffs)
    large = c._scan_symbols(c.compress_array(np.tile(photo, (3, 3)), quality=50).coeffs)

    small_layout, _ = c._choose_ac_layout(small)
    large_layout, _ = c._choose_ac_layout(large)
    assert large_layout >= small_layout, (
        f"more symbols chose a coarser layout ({large_layout} < {small_layout})"
    )


def test_quantizer_output_is_unchanged_by_context_coding(photo):
    """Phase 7 is entropy-side only: the same coefficients, coded better.

    Guards against a future change to the trellis rate model quietly turning
    a lossless coding win into a rate-distortion slide, which would make the
    BD-rate improvement unattributable.
    """

    for quality in (10, 50, 90):
        comp = c.compress_array(photo, quality=quality)
        recon = c.decompress_to_array(comp)
        mse = float(np.mean((recon.astype(np.float64) - photo.astype(np.float64)) ** 2))
        # These are the phase 4.2 values, unchanged by phase 7.
        expected = {10: 381.9600, 50: 88.0864, 90: 7.4462}[quality]
        assert mse == pytest.approx(expected, abs=1e-4)
