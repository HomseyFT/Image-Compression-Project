"""Terminal viewer tests.

The viewer is output-only, so the things worth pinning are the ones that
produce *wrong-looking* output rather than exceptions: geometry (aspect ratio
and cell counts) and colour-mode detection. Emitting truecolor escapes into a
terminal that cannot render them yields confetti, and emitting them into a
pipe corrupts the file -- both are silent failures that no round-trip test
would notice.
"""

from __future__ import annotations

import base64
import io
import pathlib
import re
import zlib

import numpy as np
import pytest
from PIL import Image

import compression as c
import icjview as v
from bench import viewer_quality as vq


@pytest.fixture(scope="module")
def icj(tmp_path_factory) -> pathlib.Path:
    tmp = tmp_path_factory.mktemp("view")
    src, dst = tmp / "in.png", tmp / "in.icj"
    img = np.random.RandomState(0).randint(0, 256, (64, 96, 3)).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(src)
    c.compress_huffman_file(str(src), str(dst), quality=60)
    return dst


# --- Geometry ----------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,cols,rows",
    [((512, 768), 80, 24), ((768, 512), 80, 24), ((10, 10), 200, 100), ((1, 1), 80, 24)],
)
def test_fit_never_exceeds_the_cell_budget(shape, cols, rows):
    h, w = v.fit(shape, cols, rows)
    assert 1 <= w <= cols
    assert 1 <= h <= rows * 2      # two pixel rows per cell


def test_fit_preserves_aspect_ratio():
    """A 3:2 image must stay 3:2, or the viewer lies about the picture."""

    h, w = v.fit((512, 768), 120, 100)
    assert w / h == pytest.approx(768 / 512, rel=0.05)


def test_fit_does_not_upscale():
    """Small images should render at native size, not be blown up and blurred."""

    assert v.fit((10, 20), 200, 100) == (10, 20)


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def glyphs(line: str) -> str:
    """Visible characters in a rendered line, escapes stripped."""

    return ANSI.sub("", line)


def test_render_emits_one_line_per_cell_row():
    """One glyph per cell, whichever glyph that is.

    This used to count ``U+2580`` specifically. That was a proxy for the cell
    count that only held while every cell was painted with the same character:
    a uniform cell is now rendered as a space over its background colour,
    which is exact and cheaper, and sub-cell modes draw from a 16- or
    64-glyph table. Counting visible characters measures what the assertion
    was always for.
    """

    img = np.zeros((40, 60, 3), dtype=np.uint8)
    out = v.render(img, cols=60, rows=20, mode=v.TRUECOLOR)
    lines = out.split("\n")
    assert len(lines) == 20                       # 40 pixel rows / 2
    assert all(len(glyphs(line)) == 60 for line in lines)


def test_render_fits_within_a_small_terminal():
    img = np.zeros((512, 768, 3), dtype=np.uint8)
    out = v.render(img, cols=40, rows=10, mode=v.TRUECOLOR)
    lines = out.split("\n")
    assert len(lines) <= 10
    assert all(len(glyphs(line)) <= 40 for line in lines)


def test_odd_pixel_heights_do_not_drop_a_row():
    """An odd resize height must be padded to whole top/bottom pairs.

    Without this the last pixel row silently vanishes, or `bottom` is shorter
    than `top` and rendering raises.
    """

    img = np.zeros((41, 33, 3), dtype=np.uint8)
    out = v.render(img, cols=33, rows=21, mode=v.TRUECOLOR)
    assert out.split("\n")


# --- Colour modes ------------------------------------------------------------


def test_truecolor_and_256_emit_different_escapes():
    img = np.random.RandomState(1).randint(0, 256, (8, 8, 3)).astype(np.uint8)
    true = v.render(img, 8, 4, mode=v.TRUECOLOR)
    idx = v.render(img, 8, 4, mode=v.COLOR256)
    assert "\x1b[38;2;" in true and "\x1b[38;5;" not in true
    assert "\x1b[38;5;" in idx and "\x1b[38;2;" not in idx


def test_mono_emits_no_escape_sequences():
    """The fallback must be safe to redirect into a file."""

    img = np.random.RandomState(2).randint(0, 256, (16, 16, 3)).astype(np.uint8)
    out = v.render(img, 16, 8, mode=v.MONO)
    assert "\x1b" not in out
    assert set(out.replace("\n", "")) <= set(v.ASCII_RAMP)


def test_grayscale_renders_without_a_channel_axis():
    img = np.random.RandomState(3).randint(0, 256, (16, 16)).astype(np.uint8)
    assert v.render(img, 16, 8, mode=v.TRUECOLOR)
    assert v.render(img, 16, 8, mode=v.MONO)


def test_non_tty_falls_back_to_mono():
    """Piping must not inject escapes into the redirected output."""

    assert v.detect_color_mode(io.StringIO()) == v.MONO


def test_no_color_env_is_respected(monkeypatch):
    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("NO_COLOR", "1")
    assert v.detect_color_mode(FakeTTY()) == v.MONO

    monkeypatch.delenv("NO_COLOR")
    assert v.detect_color_mode(FakeTTY()) == v.TRUECOLOR


def test_dumb_terminal_falls_back(monkeypatch):
    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORTERM", "")
    monkeypatch.setenv("TERM", "dumb")
    assert v.detect_color_mode(FakeTTY()) == v.MONO


# --- Loading -----------------------------------------------------------------


def test_loads_an_icj_file(icj):
    img, meta = v.load(str(icj))
    assert img.shape == (64, 96, 3)
    assert meta["format"] == "ICJ4"
    assert meta["components"] == 3
    assert meta["sampling"] == "4:2:0"
    assert meta["file_bytes"] == icj.stat().st_size


def test_loads_an_ordinary_image(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(p)
    img, meta = v.load(str(p))
    assert img.shape == (8, 8, 3)
    assert meta["format"] == "PNG"


def test_older_containers_are_reported_clearly(tmp_path, icj):
    """A stale .icj must say so, not fail as a corrupt image."""

    raw = bytearray(icj.read_bytes())
    raw[:4] = b"ICJ2"
    stale = tmp_path / "old.icj"
    stale.write_bytes(raw)

    with pytest.raises(ValueError, match="older container"):
        v.load(str(stale))


def test_missing_file_raises_not_crashes(tmp_path):
    with pytest.raises(FileNotFoundError):
        v.load(str(tmp_path / "nope.icj"))


def test_info_reports_the_compression_ratio(icj):
    text = v.format_info(str(icj), v.load(str(icj))[1])
    assert "ICJ4" in text and "4:2:0" in text
    assert "x" in text and "vs raw" in text
    # The stored quality is post-calibration; the viewer must say so rather
    # than let it read as a discrepancy against what the user asked for.
    assert "calibrated" in text


# --- CLI ---------------------------------------------------------------------


def test_cli_renders(icj, capsys):
    assert v.main([str(icj), "--width", "20", "--height", "6", "--color", "mono"]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_compare_places_images_side_by_side(icj, capsys):
    v.main([str(icj), str(icj), "--compare", "--width", "40", "--height", "6",
            "--color", "mono"])
    lines = [l for l in capsys.readouterr().out.split("\n") if l.strip()]
    assert lines and all(len(l) <= 41 for l in lines)


def test_cli_reports_a_missing_file(tmp_path, capsys):
    assert v.main([str(tmp_path / "gone.icj")]) == 1
    assert "icjview" in capsys.readouterr().err


# --- Phase 10: colour correctness --------------------------------------------


@pytest.fixture(scope="module")
def rgb_photo() -> np.ndarray:
    """A real photograph in colour.

    ``real_images`` is grayscale, and the defects this phase fixes -- cube
    quantization and gamma-incorrect resampling -- are largely invisible
    without chroma.
    """

    path = pathlib.Path(__file__).resolve().parent.parent / "images" / "kodim03.png"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m bench.fetch_corpus`."
        )
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def test_cube_levels_map_to_themselves():
    """A colour the palette contains exactly must survive quantization.

    The old mapping failed this for every level but the endpoints, because it
    assumed the cube was linear in 0..255 when its levels are
    0/95/135/175/215/255.
    """

    for level in v.CUBE_LEVELS:
        rgb = np.array([[level, level, level]], dtype=np.uint8)
        idx = int(v._rgb_to_256(rgb)[0])
        assert tuple(v._palette_rgb_by_code()[idx]) == (level, level, level)


def test_nearest_palette_beats_the_linear_cube_assumption(rgb_photo):
    """Regresses SPEC 10.0 finding 2, with the measurement that found it."""

    small = np.array(
        Image.fromarray(rgb_photo).resize((160, 100), Image.LANCZOS), dtype=np.uint8
    ).reshape(-1, 3).astype(float)

    # The mapping as it shipped in phase 9.
    old_cube = 16 + 36 * (small[:, 0] * 5 // 255) + 6 * (small[:, 1] * 5 // 255) + (
        small[:, 2] * 5 // 255
    )
    mx, mn = small.max(axis=1), small.min(axis=1)
    old_grey = 232 + np.clip(small.mean(axis=1) // 1 * 25 // 255, 0, 23)
    old = np.where(mx - mn < 12, old_grey, old_cube).astype(int)

    table = v._palette_rgb_by_code()
    old_err = np.linalg.norm(small - table[old], axis=1).mean()
    new_err = np.linalg.norm(
        small - table[v._rgb_to_256(small.astype(np.uint8))], axis=1
    ).mean()

    assert new_err < old_err * 0.75, (old_err, new_err)


def test_greys_are_candidates_not_a_special_case():
    """A near-grey that the cube represents better must not be forced to the ramp.

    The old ``max - min < 12`` branch chose a palette before asking which was
    closer, so a colour just inside the threshold could be snapped to a
    visibly worse grey.
    """

    rgb = np.array([[95, 95, 100]], dtype=np.uint8)
    idx = int(v._rgb_to_256(rgb)[0])
    chosen = v._palette_rgb_by_code()[idx].astype(int)

    table = v._palette_rgb_by_code()
    codes = v._xterm_palette()[1]
    best = min(codes, key=lambda i: ((table[i].astype(int) - rgb[0]) ** 2).sum())
    assert ((chosen - rgb[0]) ** 2).sum() == ((table[best].astype(int) - rgb[0]) ** 2).sum()


def test_resampling_happens_in_linear_light():
    """Averaging a black/white checkerboard must give mid-*linear*, not mid-code.

    Half black and half white is 0.5 in linear light, which encodes to sRGB
    ~188, not 128. A renderer that returns 128 is averaging gamma-encoded
    values -- SPEC 10.0 finding 3.
    """

    board = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    out = v._resize(np.repeat(board[:, :, None], 3, axis=2), 1, 1)
    assert 180 <= int(out[0, 0, 0]) <= 195


def test_srgb_transfer_round_trips():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    assert np.allclose(v.linear_to_srgb(v.srgb_to_linear(x)), x, atol=1e-5)


def test_ascii_dithering_improves_local_accuracy_on_a_gradient():
    """A 10-level ramp bands badly; error diffusion trades that for noise.

    Judged pixel by pixel, dithering is a *regression* -- it adds
    high-frequency error on purpose. The claim it makes is about local means,
    which is what the eye integrates, so that is what is measured.

    This is the ASCII path specifically. The colour cell paths do not dither:
    ``fg``/``bg`` are per-cell colour lists rather than images, and diffusing
    error through them lost to plain nearest-palette on 11 of 11 corpus
    images. Here the canvas really is an image, so it works.
    """

    ramp = np.linspace(10, 245, 96, dtype=np.uint8)
    rgb = np.repeat(np.repeat(ramp[None, :], 48, axis=0)[:, :, None], 3, axis=2)

    dithered = v.rasterize(rgb, 96, 24, v.MONO, v.ASCII)

    n = len(v.ASCII_RAMP)
    levels = np.linspace(0, 255, n)
    canvas = v._canvas(rgb, 96, 24, v.ASCII, 2.0)
    luma = canvas[0::2, :, 0].astype(float)
    plain = levels[np.abs(luma[..., None] - levels).argmin(-1)].astype(np.uint8)
    plain = np.repeat(np.repeat(plain, 2, axis=0)[..., None], 3, axis=2)

    ref = vq.reference(rgb, dithered.shape[0], dithered.shape[1])
    assert vq.block_delta_e(dithered, ref) < vq.block_delta_e(plain, ref)


def test_colour_cell_paths_do_not_dither(rgb_photo):
    """Regresses the speckle: error diffusion on fg/bg must not come back.

    It was implemented, measured, and removed. Nearest-palette beat it
    pixelwise on 11 of 11 corpus images, because the glyph mask interleaves
    the two planes back together and their independent error fields recombine
    as noise. The cell mask already provides the spatial modulation dithering
    would have added.
    """

    assert not hasattr(v, "_dither_to_palette")

    painted = v.rasterize(rgb_photo, 64, 24, v.COLOR256, v.QUADRANT)
    used = {tuple(c) for c in painted.reshape(-1, 3)}
    palette = {tuple(c) for c in v._palette_rgb_by_code()}
    assert used <= palette


# --- Phase 10: geometry with sub-cell divisions -------------------------------


@pytest.mark.parametrize("kind", [v.BLOCKS, v.QUADRANT, v.SEXTANT])
def test_fit_preserves_aspect_for_every_subdivision(kind):
    """Sub-cell modes change the pixel aspect; the picture must not stretch."""

    sub_x, sub_y = v.SUBDIVISIONS[kind]
    aspect = 2.2
    h, w = v.fit((512, 768), 120, 100, sub_x, sub_y, aspect)

    displayed = (w / sub_x) / (h / sub_y * aspect)
    assert displayed == pytest.approx(768 / 512, rel=0.05)


@pytest.mark.parametrize("kind", [v.BLOCKS, v.QUADRANT, v.SEXTANT])
def test_render_respects_the_cell_budget_for_every_subdivision(kind):
    img = np.random.RandomState(7).randint(0, 256, (200, 300, 3)).astype(np.uint8)
    out = v.render(img, 40, 12, v.TRUECOLOR, kind)
    lines = out.split("\n")
    assert len(lines) <= 12
    assert all(len(glyphs(line)) <= 40 for line in lines)


def test_cell_aspect_falls_back_when_the_terminal_is_silent():
    """Most terminals leave ws_xpixel zeroed; the constant is a fallback."""

    assert v.cell_aspect(io.StringIO()) == v.DEFAULT_CELL_ASPECT
    assert v.cell_pixels(io.StringIO()) == v.DEFAULT_CELL_PIXELS


def test_cli_accepts_a_cell_aspect_override(icj, capsys):
    assert v.main([str(icj), "--width", "20", "--height", "6",
                   "--color", "mono", "--cell-aspect", "2.4"]) == 0
    assert capsys.readouterr().out.strip()


# --- Phase 10: sub-cell glyph tables ------------------------------------------


def test_quadrant_table_is_complete_and_distinct():
    assert len(v.QUADRANT_GLYPHS) == 16
    assert len(set(v.QUADRANT_GLYPHS)) == 16


def test_sextant_table_skips_the_two_aliased_masks():
    """U+1FB00..U+1FB3B omits masks 21 and 42, which are U+258C and U+2590.

    Off-by-one here is silent: every glyph past the gap is simply the wrong
    shape, and the picture stays plausible.
    """

    table = v._sextant_glyphs()
    assert len(table) == 64
    assert len(set(table)) == 64

    assert table[0] == " " and table[63] == v.FULL_BLOCK
    assert table[21] == "▌" and table[42] == "▐"
    assert table[1] == "\U0001fb00"
    assert table[62] == "\U0001fb3b"

    body = [table[m] for m in range(1, 63) if m not in (21, 42)]
    assert all(0x1FB00 <= ord(ch) <= 0x1FB3B for ch in body)


def test_uniform_cells_render_as_a_bare_background():
    """No foreground work for a flat cell, and no colour error either."""

    cells = np.full((2, 3, 4, 3), 77, dtype=np.uint8)
    mask, fg, bg = v._two_colour_split(cells)
    assert (mask == 0).all()
    assert (bg == 77).all()


def test_two_colour_split_separates_equal_luma_colours():
    """Luma thresholding cannot do this; the principal-axis split must.

    Two colours of the same brightness in a cell are exactly where a
    luma-based split collapses, and it collapses to a flat average -- the
    most visible failure available.
    """

    a, b = np.array([180, 40, 40]), np.array([40, 40, 180])
    cell = np.array([[[a, a, b, b]]], dtype=np.uint8)

    mask, fg, bg = v._two_colour_split(cell)
    assert mask[0, 0] not in (0, 15)

    pair = {tuple(fg[0, 0]), tuple(bg[0, 0])}
    for want in (a, b):
        assert min(sum((np.array(p) - want) ** 2) for p in pair) < 400


# --- Phase 10: graphics protocols ---------------------------------------------


def _kitty_payload(out: str) -> tuple[dict, bytes]:
    """Parse a kitty transmission back into its controls and pixel bytes."""

    chunks = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\\\?", out)
    assert chunks, "no APC graphics sequences emitted"

    controls = dict(kv.split("=", 1) for kv in chunks[0][0].split(",") if "=" in kv)
    payload = "".join(chunk for _, chunk in chunks)
    return controls, zlib.decompress(base64.b64decode(payload))


def test_kitty_output_decodes_back_to_the_image():
    """Envelope plus round-trip: the bytes on the wire must be the picture.

    Golden byte pins were rejected for this (SPEC 10.5) -- they break on any
    zlib or Pillow change while catching nothing this does not.
    """

    img = np.random.RandomState(11).randint(0, 256, (64, 64, 3)).astype(np.uint8)
    out = v.render_kitty(img, 40, 12, cell=(8, 16))

    controls, raw = _kitty_payload(out)
    assert controls["a"] == "T" and controls["f"] == "24"
    assert controls["o"] == "z" and controls["t"] == "d"
    assert controls["C"] == "1", "cursor must not move; the caller advances"

    width, height = int(controls["s"]), int(controls["v"])
    assert len(raw) == width * height * 3

    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    assert decoded.shape[2] == 3
    assert int(controls["c"]) <= 40 and int(controls["r"]) <= 12


def test_kitty_chunks_are_continued_correctly():
    """Every chunk but the last carries m=1, and only the first has controls.

    A dropped continuation marker leaves the terminal waiting for data that
    never arrives, which hangs the display rather than corrupting it.
    """

    img = np.random.RandomState(12).randint(0, 256, (256, 256, 3)).astype(np.uint8)
    out = v.render_kitty(img, 200, 60, cell=(10, 20))

    chunks = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\\\?", out)
    assert len(chunks) > 1, "payload too small to exercise chunking"

    heads = [dict(kv.split("=", 1) for kv in h.split(",") if "=" in kv) for h, _ in chunks]
    assert all(h["m"] == "1" for h in heads[:-1])
    assert heads[-1]["m"] == "0"
    assert "a" in heads[0] and all("a" not in h for h in heads[1:])
    assert all(len(body) <= 4096 for _, body in chunks)


def test_kitty_reserves_the_rows_it_paints():
    """C=1 means the terminal will not advance; the viewer must."""

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = v.render_kitty(img, 40, 12, cell=(8, 16))

    controls, _ = _kitty_payload(out)
    assert out.endswith("\n" * int(controls["r"]))


def test_iterm_output_decodes_back_to_the_image():
    img = np.random.RandomState(13).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    out = v.render_iterm(img, 40, 12, cell=(8, 16))

    assert out.startswith("\x1b]1337;File=inline=1;")
    body = out.split(":", 1)[1].split("\x07", 1)[0]
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(body))))

    assert decoded.ndim == 3 and decoded.shape[2] == 3
    assert "preserveAspectRatio=1" in out


def test_graphics_paths_never_upscale():
    """A thumbnail must not be blown up to fill the window."""

    img = np.zeros((10, 12, 3), dtype=np.uint8)
    controls, raw = _kitty_payload(v.render_kitty(img, 200, 60, cell=(10, 20)))
    assert (int(controls["v"]), int(controls["s"])) == (10, 12)


# --- Phase 10: render-path detection (hermetic) --------------------------------


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def tty_env(monkeypatch):
    for name in ("NO_COLOR", "KITTY_WINDOW_ID", "TERM_PROGRAM", "LC_TERMINAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    return monkeypatch


def test_kitty_is_detected_from_the_environment(tty_env):
    tty_env.setenv("TERM", "xterm-kitty")
    assert v.detect_render_kind(FakeTTY()) == v.KITTY

    tty_env.setenv("TERM", "xterm-256color")
    tty_env.setenv("KITTY_WINDOW_ID", "1")
    assert v.detect_render_kind(FakeTTY()) == v.KITTY


@pytest.mark.parametrize("program", ["ghostty", "WezTerm"])
def test_kitty_protocol_peers_are_detected(tty_env, program):
    """Ghostty and WezTerm implement the same protocol under another name."""

    tty_env.setenv("TERM_PROGRAM", program)
    assert v.detect_render_kind(FakeTTY()) == v.KITTY


def test_iterm_is_detected_from_the_environment(tty_env):
    tty_env.setenv("TERM_PROGRAM", "iTerm.app")
    assert v.detect_render_kind(FakeTTY()) == v.ITERM

    tty_env.delenv("TERM_PROGRAM")
    tty_env.setenv("LC_TERMINAL", "iTerm2")
    assert v.detect_render_kind(FakeTTY()) == v.ITERM


def test_unknown_terminals_get_quadrants(tty_env):
    """Coarser glyphs beat a protocol the terminal cannot draw."""

    assert v.detect_render_kind(FakeTTY()) == v.QUADRANT


def test_non_tty_gets_ascii_regardless_of_environment(tty_env):
    """Redirected output must never receive a bitmap payload."""

    tty_env.setenv("TERM", "xterm-kitty")
    tty_env.setenv("KITTY_WINDOW_ID", "1")
    assert v.detect_render_kind(io.StringIO()) == v.ASCII


def test_no_color_outranks_graphics_protocols(tty_env):
    tty_env.setenv("TERM", "xterm-kitty")
    tty_env.setenv("NO_COLOR", "1")
    assert v.detect_render_kind(FakeTTY()) == v.ASCII


def test_mono_never_emits_escapes_even_on_a_graphics_terminal():
    """--color mono must stay redirectable whatever --render says."""

    img = np.random.RandomState(14).randint(0, 256, (32, 32, 3)).astype(np.uint8)
    out = v.render(img, 20, 8, v.MONO, v.KITTY)
    assert "\x1b" not in out


def test_compare_falls_back_to_cells(icj, capsys):
    """Bitmaps have no text rows to interleave, so side-by-side needs glyphs."""

    v.main([str(icj), str(icj), "--compare", "--width", "40", "--height", "6",
            "--render", "kitty", "--color", "truecolor"])
    out = capsys.readouterr().out
    assert "\x1b_G" not in out
    assert "\x1b[38;2;" in out


# --- Phase 10.6: the fidelity gate --------------------------------------------
#
# Viewer quality had no metric, which is why these defects survived phase 9.
# Each fix below must measurably improve CIEDE2000 against a linear-light
# reference, or it does not ship -- including the gamma fix, which is
# textbook-correct and was still asserted rather than measured.


def test_delta_e_is_zero_for_identical_colours():
    lab = vq.srgb_to_lab(np.array([[120, 60, 200]], dtype=np.uint8))
    assert vq.delta_e_2000(lab, lab)[0] == pytest.approx(0.0, abs=1e-9)


def test_delta_e_matches_a_known_sharma_pair():
    """One row of the Sharma CIEDE2000 reference table, as a sanity anchor."""

    lab1 = np.array([[50.0, 2.6772, -79.7751]])
    lab2 = np.array([[50.0, 0.0, -82.7485]])
    assert vq.delta_e_2000(lab1, lab2)[0] == pytest.approx(2.0425, abs=1e-3)


def test_rasterize_matches_the_cell_grid(rgb_photo):
    """The metric measures the real renderer, not a parallel implementation."""

    painted = v.rasterize(rgb_photo, 40, 12, v.TRUECOLOR, v.QUADRANT)
    sub_x, sub_y = v.SUBDIVISIONS[v.QUADRANT]
    assert painted.shape[0] % sub_y == 0 and painted.shape[1] % sub_x == 0
    assert painted.shape[0] <= 12 * sub_y and painted.shape[1] <= 40 * sub_x


def gamma_naive_resize(img, height, width):
    """``_resize`` as phase 9 shipped it: Lanczos straight on sRGB bytes."""

    rgb = v._to_rgb(img)
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    return np.array(
        Image.fromarray(rgb, mode="RGB").resize((width, height), Image.LANCZOS)
    )


def test_reference_does_not_move_with_the_renderer(monkeypatch):
    """The ground truth must not be built by the code under test.

    An early version of this metric routed ``vq.reference`` through
    ``v._resize``, so swapping the renderer's resampler swapped the reference
    too and every variant scored well against its own assumptions -- which
    "proved" gamma-naive resampling superior across the whole corpus. This
    pins the decoupling that fixed it.
    """

    img = np.random.RandomState(21).randint(0, 256, (64, 64, 3)).astype(np.uint8)
    before = vq.reference(img, 16, 16)

    monkeypatch.setattr(v, "_resize", gamma_naive_resize)
    assert np.array_equal(vq.reference(img, 16, 16), before)


def test_gate_linear_light_resampling_improves_fidelity(rgb_photo, monkeypatch):
    """SPEC 10.1.2 is gate-eligible: correct-in-principle is not enough.

    Only the renderer is swapped; the reference stays linear-light for both
    sides, which is the whole point of the previous test.
    """

    def delta_e():
        painted = v.rasterize(rgb_photo, 64, 24, v.TRUECOLOR, v.QUADRANT)
        ref = vq.reference(rgb_photo, painted.shape[0], painted.shape[1])
        return float(
            vq.delta_e_2000(vq.srgb_to_lab(painted), vq.srgb_to_lab(ref)).mean()
        )

    correct = delta_e()
    monkeypatch.setattr(v, "_resize", gamma_naive_resize)
    naive = delta_e()

    assert correct < naive, (correct, naive)


def test_the_two_instruments_answer_different_questions(rgb_photo):
    """Using the cross-mode metric on a within-mode question inverts it.

    ``score`` expands to the physical grid so modes are comparable, which
    adds a blockiness term far larger than anything a resampler contributes;
    ``canvas_score`` holds the mode fixed and compares at its own canvas.
    Half blocks scoring a perfect 0.0 on the latter is the tell -- two
    colours reproduce two pixels exactly -- and is why it cannot rank modes.
    """

    assert vq.canvas_score(rgb_photo, 64, 24, v.TRUECOLOR, v.BLOCKS)["delta_e"] == 0.0
    assert vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.BLOCKS)["delta_e"] > 0.0

    # The cross-mode instrument must still rank modes by sample count.
    blocks = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.BLOCKS)["delta_e"]
    sext = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.SEXTANT)["delta_e"]
    assert sext < blocks


def test_gate_nearest_palette_improves_fidelity(rgb_photo):
    """SPEC 10.1.1, measured in CIEDE2000 rather than RGB distance.

    Compares the two quantizers directly rather than two whole renders: the
    palette is the only thing under test, and isolating it keeps the result
    from being diluted by everything else the renderer does.
    """

    small = np.array(
        Image.fromarray(rgb_photo).resize((192, 128), Image.LANCZOS), dtype=np.uint8
    )
    table = v._palette_rgb_by_code()

    r, g, b = (small[..., k].astype(int) for k in range(3))
    old_cube = 16 + 36 * (r * 5 // 255) + 6 * (g * 5 // 255) + (b * 5 // 255)
    old_grey = 232 + np.clip((r + g + b) // 3 * 25 // 255, 0, 23)
    old = np.where(small.max(2).astype(int) - small.min(2).astype(int) < 12,
                   old_grey, old_cube)

    lab = vq.srgb_to_lab(small)
    old_de = float(vq.delta_e_2000(vq.srgb_to_lab(table[old]), lab).mean())
    new_de = float(vq.delta_e_2000(vq.srgb_to_lab(table[v._rgb_to_256(small)]), lab).mean())

    assert new_de < old_de, (old_de, new_de)


def test_gate_subcell_modes_beat_half_blocks(rgb_photo):
    """SPEC 10.3: more samples per cell must buy accuracy, not just detail."""

    blocks = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.BLOCKS)
    quad = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.QUADRANT)
    sext = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.SEXTANT)

    assert quad["delta_e"] < blocks["delta_e"]
    assert sext["delta_e"] < blocks["delta_e"]


def test_gate_graphics_protocols_are_near_lossless(rgb_photo):
    """Tier B is gated on the round trip, but it should dominate every glyph
    mode by a wide margin -- that is the whole point of the phase."""

    kitty = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.KITTY)
    quad = vq.score(rgb_photo, 64, 24, v.TRUECOLOR, v.QUADRANT)

    assert kitty["delta_e"] < quad["delta_e"]
    assert kitty["delta_e"] < 1.0        # below the just-noticeable threshold
