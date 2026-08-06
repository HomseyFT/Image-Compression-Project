"""Terminal viewer tests.

The viewer is output-only, so the things worth pinning are the ones that
produce *wrong-looking* output rather than exceptions: geometry (aspect ratio
and cell counts) and colour-mode detection. Emitting truecolor escapes into a
terminal that cannot render them yields confetti, and emitting them into a
pipe corrupts the file -- both are silent failures that no round-trip test
would notice.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest
from PIL import Image

import compression as c
import icjview as v


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


def test_render_emits_one_line_per_cell_row():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    out = v.render(img, cols=60, rows=20, mode=v.TRUECOLOR)
    lines = out.split("\n")
    assert len(lines) == 20                       # 40 pixel rows / 2
    assert all(line.count(v.UPPER_HALF) == 60 for line in lines)


def test_render_fits_within_a_small_terminal():
    img = np.zeros((512, 768, 3), dtype=np.uint8)
    out = v.render(img, cols=40, rows=10, mode=v.TRUECOLOR)
    lines = out.split("\n")
    assert len(lines) <= 10
    assert all(line.count(v.UPPER_HALF) <= 40 for line in lines)


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
