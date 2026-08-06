"""Golden regression: pinned rate and distortion for the reference photo.

These numbers exist to make silent rate/quality drift impossible during the
optimization phases. They are *expected* to change when the pipeline changes
on purpose -- notably:

- Phase 3 (float32 DCT) shifts ~0.007% of coefficients by +/-1.
- Phase 4.1 (compact Huffman tables) shrinks the container header.
- Phase 4.2 (trellis quantization) reduces the bitstream at equal quality.

When one of those lands, re-pin **deliberately, in the same commit**, and
state the measured delta in the commit message. Never re-pin to silence an
unexplained change: that is precisely the drift this file exists to catch.

Measured against dog.png downscaled to 256x256 (LANCZOS).
"""

from __future__ import annotations

import numpy as np
import pytest

import compression as c


# quality -> (bitstream bytes, MSE, count of non-zero quantized coefficients)
#
# Re-pinned in phase 4.2 (trellis quantization on by default). Rate drops
# sharply and MSE rises at the same quality number -- that is the intended
# trade, and on its own it proves nothing, since lowering --quality does the
# same. The justification is the equal-PSNR comparison in test_trellis.py:
# ~9-10% fewer bytes at matched quality on photographic content.
#
# Pin history:
#   phase 2 (float64 DCT):  10: (3411, 274.6319, 5393)
#                           50: (10118, 68.2794, 14697)
#                           90: (22657, 7.1094, 29711)
#   phase 3 (float32 DCT):  10: (3411, 274.7267, 5393)
#                           50: (10117, 68.2933, 14693)
#                           90: (22657, 7.1163, 29704)
#   phase 4.2 (trellis):    10: (2152, 381.9600, 3364)
#                           50: (7927, 88.0864, 11101)
#                           90: (21716, 7.4462, 28499)
#
# Re-pinned in phase 7 (AC context tables, ICJ3). Unlike every previous
# re-pin, this one is *purely* entropy-side: MSE and the non-zero coefficient
# counts are unchanged to the last digit, because the quantizer output is
# byte-identical and only the coding of it improved.
#
# Re-pinned again after the quality dial was recalibrated. Trellis slides down
# the R-D curve at a fixed quantizer, so an uncorrected q50 looked like
# libjpeg q31; requested quality is now a JPEG-equivalent target mapped to a
# finer internal quality (q10 -> 22, q50 -> 65, q90 -> 91). Rate and MSE both
# move because the quantizer moved -- this pin is not comparable to the ones
# above, which were taken on the uncalibrated scale.
#
# CAVEAT: these are *payload* bytes, and on this 256x256 fixture payload alone
# is misleading. The encoder picks the AC band layout that minimises payload
# PLUS tables, so it will happily spend payload bytes to save more in tables:
# a coarser split can cost payload while saving more in table bytes, which a
# payload-only pin reads as a regression. Whole-file rate is what `bench` and
# `test_bench.py` score; this pin exists to catch *unintended* drift, so read
# a change here alongside the BD-rate gate, not on its own.
#
# Layouts chosen at these pins: 3 bands at q10, 5 at q50 and q90.
GOLDEN = {
    10: (3325, 270.0701, 4693),
    50: (9811, 52.1434, 14430),
    90: (21305, 6.3798, 29564),
}

RATE_TOLERANCE = 0.0  # exact; float32 is deterministic for a given numpy build
MSE_TOLERANCE = 1e-4


@pytest.mark.parametrize("quality", sorted(GOLDEN))
def test_golden_rate_and_distortion(photo, quality):
    expected_bytes, expected_mse, expected_nonzero = GOLDEN[quality]

    comp = c.compress_array(photo, quality=quality)
    bitstream, _, _, _ = c._encode_blocks_huffman(comp.coeffs)
    recon = c.decompress_to_array(comp)
    mse = float(np.mean((recon.astype(np.float64) - photo.astype(np.float64)) ** 2))
    nonzero = int((comp.coeffs != 0).sum())

    assert nonzero == expected_nonzero, (
        f"quantizer output changed at q={quality}: "
        f"{nonzero} non-zero coefficients vs pinned {expected_nonzero}"
    )
    assert mse == pytest.approx(expected_mse, abs=MSE_TOLERANCE), (
        f"reconstruction quality drifted at q={quality}"
    )

    if RATE_TOLERANCE == 0.0:
        assert len(bitstream) == expected_bytes, (
            f"bitstream size changed at q={quality}: "
            f"{len(bitstream)} vs pinned {expected_bytes}"
        )
    else:
        assert len(bitstream) == pytest.approx(expected_bytes, rel=RATE_TOLERANCE)


def test_golden_covers_a_real_photograph(real_images):
    """Guard against the R-D gate silently degrading to synthetic-only.

    Trellis and quantization-table tuning must be measured on natural image
    statistics; a corpus that lost its real photographs would overfit.
    """

    assert "dog" in real_images
    assert real_images["dog"].ndim == 2
