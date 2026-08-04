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
# Re-pinned in phase 3 (float64 loop -> float32 batched DCT). The change was
# verified as pure precision, not a restructuring error: running the new
# vectorized path in float64 reproduces the old per-block loop bit-exactly at
# every quality, and the float32 deltas are max |1| on 0.003%-0.078% of
# coefficients. Full-image PSNR moved 35.41 dB -> 35.41 dB.
#
# Previous (float64) pins, for reference:
#   10: (3411, 274.6319, 5393)
#   50: (10118, 68.2794, 14697)
#   90: (22657, 7.1094, 29711)
GOLDEN = {
    10: (3411, 274.7267, 5393),
    50: (10117, 68.2933, 14693),
    90: (22657, 7.1163, 29704),
}

RATE_TOLERANCE = 0.0  # exact; float32 is deterministic for a given numpy build
MSE_TOLERANCE = 1e-4


@pytest.mark.parametrize("quality", sorted(GOLDEN))
def test_golden_rate_and_distortion(photo, quality):
    expected_bytes, expected_mse, expected_nonzero = GOLDEN[quality]

    comp = c.compress_array(photo, quality=quality)
    bitstream, _, _ = c._encode_blocks_huffman(comp.coeffs)
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
