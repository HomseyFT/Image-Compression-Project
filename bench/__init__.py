"""Rate-distortion benchmarking for the ICJ codec.

Run ``python -m bench`` for a markdown report.

The public surface is deliberately small:

- :class:`~bench.bdrate.Curve` -- a rate-distortion curve
- :func:`~bench.bdrate.bd_rate` -- % bitrate change at equal quality
- :func:`~bench.bdrate.check_monotone` -- refuses to score unscorable content
- :mod:`bench.codecs` -- adapters producing curves from images
"""

from .bdrate import (
    Curve,
    InsufficientOverlapError,
    NonMonotoneCurveError,
    bd_rate,
    check_monotone,
)

__all__ = [
    "Curve",
    "InsufficientOverlapError",
    "NonMonotoneCurveError",
    "bd_rate",
    "check_monotone",
]
