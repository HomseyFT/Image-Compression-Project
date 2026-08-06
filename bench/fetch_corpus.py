"""Download the Kodak test corpus into ``tests/images/``.

``python -m bench.fetch_corpus``

Tuning against a single image risks overfitting -- the concern already raised
against quantization-table tuning in SPEC.md 4.3. The Kodak set (24 uncompressed
photographs) is the conventional choice for image-codec R-D evaluation.

The images are **not** committed: ``tests/images/`` is gitignored. Both
``tests/conftest.py::real_images`` and ``python -m bench`` pick up whatever is
present, so the corpus is optional and the suite degrades to ``dog.png`` alone
without it.
"""

from __future__ import annotations

import pathlib
import sys
import urllib.error
import urllib.request

BASE_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
COUNT = 24
DEST = pathlib.Path(__file__).resolve().parent.parent / "tests" / "images"


def fetch(dest: pathlib.Path = DEST, count: int = COUNT) -> int:
    dest.mkdir(parents=True, exist_ok=True)

    ok = 0
    for i in range(1, count + 1):
        target = dest / f"kodim{i:02d}.png"
        if target.exists():
            print(f"  have  kodim{i:02d}.png")
            ok += 1
            continue

        url = BASE_URL.format(i)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  FAIL  kodim{i:02d}.png: {exc}", file=sys.stderr)
            continue

        # Write via a temporary name so an interrupted run cannot leave a
        # truncated PNG that later looks like a valid corpus entry.
        tmp = target.with_suffix(".png.part")
        tmp.write_bytes(data)
        tmp.rename(target)
        print(f"  got   kodim{i:02d}.png ({len(data):,} B)")
        ok += 1

    return ok


def main() -> int:
    print(f"Fetching the Kodak corpus into {DEST}")
    ok = fetch()
    print(f"\n{ok}/{COUNT} images available.")
    if ok < COUNT:
        print("Re-run to retry the failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
