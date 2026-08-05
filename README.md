# Image-Compression-Project

A minimal, self-contained JPEG-style compressor for **grayscale** images, built
on NumPy and Pillow. It uses an 8×8 block DCT, a JPEG-like luminance
quantization matrix, JPEG-style entropy coding with per-image Huffman tables,
and **rate-distortion optimized (trellis) quantization**.

On the sample image, trellis buys **17.6% fewer bytes at equal reconstruction
quality** — a compression ratio of 22.4× versus 18.4× at 35.4 dB PSNR.

## Install

```bash
pip install -e .          # numpy + pillow
pip install -e '.[dev]'   # + pytest
```

## CLI

```bash
python compression.py compress   input.png  output.icj --quality 50
python compression.py decompress output.icj reconstructed.png
```

Quality is an integer in `[1, 100]`, higher = better quality and larger output
(default 50). Out-of-range values are rejected at the CLI; the library API
clamps instead.

## Programmatic use

```python
import numpy as np
from PIL import Image
import compression as c

arr  = np.array(Image.open("dog.png").convert("L"), dtype=np.uint8)
comp = c.compress_array(arr, quality=50)      # CompressedImage
rec  = c.decompress_to_array(comp)            # uint8 ndarray, same shape

base = c.compress_array(arr, quality=50, trellis=False)   # plain quantization
```

`CompressedImage` carries the quantized coefficients, original shape, padded
shape, and quality — everything needed to reconstruct the image.

## How it works

**Transform.** `_get_dct_matrices` builds and caches the orthonormal DCT-II
basis `D` and its transpose as contiguous `float32`. `_to_blocks` reshapes a
padded image into `(blocks_y, blocks_x, 8, 8)`, and `_forward_dct_2d` /
`_inverse_dct_2d` apply `D @ X @ D.T` — `matmul` broadcasts over the leading
dimensions, so a whole image is transformed in one batched BLAS call.

Working in `float32` rather than `float64` is ~2.7× faster and perturbs about
0.007% of quantized coefficients by ±1, well below the quantization step.

**Quantization.** `_build_quant_matrix` scales the standard JPEG luminance
matrix by the usual quality mapping and clips to `[1, 255]`.

**Trellis (`_trellis_quantize`).** Instead of rounding each coefficient to the
nearest level, a Lagrangian dynamic program runs over the zigzag scan of every
block and chooses — per coefficient — to zero it, keep the nearest level, or
drop one magnitude step, minimising `D + λR`. `R` is the *actual* Huffman cost
including run-length structure, and the end-of-block position is itself a free
decision, so zeroing a coefficient can collapse a run symbol or move the EOB
earlier and pay for itself in bits. The DP state is the pending zero-run;
because the DCT is orthonormal, coefficient-domain squared error equals
pixel-domain squared error, so distortion is scored without an inverse
transform per candidate.

λ is tied to the quantizer via `_trellis_lambda`, so `--quality` keeps its
usual meaning — each quality level simply becomes cheaper in bits.

Trellis is **encoder-side only**: it emits an ordinary coefficient array, so
the decoder and container format are untouched. `trellis=False` reproduces
plain round-to-nearest exactly.

**Entropy coding.** `_scan_symbols` extracts the whole image's DC/AC symbol
stream in one vectorized pass (DC differentials, AC run/size symbols, ZRL,
EOB), `_build_huffman_table` builds optimal per-image canonical tables, and
`_pack_bits` emits the bitstream via `np.packbits`.

## ICJ2 container

```
magic      4 bytes   ASCII 'ICJ2'
height     4 bytes   big-endian unsigned
width      4 bytes   big-endian unsigned
quality    1 byte    1..100 (always clamped before writing)
blocks_y   2 bytes   big-endian unsigned
blocks_x   2 bytes   big-endian unsigned
dc_len     2 bytes   byte length of the DC table
dc_table   dc_len bytes
ac_len     2 bytes   byte length of the AC table
ac_table   ac_len bytes
bit_len    4 bytes   big-endian length of the payload
bitstream  bit_len bytes, big-endian bit packing, zero-padded
```

Each table is a 2-byte symbol count followed by that many 2-byte
`(symbol, code length)` pairs. Canonical codes are a pure function of the
lengths, so the codes themselves are not stored.

ICJ2 replaced ICJ1, which could persist an unclamped quality byte and decode
against the wrong quantization matrix. ICJ1 files are rejected.

## Tests

```bash
python -m pytest
```

697 tests, ~3s. Covers the lossless coefficient round trip, padding and shape
restoration, quality clamping, container fuzzing (bad magic, truncation,
corrupt tables), bit I/O, Huffman construction, trellis invariants, and pinned
golden rate/distortion values.

The synthetic corpus in `tests/conftest.py` is generated from a seeded RNG, so
no binaries live in the repo. Real photographs dropped into `tests/images/`
are picked up automatically.

## Design notes: what was measured and rejected

Several plausible optimizations were measured and **discarded**, which is why
they are absent:

- **Arithmetic / range coding.** The per-image Huffman tables already land
  within **0.76%** of the order-0 entropy bound. Replacing the entropy coder
  cannot recover more than that.
- **Deadzone quantization.** Looked like a 12% win, but that was pure
  rate-distortion sliding: against an equal-size baseline it scored +0.05 /
  +0.01 / −0.13 dB. Trellis is the version of this idea that actually works,
  because it prices the real run-length structure.
- **Improved DC prediction.** A row-wise predictor moved DC cost 45,839 →
  45,802 bytes (0.08%). DC is only 13.5% of the stream.

**Methodology note.** A lossy change must be scored as rate at *equal
quality*, never bytes alone — any quality knob makes files smaller. And only
content with a monotonically rising R-D curve can be scored at all: smooth
synthetic images have flat spots (quality 50 → 55 gains +0.00 dB while the
rate grows), which makes "rate at a given PSNR" unstable and manufactures
phantom gains of 5%+. An early λ fit was skewed by exactly this; the tell was
that the apparent regression persisted as λ → 0, where the DP provably
reproduces the baseline bit-for-bit.

## Limitations

- Grayscale only — no color or chroma subsampling.
- Not JFIF-interoperable; ICJ2 is its own container.
- Trellis roughly triples encode time (~1.5s → ~4.8s on a 2500×2500 image).
