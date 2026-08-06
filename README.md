# Image-Compression-Project

A minimal, self-contained JPEG-style compressor for **grayscale and colour**
images, built on NumPy and Pillow. It uses an 8×8 block DCT, JPEG-like
quantization matrices, YCbCr with chroma subsampling,
**rate-distortion optimized (trellis) quantization**, and JPEG-style entropy
coding with per-image Huffman tables **split across frequency-band contexts**.

Across an 11-image corpus it needs **17.8% fewer bytes than libjpeg-turbo at
equal quality** in colour, and **14.7% fewer** in grayscale (mean BD-rate,
`optimize=True`, native resolution, matched chroma subsampling).

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

Colour is automatic: an RGB input is coded as YCbCr, a greyscale one as a
single plane.

Quality is an integer in `[1, 100]`, higher = better quality and larger output
(default 50). Out-of-range values are rejected at the CLI; the library API
clamps instead.

## Programmatic use

```python
import numpy as np
from PIL import Image
import compression as c

rgb  = np.array(Image.open("images/kodim01.png").convert("RGB"))
comp = c.compress_array(rgb, quality=50)      # CompressedImage, 3 planes
rec  = c.decompress_to_array(comp)            # uint8 (H, W, 3)

grey = np.array(Image.open("images/dog.png").convert("L"), dtype=np.uint8)
comp = c.compress_array(grey, quality=50)     # 1 plane; (H, W) out

base = c.compress_array(rgb, quality=50, trellis=False)              # plain quantization
c444 = c.compress_array(rgb, quality=50, sampling=c.SAMPLING_444)    # no chroma subsampling
fast = c.compress_array(rgb, quality=50, trellis_iterations=1)       # ~2x faster encode
```

`CompressedImage` carries one coefficient array per component plus the
original shape, padded shapes, quality and sampling scheme — everything needed
to reconstruct. `comp.coeffs` remains an alias for the luma plane.

## How it works

**Transform.** `_get_dct_matrices` builds and caches the orthonormal DCT-II
basis `D` and its transpose as contiguous `float32`. `_to_blocks` reshapes a
padded image into `(blocks_y, blocks_x, 8, 8)`, and `_forward_dct_2d` /
`_inverse_dct_2d` apply `D @ X @ D.T` — `matmul` broadcasts over the leading
dimensions, so a whole image is transformed in one batched BLAS call.

Working in `float32` rather than `float64` is ~2.7× faster and perturbs about
0.007% of quantized coefficients by ±1, well below the quantization step.

**Colour.** RGB is converted to YCbCr (BT.601 full range, the JPEG
convention) and chroma is box-averaged down by the sampling factors, then
bilinearly upsampled on decode. Planes are coded **planar** — all of Y, then
all of Cb, then all of Cr — not interleaved into MCUs. Cb and Cr share one
Huffman table set, as in JPEG.

Planar costs nothing here and buys two things: raster DC prediction, worth
~3% on DC (an MCU walk breaks left-neighbour adjacency on a quarter of
transitions), and **no MCU padding**. Interleaved JPEG must pad luma to 16×16
at 4:2:0 so subsampled chroma lands on block boundaries; planar planes pad to
8 independently, so that coupling — the classic place to get colour wrong —
simply does not exist. What planar gives up is streaming decode, since the
whole Y plane must be buffered before any pixel can be emitted. With no
streaming requirement that costs ~6 MB and nothing else.

**Quantization.** `_build_quant_matrix` scales the standard JPEG luminance or
chrominance matrix by the usual quality mapping and clips to `[1, 255]`.

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

**λ is fitted per component class, and this matters more than it looks.**
Running chroma at the luma-derived λ costs **4.9 percentage points** of
BD-rate — worse than switching trellis off on chroma entirely. λ is
proportional to `mean(Q²)`, and the chroma table is mostly 99s, so `mean(Q²)`
is 8125 against luma's 4500 at q50: the formula hands chroma a *larger* λ
exactly where it needs about a seventh of one. Chroma is already coarsely
quantized, so there is little redundant precision left to trade for bits, and
aggressive zeroing destroys what chroma signal survives.

Trellis is **encoder-side only**: it emits an ordinary coefficient array, so
the decoder and container format are untouched. `trellis=False` reproduces
plain round-to-nearest exactly.

**Entropy coding.** `_scan_symbols` extracts the whole image's DC/AC symbol
stream in one vectorized pass (DC differentials, AC run/size symbols, ZRL,
EOB), `_build_huffman_table` builds optimal per-image canonical tables, and
`_pack_bits` emits the bitstream via `np.packbits`.

**AC context modeling.** Rather than one Huffman table for every AC symbol in
the image, symbols are coded with a table per **frequency band**. A symbol's
context is the zigzag position where its zero run *starts* — not where the
coefficient lands, which the decoder cannot know until it has decoded the run.
Both sides derive it from the position about to be filled, so no side
information is transmitted.

Splitting only pays when there are enough symbols to amortise the extra
tables, and the break-even moves with image size: 15 bands is worth +21% on AC
for a 2500×2500 photo and **−7% on a 256×256 one at low quality**, where the
tables cost more than the entire payload. So the encoder prices five candidate
layouts (1, 3, 5, 8, 15 bands) on the actual symbol stream and writes the
winner's id into the container. Layout 0 is a single table — exactly ICJ2's
behaviour — which makes ICJ3 provably never worse on any input.

Conditioning on the *previous symbol* was measured and rejected: it looks
worth 14.2% by conditional entropy, but collapses to under 2% once table cost
is paid, because a 256-context split is unaffordable.

## ICJ4 container

```
magic      4 bytes   ASCII 'ICJ4'
height     4 bytes   big-endian unsigned
width      4 bytes   big-endian unsigned
quality    1 byte    1..100 (always clamped before writing)
format     1 byte    high nibble = components (1 or 3)
                     low  nibble = chroma sampling (0=4:4:4, 1=4:2:2, 2=4:2:0)
per class  1 set for grayscale, 2 (luma, chroma) for colour:
  dc_table   self-delimiting
  layout     1 byte    which AC band layout was chosen
  present    2 bytes   bitmap; bit i set if context i has a table
  ac_tables  one self-delimiting table per set bit, in context order
bit_len    4 bytes   big-endian length of the payload
bitstream  bit_len bytes, big-endian bit packing, zero-padded
```

Each table is a 2-byte symbol count followed by that many 2-byte
`(symbol, code length)` pairs. Canonical codes are a pure function of the
lengths, so the codes themselves are not stored — and since a table opens with
its own count, it is self-delimiting and needs no byte-length prefix.

Contexts that never occur are omitted entirely rather than written as empty
tables. On a one-band image that saves 56 bytes of pure overhead, which
matters at the small end: a 3×5 file is 33 bytes.

**Block dimensions are not stored** — they follow from the image size and
sampling scheme. ICJ3 wrote them redundantly; dropping them is what lets a
colour-capable container be *smaller* on grayscale than its grayscale-only
predecessor (218,333 vs 218,336 bytes on the sample), so colour costs the
grayscale path nothing.

ICJ4 added colour. ICJ3 replaced ICJ2's single AC table. ICJ2 replaced ICJ1,
which could persist an unclamped quality byte and decode against the wrong
quantization matrix. Older files are rejected, and an unknown `layout` id,
component count or sampling code is rejected rather than silently mis-decoded.

## Tests

```bash
python -m pytest
```

808 tests, ~7s. Covers the lossless coefficient round trip, padding and shape
restoration, quality clamping, container fuzzing (bad magic, truncation,
corrupt tables, unknown band layout), bit I/O, Huffman construction, trellis
invariants, AC context causality, colour transform and resampling, all three
subsampling schemes, pinned golden rate/distortion values, and the BD-rate
harness (including that it refuses unscorable curves).

Two tests carry more weight than the rest. `test_contexts_match_a_sequential_decoder`
checks the encoder's bulk, vectorized context assignment against a naive loop
that only looks backwards — if those disagree the stream mis-decodes into
garbage instead of failing. And `tests/test_context.py` holds an independent,
deliberately slow reference decoder that the optimized one is checked against,
because the fast decoder is written as one inlined loop and is exactly the kind
of code that harbours an off-by-one.

The synthetic corpus in `tests/conftest.py` is generated from a seeded RNG, so
no binaries live in the repo. Real photographs dropped into `tests/images/`
are picked up automatically.

## Design notes: what was measured and rejected

Several plausible optimizations were measured and **discarded**, which is why
they are absent:

- **Arithmetic / range coding.** The per-image Huffman tables land within
  **0.76%** of the order-0 entropy bound, so replacing the coder cannot
  recover more than that *at order 0*. Context modeling beats that bound
  outright — but it turned out not to need an arithmetic coder to do it, since
  splitting the Huffman tables by frequency band captures the gain and also
  reduces the integer-code-length penalty. A sequential, un-vectorizable
  adaptive coder in Python was not worth the small remaining headroom.
- **Deadzone quantization.** Looked like a 12% win, but that was pure
  rate-distortion sliding: against an equal-size baseline it scored +0.05 /
  +0.01 / −0.13 dB. Trellis is the version of this idea that actually works,
  because it prices the real run-length structure.
- **Improved DC prediction.** A row-wise predictor moved DC cost 45,839 →
  45,802 bytes (0.08%). DC is only 13.5% of the stream.

- **A single fixed AC context layout.** Splitting AC symbols into 15 frequency
  bands is worth +21% on a 2500×2500 photo and **−7%** on a 256×256 one at low
  quality, where 15 tables cost more than the whole AC payload. The encoder
  prices five layouts and picks; layout 0 is a single table, so the format is
  never worse than its predecessor.

**Methodology note.** A lossy change must be scored as rate at *equal
quality*, never bytes alone — any quality knob makes files smaller. Only
content with a monotonically rising R-D curve can be scored at all: smooth
synthetic images have flat spots (quality 50 → 55 gains +0.00 dB while the
rate grows), which makes "rate at a given PSNR" unstable and manufactures
phantom gains of 5%+. An early λ fit was skewed by exactly this; the tell was
that the apparent regression persisted as λ → 0, where the DP provably
reproduces the baseline bit-for-bit.

Three further traps, all of which caught something real here: score against a
**multi-image corpus** (one image put the headline 8 points too high, and put
the value of a second trellis pass 2.3× too low); hold **confounds fixed**
(comparing subsampling schemes against a single reference measures chroma
retention, not coding); and remember that **a constant fitted for one
component class need not transfer to another** — the chroma λ needed refitting
by a factor of ~7, and the naive value was worse than no trellis at all.

## Benchmarks

```bash
python -m bench                      # BD-rate report, grayscale, 512px, fast
python -m bench --color              # colour at 4:2:0
python -m bench --max-side 0         # native resolution
python -m bench.fetch_corpus         # top up images/ to the full Kodak set
```

The corpus lives in `images/` and is tracked, because the golden pins and the
BD-rate gates are meaningless without it — a missing corpus **fails** the
suite rather than skipping.

BD-rate is the standard "average % bitrate change at equal quality" metric —
negative means fewer bits for the same PSNR. Corpus of 11 photographs
(`dog.png` plus Kodak 01–10) at native resolution, quality 20–90:

| image | grayscale | colour 4:2:0 |
|---|---|---|
| dog (2500×2500) | −23.41% | −23.36% |
| kodim01 | −13.21% | −15.00% |
| kodim02 | −16.80% | −22.19% |
| kodim03 | −15.29% | −20.03% |
| kodim04 | −12.72% | −16.97% |
| kodim05 | −12.03% | −13.28% |
| kodim06 | −15.09% | −17.44% |
| kodim07 | −11.56% | −15.49% |
| kodim08 | −12.91% | −13.91% |
| kodim09 | −15.58% | −20.16% |
| kodim10 | −13.58% | −18.21% |
| **mean** | **−14.74%** | **−17.82%** |

Colour is scored against libjpeg **at the same subsampling**, which matters:
comparing our 4:2:2 against libjpeg's default 4:2:0 makes us look better
(−13.08% vs −12.35%) purely for retaining more chroma, not for coding it
better. Controlling for it reverses the ordering — our margin over libjpeg is
largest at 4:4:4 (−14.31%), where there is the most chroma detail for the
trellis and context tables to work on:

| vs. libjpeg at matched subsampling | 4:4:4 | 4:2:2 | 4:2:0 |
|---|---|---|---|
| mean BD-rate | **−14.31%** | −12.81% | −12.35% |

(These use a shorter quality sweep than the table above, hence the small
difference in the 4:2:0 figure.)

**Quote the mean, not `dog`.** At −23.41% it is the best image in the set by a
wide margin — the largest and the easiest — and citing it alone overstates the
codec by two thirds. The spread across genuinely comparable images (Kodak, all
768×512) is −11.6% to −16.8%, which is the honest range.

Resolution matters too, and should always be stated: the same corpus scores
−13.94% at 512px, because per-image Huffman tables are a larger share of a
smaller file.

The harness **refuses to score** content whose R-D curve has flat spots — the
synthetic `flat` and `gradient` fixtures raise rather than returning a number.
That is deliberate; see the methodology note above.

## Limitations

- Not JFIF-interoperable; ICJ4 is its own container.
- No streaming or progressive decode — a consequence of planar coding, since
  the whole luma plane must be buffered before any pixel can be emitted.
- Trellis dominates encode time: **0.06 s → 6.8 s** on a 2500×2500 image, a
  ~115× slowdown. (An earlier README claimed "roughly triples encode time,
  ~1.5s → ~4.8s". That was measured before the entropy coder was vectorized,
  which cut the baseline to 0.06 s and blew the ratio out; the figure was
  never re-taken.)
- Decode is **0.41 s** (was 0.68 s before the phase 7 rewrite). Huffman
  decoding is inherently sequential and cannot be vectorized the way the
  encoder was, so the only lever was the constant factor — inlining the bit
  accumulator cut 902k function calls to 106k. That is close to the ceiling
  for this approach: the remainder is ~680 ns per symbol of irreducible
  CPython loop work plus 0.09 s materializing the output array, and going
  meaningfully faster wants a C extension.
