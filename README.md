# Image-Compression-Project
A minimal, self-contained JPEG-style compressor for **grayscale** images,
implemented in pure Python on top of NumPy and Pillow. Uses an 8×8 block
Discrete Cosine Transform (DCT) and a JPEG-like luminance quantization
matrix to trade reconstruction quality for file size.
## Layout
```
compression.py   # all codec logic + CLI entry point
dog.png          # sample input image
README.md        # this file
```
Everything lives in a single module (`compression.py`). It is organised in
three logical sections:
1. **Core DCT + quantization pipeline** (shared by both codecs)
    - `_get_dct_matrix(n)` — builds and caches the orthonormal DCT-II basis
      matrix `D` **and** its transpose `D.T` as contiguous `float32` arrays.
      Caching both avoids per-call transposition and keeps BLAS GEMM happy.
    - `_forward_dct_2d(blocks)` / `_inverse_dct_2d(blocks)` — apply the
      separable 2D DCT (`D @ X @ D.T`) and its inverse to an array of shape
      `(..., 8, 8)`. `matmul` broadcasts over all leading dims, so a whole
      image is transformed in a single batched BLAS call.
    - `_build_quant_matrix(quality)` — scales the standard JPEG luminance
      matrix by the usual `quality -> scale` mapping and clips to `[1, 255]`.
    - `compress_array` / `decompress_to_array` — end-to-end in-memory
      pipeline: level shift, pad to 8×8, forward/inverse DCT, quantize.
2. **`.npz` codec** (`compress_image_file` / `decompress_image_file`)
    - Stores the quantized coefficient tensor plus metadata in a
      `numpy` `.npz` archive. No entropy coding; useful as a reference.
3. **ICJ1 Huffman codec** (`compress_huffman_file` / `decompress_huffman_file`)
    - Adds JPEG-style DC differential coding, AC run/size coding, and a
      per-file canonical Huffman table, packed into a small custom binary
      container with magic bytes `ICJ1`.
## CLI usage
```bash
# .npz codec (DCT + quantization only)
python compression.py compress   input.png  output.npz  --quality 50
python compression.py decompress output.npz reconstructed.png
# ICJ1 codec (DCT + quantization + Huffman)
python compression.py compress_huff   input.png  output.icj  --quality 75
python compression.py decompress_huff output.icj reconstructed.png
```
Quality is a JPEG-style integer in `[1, 100]`, higher = better quality /
larger output. Default is 50.
## Programmatic usage
```python
import numpy as np
from PIL import Image
import compression as c
arr = np.array(Image.open("dog.png").convert("L"), dtype=np.uint8)
comp = c.compress_array(arr, quality=50)        # CompressedImage dataclass
rec  = c.decompress_to_array(comp)              # uint8 ndarray, same shape
```
`CompressedImage` carries the quantized coefficients, original shape,
padded shape, and quality — everything needed to reconstruct the image.
## ICJ1 container format
```
magic      4 bytes   ASCII 'ICJ1'
height     4 bytes   big-endian unsigned
width      4 bytes   big-endian unsigned
quality    1 byte    1..100
blocks_y   2 bytes   big-endian unsigned
blocks_x   2 bytes   big-endian unsigned
dc_table   12  * 5 bytes  (1 byte code length, 4 bytes big-endian code)
ac_table   256 * 5 bytes
bit_len    4 bytes   big-endian length of following payload
bitstream  bit_len bytes, big-endian bit packing with 0xFF byte stuffing
```
Byte stuffing is not strictly required by the container but is included so
the bitstream could be embedded into a real JPEG wrapper without escaping.
## Change log
### Fix: broken DCT after time-optimization attempt
An earlier attempt to speed up `compress_array` using two `np.einsum` calls
(`"ui,bjix->bjux"` then `"vj,bjux->buvx"`) re-used the label `j` for two
incompatible axes — the block-row count and the DCT matrix inner dimension
— which raised:
```
ValueError: Size of label 'j' for operand 1 (8) does not match previous
terms (313).
```
for any image whose width is not exactly 64 pixels. Both einsum lines were
also dead code because the next line overwrote the result.
**Resolution:**
- Extracted the separable 2D DCT into `_forward_dct_2d` /
  `_inverse_dct_2d` helpers so the exact same (correct, BLAS-backed)
  implementation is used by both `compress_array` and
  `decompress_to_array`. This removes duplication and prevents the two
  paths from drifting out of sync.
- Extended `_get_dct_matrix` to cache **both** `D` and `D.T` as
  contiguous `float32` arrays, so every forward/inverse call is a pair of
  batched GEMMs with no intermediate copies.
- Verified both codecs end-to-end on `dog.png` at quality 50: the `.npz`
  and ICJ1 pipelines produce bit-identical reconstructions (MSE 18.70,
  PSNR ≈ 35.4 dB versus the original), as expected since they share the
  core transform and quantization.
