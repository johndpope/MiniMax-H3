# Level-of-Token Diffusion, drop-in for MiniMax-H3

Method code for [Level-of-Token Diffusion](https://arxiv.org/abs/2610.05816) (Nakayama et al., 2026). A LoT layout replaces the uniform visual token grid with rectangles of mixed size. The transformer sees the short sequence. A patch-wise asymmetric flow, taken from LakonLab, turns that sequence back into a dense velocity on the original lattice.

This track is separate from Separable Causal Diffusion. Nothing here imports `scripts/scd`.

This is local research. It does not train FLUX or Wan, and it does not load H3 weights.

## What is implemented

- Partitions of the uniform token lattice, including the paper's quadtree (eq. 23), mask levels (eq. 19), blur radii (eqs. 21–22), and a square reading of the VRS tolerance.
- Procrustes lifts and RMS scales (eqs. 14–15), using the same SVD as `LakonLab/tools/asymflow_subspace_procrustes.py`.
- Data scaling in y-space (eq. 16) so every token shares one timestep. Per-extent timestep calibration (eq. 18) is not used.
- Shape features (eq. 11) and finest-grid RoPE centers (eq. 13). Unit extents sit on the integer lattice the pretrained model already uses.
- Extent input and output heads, initialized as `W_in A^T` and `A W_out`.
- Velocity recovery through `AsymFlowMixin.asymflow_velocity` with `s = 1`, `k = 1`, which is eq. 9. The module is loaded from `../LakonLab/lakonlab/models/architectures/asymflow/common.py` so the package import (and its `mmcv` dependency) is skipped.
- Clean-data loss (eq. 17).

Bases start as a semi-orthonormal mean lift (`A^T A = I`, identity when the extent is 1). Call `bank.fit_` on aligned `(dense patch, multi-scale VAE token)` pairs before training. Those buffers stay frozen.

## H3 contract

H3 patchifies 24-channel video latents with `(t, h, w) = (1, 2, 2)`, so each uniform token has dimension 96. LoT extents count those tokens. A `2×2` LoT token covers a `1×4×4` block of VAE latents.

`make_h3_adapter()` builds the Appendix A.10 extent set: temporal extent 1, spatial sides in `{1, 2, 4}`, including rectangles. The backbone keeps text and audio. It receives visual embeddings and returns one hidden state per LoT rectangle, in `layout.rects` order.

```python
import sys
sys.path.insert(0, "scripts/lot")

from flow import clean_from_velocity
from h3 import make_h3_adapter, patchify

adapter = make_h3_adapter()  # token dim 96, hidden 5376; no checkpoint loaded
# adapter.init_from_pretrained(weight_in, weight_out, bias_in, bias_out)
# weight_in is (5376, 96), weight_out is (96, 5376), nn.Linear layout.

tokens = patchify(latent)          # (B, T, H/2, W/2, 96)
y0 = adapter.bank.scale_clean(tokens, layout)

def backbone(hidden, centers, sigma, **cond):
    # hidden: (B, L, 5376), centers: (L, 3) in (t, h, w) on the uniform grid.
    # Run the H3 blocks on this shorter visual sequence. Return (B, L, 5376).
    ...

velocity = adapter(y_t, sigma, layout, backbone)   # dense, y-space
y0_hat = clean_from_velocity(y_t, velocity, sigma)
x0_hat = adapter.bank.unscale(y0_hat, layout)
```

Image-style square layouts up to 8×8 are `IMAGE_EXTENTS` in `h3.py`. Pass them to `make_h3_adapter(extents=IMAGE_EXTENTS)`.

Set `include_time=True` only if a rectangle spans more than one uniform frame. The paper's video model does not.

## Checks

```bash
python3 scripts/lot/sanity.py
python3 scripts/lot/test_lot.py
python3 scripts/lot/gpu_smoke.py
python3 scripts/lot/infer.py --ckpt scripts/lot/runs/day/last.pt --steps 8
```

`sanity.py` is CPU-only. It checks that a mixed layout attends over 6 tokens instead of 64, that the velocity is still the full 8×8 canvas, and that an 8×8 square is rejected by the H3 extent bank. It does not write a checkpoint. `train_synth.py` also writes none unless `--save-every` is positive.

The H3 tail lives in `h3_splice.py`. Fizgig `MiniMaxH3DiT._lot` defaults to `None`. When it is a `LotSplice`, only the target-video rows are shortened, sigma passed into the velocity step is `1 - t`, and `forward_cached` raises. Setting `_lot` and `_tread` together raises. That path does not load a checkpoint by itself.

H3's head predicts `x0 - eps`, the negation of what eq. 9 expects. `LotSplice.project_rows` passes `x0_minus_eps=True`, which flips the head before recovery and the velocity after, so the Fizgig sampler steps the result unchanged. A 1×1 basis has no orthogonal complement, so the 1×1 parity cannot catch a wrong sign; `test_h3_head_sign_recovery` does.

After a Procrustes fit the scales are not 1, and the DiT input must be y-space. Build the state from `splice.to_y(x0)`, construct `LotSplice(..., y_space=True)`, and map the clean estimate back with `splice.to_x`. A splice without `y_space=True` raises once any scale in its layout differs from 1.

Phase 4 is `procrustes_h3.py --pairs DIR [--smoke] [--out FILE]`. `DIR` holds user-supplied `{et}x{eh}x{ew}.pt` files with `dense (N, 96·et·eh·ew)` and `reference (N, 96)` in `gather_extent` row order; there is no `1x1x1.pt`. A missing directory exits. The fit reads only the two pretrained head maps from the checkpoint on CPU, calls `fit_extent`, and checks `AᵀA = I`, finite positive scales, and an unchanged 1×1 head. `--smoke` runs one frozen y-space forward on the 384×640, `latent_t=7` canvas and asserts shape, finite values, and the packed length, not picture quality. It writes nothing unless `--out` is given. The GPU scripts refuse to start while another one is running (`gpu_guard.py`).

`make_pairs_h3.py` builds those pair files from decoded H3 stills with H3's own encoder: the fine side is `encode(x)` gathered in `eh×ew` blocks, the reference is `encode(resize(x, H/eh, W/ew))`, so one reference token covers one block's pixels. The default source is the 538 Nikki Ref2VA clips (`comfy_t00.png`), 768×1152, which divide by 128 so every extent tiles. Output goes to `runs/pairs_<name>` (gitignored).

`render_h3.py --pairs DIR` samples same-seed 768×1152 stills three ways (dense, LoT with the mean lift, LoT with the fit) from cached H3 text embeddings, so Qwen3-VL is never loaded, and decodes them with the fp16 H3 decoder. `time_decode_h3.py` times one clip decode at the DiT timing shape; LoT does not shorten the VAE.

`infer.py` loads a `train_synth.py` checkpoint, integrates noise from `t = 1` to `t = 0`, and writes a latent plus a channel-0 preview. It does not call the H3 DiT.

A day-long synthetic run, still outside Separable Causal Diffusion, is the `lot-day` workflow: sanity checks, a 200-step probe, then training for `--minutes` (default 480). Checkpoints go to `scripts/lot/runs/`, which is gitignored.

The tests cover partition rules, Procrustes, exact recovery of `eps - x0` when the asymmetric target is correct, pretrained-head parity on a dense layout, and the H3 patch round-trip. They do not load a 50-layer H3 checkpoint.
