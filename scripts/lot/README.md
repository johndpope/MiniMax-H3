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
python3 scripts/lot/test_lot.py
python3 scripts/lot/gpu_smoke.py
```

A day-long synthetic run, still outside Separable Causal Diffusion, is the `lot-day` workflow: sanity checks, a 200-step probe, then training for `--minutes` (default 480). Checkpoints go to `scripts/lot/runs/`, which is gitignored.

The tests cover partition rules, Procrustes, exact recovery of `eps - x0` when the asymmetric target is correct, pretrained-head parity on a dense layout, and the H3 patch round-trip. They do not load a 50-layer H3 checkpoint.
