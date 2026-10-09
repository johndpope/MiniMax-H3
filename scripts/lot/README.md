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

`train_h3.py` is phase 5 and refuses to train unless `LOT_H3_TRAIN=1`. It trains the LoT adapter (extent heads, shape MLP) and a rank-16 LoRA on `qkv_proj`, `out_proj`, `fc1`, `fc2` in all 50 blocks; AdaLN, the token refiner, the int8 base and the extent bank stay frozen. Each step samples a layout (20% dense 1×1, uniform extents, or a detail-driven mosaic of 4×4 super-cells), moves the clean latent to y-space, draws sigma with H3's own density, and takes `lot_h3_clean_loss`: eq. 17 with H3's `x0 - eps` head, which is Fizgig's velocity MSE in y-space. Data is a Fizgig still cache with cached text (`cache_iso3d`), so Qwen is not loaded. `--check` runs one forward and backward per layout kind, writes nothing, and needs no flag.

`infer.py` loads a `train_synth.py` checkpoint, integrates noise from `t = 1` to `t = 0`, and writes a latent plus a channel-0 preview. It does not call the H3 DiT.

A day-long synthetic run, still outside Separable Causal Diffusion, is the `lot-day` workflow: sanity checks, a 200-step probe, then training for `--minutes` (default 480). Checkpoints go to `scripts/lot/runs/`, which is gitignored.

The tests cover partition rules, Procrustes, exact recovery of `eps - x0` when the asymmetric target is correct, pretrained-head parity on a dense layout, and the H3 patch round-trip. They do not load a 50-layer H3 checkpoint.

## Training H3 for LoT from your own mp4s

This trains the LoT adapter and a rank-16 LoRA on a frozen int8 H3 base (`train_h3.py`, phase 5). It runs on one 24 GB card. Each step below was run on this machine (RTX PRO 4000 Blackwell, 24 GB) before being written down.

**What you need**

- The pruned int8 DiT: `/media/2TB/Fizgig/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` (the path is set in `procrustes_h3.py`).
- The H3 video VAE: `/media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors`.
- The Qwen3-VL text encoder, for caching captions only: `/media/2TB/minimax-h3-nvfp4/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`.
- Fizgig at `/media/2TB/Fizgig`, with LoT hooks on branch `immiscible-h3-noise`. Its cache scripts need Fizgig's requirements; on this machine the `sdwebui` conda env has them (`~/miniconda3/envs/sdwebui/bin/python`). The LoT scripts themselves run in the base `python3`.

### 1. Cut clips to H3's spec

Fizgig refuses off-spec clips instead of fixing them (`fizgig/minimax/clip.py`). The spec: `.mp4`, exactly 24 fps, a frame count on the 17n+5 grid, and both sides multiples of 32. Audio must be 32 kHz stereo, or no track at all. Train at **22 frames** (`latent_t = 7`), **640×384** landscape or **384×640** portrait. Both give a 12×20 token grid that every LoT extent tiles.

```bash
SRC=talk.mp4; OUT=/data/lot_clips; W=640; H=384   # use W=384 H=640 for portrait footage
mkdir -p $OUT
for start in 0 2 4 6 8; do                        # one 22-frame window every 2 s
  ffmpeg -v error -y -ss $start -i "$SRC" \
    -vf "fps=24,scale=$W:$H:force_original_aspect_ratio=increase,crop=$W:$H" \
    -frames:v 22 -an -c:v libx264 -crf 16 -pix_fmt yuv420p \
    "$OUT/$(basename "${SRC%.*}")_$(printf %03d $start)_mute.mp4"
done
```

`-an` drops the audio, and the `_mute` suffix tells Fizgig the clip trains video only. `train_h3` does not train audio; it packs noised silence, as Fizgig does for stills. Write one caption per clip as a same-stem `.txt` next to it (`talk_000_mute.txt`). Captions are free text.

### 2. Cache latents and text (Fizgig)

```toml
# /data/lot_clips/dataset.toml
[general]
resolution = [640, 384]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true
caption_extension = ".txt"
num_repeats = 1

[[datasets]]
image_directory = "/data/lot_clips"
cache_directory = "/data/lot_clips/cache"
```

```bash
PY=~/miniconda3/envs/sdwebui/bin/python
cd /media/2TB/Fizgig
$PY src/fizgig/scripts/minimax_cache_latents.py --dataset_config /data/lot_clips/dataset.toml \
    --vae /media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
$PY src/fizgig/scripts/minimax_cache_text.py --dataset_config /data/lot_clips/dataset.toml \
    --text_encoder /media/2TB/minimax-h3-nvfp4/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

Each clip becomes `<stem>_0640x0384_minimaxh3.safetensors` with key `latent_7x24x40`, shape `(24, 7, 24, 40)`, plus `<stem>_minimaxh3_te.safetensors`. The text step is the only one that loads Qwen, in its own process. Run it alone on the GPU, never next to training.

### 3. Fit the extent bank once

```bash
python3 scripts/lot/make_pairs_h3.py --src DIR_OF_CLIP_FOLDERS_WITH_comfy_t00.png   # or reuse the Nikki pairs
python3 scripts/lot/procrustes_h3.py --pairs scripts/lot/runs/pairs_nikki_scrya_ref2va_768x1152
```

Every fit line must print `ok=1`. The bank (`A`, `s`) stays frozen during training. The existing Nikki pairs (`runs/pairs_nikki_scrya_ref2va_768x1152`) are a valid default.

### 4. Dry-check memory, then train

```bash
P=scripts/lot/runs/pairs_nikki_scrya_ref2va_768x1152
# forward + backward per layout kind, no optimizer step, writes nothing, no flag needed
python3 scripts/lot/train_h3.py --pairs $P --cache /data/lot_clips/cache --swap 8 --shift 3 --check

LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs $P \
    --cache /data/lot_clips/cache \
    --cache /media/2TB/lora-data/fizgig_minimax_h3/cache_iso3d \
    --swap 8 --shift 3 --steps 2000 --save-every 500 \
    --init scripts/lot/runs/train_h3 \
    --out scripts/lot/runs/train_h3_clips
```

- `--cache` repeats. Mixing clips with stills of other styles stops the LoRA's style learning from masking LoT learning (see "What the first run showed").
- **Memory, measured:** stills fit at `--swap 4` (20.5 GB peak, ~2.6 s/step). 22-frame clips OOM at `--swap 4` and fit at **`--swap 8`** (20.0 GB peak, 13–17 s/step). Always run `--check` with your exact flags first.
- `--shift 3` sets the noise density for LoT steps; `--dense-shift` (default 12, H3's own) sets it for the 20% dense anchor steps. `--init` warm-starts from an earlier run's `adapter.pt` / `lora.safetensors` with a fresh optimizer.
- Each step's loss is `lot_h3_clean_loss`, the paper's eq. 17 with H3's `x0 − ε` head. It equals Fizgig's velocity MSE taken in y-space. A held-out eval runs every 250 steps (dense, all-2×2, and mosaic layouts, plus mosaic error per extent) and goes to `log.jsonl`. Outputs go to `--out` (gitignored under `runs/`).

### 5. Look at the result

```bash
python3 scripts/lot/render_h3.py --pairs $P --trained scripts/lot/runs/train_h3_clips --swap 4 \
    --layout bands --variants dense,dense_lora,lot_fit,lot_trained
```

This renders the trainer's held-out prompts as same-seed stills. `dense_lora` shows what the LoRA alone did to the base. Compare `lot_trained` with it, not only with `dense`.

### What the first run showed

The first run was 2,000 steps on 777 isometric stills, shift 12 everywhere. Held-out loss went dense 0.520 → 0.297, all-2×2 0.270 → 0.194, mosaic 0.387 → 0.262. Most of the dense drop is style learning, since all the data shares one look. In renders, training removed the frozen model's streaks and grid texture, but the coarse regions came out soft (`assets/lot-h3-trained-*.png`). Shift 12 trains at σ > 0.9 for 57% of steps and below 0.3 for only 3.5%, which is where detail forms. Hence `--shift 3` for LoT steps (12.6% below 0.3, 25% above 0.9).
