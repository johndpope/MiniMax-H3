# Level-of-Token on H3: cheatsheet

LoT gives the H3 DiT fewer, larger tokens where detail is low, then recovers the full-resolution velocity (Nakayama et al., [arXiv 2610.05816](https://arxiv.org/abs/2610.05816)). The VAE latent stays full size; only the transformer sequence shrinks. Full walkthrough: [README → Training H3 for LoT from your own mp4s](README.md#training-h3-for-lot-from-your-own-mp4s). Results and history: [issue 1](https://github.com/johndpope/MiniMax-H3/issues/1).

Run everything from the repo root. One GPU job at a time; the scripts refuse to start beside another one and never kill it.

## 0. Does it work here?

```bash
python3 scripts/lot/test_lot.py            # CPU, no weights -> "ok 25"
python3 scripts/lot/parity_h3.py           # GPU: LoT 1x1 == dense H3 -> rel_rms=0 ok=1
```

## 1. Clips (any mp4)

```bash
W=640; H=384   # portrait footage: W=384 H=640
ffmpeg -v error -y -ss 0 -i talk.mp4 \
  -vf "fps=24,scale=$W:$H:force_original_aspect_ratio=increase,crop=$W:$H" \
  -frames:v 22 -an -c:v libx264 -crf 16 -pix_fmt yuv420p /data/lot_clips/talk_000_mute.mp4
echo "a woman talking to camera in a kitchen" > /data/lot_clips/talk_000_mute.txt
```

Spec: 24 fps · 22 frames (17n+5 grid) · sides ×32 · no audio, or 32 kHz stereo · same-stem `.txt` caption.

## 2. Cache (Fizgig, `sdwebui` env)

```bash
PY=~/miniconda3/envs/sdwebui/bin/python; cd /media/2TB/Fizgig
$PY src/fizgig/scripts/minimax_cache_latents.py --dataset_config /data/lot_clips/dataset.toml \
    --vae /media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
$PY src/fizgig/scripts/minimax_cache_text.py --dataset_config /data/lot_clips/dataset.toml \
    --text_encoder /media/2TB/minimax-h3-nvfp4/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
cd -
```

`dataset.toml`: `image_directory = "/data/lot_clips"`, `cache_directory = "/data/lot_clips/cache"`, `resolution = [640, 384]` (template in the README). The text step loads Qwen; run it alone on the GPU.

## 3. Extent bank (once)

```bash
P=scripts/lot/runs/pairs_nikki_scrya_ref2va_768x1152    # existing Nikki pairs, or build your own:
# python3 scripts/lot/make_pairs_h3.py --src DIR_WITH_<clip>/comfy_t00.png
python3 scripts/lot/procrustes_h3.py --pairs $P --smoke  # every line ok=1
```

## 4. Train

```bash
python3 scripts/lot/train_h3.py --pairs $P --cache /data/lot_clips/cache --swap 8 --shift 3 --check

LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs $P \
  --cache /data/lot_clips/cache --cache /media/2TB/lora-data/fizgig_minimax_h3/cache_iso3d \
  --swap 8 --shift 3 --steps 2000 --save-every 500 \
  --init scripts/lot/runs/train_h3 --out scripts/lot/runs/my_run
```

Long runs: start detached with `setsid nohup ... > /tmp/my_run.log 2>&1 &`. Then watch:

```bash
grep -E "train_eval|kind=train step" /tmp/my_run.log | tail -3             # progress, s/step
tail -f scripts/lot/runs/my_run/log.jsonl | grep --line-buffered '"eval"'  # held-out evals
```

## 5. Look

```bash
python3 scripts/lot/render_h3.py --pairs $P --trained scripts/lot/runs/my_run --swap 4 \
  --layout bands --variants dense,dense_lora,lot_fit,lot_trained   # -> runs/render/grid.png
python3 scripts/lot/bench_h3.py          # 37-frame 768x1344 DiT timing, dense vs LoT
python3 scripts/lot/time_decode_h3.py    # VAE decode (LoT does not speed this up)
```

## Flags that matter

| Flag | Default | What it does |
|---|---|---|
| `LOT_H3_TRAIN=1` | unset | Required to train. `--check` runs without it and writes nothing. |
| `--cache DIR` | `cache_iso3d` | Fizgig cache of latents + text. Repeat to mix clips and stills. |
| `--swap N` | 4 | Blocks streamed from CPU. Stills: 4. 22-frame clips: **8** (4 OOMs). |
| `--shift` | 12 | Noise density for LoT steps. **3** trains low σ, where detail forms. |
| `--dense-shift` | 12 | Noise density for the 20% dense anchor steps (H3's own). |
| `--init DIR` | none | Warm-start `adapter.pt` + `lora.safetensors`, fresh optimizer. |
| `--dense-p` | 0.2 | Share of steps on the plain dense layout. |
| `--rank` | 16 | LoRA rank on qkv / out / fc1 / fc2 × 50 blocks (200 modules). |

## Measured on one 24 GB card (RTX PRO 4000 Blackwell)

| Thing | Number |
|---|---|
| DiT forward, 37-frame 768×1344 (48 blocks streamed) | dense 51.9 s · LoT 19.6 s · **2.64×** |
| DiT forward, 768×1152 still, all resident | dense 0.69 s · bands 0.42 s (1.65×) · all-2×2 0.30 s (2.26×) |
| VAE decode, 37 latent frames | 31.8 s, same with or without LoT |
| Train step | stills ~2.7 s (swap 4) · 22-frame clips 13–17 s (swap 8) |
| Training peak | stills 20.5 GB · clips 20.0 GB |

## Gotchas

- **Sign:** H3's head predicts `x0 − ε`. The clean estimate is `y_t + σ·out`, not `flow.clean_from_velocity` (the toy model's `ε − x0`).
- **y-space:** after a Procrustes fit the scales are no longer 1, so the DiT input is `splice.to_y(x0)` noised, and output goes back through `to_x`. A splice without `y_space=True` refuses fitted scales.
- **Swap costs about 0.37 s per streamed block per forward.** On small inputs it hides the LoT speedup, so keep everything resident (`--swap 0`) when it fits.
- **Frozen base + coarse tokens = artifacts.** LoT needs the fine-tune; a fitted bank alone gives streaks and grid texture.
- **Style vs LoT:** single-style data makes the dense loss drop as well. Compare `lot_trained` with `dense_lora`, not with `dense`.
- **Fizgig hooks** live on Fizgig branch `immiscible-h3-noise`; `master` has no `_lot`.
