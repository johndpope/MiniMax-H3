# Claude handoff: Level-of-Token on MiniMax-H3

Written 2026-10-09 from the machine, for a new Claude session. Updated later on 2026-10-09: the 37-frame timing finished and passed, an H3 head-sign bug in the LoT velocity recovery was found and fixed, and the phase-4 code landed (pairs still absent). This file is local research under `docs/`. Do not push it, and do not push anything in `docs/` or `scripts/scd/`, to MiniMax-AI. The fork is `johndpope/MiniMax-H3`. Issue comments go only to issue 1 on that fork (`gh` defaults to MiniMax-AI).

The approved design is `docs/LOT_H3_COMPUTE_PLAN.md` (design id `27928a4d`, open-issue count 0). Do not reopen those decisions. This handoff says what is already built, which commands printed which lines, and what is still unfinished.

Workspace memory `topics/scripts-lot.md` is stale. It still says the adapter is not spliced into Fizgig. The splice is in. Trust this file and the code.

## What Level-of-Token has to do on H3

Paper: Nakayama et al., arXiv:2610.05816, https://georgenakayama.github.io/lotdiffusion/. Patch-wise asymmetric flow is LakonLab `AsymFlowMixin`, loaded by importlib from `../LakonLab/lakonlab/models/architectures/asymflow/common.py`. Do not import the `lakonlab` package. That pulls mmcv. `s = 1`, `k = 1`. Extent scale divides the clean latent (eq. 16). Per-extent timestep calibration (eq. 18) is unused.

LoT partitions the uniform token lattice into axis-aligned rectangles. The pixel canvas and the VAE latent stay full resolution. The transformer attends over `layout.count` tokens, not over one token per finest cell. Gather and scatter put the velocity back on the full lattice. Mixed cell sizes live inside one canvas. Five separate uniform tilings are not the result. `IMAGE_EXTENTS` includes 8×8 and is image-only. H3 video rejects an 8×8 cell.

### Geometry the splice must keep

| Fact | Value |
| --- | --- |
| Visual VAE | f16t4d24: 16× spatial, 4× temporal, 24 latent channels |
| Patch into the DiT | `(1, 2, 2)`. Token dim 96. A 2×2 LoT token covers a 1×4×4 VAE-latent block |
| DiT | Fizgig `MiniMaxH3DiT`, 50 blocks, hidden 5376, 56 heads, head dim 128, ffn 14336 |
| H3 extents | `et = 1`, spatial sides `{1, 2, 4}`, rectangles included. Nine keys. Video layouts use temporal extent 1 only |
| RoPE | Not integer lattice indices. Spatial `(arange(n) * (ratio / n) + (1 - ratio) / 2) * 32`. Time `5/3 * (1, 4, 4, 4, 4)` |
| Pack order | text, keyframes, refs, audio, target video. Audio does not advance the video clock |
| Keyframe clock | Post-ref cursor + `(5/3) * frame_index`. Image refs advance the cursor by 1. A video-kind ref `(h, w, t)` with `t > 1` advances by the span sum |
| Sigma into AsymFlow | `1 - t`, clamped to `[0, 1]`. `t` stays the cleanness input to the time embedder. A dense 1×1 identity basis ignores sigma, so parity cannot catch a swapped timestep |

### Where the splice sits

Fizgig file, outside this git repo: `/media/2TB/Fizgig/src/fizgig/minimax/model.py`.

- `self._lot = None` in `__init__` (line 630). `None` is the dense forward. The DiT does not import `scripts/lot`.
- After the batch-size check, LoT and TREAD together raise `RuntimeError`.
- Target video only: `lot.video_tokens` then `lot.embed_rows` instead of `video_patch_proj`. Keyframes and refs still use `video_patch_proj`.
- `lot.replace_video_positions` replaces the dense video tail. Origin is `dense_pos[-n_dense, 0]`.
- Before the video return: `sigma = (1 - t_val).clamp(0, 1)`, then `lot.modulate` (RMSNorm + AdaLN, shared `final_layer`), then `lot.project_rows`. That replaces `final_layer.video_out` only. Do not call `LotVisualAdapter.forward`. The blocks sit between embed and the head.
- `forward_cached` raises if `_lot` is set. A 6-step cache of the 37-frame clip is about 115 GiB. Do not dump per-block activations.
- `project_rows` casts modulated states and tokens onto the extent-linear dtype and device before `velocity_from_states`.
- `project_rows` passes `x0_minus_eps=True`. H3's head predicts `x0 - eps` (Fizgig `sampling.py:16`, `trainer.py:11`), eq. 9 expects `P eps - x0`. The flag negates the head before recovery and the velocity after. Before this fix every extent larger than 1×1 recovered `((2 - σ) x0ᶜ + σ εᶜ) / σ` on the complement instead of the velocity. 1×1 parity cannot see the sign (a unit basis has no complement). Timing was unaffected; any LoT output from a non-unit extent produced before the fix is wrong. `test_h3_head_sign_recovery` covers it.
- y-space: `LotSplice(adapter, layout, y_space=False)` raises in `video_tokens` once any scale in the layout is not 1. After a Procrustes fit, build the state with `splice.to_y(x0)`, pass `y_space=True`, and map the clean estimate back with `splice.to_x`. One H3 step to σ=0 is `y_t + σ * out`.

Repo-side modules: `scripts/lot/h3_splice.py` (`LotSplice`), `scripts/lot/h3.py` (`gate_frame_layout(height, width)`, `clip_layout`, `make_h3_adapter`), `scripts/lot/procrustes_h3.py` (phase 4), `scripts/lot/gpu_guard.py` (one GPU job at a time), `scripts/lot/h3_positions.py`, `scripts/lot/flow.py` (vectorized gather, scatter, scale), `scripts/lot/adapter.py`.

### The clip that is allowed to count as a speedup

t2va, 768×1344, `latent_t=37`. Do not pass 120 pixel frames through Fizgig `latent_frames_for_pixels`. That snaps down to 32 latent frames. `pixel_frames_for_latent(37) = 124`. Latent `(1, 24, 37, 48, 84)`. Token grid 24×42 = 1008 per frame. Width 42 is not divisible by 4, so the mixture is bands, not 4×4:

- Rows 0–7: 1×1 (336)
- Rows 8–15: 2×2 (84)
- Rows 16–23: 4×2 (42)

462 tokens per frame. Extents `{(1,1,1), (1,2,2), (1,4,2)}`.

| | Dense | Mixed |
| --- | --- | --- |
| Video tokens | 37,296 | 17,094 |
| Text | 512 | 512 |
| Audio rows | 414 (`audio_latents_for_frames(124) * 2`) | 414 |
| Packed S | 38,222 | 18,020 |

Video-token compression 2.182×. Packed compression 2.121×. MAC estimate about 3.08× because attention is past the 26,880-token crossover. Acceptance is median wall time at least 1.5×, and not above the MAC ratio plus 10%, same tensors, flag off versus flag on. Under 1.5× means overhead ate the win. Profile before any training.

The 7-latent-frame fallback (22 pixel frames, text 128, packed dense 7,258) is under that crossover. Record it. Do not call it the gate.

### Checkpoint and memory

Only this file, about 20,970,379,616 bytes:

`/media/2TB/Fizgig/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`

There is also a symlink under `/media/2TB/Fizgig/models/`. Load with `load_minimax_h3_dit(..., base_quant="int8", blocks_to_swap=N)` then `enable_block_swap(N, h2d_only=True)`. The loader banner says the 200 int8 ConvRot linears are about 21 GB if kept resident. `adaln_t_table` is float32 `[1025, 8]`. `time_embed_dim` 2688 is the unpruned config and is wrong for this file. The eager int8 GEMM does `qdata.to(dtype)` then `y.float() * wscale` (`convrot.py`). It does not materialize 38.5 GB of bf16 weights. The 4,385,144,832-byte allocation in the logs is `38222 * 28672 * 4`, rounded up to a 2 MiB boundary: the fp32 `fc1` output (`ffn * 2 = 28672`) on the dense 37-frame pack.

One GPU: NVIDIA RTX PRO 4000 Blackwell, 24467 MiB. Do not call `torch.cuda.empty_cache`. Do not kill processes to free the card. Do not load Qwen3-VL-32B or the full bf16 H3 weights. Do not load `scripts/lot/runs/day/last.pt` as H3 weights. `/media/2TB` had about 62 GB free. Do not write checkpoints or activation dumps. `scripts/lot/runs/` is gitignored and is about 19 MB; `day/last.pt` is the stopped toy trainer (3,147,571 bytes). Leave it.

Math SDPA at S=38,222 is on the order of 160 GB. Timing must keep math SDPA off (`torch.backends.cuda.enable_math_sdp(False)`) and abort if neither flash nor mem-efficient is available. The small parity canvas may use math SDPA. Do not turn on `set_int8_attention` for parity.

### Parity rule

Closed compare: one shared `video_latent`, one explicit `audio_rows` (omitting both draws a fresh `torch.randn`; `audio_noise` is scaled and is not a substitute), same bf16 text already at hidden size, same `t`. Pass when `rel_rms <= max(1e-4, 2 * floor)` and max abs `<= 1e-2`. Floor is two flag-off forwards. Floor above `1e-3` is inconclusive. The implemented parity uses latent `(1, 24, 1, 16, 16)`, text `(1, 4, hidden)`, audio `(4, audio_latents_dim)`, `t = 0.6`, dense 1×1 layout. The plan's larger sketch was `(1, 24, 2, 32, 32)`. The script that ran is the smaller one.

### Phases still gated

1. Contract tests. Done. `python3 scripts/lot/test_lot.py` now prints `ok 21`.
2. Fizgig splice, flag default off, closed 1×1 parity. Done, and re-run after the sign fix. See the parity lines below.
3. Mixed-layout timing on latent_t=37. Done. Passed: `ratio=2.643`. See "Timing result".
4. Procrustes. Code done (`procrustes_h3.py`), not run: there is still no pair directory. Do not invent pairs. Do not encode 768×1344 (one 17-frame group is about 55 GiB). `plan_clip_bucket(22, 1344, 768)` returns `(800, 448)`. Call `fit_extent`, not bare `bank.fit_`.
5. Fine-tune only if `LOT_H3_TRAIN=1`, only 384×640, `latent_t=7` (packed S=1882). Exit if `14.5 + ft_clip_activation_gb + 2.0 > 18`. LoRA rank 16 on `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` only. AdaLN excluded. Loss `lot_clean_loss`. Precached bf16 text. The user has not set that flag.

`FL2VA/` and `Ref2VA/` stay byte-identical except `model_index.json`. `diff -rq FL2VA Ref2VA` should report only that file. Do not edit `scripts/scd`. Do not commit or push unless asked.

## Timing result — gate passed

pid 2459145 exited 0. `/tmp/lot_bench_h3_t37.log` ends:

```
LOT_H3 kind=progress phase=lot step=2
LOT_H3 kind=timing task=t2va latent_t=37 swap=48 dense_tokens=38222 lot_tokens=18020 dense_seq=(38222, 5376) lot_seq=(18020, 5376) dense_ms=51901.1 lot_ms=19636.6 ratio=2.643 peak_mb=15855.3 dtype=bf16 ok=1
EXIT:0
```

- `latent_t=37`, ratio 2.643 ≥ 1.5, and under the ceiling (MAC ≈ 3.08, +10% ≈ 3.39). The gate passes.
- `dense_seq`/`lot_seq` come from the `blocks[0]` hook: the packed sequence really went 38,222 → 18,020.
- Caveat: a 4,385,144,832-byte allocation failed once during the *dense warmup* and the allocator retried. If retries also hit the timed dense steps, `dense_ms` (and so the ratio) is somewhat inflated. That run did not record it. `bench_h3.py` now prints `dense_retries=` / `lot_retries=` (allocator retries inside the timed steps only); quote those on any re-run.
- The timing predates the head-sign fix. The fix is two negations per extent and does not change the timed work.

## Commands and the lines they printed

Run from `/home/johndpope/Documents/GitHub/MiniMax-H3` unless noted. Outputs below were printed by those processes. The parity and CPU lines are from the build session. The two bench logs were re-read from `/tmp` while writing this file.

CPU contract, no checkpoint, no H3 weights:

```bash
python3 scripts/lot/sanity.py && python3 scripts/lot/test_lot.py
```

```
SANITY ok 2 tokens 6/64 macs 304128/4194304 writes=0
ok 17
```

The MAC line is the toy `LotBackbone` (hidden 32 in that check, not H3). It is not a speedup.

```bash
python3 scripts/lot/gpu_smoke.py
```

```
gpu NVIDIA RTX PRO 4000 Blackwell peak_mb 64.4 tokens 16/48 video_start 8 positions (24, 3) loss 292.327484 finite True
```

`test_lot.py` was run again after the `project_rows` dtype cast and printed `ok 17` again.

Closed 1×1 int8 parity. This loaded the pruned checkpoint. It does not measure a speedup (64 tokens on both sides).

```bash
python3 scripts/lot/parity_h3.py
```

```
LOT_H3 kind=parity task=t2va dense_tokens=64 lot_tokens=64 rel_rms=0 rel_rms_floor=0 max_abs=0 peak_mb=9622.2 dtype=bf16 ok=1
```

Syntax check, then the first timing run (the script at that moment used `blocks_to_swap=32` and fell back to latent_t=7):

```bash
python3 -m py_compile scripts/lot/bench_h3.py
python3 -u scripts/lot/bench_h3.py > /tmp/lot_bench_h3.log 2>&1
```

`/tmp/lot_bench_h3.log` ends with:

```
LOT_H3 kind=attn flash=1 mem_efficient=1 math=0
LOT_H3 kind=timing task=t2va latent_t=37 ok=0 error=oom
LOT_H3 kind=timing task=t2va latent_t=7 dense_tokens=7258 lot_tokens=3436 dense_ms=15110.2 lot_ms=14661.6 ratio=1.031 peak_mb=18759.7 dtype=bf16 ok=1
EXIT:0
```

The 37-frame failure was three attempts to allocate 4385144832 bytes with about 1.85–1.99 GB free (CUDACachingAllocator warnings at 17:04:13). Ratio 1.031 is the 7-frame fallback. Packed 7258 is under the 26880 crossover. It is not an H3 speedup.

The second timing run is the live one above. It was started as:

```bash
python3 -u scripts/lot/bench_h3.py > /tmp/lot_bench_h3_t37.log 2>&1
```

after `BLOCKS_TO_SWAP` was set to 48. Resident weights after that swap printed `head_mb=742.2 tail_mb=895.9 allocated_mb=3172.9`.

Closed 1×1 parity re-run after the head-sign fix and the `gpu_guard` change (`/tmp/lot_parity_h3_sign.log`):

```
LOT_H3 kind=parity task=t2va dense_tokens=64 lot_tokens=64 rel_rms=0 rel_rms_floor=0 max_abs=0 peak_mb=9622.2 dtype=bf16 ok=1
EXIT:0
```

Phase 4 without pairs exits (CPU, nothing loaded):

```bash
python3 scripts/lot/procrustes_h3.py --pairs /nonexistent/pairs
```

```
pair directory /nonexistent/pairs does not exist. Phase 4 does not invent pairs.
```

Useful checks that are not a benchmark:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -eo pid,etime,cmd | grep -E 'bench_h3|parity_h3|train_synth' | grep -v grep
diff -rq FL2VA Ref2VA    # expect only model_index.json
```

Gate the GPU tests on `nvidia-smi` or `ps`, not on `scripts/lot/runs/day/pid`. That file still contains dead pid 2294846. `bench_h3.py`, `parity_h3.py`, and `procrustes_h3.py --smoke` call `gpu_guard.refuse_if_busy`, which refuses (never kills) when another of `train_synth.py`, `bench_h3.py`, `parity_h3.py`, `procrustes_h3.py` is running under a Python interpreter.

## Git

Branch `lot-progress`, pushed through `94434ac` (`lot: splice LoT into the H3 DiT, fix the head sign, add phase 4`). `origin` is `https://github.com/johndpope/MiniMax-H3.git`. `upstream` is MiniMax-AI. Do not push upstream.

Committed in `94434ac`: everything under `scripts/lot/` except `runs/`, `.grok/workflows/lot-day.rhai` (CPU sanity only; it no longer launches `train_synth.py`), `docs/LOT_H3_COMPUTE_PLAN.md`, and this handoff. Still untracked and not part of this port: `scripts/scd/*`, `IMF/`, `assets/scrya/`, `Ref2VA/Ref2VA.combined`, `lora_gauss_collapse`, `.grok/workflows/vfm-stack.rhai`, `scripts/start_wandb_tui.sh`.

`train_synth.py` `--save-every` defaults to 0 and does not write `last.pt` unless that flag is positive. Do not relaunch the 480-minute toy run.

Fizgig `model.py` is not in this repo. The splice hooks are committed in Fizgig as `b7f485f` on branch `lot-h3-splice` (pushed to `johndpope/Fizgig`), and fast-forwarded into `immiscible-h3-noise` (also pushed). Not merged to `master`; a `master` checkout has no `_lot`.

Do not stage `scripts/scd/`, `IMF/`, `wandb`, or `scripts/lot/runs/`.

Earlier pushed LoT commits on the fork include `5c75a3c` (adapter and synthetic trainer), `e3db6c6` (Euler inference), `1241444` (same-seed compare), `4d65f30` (mixed grid). Unpushed SCD commit `22a4814` was reset off the branch and must not be resurrected.

Issue 1 comments already posted, do not repeat them: `6074574741`, `6074662733`, `6074686915`, `6074914028`, `6075056553`, `6075068523`, `6075595059`, `6076658364` (commit `94434ac` plus the toy loss table; says no H3 images or losses exist yet). The 37-frame timing, the parity re-run, and the head-sign fix were posted as one comment: `6075595059` (https://github.com/johndpope/MiniMax-H3/issues/1#issuecomment-6075595059). Post the next comment only after a new verified `LOT_H3` line (for example a phase-4 `kind=procrustes` / `kind=smoke` line), and quote it.

## What to do next

1. Phase 4 is blocked on pairs, which is Open Question 1 in the compute plan: the user picks the encoder that fills the directory. When the user points at a directory: `python3 scripts/lot/procrustes_h3.py --pairs DIR`, and then `--smoke` on the GPU if the fit lines print `ok=1`. Quote the `kind=procrustes` and `kind=smoke` lines. Add `--out` only if the user asks to keep the fitted adapter.
2. Optional: re-run `bench_h3.py` once to record `dense_retries`/`lot_retries`. If the retry count is nonzero, the 2.643 is an upper bound. Do not run it while another GPU script is alive.
3. Phase 5 still needs `LOT_H3_TRAIN=1`. Its loss and sampler must use H3's sign: `y0_hat = y_t + σ * out`, not `flow.clean_from_velocity` (that is the `eps - x0` convention of the toy pipeline).

## Hard rules, short

- One GPU. No `empty_cache`. No killing other processes. No Qwen. No full bf16 H3. No `runs/day/last.pt` as weights.
- No SCD. No MiniMax-AI push. No commit unless asked.
- No checkpoint writes. No activation dumps. No `forward_cached` with LoT set.
- No invented Procrustes pairs. No fine-tune unless `LOT_H3_TRAIN=1`.
- The `latent_t=37` speedup is 2.643×, measured on random tensors and before the retry counter existed. Do not quote a quality claim from it.
