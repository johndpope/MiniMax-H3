# MiniMax H3 × SCD Port — Design

**Status:** design only (not implemented) — revised 2026-08-10 after a code audit of the H3 stacks  
**Audience:** Scrya agents + train infra  
**Sources:** CastleHill `ltx2-castlehill` SCD (LTX-2), Comfy `comfy/ldm/minimax/model.py`, Fizgig `fizgig/minimax/`, SCD paper arXiv:2602.10095 — Bai, He, Li, Shechtman, Huang, Wu, *"Causality in Video Diffusers is Separable from Denoising"*  

---

## 1. Goal and non-goals

### Goal

Port **Separable Causal Diffusion (SCD)** ideas onto **MiniMax H3** so we can:

1. **Amortize** early DiT layers across denoise steps (faster multi-step gen).
2. Optionally run **autoregressive long video** with **flat peak VRAM** (chunk + KV-cache).
3. Keep **H3 look** (nvfp4/int8 deploy path, existing LoRAs as much as possible).

### Non-goals (v0)

- Drop-in load of CastleHill **LTX SCD LoRAs** into H3 (impossible — different base).
- Replace Fizgig **still** LoRA training (SCD is video/temporal).
- Full **audio-native SCD** in v0 (joint AV packing is the hard part; see phases).
- Matching Grok quality via architecture alone.

---

## 2. What SCD actually is (CastleHill / paper)

```
Standard DiT video denoise:
  every step:  all layers × all frames × full attention     → O(T × L × steps)

SCD:
  encoder (early L_e layers): once per frame on clean (σ≈0), frame-causal, KV-cache
  decoder (late L_d layers):  N steps per frame on noisy latents @ σ
  combine: token_concat (paper best) — encoder prefix + noisy tokens
```

CastleHill concrete split for LTX-2 (**48** layers): **32 enc + 16 dec** (~2:1).  
Train: flow-matching velocity, gradients through both stacks, per-frame decoder, scheduled sampling.  
Infer: AR chunks, encoder once, decoder multi-step, streaming VAE decode.

**Payoff if it works:** long duration + amortized compute — **not** sharper stills.

**The paper's own numbers differ from the CastleHill reading above, in ways that matter (read 2602.10095 §14 before Phase 1):**

- **The decoder is not the tail of the encoder.** Base is WAN-2.1 T2V-1.3B, 30 layers. Encoder = layers 0–24. Decoder = "its **first 5 and last 5** layers", i.e. 0–4 plus 25–29 — *"resulting in a total of 35 layers"* from a 30-layer base, so layers 0–4 are instantiated twice and only share an initialization. This is a re-composition, not a partition, and it means **no identity path exists**: `concat(enc, dec)` cannot reproduce the base forward by construction. Phase 1's parity test has to be rewritten around this (§7).
- **The split came from a leave-one-out ablation, not from σ-invariance.** Figure 7: *"We separately remove each layer in WAN2.1 T2V-1.3B and calculate the validation diffusion loss averaged across 5 noise levels."* Earliest and latest layers cost the most to remove; middle layers cost the least — hence a decoder built from both ends and an encoder that absorbs the cheap middle. Our Phase 0 axis (a) answers a *different* question, so its knee is not directly comparable (§7 results).
- **Adaptation is full fine-tuning, 55K steps**, with flow-matching + teacher forcing and a self-forcing distillation pass for the few-step decoder. The LoRA route is **CastleHill's** choice on LTX-2, not the paper's. D3 is therefore a bet that a lighter adapter reaches the same place on a much larger base — reasonable, but it is our extrapolation and should be labelled as one.
- **Context corruption during training:** `c̃ᵢ = cᵢ + η·ζ`, `ζ ~ N(0, I)`. Modest noise on the encoder context improves robustness and context-following. Cheap, and worth carrying into Phase 3.
- **Frame-wise token concat beats channel-wise** (Appendix Table 4) — independent support for D5.
- **Reported gains are uneven.** TECO-Minecraft: 4× lower latency (0.52 vs 2.4 s/frame). Fine-tuned VBench vs a Self-Forcing baseline: 11.1 vs 8.9 FPS and 0.29 vs 0.45 s latency — ~25% throughput, ~35% latency. The headline 4× is the AR-native benchmark; the general-video number is much more modest, which is exactly the §2.2 warning in the paper's own data.

### 2.1 Caveat: the paper studied *causal* diffusers

The paper's separability finding is measured on **autoregressive** video diffusers — models already trained to reason causally over time. H3 is **bidirectional full-attention**; its early layers were never trained to do causal temporal reasoning, they reason over the whole clip at once.

**Confirmed in the paper, and it is sharper than it looks.** WAN-2.1 T2V-1.3B is itself a bidirectional model, but the probing experiments were not run on it as shipped: *"we adopt WAN-2.1 T2V-1.3B, one of the most capable open-source text-to-video models, and **convert it to a frame-wise AR generator via teacher forcing**"*. Both headline findings — feature similarity across steps, and sparse cross-frame attention in deep layers — are therefore measured **after** causal adaptation. The paper's own framing agrees that the sparsity is emergent rather than designed (*"although training uses a standard frame-wise causal mask that permits dense cross-frame attention, long-range sparsity nonetheless emerges in deeper layers as an intrinsic property of the learned model"*) — but note *"training uses a … causal mask"*: the model in which it emerges is a causal one.

The consequence for us is concrete: **there is no published measurement of these properties on a raw bidirectional base**, so a Phase 0 probe of frozen H3 is measuring a state the paper never claimed anything about. A null result there is uninformative about the port; it only sizes the gap the adaptation has to close. Our §7 axis (c) is exactly this case.

So σ-invariance alone is **necessary but not sufficient** for H3. The transfer question is whether early-layer features survive *causal masking*, which the paper never had to ask. Phase 0 must test both axes (§7).

CastleHill is the existence proof that this is survivable — LTX-2 is also bidirectional — but it needed a LoRA plus scheduled sampling to get there, not just a split. The paper needed 55K steps of full fine-tuning. Neither reached separability by splitting a frozen model.

### 2.2 Where the speed actually comes from on H3

CastleHill measured SCD as **speed-neutral** per frame on LTX-2 (`scd-achievements.md`: 48 layers × 30 steps × 336 tokens = 1.7 s/frame; SCD 16 × 30 × 672 = 1.7 s/frame). The wins came from DDiT, distillation, and BezierFlow stacked on top. **Do not assume a speedup transfers — derive it.**

For H3's dims, attention MACs (∝ S²) and linear MACs (∝ S) cross at **S ≈ 27k tokens**:

- linear per token per layer = **385.4M** MACs — qkv 115.6M (5376→3·7168) + out 38.5M (7168→5376) + fc1 154.1M (5376→2·14336, gated) + fc2 77.1M (14336→5376), read off `fizgig/minimax/model.py:255-286`
- attention per token per layer = 2 · S · 7168 MACs
- crossover = 385.4M / (2 · 7168) = **26,880 tokens**

### 2.2.1 Measured — Tier 0, RTX PRO 4000 Blackwell, bf16

`scripts/scd/tier0_bench.py` (raw JSON in `docs/tier0_results.json`). No weights: primitives timed at real dims with `torch.randn`, then composed into stock and SCD cost models. Encoder attention is a real FlexAttention block-causal-per-frame mask, not an estimate.

| Config | Tokens/frame | S | Attention share | **Measured speedup** (N=16) | FLOP model |
|--------|--------------|---|-----------------|------------------------------|------------|
| 512², T=4 (Phase 3 train) | 256 | 1,024 | 3.7% | **1.41×** | 1.41× |
| 768p, 5s | 1008 | 31,248 | 53.8% | **2.67×** | 2.72× |
| 768p, 10s | 1008 | 61,488 | 69.6% | **3.90×** | 4.01× |
| 768p, 15s | 1008 | 91,728 | 77.3% | **5.03×** | 5.24× |

Measured tracks the FLOP model to within 4%, so the ratio is compute-bound as predicted — no roofline trap. (Per-frame decoder GEMMs are M=2016, K=5376, N=28672 → ~4000 FLOP/byte, far past any consumer or datacenter ridge, so re-reading decoder weights 61×N times stays compute-bound.)

**Sensitivity to step count N** — more steps favour SCD, because the one-shot encoder amortizes further:

| Config | N=8 | N=16 | N=30 |
|--------|-----|------|------|
| 768p, 10s | 3.50× | 3.90× | 4.12× |
| 768p, 15s | 4.43× | 5.03× | 5.38× |

**Two corrections this forced on the first draft:**

1. The draft's 462M linear MACs double-counted `out_proj` at qkv's size. The real figure is 385.4M, which moves the crossover **down** to 26.9k — SCD reaches break-even earlier than assumed, and every 768p config clears it.
2. The draft predicted **~1× or worse** at 512²/T=4. Measured is **1.41×**. token_concat doubling does not dominate, because the decoder runs 2×17 = 34 token-layers against stock's 50 — SCD wins ~1.45× even when fully linear-bound. This does *not* mean the training-resolution number validates the 768p number; the two regimes have different bottlenecks. But there is no regime here where SCD is slower.

LTX-2 at 336 tokens/frame never reached the crossover. **H3 at 768p is 1.2–3.4× past it.** That — not the layer split — is why SCD is worth porting to H3.

---

## 3. MiniMax H3 surface area (as implemented)

| Fact | Value / implication |
|------|---------------------|
| DiT class | `MiniMaxH3Model` — **50** `DiTBlock`s, hidden 5376, 56 heads × 128 (inner 7168), ffn 14336 |
| Token refiner | 2 extra `RefinerBlock`s, **text only**, outside the 50 — exclude from the split |
| Sequence | Packed **`[text \| cond \| ref_audio \| ref_img \| audio \| video]`**, single unbatched `[S, C]` stream. Target audio then target video are **always the last two segments** (`comfy/.../model.py:369-381`) |
| Frame spans | Video segment is contiguous and frame-major: `n_video = latent_t * frame_rows`, `frame_rows = (h//2)*(w//2)`. **Frame boundaries are already trivially derivable** |
| Sequence size | 768p 16:9 (1344×768) → 84×48 latent → **1008 rows/latent frame**. 10s = 61 latent frames ≈ **62k rows**; 15s ≈ **92k rows** |
| Sparse attention | **Not shipped.** README: native sparse attention exists in training, "not included in the initial open-source release" — every open H3 inference is dense O(S²) |
| Video latent | 24ch, patch **1×2×2**, 24 fps domain |
| Visual VAE | **Temporally causal already** (f16t4d24, `vae_clip_length: 17`, `vae_token_drop: 3`) — streaming AR decode is near-free, unlike LTX |
| Audio | 32ch stereo, 40 Hz, **joint** denoise with shifted σ (video 12 / audio 3) |
| AdaLN | `AdalnProj = Linear(2688 → 6·5376·3)` ≈ 260M/block × 50 ≈ **13B of the 33B**. Pruned/curve checkpoints replace it with an `adaln_t_table` (`fizgig/minimax/model.py:54-59`, Comfy `adaln_curve_grid`) |
| Conditioning | Qwen3-VL text; fl2va keyframes / r2v refs as cond rows |
| Attention | Full packed attention + RoPE; `mask=None` **hardcoded** (`comfy/.../model.py:171`), no KV API. RoPE is a pure function of `position_ids [S,3]` (`model.py:451-458`) |
| Train (Scrya today) | Fizgig still-oriented (`T=1`); clip LoRA via other stacks |
| Infer (Scrya today) | Comfy `nodes_minimax_h3.py` — full stack every step |

**Which stack to build on:** Comfy's H3 is **inference-only** — `_mod_gate` accumulates residuals in place (`x[a:b].addcmul_(...)`, `model.py:209-213`), autograd-hostile. Fizgig's equivalents are out-of-place and autograd-safe, and `_run_block` already implements per-block CPU offload with recompute. **Build the SCD wrapper on Fizgig; port to Comfy after.** Parameter names match the checkpoint in both.

**H3-native split (mirror paper ~2:1):**  
`encoder_layers = 33`, `decoder_layers = 17` (or 34/16). Configurable — but the split point should be **chosen from the Phase 0 sweep**, not assumed.

---

## 4. Gap analysis: LTX SCD → H3

Difficulty ratings below are **post-audit** — several items the first draft called Hard are already solved by H3's packing code.

| SCD requirement | LTX (CastleHill) | H3 today | Port work |
|-----------------|------------------|----------|-----------|
| Split transformer blocks | `transformer_blocks[:32/32:]` | `blocks[:33/33:]` | Easy API, same idea |
| Frame-level causal mask — *spans* | Built for LTX frame tokens | Video is the last segment, contiguous, frame-major; `PackedLayout.segments` gives the table | **Easy** (was rated Hard) |
| Frame-level causal mask — *kernel* | LTX passes `self_attention_mask` | `mask=None` hardcoded; a dense `[S,S]` mask at S=62k is **3.8 GB** | ~~**Hard — the main risk.**~~ ✅ **Done** — `scd_attention.block_mask`, a compiled FlexAttention `create_block_mask` with a `mask_mod` over `position_ids[:,0]`, exactly as predicted. It was not the main risk: the KV cache it feeds is 53 GB at the same size |
| Encoder σ=0 clean pass | Video latents clean | Video (+ audio, see D1) | Medium |
| KV-cache in attention | LTX Attention supports cache | H3 `Attention.forward` has **no** kv_cache | **Hard** — fork attention (Fizgig's is plain SDPA, easier to fork than Comfy's fused in-place kernels) |
| KV-cache *memory* | ~13 GB @ 30s, fits 32 GB | **~0.95 GB per latent frame** at 768p bf16 → 58 GB @ 10s, 87 GB @ 15s | **Hard — new blocker**, see §8.1 |
| token_concat combine | Prefix encoder feats | Same math; H3 rows are already 2D `[S,C]` | Medium |
| RoPE under token_concat | CastleHill duplicates/extends pos | `rope_freqs()` is a pure function of `position_ids [S,3]` — just concat position rows | **Easy** (was rated "port carefully") |
| Per-frame decoder train | 1 frame / forward | Frame slice = `[vstart + i·frame_rows, vstart + (i+1)·frame_rows)` | Medium |
| AR inference loop | `scd_inference.py` | New H3 AR driver; VAE is already causal so streaming decode is cheap | Medium–large |
| Audio joint stream | N/A in SCD v1 LTX focus | Audio rows precede video and share the same `cursor` — **dropping them does not shift video RoPE** | Medium (see D1) |
| Comfy deploy | Custom LTX path | Must register SCD nodes or replace sampler | Medium |
| Existing H3 LoRAs | — | Trained on **full 50-block** graph | **Likely incompatible** with SCD split without retrain |

---

## 5. Design decisions (locked for v0)

### D1 — Video-only *decoding*, but audio rides the shared encoder cache

**v0:** Run the SCD **decoder** only on video token rows.  
Audio: run the **encoder at σ=0 over clean video *and* audio rows**, then decode audio with a small second decoder pass **reusing the same encoder KV cache**.

**Rejected:** the first draft's option (a) — "generate audio with a standard joint forward." That is a second full 50-layer pass over the *same 62k-token sequence*, which erases the speed win for any AV output, i.e. all of them. It would also invalidate the Phase 2.5 benchmark.

**Principle:** the encoder KV cache is the asset. Every modality decoder should amortize against it rather than re-running the stack.

**Verified safe:** audio rows precede video and derive from the same `cursor`, so including or excluding them does not perturb video RoPE positions (`comfy/.../model.py:369-381`). Either choice is implementable without touching the position math.

**Rationale:** CastleHill's payoff was visual AR length; H3's product is native stereo AV. Full *causal* AV SCD (audio also frame-causal) remains out of scope for v0.

> **Amended 2026-08-11 by the Phase 2 mask-cost measurement.** "Audio rides the shared encoder cache" is still the right shape, but the cache was emptier than this decision assumed. Treating audio as *context* makes it blind to video by construction, and the encoder's audio rows measured centered cos **0.047 / −0.074** against a bidirectional pass — no video conditioning at all, so the second decoder pass would have inherited no audio–video fusion and had to create every bit of sync in 7 blocks.
>
> **Audio is therefore ordered, not context**, on its own 40 Hz spans (Phase 2's `av` clock), which brings those rows to **0.445 / 0.339** while keeping the mask causal in both directions and the KV cache sound. Video pays ~0.03 of centered cos. The "full *causal* AV out of scope for v0" clause below is what this overturns: audio is now causal in the ENCODER. It stays bidirectional in the DECODER, which is the part D1 was really about, and the speed argument is untouched.

### D2 — Two products, two codebases (until proven)

| Track | Code home | Purpose |
|-------|-----------|---------|
| **A. LTX SCD (now)** | `ltx2-castlehill` | Long AR **today** |
| **B. H3 SCD (research)** | new package under Scrya or Fizgig fork | H3 look + long AR later |

Do **not** merge CastleHill into Comfy until B has a kill-criteria pass.

### D3 — Train as LoRA on frozen base (mirror CastleHill)

**This mirrors CastleHill, not the paper.** 2602.10095 full-fine-tunes 55K steps on a 1.3B base (§2); CastleHill got there with a LoRA on LTX-2. We are betting the lighter route also works on a base ~20× larger, which is plausible — bigger bases usually need *less* adapter capacity per unit of behaviour change — but it is an extrapolation from one data point, and §7 axis (c) says the adapter has to *create* deep-layer locality rather than reinforce it. If Phase 3 stalls, raising rank or unfreezing the decoder half is the first thing to try, not the last.

- Freeze H3 base (int8 train DiT preferred, same as Fizgig).  
- LoRA on **both** encoder and decoder blocks (or decoder-heavy ablation).  
- **Exclude encoder `adaln_proj` from LoRA targets.** The encoder runs at a single σ=0, so its modulation is one constant vector — the gradient is rank-1 and the adapter is wasted capacity. Target `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` only on the encoder half.
- Optimizer: start **AdamW**; Muon only after token_concat stable (CastleHill lesson: Muon lr must drop with doubled seq).

### D4 — Per-frame decoder is mandatory

CastleHill: multi-frame decoder train → grid artifacts at 1-frame infer.  
H3 SCD v0 trains decoder on **one latent frame** per step, matching AR infer.

### D5 — Combine mode: `token_concat` only in v0

No `add` path in production configs (CastleHill: add → mush).

### D6 — Existing `scrya_iso_*` LoRAs

Treat as **non-portable** onto SCD graph without retrain.  
Plan: retrain iso/scene LoRAs **on SCD graph** after backbone works, or keep full-graph H3 for short clips + SCD for long.

---

## 6. Target architecture

### 6.1 Module map

```
packages/minimax-h3-scd/   (proposed)
  scd_model.py          # MiniMaxH3SCDModel wraps MiniMaxH3Model
  attention_kv.py       # Attention + optional KV cache + causal mask
  packing.py            # frame spans in packed [text|audio|video] layout
  combine.py            # token_concat / shift_encoder_features
  train_strategy.py     # SCDTrainingStrategy for H3 flow (video σ)
  infer_ar.py           # chunked AR loop + streaming VAE decode
  comfy_nodes.py        # optional: MiniMaxH3SCDImageToVideo / AR sampler
```

**Wrap, don’t rewrite** Comfy/Fizgig weights loaders. **Decided (was open question §13.3):**

- **Build on Fizgig.** Its `_mod_scale_shift` / `_mod_gate` are out-of-place and autograd-safe; its `Attention` is plain `F.scaled_dot_product_attention`, so adding a mask + KV cache is a contained change. Comfy's is inference-only (in-place residual accumulation) and uses fused `rms_rope_split_half_` kernels that a cache retrofit would have to bypass.
- Parameter names match the checkpoint in both stacks, so a LoRA trained through the Fizgig wrapper maps back onto the Comfy graph.
- Infer: offline CLI first; Comfy nodes only after the research gates pass.

### 6.2 `MiniMaxH3SCDModel`

```text
base: MiniMaxH3Model (50 DiTBlocks)

encoder_blocks = blocks[0 : L_e]      # default L_e=33
decoder_blocks = blocks[L_e : 50]     # default 17

forward_encoder(clean_video_latents, text, …, kv_cache):
  pack sequence with video rows at σ=0 (clean)
  frame-causal mask on video-video attention only
  run encoder_blocks once; write KV; return encoder features per frame
  shift features by 1 frame (frame t decoder sees t-1 context)  # paper

forward_decoder_per_frame(noisy_frame_t, σ, enc_feat_{t-1}, text, …):
  pack noisy video frame tokens (+ text; audio policy per D1)
  token_concat(enc_feat, noisy_tokens)
  run decoder_blocks with AdaLN at σ
  return velocity (or H3-native prediction head) for that frame

# optional multi-frame decoder for ablations only — not default train
```

### 6.3 Packing / masks (the H3-specific hard part)

Define explicit **segment layout** for each forward:

```text
[ text_tokens | (optional cond) | audio_tokens? | video_frame_0 | … | video_frame_{T-1} ]
```

Must implement:

1. **Index ranges** — *already available.* Video is the final segment; frame `i` occupies
   `[vstart + i·frame_rows, vstart + (i+1)·frame_rows)` where `frame_rows = (latent_h//2)·(latent_w//2)`.
   Read it off `PackedLayout.segments`; do not recompute.
2. **Causal mask:** video frame `i` attends to video frames `≤ i`, and always to text / cond / ref rows.
   **A dense `[S,S]` mask is not an option** — 62k² bool = 3.8 GB. Use
   `torch.nn.attention.flex_attention.create_block_mask` with a `mask_mod` closure over the frame
   index derived from `position_ids[:,0]`. This is the single largest piece of new work in the port.
   **Shipped as `scd_attention.block_mask` (2026-08-11).** The `mask_mod` did turn out to be a
   closure over `position_ids[:,0]` exactly as written here — `row_time` reads that axis — but the
   rule is not "video ≤ i, plus always text/cond/ref". It is just `k_time <= q_time`, with context
   at −inf; "always visible" and "blind to the ordered rows" are one inequality read from its two
   ends. And this was **not** the largest piece of new work: the mask is 3.8 GB at 768p/15s and the
   KV cache it feeds is 53 GB.
3. **Encoder cache key:** cache K/V only for encoder layers; append one frame at a time in AR.
   See §8.1 — the cache does not fit naively at 768p and needs a window or quantization from day one.
4. **RoPE:** *solved.* `rope_freqs()` is a pure function of `position_ids [S,3]`, so the `token_concat`
   prefix is handled by concatenating position rows — the CastleHill `_duplicate_pe` analog is
   two lines here, with no hidden state to keep consistent.

### 6.4 Training strategy (H3 flow)

Reuse CastleHill curriculum, retargeted to H3:

| Hyper | v0 default | Notes |
|-------|------------|--------|
| encoder_layers | 33 | ~2:1 of 50 |
| decoder_input_combine | token_concat | paper + CastleHill |
| per_frame_decoder | true | required |
| clean_context_ratio | 0.1 | paper |
| first_frame_cond_p | 0.5 | aligns with fl2va |
| scheduled_sampling | 0→50% AR after warmup | CastleHill schedule |
| resolution | 512² then 768² | 24 GB path |
| frames / clip | 4–8 latent frames | grow after stable |
| batch | 1 | PRO 4000 |
| steps | 2k–3k LoRA | like scd_isometric |

Loss: H3’s training objective (velocity / flow as in Fizgig `trainer.py`) on **video** prediction; do not invent a second loss.

### 6.5 Inference modes

| Mode | Use | VRAM |
|------|-----|------|
| **M0 Clip-SCD** | One short clip (≤6s): encoder once per frame, decoder N steps — **no AR** | ~same or less than full stack |
| **M1 AR-SCD** | Chunk AR (e.g. 4 latent frames, 1 overlap) for 30s+ | flat vs length |
| **M2 Hybrid** | Full H3 for first 1s keyframe quality, SCD AR for tail | product pragmatism |

Ship **M0** before M1 (proves split + combine without AR exposure bias).

---

## 7. Implementation phases

### Phase 0 — Prove the hypothesis (1–2 weeks, offline, **local hardware**)

Three axes, not one. The first draft tested only (a).

**(a) σ-invariance.** Feature cosine similarity of each block's output across a σ grid, on the same video.
- Early layers highly similar across σ → the split is justified. σ-sensitive → SCD payoff is low.
- Hypothesis worth noting: H3's released checkpoints are **CFG-distilled**, which flattens trajectories, so σ-invariance may read *stronger* here than in the paper's models.

**(b) Causal-mask sensitivity — the transfer test.** Re-run each block with a frame-causal mask and measure feature drift vs unmasked. This is what §2.1 says the paper never had to ask, and it is the axis most likely to kill the port. High drift in early layers means the encoder half has to *relearn* temporal reasoning, not just inherit it — which is a training bill, not a wrapper.

**(c) Late-layer cross-frame attention sparsity.** The paper's second finding. Measure the fraction of attention mass that stays intra-frame in blocks 33–49. If deep layers are already mostly intra-frame, the per-frame decoder is nearly free; if not, expect quality loss at 1-frame decode.

**Split point comes out of this sweep** — pick the layer where σ-sensitivity or causal drift rises, rather than assuming 33/17.

**Cost: $0.** Forward passes only, 512²/T=4–8 (~256 rows/frame). Runs on the local PRO 4000s via Fizgig's block-streaming offload.

**Script written — `scripts/scd/phase0_probe.py`.** Loads the full bf16 checkpoint NF4-quantized (~11 GB resident) with the tail blocks CPU-parked, then sweeps all three axes and emits JSON + an optional wandb run. It reports both a *cumulative* causal drift (whole stack masked) and an *isolated* per-block drift (unmasked activations in, one block masked), because only the isolated curve identifies **where** to split; the cumulative one shows what the split will cost end to end.

```bash
python3 scripts/scd/phase0_probe.py --self-test          # no checkpoint needed
python3 scripts/scd/phase0_probe.py \
    --checkpoint /path/to/MiniMax-H3-FL2VA/FL2VA/transformer \
    --latents clip_latents.safetensors --text clip_te.safetensors \
    --out docs/phase0_results.json --wandb h3-scd
```

`--self-test` passes today: it asserts **prefix equivalence** — under the causal mask, frame 0's rows are bit-identical (1.2e-7) to running a sequence in which later frames do not exist, while the mask still materially changes output (4.0e-2). A mask that is subtly non-causal would produce a plausible-looking drift curve that means nothing, so this is asserted rather than eyeballed. The tiny model is seeded, so both numbers are reproducible run to run rather than a single sample.

**Multi-frame latents — ✅ resolved (2026-08-10).** Fizgig's VAE encoder used to hard-reject `T>1`, slicing `moments[:, :, -1:]` for stills. That slice is a no-op at T=1, so removing it leaves the image path byte-identical while letting clips through; the 3D stack underneath was already causal (verified in float64: perturbing pixel frames 13–16 of a T=17 clip moved only latent frame 4, frames 0–3 by exactly 0). Axes (b) and (c) are meaningless on a single frame, so this was load-bearing.

Independently, Fizgig upstream landed `694c2f9` on 2026-08-07, which made the *rest* of the H3 stack multi-frame: `image_position_ids` gained a `latent_t` argument emitting t-major frame blocks on the `(1,4,4,4,4)×5/3` grid, the DiT's audio block is now sized from pixel rather than latent frames, and the VAE gained `decode_clip`. Encode was the one remaining `T=1` pin. The probe originally carried its own `video_position_ids` because upstream's pinned the video at a single frame; that reimplementation is now **deleted** and the probe calls `mm.image_position_ids(..., latent_t=...)` directly — verified bit-identical across five shapes (`latent_t` 1→37, with and without audio, up to S=9307) before the swap. One authority for the rotary layout, not two.

Note the two grids are not inverses: encode gives `T_latent = ceil(T_pixel/4)` (measured on the released fp16 weights at T = 1, 2, 4, 5, 9, 17, 21, 22, 25, 29 — every one matches), while `decode_clip`'s sampling grid is `5n+2` latents ↔ `17n+5` pixels. Fine for the probe, which only encodes; do not round-trip without checking.

`scripts/scd/encode_clip.py` turns an ordinary video file into the cache the probe wants:

```bash
python3 scripts/scd/encode_clip.py clip.mp4 --latent-t 8 --size 512 512 \
    --out clip_latents.safetensors
```

**Text states — ✅ done.** `scripts/scd/encode_text.py` produces the `--text` half from the local nvfp4-AWQ Qwen3-VL-32B (`te_quant="auto"` keeps the packed nvfp4 rather than re-quantizing to NF4, so no extra ~9% error), emitting the raw layer-50 output that H3 conditions on. Verified end to end: a 267-char prompt → 66 tokens → `[66, 5120]` bf16, read back through the probe's own `load_text`.

**Tokenizer bug found and fixed.** `fizgig/assets/qwen3vl_tokenizer` shares `tokenizer.json` byte for byte with this repo's `text_encoder/`, but its `tokenizer_config.json` omitted seven H3 special tokens — `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_{start,end}|>`, `<|caption_{start,end}|>`. Plain prose tokenizes identically, so nothing ever complained; H3 prompts are not plain prose, because dialogue is wrapped in `<d>[Language] ...</d>` (§"Prompt structure"). Measured: `<d>` became `[90707, 30768, ...]` instead of the single id `151669`, and `<|lyrics_start|>hold on<|lyrics_end|>` expanded from 4 tokens to 16. **This was a live bug in Fizgig's own H3 caption caching**, not just a probe inconvenience — any caption carrying dialogue markup trained against subtly wrong conditioning.

Fixed in `fizgig/minimax/embedder.py` (`_add_h3_special_tokens`, applied at TE load), *not* by patching the asset: that tokenizer directory is shared with the krea2 encoder, and editing it would change tokenization on an unrelated model path. The tokens exist in neither `vocab.json` nor `tokenizer.json` — transformers numbers them sequentially on load — so **the order is load-bearing** and is pinned to MiniMax's own ordering, giving ids 151669–151675, all inside the 151936-row embedding table. Verified: patched-bundled and repo tokenizers now agree on every test string with pre-existing ids unmoved, the shared asset still does not resolve `<d>` (krea2 unaffected), and the two encoders produce **bit-identical** `[66, 5120]` conditioning end to end.

`encode_text.py` still defaults `--tokenizer` to this repo's `text_encoder/` and hard-fails if a special token present in the prompt is unknown to whatever tokenizer it was handed — belt and braces, since the failure mode is silent.

encode_clip.py asks for exactly `1 + 4*(latent_t - 1)` pixel frames. A partial trailing group would leave the last latent frame seeing fewer pixel frames than its siblings — the same asymmetry axis (b) is trying to measure, injected into the input. Two incidental findings: the released fp16 VAE has 61 stray bytes past its last tensor and is rejected by the strict safetensors parser (Fizgig's `MemoryEfficientSafeOpen` reads it fine), and fp16 encoding is the default because fp32 only upcasts fp16 weights while tripling activation memory — measured cost 0.23% relative L2.

**All three Fizgig fixes are upstreamed** (2026-08-10) — [#56](https://github.com/shootthesound/Fizgig/pull/56) tokenizer, [#57](https://github.com/shootthesound/Fizgig/pull/57) sharded loading, [#58](https://github.com/shootthesound/Fizgig/pull/58) multi-frame encode — one concern per PR, per Fizgig's `CONTRIBUTING.md`. The local checkout is at upstream `master` with the three cherry-picked on top while they are in review, so nothing here depends on a private fork.

**Checkpoint — downloading (2026-08-10), ~66 GB.** Two traps:

- **`hf download MiniMaxAI/MiniMax-H3` unfiltered pulls 498 GB, not 66.** The repo carries three copies of every weight: `FL2VA/`, `Ref2VA/`, and the root diffusers-modular layout (§"Two checkpoint layouts"). Phase 0 needs only the FL2VA DiT: `--include "FL2VA/transformer/*"`, 13 shards. Set `HF_HUB_ENABLE_HF_TRANSFER=1` — it roughly doubled throughput here (4 → 8 MB/s).
- **The Hub ships 13 shards; Fizgig's loader took one file — ✅ resolved.** `load_minimax_h3_dit(path)` now dispatches on `os.path.isdir` to a new `ShardedSafeOpen` (`fizgig/krea2/safetensors_utils.py`) presenting a shard directory behind the same `keys()`/`get_tensor()` interface, so `--checkpoint` accepts either form and no 66 GB merge copy is needed. **Key names were the real risk and they are fine**: diffing the Hub shard index against `MiniMaxH3DiT`'s state dict gives 535/535 exact matches, zero renames either way — Fizgig's names came from the ComfyUI conversion, so this was not a given.

NF4 remains the default because dense bf16 needs ~66 GB of the box's 91 GB RAM. Quantization error is common-mode (both sides of every cos-sim go through the same weights) but `--base-quant none` exists to check that.

Deliverable: the JSON above plus a wandb report (layers 0–49 × σ grid × {unmasked, causal}). Video and audio rows are scored separately — AdaLN is modality-specific. Note that *isolated* audio drift is 1.0 by construction (only video queries are masked, per D1); it is a control that the mask touches nothing it should not. The audio number to read is the **cumulative** one, which does move, because the video values audio attends to have changed.

#### Phase 0 results (2026-08-10) — `docs/phase0_validation_ondist.json`

Run on the real FL2VA DiT, NF4 base, a 512², 7-latent-frame clip from the scrya set (`S=2106` = 230 text + 84 audio + 1792 video, `frame_rows=256`, 50 blocks, ~25 s/sweep). Both noise levels are drawn on H3's own training density — reference σ=0.571 (`u=0.1`), test σ=0.991 (`u=0.9`) — sharing one noise draw, so the delta is σ sensitivity rather than a different sample. `docs/phase0_validation.json` holds the earlier run that used σ=0 as the reference on an off-grid 8-frame clip; it is kept only as the record of what that reference did to the numbers (see "σ-density correction" below).

**What these numbers are and are not.** Every measurement below is on the **frozen base, zero-shot**. Neither SCD nor CastleHill claims a frozen model is already separable — the method is a LoRA plus scheduled sampling that *teaches* the model to rewire (§2.1, D3). So none of these axes is a go/no-go gate, and reading a low number as "fails" is a category error. What Phase 0 buys is a **budget and a placement**: how far the base already sits from the target wiring, which half has to move, and where to split. Treat the axes as cost estimates for the adapter, not as verdicts on feasibility. The one hard output is the split point — **encoder bounded at 30 blocks, not the assumed 33/17, with the decoder's index set fixed by the leave-one-out run below rather than by the knee**.

**Read the first probe's raw numbers as void.** `phase0_probe.py` scored every axis as plain cosine similarity on the residual stream, and on this model that metric is broken: the residual stream is almost entirely one shared vector. Measuring `‖mean row‖ / mean‖row‖` over the video rows gives **0.918 at block 0, ~0.81 through the middle, and 0.997 at block 49**, against 0.024 for uncorrelated rows. Raw cos-sim on rows that are 99% a common vector reports ~1.0 for two runs whose informative parts are unrelated. That single artefact manufactured the probe's headline result — that σ-invariance *rises* with depth, which is backwards from what SCD needs and should have been the tell. `scripts/scd/phase0_validate.py` re-measures everything with relative L2 and with cos-sim after subtracting the clean run's mean row ("centered cos"), and adds two controls.

**(a) σ-invariance — ✅ passes, and the knee at 29/30 is now the strongest number in Phase 0.** Two independent row sets put it in the same place.

- **Text rows** (only noised indirectly, through attention — the shared context an SCD encoder would cache): centered cos 0.916–0.925 for blocks 0–12, a step to a **0.973–0.975 plateau held flat from block 13 through 27**, then 0.960 (28), 0.910 (29), and a collapse — **0.777 (30)**, 0.677 (31), 0.543 (32), 0.373 (34), 0.269 (36), 0.170 (39), 0.037 (49). The plateau is flat to within ±0.002 over fifteen blocks and the first step off it is a 0.13 drop. That is a knee, not a slope.
- **Video rows**: 0.226 at block 0 (the residual there is still mostly the patch-projected noisy latent, so this is arithmetic), then **0.807 at block 1 and a 0.78 plateau through block 25**, 0.711 (26), 0.657 (29), **0.402 (30)**, 0.345 (34), 0.246 (39), 0.082 (49). Same cliff, same block, on rows that literally carry the noise.

That video-row curve is the single largest correction from the σ-density fix: on the old σ=0 reference it read **0.061 mean over blocks 1–25**, i.e. flat noise, and the write-up concluded video rows carried no σ-invariant content at all. On-distribution the same mean is **0.779**. The earlier reading was an artefact of comparing against a noise level H3 never trains on, not a property of the model.

So the encoder is **blocks 0–29** — 30 blocks, empirical, against the assumed 33. Audio rows sit at ~0.16 mid-stack and drift negative late; they are being denoised on their own remapped σ, so they are decoder-side state, not cacheable context.

**(b) Causal-mask drift — ✅ passes, and this was the axis expected to kill the port.** Isolated frame-causal masking costs almost nothing: centered cos ≥0.97 through block 45 with a single dip to 0.922 at block 46, relative L2 ≤0.035 everywhere except block 0 (0.094). Two controls make that trustworthy rather than merely convenient:

> **Correction (2026-08-11), and read this before quoting the numbers below.** These were measured with a mask that restricts only *video* queries, leaving text and audio rows attending to every frame. That is frame-causal for **one block** and not for a stack — the context rows carry the future backward, and the encoder therefore has to run a stricter mask that blinds them to video (Phase 2, and protocol rule 9). The per-block *isolated* numbers here are unaffected: for a video query the two masks are identical row for row, so this paragraph measures what it says it measures. What does not follow is the sentence the reader wants to write next — that masking the *encoder* is nearly free. That is a cumulative question under the stricter mask, and it is answered separately in "Phase 2 mask cost" below.

- *Does the mask reach the real forward path?* Re-run with a deliberately brutal mask — a video row sees text, audio, and only its **own** frame. Drift jumps to relative L2 0.202 and centered cos 0.869 at block 0, bottoming at 0.740 (block 46), a 2–6× larger effect than the causal mask at every block. The mask is live; the small frame-causal number is a property of the model, not dead plumbing.
- *Why is it small?* Decompose where video queries actually spend attention: **text 0.247, audio 0.111, past 0.279, own 0.094, future 0.270**. Over a quarter of the mass does sit on future frames — but attention over video is close to uniform (own-frame mass 0.094 against a 0.092 uniform baseline: no intra-frame locality at all), and with common-mode 0.80–0.997 the value vectors are near-interchangeable. Deleting a quarter of a near-uniform average over near-identical values barely moves the output. Convenient for us, and worth remembering it is a property of a bidirectionally-trained model with massive activations, not evidence that H3 has latent causal structure.

**(c) Late-layer sparsity — absent, and this is the expected result, not a failure.** Measured on the paper's own metric (mass from queries in frame *i* to keys in frames *j<i*), past-frame mass is **flat with depth**: 0.288 mean over blocks 0–24 vs 0.270 over blocks 25–49, ending near where it starts (0.350 at block 0, 0.308 at block 49). Intra-frame mass is likewise flat and sits *at* uniform — **0.096** (blocks 0–24) and **0.091** (25–49) against a **0.092** baseline. No trend in either direction.

This does **not** contradict the paper, because it is not the paper's experiment. Per §2.1, their curve was measured on WAN-2.1 *after conversion to a frame-wise AR generator via teacher forcing*; deep-layer sparsity is described as emerging under causal training. Frozen bidirectional H3 has had no such training, so flat cross-frame mass is what the paper's own account predicts. What the number gives us is a **budget**: the locality the decoder needs does not exist yet and the adaptation has to create it, so decoder-half capacity is not optional and D4's per-frame decode should be costed as a thing training must buy, not a property to inherit. The measurement to actually compare against Figure 3 is this same probe re-run *after* Phase 3 — if the curve is still flat then, that is a real negative.

**On the split point.** The paper did not choose 25/30 from σ-invariance; it used a leave-one-out ablation on validation diffusion loss across 5 noise levels (§2), finding earliest and latest layers most load-bearing. Our 29/30 answers a different question and the two should not be conflated. That ablation has now been run — see below.

#### Phase 0 leave-one-out (2026-08-10) — `docs/phase0_leaveout.json`

`scripts/scd/phase0_leaveout.py` reproduces the paper's actual split criterion on H3: skip block *i* (identity on the residual stream, which is what "remove" means for a pre-norm residual transformer), run the rest, and score the real flow-matching objective through `final_layer` — MSE of `video_out` against `x0 - noise`, Fizgig's training target. Reported per block as relative cost, `loss_without_i / loss_baseline - 1`, averaged over the 5 noise levels. 3.2 min for the full 50-block sweep on one 4090 (caching each block's input during the baseline pass lets each leave-one-out run resume at *i+1*, ~2×).

**The paper's Figure 7 shape reproduces on H3, and it is far more extreme.** Mean relative cost by third: **early 0.0948, middle 0.0004, late 0.0471** — early-high, middle-flat, late-high, matching the paper. But the load is concentrated in a handful of blocks rather than spread over the ends: block **0 costs +96%**, **49 +82%**, **1 +55%**, **48 +4.5%**, and every other block is within ±0.03. Those four blocks sum to +2.38 relative loss; the remaining 46 sum to **−0.06**. Twenty-six blocks are *negative* — removing them slightly improves the loss (min −0.029 at block 47) — which at this magnitude is measurement floor on one clip, not a real gain, but it does bound the middle's contribution at approximately nothing.

**This settles the decoder construction; it does not compete with 29/30.** A both-ends decoder is not a stylistic choice inherited from the paper; on H3 it is the only construction the data supports, because a 20-block suffix decoder {30–49} would contain exactly one load-bearing block (49) and leave blocks 0–1 — the two most expensive in the model — frozen inside the once-per-frame encoder. Conversely the σ-invariance knee at 29/30 says nothing about where the *cheap* blocks are; it says where the cacheable context prefix ends. The two criteria are compatible and answer different halves of the question: **axis (a) bounds the encoder** (the cache stops being σ-stable after 29, so `L_e ≤ 30`), **leave-one-out places the decoder** (it must contain 0, 1, 48, 49). A candidate consistent with both is encoder = 0–29, decoder = {0–1} ∪ {45–49}, 37 layer instances — same over-provisioning ratio as the paper's 35/30. Do not freeze this until the ledger clears the protocol's n≥5 bar and reports `STABLE`; the negative-cost blocks are the tell that the noise floor is comparable to the middle's signal.

**σ-density correction — found here, and it forced a re-run of axes (a)–(c).** The first leave-one-out run used a flat σ grid `[0.1 … 0.9]` and produced a baseline loss *worse than predicting zero* (2.245 vs 2.041) with cosine similarity to the target of **−0.44 at σ=0.1**. The model is not broken — the grid is off-distribution. H3 samples training σ as `shift_sigma(u, 12) = 12u/(1+11u)` with *u* uniform, which puts the median at ~0.92 and **~3% of steps below σ=0.3**. Sampling *u* uniformly instead — the 5 levels become σ = 0.571, 0.837, 0.923, 0.966, 0.991 — the model behaves correctly (cos 0.27→0.73, MSE 0.52× trivial).

Axes (a)–(c) had been measured with **σ=0 as the reference**, a point of essentially zero training density, so `phase0_validate.py` grew `--u-ref`/`--u` and was re-run with the pair drawn on H3's own density (σ 0.571 → 0.991). The numbers above are that re-run. Two outcomes: the **knee index did not move** — 29/30 on the old reference, 29/30 on the new one, and now visible on video rows as well as text — while the **video-row σ-invariance level moved by an order of magnitude**, 0.061 mean over blocks 1–25 before, 0.779 after. The conclusion that survived was the one being tested; the conclusion that was overturned was an incidental claim the earlier write-up had made confidently. Worth remembering when reading anything else measured against an arbitrary reference point.

Two smaller corrections found in the same debugging pass: all earlier Phase 0 runs used `latent_t=8`, which is **off the DiT's `5n+2` grid** (2, 7, 12, …) and makes `model.forward()` raise; the block-by-block path tolerates it, so it went unnoticed. Testing confirmed it was not the cause of the anti-correlation, but both scripts now default to 7 and call `pixel_frames_for_latent()` as an assertion. Ratios throughout are normalised by the per-σ baseline because raw loss varies ~2× across the grid and would otherwise let σ=0.571 dominate the average.

Consequences to fold into the design: D1's assumption that audio rides the shared encoder cache is not supported by axis (a) — audio rows are decoder-side state (see open question 1); D4 now carries an explicit training cost rather than an assumed-free one; Phase 1's identity-path test needs rewriting because the paper's decoder is a re-composition of both ends rather than a suffix (§2), so `concat(enc, dec)` provably cannot equal the base forward.

Caveat on generality: the curves quoted above are one clip, one prompt, one resolution, one σ pair for axes (a)–(c), and all of it zero-shot on a frozen base. Enough to place scaffolding and size the adapter; not enough to commit training spend against. `scripts/scd/run_phase0.py` runs both probes on one model load and appends each run to `docs/phase0_ledger.jsonl` with its full config, grouped so that runs of different configurations can never pool. What follows is what that ledger says once the knee is asked to survive something. Breadth over content turned out to be the least informative thing to buy: the axes that moved the answer were the ones inside the measurement, not the ones in the clip set.

Three clips are in the ledger so far, summarised in `docs/phase0_summary.json` and chosen to be as unlike each other as the corpus allows: a live-action studio portrait, a photorealistic isometric room diorama on an arcing camera, and a miniature Formula One tracking shot. All three return **knee text 30, knee video 29, load-bearing {0, 1, 48, 49}** — identical, not merely close, on content dissimilar enough that reshuffling was the expected outcome.

Two things stop that from being as strong as it reads. First, **only the text knee survives an audit of its own threshold.** `knee(curve, frac)` is "first block below `frac` of the plateau", and `frac=0.85` was never itself swept. Sweeping it (`knee_*_by_frac` in the summary, over 0.70–0.95) moves the text knee by 2 blocks — 31, 31, 30, 30, 29, 29 — and the video knee by 5 — 30, 30, 30, 29, 26, 25, monotone. A knee that tracks its threshold is reporting the threshold: the video σ-invariance curve has no plateau to depart from, so **"knee video 29" is not independent corroboration of the text knee and should not be read as such.** The block-30 split rests on the text curve alone. Second, the three clips vary only in *content*: all are `latent_t=7` at 512×512, so the three-way agreement was measured along one axis while geometry — the thing that actually changes the sequence length the blocks operate on — was held fixed.

A fourth row, `isodiorama640`, was added to test that second point directly. It is the same source clip over the same 29-frame window with a byte-identical text embedding, re-encoded at 640×640: 40×40 latent, **3142 tokens against 2134**, of which 2800 are video against 1792. Video geometry is the only variable. It returns **the same knee text 30, knee video 29, load-bearing {0, 1, 48, 49}**, with a frac profile within one block of the 512 group at every threshold. Because `latent_hw` is part of the ledger's grouping key, this lands as a separate `n=1` group rather than being pooled into the 512 numbers — the summary reports the 512 group, and the geometry run is read beside it, not averaged into it.

That is a genuine strengthening: a 56% larger video-token budget moving nothing suggests the split is a property of the block stack rather than of the sequence it is fed.

**The σ pair, by contrast, moves it a lot** — and this is the axis that matters most, because SCD reuses one encoder pass across denoising steps, so how far apart the two probe points sit is not an incidental setting. Three further pairs on `isodiorama`, all with both ends on-distribution, sweeping the separation in *u* (uniform on H3's own training density) rather than in raw σ:

| u pair | σ pair | Δu | knee text | text curve drop | knee across frac 0.70–0.95 |
|---|---|---|---|---|---|
| 0.1 → 0.9 | 0.571 → 0.991 | 0.8 | **30** | 0.78 | 31 … 29 (spread 2) |
| 0.1 → 0.5 | 0.571 → 0.923 | 0.4 | **31** | 0.64 | 33 … 29 (spread 4) |
| 0.7 → 0.9 | 0.966 → 0.991 | 0.2 | **35** | 0.48 | 40 … 32 (spread 8) |
| 0.3 → 0.5 | 0.837 → 0.923 | 0.2 | **37** | 0.32 | 45 … 32 (spread 13) |

The knee is monotone in Δu, which on its own would read as "narrower σ separation leaves more of the stack shareable". The last two columns say otherwise, and they say it in a way that was cheap only because the `frac` sweep from the previous finding was already in place. As Δu shrinks the text curve's total drop over all 50 blocks collapses from 0.78 to 0.32, and the knee's own threshold-sensitivity blows up from 2 blocks to 13. **Narrow pairs do not push the knee later; they stop resolving a knee at all.** There is less divergence to measure, so "first block below 85% of the plateau" starts firing on the shape of the noise. The one property that made the text knee trustworthy at the baseline — a real plateau, hence threshold-independence — is present only at wide separation.

That resolves the axis in the reassuring direction rather than the alarming one. The widest on-distribution pair is simultaneously the most demanding question (a block σ-invariant between 0.571 and 0.991 is invariant everywhere between) and the only one the probe can actually answer, and the two well-resolved pairs agree at 30 and 31. **Block 30 stands, but it must be stated as the knee under the widest on-distribution contrast, not as an intrinsic property of the stack** — and any future re-measurement has to quote its Δu or the number means nothing.

`latent_t` is now the one unsampled axis, and testing it needs new prompts, since these describe only the first 29 pixel frames. The set is below the protocol's n≥5 bar in any case, so the summary reports `n<5, not enough to call anything stable` and no block index should be hardcoded into Phase 1 on the strength of it. The remaining nine clips are already encoded and prompted under `scripts/scd/clips/`, with provenance in `MANIFEST.tsv`; finishing the sweep is a ~45-minute GPU job rather than new work, but it only re-samples content, which is now the best-covered axis and the one that has moved least.

How these measurements are made, and what has to be true before a finding is allowed into this document, is written down in [`SCD_RESEARCH_PROTOCOL.md`](SCD_RESEARCH_PROTOCOL.md). The numbers quoted above are registered in `docs/phase0_claims.json` and verified against their source JSON by `scripts/scd/check_findings.py` in CI, so the prose here cannot drift from the data when a probe is re-run.

### Phase 1 — `MiniMaxH3SCDModel` skeleton (2–3 weeks)

- Compose blocks per §2: encoder = a prefix, decoder = a re-composition of the **cheapest-to-lose-last** ends, not the encoder's tail. Starting point from §7's two probes: encoder 0–29, decoder {0–1} ∪ {45–49}. The decoder must contain blocks 0, 1, 48, 49 — they carry +96%, +55%, +4.5%, +82% of the loss respectively and everything else in the model is within ±0.03.
- **The identity test as originally written is impossible** and must be dropped: if the decoder re-uses early blocks, `concat(enc, dec)` has more layer instances than the base and cannot reproduce a 50-block forward. Replace with two weaker but real invariants: (i) every block instance loads weights bit-identical to its source block in stock H3, and (ii) with the decoder bypassed entirely, the encoder prefix reproduces stock H3's first `L_e` block outputs exactly.
- Unit tests: weight load, prefix parity, shared-init blocks are independent `nn.Module`s (they diverge under training and must not alias).
- **No training yet.**

**Skeleton landed (2026-08-11).** `scripts/scd/scd_model.py` + `scripts/scd/test_scd_model.py`, **9/9 in 1.2 s on CPU with no weights and no GPU** — the tests run a hidden-64 stand-in through the same code path, because composition, weight copying and aliasing are shape-independent and a test that needs the 62 GB checkpoint is a test nobody runs. `test_production_split` builds a 50-block model and applies the module's real defaults, so the split Phase 2 inherits is exercised rather than only a toy one.

Three things worth carrying forward:

- **The composition consumes the base.** Tail blocks past the prefix are *moved* into the decoder, not copied; only blocks the encoder still holds (0, 1) are deep-copied. A spare copy of five 5376-wide blocks does not fit beside the base on a 24 GB card, so §8's VRAM plan depends on this and `test_tail_blocks_are_moved_not_copied` pins it.
- **`encode()` does not re-implement the base's preamble.** It captures the per-block arguments from the base's own first block through a forward hook. That preamble decides segment order, modulation rows, audio row count and RoPE positions; a second copy of it would drift from the real one the way Phase 0's transcribed numbers drifted from their JSON. There is deliberately no pixel-output path yet — the final layer needs `video_t_index`, which the preamble derives from a sorted-unique over distinct timesteps, and Phase 2's causal mask rewrites this graph anyway.
- **Prefix parity failed first, for a real reason.** The base draws fresh `torch.randn` silence audio rows on every call when `audio_noise` is None, so two forwards with identical arguments pack different sequences and disagree by ~0.6. Any bit-exact comparison across two base calls has to pass audio rows explicitly.

Both non-trivial invariants were mutation-tested rather than trusted for being green: removing the `deepcopy` fails only `test_shared_init_blocks_are_independent` — **`test_weight_parity` still passes on that mutant**, which is precisely why a `state_dict` comparison is not sufficient and `aliases()` compares `data_ptr` — and an off-by-one prefix fails `test_prefix_parity`.

`check_findings.py` now also reads `scd_model.py`'s split constants with `ast` (no torch needed, so CI runs it) and errors if `DEFAULT_DECODER_SOURCE` stops containing the load-bearing set that `phase0_summary.json` reports, or warns if `DEFAULT_ENCODER_DEPTH` drifts from the measured text knee. That closes the last leg of the loop: data → docs was already checked by the claims registry, and this is data → code.

### Phase 2 — Causal mask + encoder KV (2–4 weeks)

- Implement video frame spans + causal mask.  
- Encoder KV-cache for multi-frame.  
- Tests: frame t cannot attend t+1; cache append correctness.

**Mask + cache landed (2026-08-11).** `scripts/scd/scd_attention.py` (`FrameSpans`, `causal_mask`, `KVCache`, a masked/cached `attention`) plus `MiniMaxH3SCD.encode(mask=, cache=)` and `encode_chunked()`. `scripts/scd/test_scd_attention.py` is **10/10 in 1.1 s on CPU**, no weights and no GPU, on the same hidden-64 stand-in Phase 1 uses.

**The finding: Phase 0's mask is frame-causal for one block and not for a stack.** That mask restricts only *video* queries; text and audio rows keep attending to every frame. Comparing the prefix of a masked full-clip run against a run in which the later frames do not exist:

| blocks | video rows | context rows |
|---|---|---|
| 1 | 1.2e-07 (roundoff) | **5.1e-02** |
| 2 | **3.7e-03** | — |
| 6 | 9.0e-03 | — |

(Max abs delta, float32, on the hidden-64 stand-in with 7 latent frames — a structural property, so width and weights do not enter it.) The context rows absorb the whole clip at block 1 and the video rows read them back at block 2. So the encoder's mask must **also blind the context rows to video**; under that mask the same comparison stays at 3.6e-07 for both row kinds through all 6 blocks. This is not a preference: with the leak, chunk 2's context rows are not the ones chunk 1 cached, and the KV cache is wrong in a way no shape check can see. `test_loose_mask_leaks_across_blocks` asserts the failure in both directions so the rejected mask stays tested rather than remembered (protocol rule 9).

**What this does *not* invalidate.** For a video query the two masks are identical row for row, so §7 axis (b)'s **isolated per-block** drift numbers stand exactly as measured. What was overstated was reading them as a property of the masked *stack*.

Design notes worth carrying forward:

- **One rule covers both row kinds.** Context rows carry frame index `-1` and video rows `0..T-1`, so `visible = (k < 0) | (k <= q)` is simultaneously "context is always visible" and "context is blind to video" — the second falls out because no video key satisfies `k <= -1`.
- **The mask never re-implements a block.** `run_block` substitutes `attn.forward` for the duration of one call and restores it in `finally`, so AdaLN modulation, the gated residuals and the MLP stay the base's own code. Phase 0's probe hand-copied that block body, which is the drift this phase exists to stop repeating.
- **The cache stores post-RoPE keys**, the point at which they stop depending on anything a later chunk can change. `test_cache_holds_post_rope_keys` separates this from the variant that caches pre-rotation and forgets to re-rotate — which yields plausible video at the wrong positions, not garbage.
- **Audio was context by scope, not by nature — and that turned out to be the wrong scope.** Filed here as a v1 item on the assumption that a video-only encoder was enough for v0; the real-weights numbers said otherwise within the day, and audio now runs ordered on its own 40 Hz spans. `row_time`'s `video_start` argument still builds the video-only clock, kept so the rejected option keeps a measurement rather than a memory.
- **`encode_chunked` runs the preamble once over the whole clip**, which real AR inference cannot do — later frames do not exist yet. It is the correctness harness for the cache, not the inference driver; Phase 5's driver has to build positions incrementally.

Mutation-tested rather than trusted for being green — a mask that drops self-attention, a mask that lets video see everything, a cache that overwrites instead of appending, a cache written but never read, a one-row error in `video_start`, and an unmasked chunk: six mutants, six distinct failures.

#### Both scale blockers closed (2026-08-11)

Two things were left open above and are now done, because Tier 1 cannot start without either.

**The mask is no longer dense.** `scd_attention.block_mask` builds the same `k_time <= q_time` rule as a FlexAttention `BlockMask`, and `encode_chunked(..., block=True)` selects it. `create_block_mask` is compiled, per Tier 0's finding that the eager path materializes a dense int64 `[S, S]` on its way to the sparse one and OOMs at 30 GB for S=62k. Queries and keys are separate time vectors rather than one sequence, because a chunk's queries are the new rows while its keys are the whole cache — the rectangular case is the one the encoder actually builds. `test_block_mask_matches_dense` pins the two paths together through `encode_chunked` at 1e-4; a `<` for `<=` in the `mask_mod` moves it to 1.6e-01.

Not a default. At test sizes the block mask is a compile for no benefit, and making the choice a size threshold means nobody can tell from a call site which attention ran.

**The window policy is implemented** — `encode_chunked(..., window=W)` keeps the last `W` latent frames and evicts the rest, and this closes the *larger* of the two problems. The dense mask was 3.8 GB; **the encoder KV cache at the same size is 53 GB**, because 30 encoder blocks hold 840 KiB per row:

| | 768p/5s (S≈21k) | 768p/10s (S≈41k) | 768p/15s (S≈62k) |
|---|---|---|---|
| dense `[S,S]` bool mask | 0.4 GB | 1.7 GB | 3.8 GB |
| unbounded encoder KV, bf16 | 18 GB | 35 GB | **53 GB** |

So the mask was the visible blocker and the cache was the real one. Measured on the tiny model, a 2-frame window holds the cache at **exactly 53 rows across latent_t 7, 12 and 17** while the sequence grows 191 → 327 → 463. That is §8.1's flat-VRAM claim as an equality, not a trend.

Two things this got wrong first, both worth recording. The window must exempt the CONTEXT rows explicitly: they sit at −inf, which is behind every horizon, so a window that evicts on time alone drops the text and reference rows first and the encoder loses its conditioning one chunk in — with no shape change to notice. The first version of the test asserted a row-count floor, which that mutant clears comfortably; only an exact expected count (53 vs 48) catches it. The test also initially counted audio as context, which is true under the video-only clock and false under the shipping AV clock — the failure was the test's, not the code's.

This **evicts** rather than copying CastleHill's reset-and-re-encode-the-overlap. The two differ in what the retained rows know: an evicted-window row keeps the K/V it was computed with over the full history it actually saw, while a re-encoded overlap row is recomputed against the overlap alone and so knows strictly less. Eviction is also cheaper. What it gives up is the reset's one real property — that a chunk boundary is a clean restart — which matters only if drift accumulates, and is a thing to measure in Phase 5 rather than assume.

Unlike everything else in Phase 2, the window is an **approximation**, not a reassociation: `window=None` is exact against a single masked pass and a windowed run is not. `test_wide_window_is_the_unbounded_cache` pins a window at least as wide as the clip to reproduce `window=None` bit for bit, so the window is a restriction of the exact path rather than a second, subtly different encoder. What a *narrow* window costs on real weights is unmeasured and is the first thing Tier 1 should report.

#### Phase 2 mask cost (2026-08-11) — `docs/phase2_mask_cost.json`

The stack question §7 axis (b) could not answer: what the strict mask costs *cumulatively*, on real weights. `scripts/scd/phase2_mask_cost.py` scores every block's output under each mask against the **unmasked bidirectional pass at the same inputs** — isodiorama, 7 latent frames, `S=2134` = 342 context + 1792 video, NF4 base, 50 blocks, σ=0.571 and σ=0.923 on H3's own density, 229 s for both. It calls `scd_attention.causal_mask`, the mask the encoder actually runs, not a second copy written for the measurement.

Centered cos throughout, using `phase0_validate`'s own definitions. This matters more here than anywhere in Phase 0 and the reason is in the last row of the table.

| at block 29 (encoder output) | σ=0.571 | σ=0.923 |
|---|---|---|
| video centered cos — loose | 0.992 | 0.996 |
| video centered cos — **strict** | **0.952** | **0.972** |
| video relative L2 — loose | 0.039 | 0.043 |
| video relative L2 — **strict** | **0.096** | **0.100** |
| audio centered cos — loose | 0.986 | 0.968 |
| audio centered cos — **strict** | **0.047** | **−0.074** |

**Video: the correction survives, and the split point gets independent support.** Strict costs ~2.5× the relative L2 of loose and ~6× the centered-cos deficit, but 0.95–0.97 at the encoder boundary is a budget, not a wall — and the shape is better than the endpoint suggests. Under the strict mask the video curve is **flat at 0.998–0.999 from block 1 through block 25**, reaches 0.952 at 29, then falls off a cliff: 0.794 (30), 0.471 (35), 0.412 (45). The mask is nearly free over exactly the span the encoder occupies and expensive immediately past it. That knee lands on the same block as §7 axis (a)'s σ-invariance knee, from a completely independent measurement, which is the first corroboration the 29/30 split has had. Block 0 is identical under both masks to four figures (0.9446 / 0.9160), exactly as the depth-1 property predicts.

**Audio: the strict mask deletes it, and the raw number hides that.** Raw cos at block 29 reads 0.980 (σ=0.571) and 0.996 (σ=0.923) — recovery, apparently, from a dip at block 1. Centered, the same rows read **0.047 and −0.074**: uncorrelated with the bidirectional reference, and at the higher σ slightly *anti*correlated. Common mode on those rows is **0.996**, so the raw cosine was measuring the shared vector and essentially nothing else. This is §7's cos-inflation artefact reproducing on a new axis, and it was one edit away from being written into this document as a finding.

The number itself is not a bug — it is the strict mask's definition showing up in the output. Context rows are blind to video, audio rows *are* context (§4 keeps v0's audio bidirectional), so the encoder's audio rows see text and audio and nothing else. Zero video conditioning produces exactly this. What it costs is concrete: **the encoder cache carries no audio–video fusion, so every bit of AV sync has to be created by the decoder's 7 blocks.** §4 assumed the audio pass could read a jointly-encoded cache; it cannot.

**Resolved the same day: audio gets its own clock (the `av` mask).** Audio rows are ordered on their own 40 Hz spans instead of exempt from order, so audio at time *t* sees video ≤ *t* and video at time *t* sees audio ≤ *t*. Causal in both directions means nothing carries the future backward, so the leak table above is not reopened — `test_av_clock_composes_across_blocks` asserts prefix-exactness through the whole encoder, the same bar the strict mask has to clear. The rejected cheap alternative, letting audio queries see all video while video stays causal, is the loose mask under another name and fails for the same reason: audio at the end of the clip has absorbed the whole clip, and any video row that can see it has read the future.

Measured on the same clip and grid, all three clocks in one run:

| at block 29 (encoder output) | loose | strict (v0) | **av (v1)** |
|---|---|---|---|
| video centered cos, σ=0.571 / 0.923 | 0.992 / 0.996 | 0.952 / 0.972 | **0.915 / 0.949** |
| video relative L2 | 0.039 / 0.043 | 0.096 / 0.100 | **0.128 / 0.133** |
| audio centered cos | 0.986 / 0.968 | 0.047 / −0.074 | **0.445 / 0.339** |

So it is a trade, and it should be described as one. Audio goes from **uncorrelated to substantially correlated** — 0.05 → 0.44 — and the curve now *plateaus* around 0.44/0.34 from block 10 onward instead of sitting at noise. Video pays about 0.03 of centered cos and a third more relative L2, because ordering audio also takes *future* audio away from the video rows, which the strict mask let them keep.

The residual gap to `loose` is not a defect to engineer away. Causal audio genuinely cannot see future video; the bidirectional pass is a **ceiling no causal mask can reach**, not a target being missed. The question the number answers is whether the encoder cache carries real audio–video information for D1's second pass to reuse, and 0.44 says yes where 0.05 said no.

The implementation turned out to be smaller than the decision. H3's packer **already** puts audio and video on a shared rotary axis — `image_position_ids` advances audio 1.0 per latent and lays video frames on `_video_t_grid`'s (1,4,4,4,4)×5/3 spans, chosen so 17 pixel frames = 5 latents = 28.33 rotary units ≈ 28 audio latents. So the second clock is not built, it is **read**: `row_time` takes `pos[:, 0]` and sets context rows to −∞, and the whole mask rule collapses to `k <= q`, with "context is always visible" and "context is blind to the ordered rows" becoming the same inequality read from its two ends. A rows-per-frame ratio computed by hand would have been a rounding error with a plausible story; this cannot drift from the packer because it *is* the packer's answer.

What it costs is in `encode_chunked`. Chunks are cut on time, and under the AV clock time order is no longer row order — audio and video interleave in time while sitting in separate slabs of `[context | audio | video]`. So a chunk is a gather, the cache fills in time order, and the result is scattered back to packed order at the end. Under the video-only clock the two coincide, which is exactly why the mutant that reverts to row slicing passes every video-only check and fails only `test_chunked_matches_full` on the AV clock.

Past block 29 the two masks separate hard (video centered cos 0.63 strict vs 0.89 loose at block 49), which is a property of blocks the SCD encoder does not run and is recorded only so the curve is not mistaken for an encoder number later.

### Phase 2.5 — **Speed POC on untrained weights** (3–5 days) ← new, and it gates all training spend

Wall-clock does not care whether the output is good. Benchmark the SCD execution graph **before** spending a dollar on training. Output will be garbage; the timings are real.

**Tier 0 — no weights, no rental. ✅ DONE (2026-08-10).** `scripts/scd/tier0_bench.py`, results in §2.2.1 and `docs/tier0_results.json`. **Passed: 3.90× at 768p/10s, 5.03× at 768p/15s (N=16), measured within 4% of the FLOP model.** Re-run on any new hardware with:

```bash
python3 scripts/scd/tier0_bench.py --steps 8 16 30 --json docs/tier0_results.json
```

Two findings worth carrying forward: (a) `create_block_mask` **must** be compiled — the eager path materializes a dense int64 `[S,S]` and OOMs at 30 GB for S=62k, confirming §6.3; (b) the win grows with N, so any later step-count increase to recover quality is partly self-funding.

**Tier 1 — real weights, still untrained, local or 1 rented hour.** Run the split graph end-to-end: encoder once → per-frame decoder × N steps. Measure **s/frame, peak VRAM, and KV-cache growth** vs stock H3 at 768p, 10s and 15s.

**Kill criterion:** if untrained SCD is not materially faster than stock H3 at 768p/15s, stop — training cannot fix a graph that is not faster, and everything downstream is sunk cost.

**Watch for:** any need to raise N (denoise steps) to recover quality later will eat the win proportionally. Record the step count assumption explicitly in the benchmark.

**Unblocked 2026-08-11.** Both prerequisites are in: the mask is a compiled FlexAttention `BlockMask` and the KV cache takes a sliding window, so run Tier 1 with `block=True, window=4` — the shipping configuration, not the exact-but-impossible one. Two numbers this must report that the Tier 1 brief above does not ask for:

- **What the window costs.** It is the only approximation in the encoder, and it is currently unmeasured at any size. Sweep `window` 2/4/8/12 against `window=None` at a length where the unbounded cache still fits, and quote centered cos per §7's rule.
- **KV-cache growth as a flat line, not a slope.** The tiny model holds it at exactly 53 rows across three clip lengths; that is the claim to reproduce at 768p, since it is what makes duration unbounded and it is the first thing an off-by-one in the horizon would break.

### Phase 3 — token_concat decoder + train loop (3–5 weeks)

- `forward_decoder_per_frame` + Fizgig-like LoRA train on small iso set (e.g. 20–50 clips @ **512**, short T).  
- Log v_std / reconstruction; reject if grid artifacts (CastleHill diagnostic).  
- Target: **24 GB** PRO 4000, Comfy closed, batch 1.

**Kill criterion:** After 1–2k steps, AR or multi-step sample is **not pure noise** and beats “decoder-only random init” baseline.

### Phase 4 — M0 Clip-SCD inference in CLI (1–2 weeks)

- N-step denoise with amortized encoder.  
- Benchmark: wall time and VRAM vs stock H3 same res/length — **at 768p/10–15s, not on a short clip.**

**Success:** quality-matched confirmation of the Phase 2.5 timing (side-by-side human + optional VBench), plus flat peak VRAM vs length.

**Do not gate on short-clip speed.** Per §2.2, SCD is expected to be ~1× or worse below the ~32k-token crossover; CastleHill measured exactly that on LTX-2. A short-clip speed gate would shoot a working port for missing a number it was never going to hit.

### Phase 5 — M1 AR longform (optional, 4+ weeks)

- Chunk loop, overlap, scheduled sampling mature.  
- Streaming VAE decode — **cheaper than CastleHill's**: H3's visual VAE is already temporally causal (`vae_clip_length: 17`, `vae_token_drop: 3`), so no `streaming_vae_decode_and_save` hack is needed.
- KV compression / sliding window is **mandatory, not conditional** — see §8.1. ~~Pull it forward into Phase 2 if the Tier 1 benchmark OOMs at 15s.~~ Pulled forward on the arithmetic instead of on an OOM: 53 GB at 768p/15s does not need measuring to be disqualifying. The sliding window ships in Phase 2; **KV quantization does not**, and remains the next lever if a 4-frame window is not enough context.

### Phase 6 — Comfy nodes (optional)

- `MiniMaxH3SCDSampler` / replace ImageToVideo backend.  
- Do **not** block research on Comfy UX.

---

## 8. VRAM plan (2× PRO 4000, 24 GB each)

**Correction to the first draft:** H3 is **33B**, not LTX-2's 19B. bf16 ≈ 66 GB, int8 ≈ 33 GB. With the AdaLN island pruned (`adaln_t_table`), ~20B → ~20 GB int8, ~10 GB nvfp4. The earlier claim that Phase 3 LoRA training "fits, expect tight" on **one** 24 GB card does not hold — int8 weights alone exceed the card. It is only reachable via Fizgig's `_run_block` CPU-offload streaming, which was tuned for `T=1` stills, not token_concat-doubled video.

**But there are two cards** (CastleHill wandb metadata records `gpu_count: 2`, 25.6 GB each), and **SCD's encoder/decoder boundary is a natural pipeline-parallel cut** — CastleHill already ships split-GPU mode (encoder→GPU0, decoder→GPU1). At int8 with pruned AdaLN, 33 encoder + 17 decoder blocks fit across 48 GB with room for activations at 512².

| Stage | Expected |
|-------|----------|
| Phase 0–2 | Fits easily (forward only, 512²) |
| Phase 2.5 Tier 0 | Random tensors — trivial |
| Phase 2.5 Tier 1 @ 768p/15s | **KV cache dominates — see §8.1.** May need rental |
| Phase 3 train @ 512, T=4–8, LoRA | Worth **one** attempt on 2× 24 GB split-GPU; otherwise 1× H100 80 GB |
| Phase 3 train @ 768, T≥80 | **≥80 GB**, not 48 |
| Phase 4 M0 infer @ 768p short | Fits at int8/nvfp4 |
| Phase 5 AR 30s+ | **This is the SCD prize** — but only with §8.1 solved |

### 8.1 Encoder KV cache — sizing, and why "nonstop" works

**Resolved by CastleHill's chunk policy — but you must copy the policy, not just the split.**

`scd_inference.py:1242-1246` allocates a **fresh KV cache per chunk** (4 latent frames, 1-frame overlap re-encoded as context for the next chunk). The cache therefore never grows with video length, which is what makes duration unbounded:

> *"SCD generation duration is unbounded by GPU; CPU RAM is the ceiling. Encoder KV-cache resets per chunk, so VRAM stays flat regardless of length (verified to 600 chunks / 10 min)."* — `ltx2-castlehill/README.md:175`

With a bounded window, H3's cache is small:

| Window | Encoder KV @ 768p bf16 | Temporal context |
|--------|------------------------|------------------|
| 4 latent frames (CastleHill default) | **3.8 GB** | ~0.67 s |
| 8 latent frames | 7.6 GB | ~1.3 s |
| 12 latent frames | 11.4 GB | ~2.0 s |

**H3 upside:** CastleHill chose 4 frames partly to fit LTX on 32 GB. On 80 GB, H3 can afford a 12-frame window — **3× the attention context CastleHill had**, which should mean less drift over minutes. Window size becomes a quality dial, not just a memory constraint.

**The real ceiling is CPU RAM, not VRAM.** CastleHill's ~8-minute failure was the post-generation pixel buffer (~22 GB at 768×448/24fps/10min) thrashing swap during VAE decode — fixed by streaming each VAE batch into libx264. H3 at 768p with stereo audio hits this sooner; its causal VAE makes the streaming fix easy, but **budget the encoder for it in Phase 5**.

### 8.1b Why unbounded context is not an option

For reference, if you *did* keep full-history context instead of chunk-resetting:

Per token, per encoder layer, the cache holds K and V at `heads × head_dim = 7168` each:

```
2 × 7168 × 2 bytes (bf16) = 28 KB / token / layer
× 33 encoder layers                = 946 KB / token
× 1008 tokens per latent frame     ≈ 0.95 GB per latent frame
```

| Length | Latent frames | Encoder KV (bf16) | fp8 | int4 |
|--------|---------------|-------------------|-----|------|
| 512², T=4 | 4 | 0.97 GB | 0.48 | 0.24 |
| 768p 10s | 61 | **58 GB** | 29 | 14.5 |
| 768p 15s | 91 | **87 GB** | 43 | 22 |

So full-history AR is off the table at 768p, and the chunk-reset policy of §8.1 is **load-bearing, not an optimization**. Anyone "simplifying" the port by keeping one long cache will OOM at ~10s.

If longer effective context is wanted later, in order of preference:
1. **Widen the chunk window** (4 → 12 frames) — cheapest, and H3 has the headroom on 80 GB.
2. **fp8 / int4 KV** (RotorQuant-style) — 2–4×, composes with (1).
3. **Cache a subset of encoder layers**, recompute the rest.
4. Run the AR path at lower resolution and recover detail with H3-Regenerate-2K.

Fix the window policy in Phase 2 so the Phase 2.5 benchmark measures the real design. **Done (2026-08-11)** — `encode_chunked(..., window=W)`, evicting rather than resetting; see the Phase 2 scale-blockers note. The table above is also optimistic about H3 in one respect: it assumes 33 encoder layers at 1008 tokens per latent frame, and the split landed at **30**, which is where the 840 KiB/row and 53 GB @ 768p/15s figures come from.

---

## 9. Data

Reuse existing Scrya packs, **short then long**:

| Pack | Role |
|------|------|
| `iso_room_people_video` square clips | Domain teachers (downsample to 512) |
| Path-B GS action clips | Motion diversity (square crop) |
| CastleHill iso Grok set | Cross-check temporal SCD recipes |
| Holdout stills / first frames | QA boards (`h3_qa_compare_board`) |

Captions: keep triggers for style LoRAs; SCD LoRA is **architecture**, may use same captions.

---

## 10. Risks and CastleHill lessons (do not relearn)

1. **`add` combine → mush** — ban in defaults.  
2. **Train multi-frame decoder / infer 1-frame → grid** — per_frame_decoder=true.  
3. **Muon too hot with token_concat** — lower lr + warmup.  
4. **Teacher forcing only → bad AR** — scheduled sampling.  
5. **X-Cache** may not transfer if H3 AR uses per-frame fresh noise (CastleHill finding).  
6. **H3 LoRA incompatibility** — budget retrain of iso LoRAs post-SCD.  
7. **Audio packing bugs** — keep audio out of the v0 *decoder* path (but on the shared encoder cache, per D1).

New, from the code audit:

8. **Encoder KV cache does not fit** at 768p — 58–87 GB naive. Not optional to solve (§8.1). **Solved 2026-08-11** by the sliding window in `encode_chunked`; at the shipped 30-block split the naive figure is 53 GB at 768p/15s and 840 KiB/row. It was also, by an order of magnitude, the bigger of the two memory blockers — risk 10 got the attention because a mask is easier to picture than a cache.
9. **Short-clip benchmarks understate by ~3.6×** — Phase 3's 512² training scale measures 1.41× against 5.03× at 768p/15s. Reporting the training-scale number will read as a near-failure. Always benchmark at 768p/10–15s (§2.2.1).
10. **Dense attention mask is impossible** at 62k². Confirmed empirically in Tier 0: an uncompiled `create_block_mask` OOMs trying to allocate **30 GB** at S=62k and **67 GB** at S=92k (int64 `[S,S]`). Compile it. FlexAttention block masks or nothing (§6.3). **Closed 2026-08-11** — `scd_attention.block_mask`, compiled, pinned to the dense path at 1e-4.
11. **Comfy's H3 is autograd-hostile** (in-place residual accumulation) — do not try to train through it.
12. **The paper's premise was measured on causal models**, H3 is bidirectional. Phase 0 axis (b) exists to catch this; it is the most likely cause of a late failure. Confirmed against the paper (§2.1): the probing was run on a teacher-forcing AR conversion of WAN-2.1, so **no published measurement covers a raw bidirectional base**. Corollary risk: zero-shot probes of frozen H3 are budget estimates, not gates — do not cancel or greenlight on them.

13. **Do not treat the decoder as the encoder's tail.** The paper's decoder re-uses the *first* 5 and *last* 5 layers (§2). Building a suffix decoder because the diagram in §2 shows one would discard the leave-one-out finding that early layers are among the most load-bearing, and would silently make Phase 1's parity test unfalsifiable.

---

## 11. Success metrics

| Metric | Gate |
|--------|------|
| Phase 0 σ-invariance | Early layers: high cos-sim across σ; late: low |
| Phase 0 causal drift | Early-layer features survive frame-causal masking; if not, price in a real retrain |
| Phase 0 late-layer sparsity | Blocks 33–49 mostly intra-frame attention |
| **Phase 2 mask cost** | ✅ **PASSED on the `av` clock.** Video centered cos 0.915–0.949 at the encoder cut, flat at 0.997+ through block 25 with a cliff at 30 that independently corroborates the split point. Audio 0.445/0.339 — against 0.047/−0.074 when audio was context, which is what forced the clock change and the §5 D1 amendment |
| **Phase 2 cache scaling** | ✅ **PASSED on the tiny model.** A 2-frame window holds the cache at exactly 53 rows while the sequence grows 191 → 327 → 463. Equality, not a trend — but at tiny scale; the 768p version of this number is Tier 1's to produce |
| **Phase 2.5 Tier 0** | ✅ **PASSED** — 3.90× at 768p/10s, 5.03× at 15s (gate was ≥2×) |
| **Phase 2.5 Tier 1** | Untrained SCD graph materially faster than stock H3 at 768p/15s; peak VRAM flat vs length |
| Phase 3 sample | Not noise; human “coherent frame” |
| Phase 4 speed/VRAM | Phase 2.5 timing **holds at matched quality**. Measured at 768p/10–15s only — *not* on short clips (§2.2) |
| Phase 5 length | 60s+ continuous with flat VRAM; CPU-RAM streaming decode in place |
| Product | One demo: locked iso room, character present, ≥30s, H3 look |

**Explicitly not a gate:** short-clip speed. Tier 0 measured 1.41× at 512²/T=4 — not slower, as the first draft assumed, but a 3.6× understatement of the 768p/15s result. A short-clip number is uninformative in *either* direction; it neither kills nor validates the port.

---

## 12. Recommendation

| Priority | Action |
|----------|--------|
| ~~This week, $0~~ | ✅ Phase 2.5 **Tier 0** microbenchmark — **done, passed at 3.90×/5.03×** |
| ~~Next, $0~~ | ✅ Phase 0 three-axis sweep, ✅ Phase 1 skeleton, ✅ Phase 2 mask + cache — all local, all $0 |
| **Next, $0** | Phase 2.5 **Tier 1** — the untrained SCD graph at 768p/15s on local hardware, at `block=True, window=4`. No longer blocked: the block mask and the sliding window both landed 2026-08-11, and without them the run would have needed 3.8 GB of mask and 53 GB of cache |
| **Now** | Keep shipping **H3 LoRAs** (still/short clip) + use **LTX SCD** for long AR if needed |
| **Research bet** | Fund **Phases 1–4** only after Tier 0 and Phase 0 both pass |
| **Do not** | Start Comfy UI / full AR before the hypothesis and speed tests |
| **Do not** | Rent anything before Phase 2.5 Tier 1 |

### Cloud plan

Phases 0–2 all run on hardware already owned. Rent only once there is a mask + cache that passes parity tests.

| Phase | Where | Cost |
|-------|-------|------|
| 0, 1, 2, 2.5-Tier0 | **Local** (2× PRO 4000, split-GPU along the encoder/decoder cut) | **$0** |
| 2.5-Tier1 @ 768p/15s | Local — the windowed cache is what makes it local. a 4-frame window is ~3.4 GB (4 × 1008 rows × 840 KiB) against 53 GB unbounded; the earlier "else 1 rented hour" was pricing the unbounded run | **$0** |
| 3 — LoRA train | 1× H100 80 GB; ~3 hr/run at 512², T=4–8 | ~$2–3.5/hr → **~$15/run** |
| 4–5 — M0/M1 at 768p/15s+ | H100 80 GB, or H200 141 GB for 2K | same |

30–50 training iterations lands under **$1.5k**. The binding cost is engineering time (12–16 weeks by §7's own phasing), not GPU hours. Provision a ~200 GB persistent volume so 66 GB of weights is not re-downloaded per rental.

**Bottom line:** An SCD port for MiniMax is **feasible as a research fork**, and unlike the LTX case there is a **measured, length-gated speed argument** — 3.90× at 768p/10s and 5.03× at 15s on a PRO 4000, within 4% of the FLOP model — on top of the length/VRAM argument. The reason is structural: H3 ships **without** the sparse attention its own README describes, and its sequences sit 1.2–3.4× past the attention/FFN crossover. The remaining risk is now entirely **quality**, not speed: whether a LoRA can recover the base model's output after the split. That is Phase 0 and Phase 3, and it costs $0 to start. It is still **not** a shortcut for better iso still LoRAs.

---

## 13. Open questions (resolve in Phase 0–1)

Resolved by the 2026-08-10 code audit:

- ~~Exact tokens-per-latent-frame~~ → **1008** at 768p 16:9 (1344×768), **256** at 512². `frame_rows = (latent_h//2)·(latent_w//2)`.
- ~~Fizgig vs Comfy weight key names~~ → **Build on Fizgig**; names match the checkpoint in both, Comfy is autograd-hostile (§6.1).
- ~~RoPE consistency under token_concat~~ → pure function of `position_ids`; concat position rows (§6.3).

Still open:

1. Whether **cond/keyframe rows** participate in the causal mask like text (recommend: full attend like text).  
2. Prediction head: velocity vs H3 checkpoint’s native target (must match Fizgig train).  
3. Can nvfp4 inference load an SCD LoRA trained on the int8 train DiT?  
4. **Chunk window size** for H3 — CastleHill's 4 latent frames was a 32 GB constraint; H3 on 80 GB can afford 12 (§8.1). Ablate against drift.
5. Does the **CFG-distilled** checkpoint change the σ-invariance picture (§7 Phase 0a) — and what is H3-Base's actual default step count `N`? The amortization win scales with `N`.

---

## 14. References

- **SCD paper** — Bai, He, Li, Shechtman, Huang, Wu, *"Causality in Video Diffusers is Separable from Denoising"*, [arXiv:2602.10095](https://arxiv.org/abs/2602.10095). Base WAN-2.1 T2V-1.3B (30 layers) → encoder 0–24, decoder {0–4} ∪ {25–29}, 35 layer instances; full fine-tune 55K steps; split by leave-one-out validation loss (Fig. 7); analysis run on a teacher-forcing AR conversion, not the shipped bidirectional model (§2.1). The PDF is ~10 MB+ and defeats a plain fetch — use the `arxiv.org/html/2602.10095v1` rendering.

### Local

| Path | Role |
|------|------|
| `/home/johndpope/Documents/GitHub/ltx2-castlehill/docs/scd-achievements.md` | SCD production lessons |
| `/home/johndpope/Documents/GitHub/ltx2-castlehill/packages/ltx-core/.../scd_model.py` | Split + KV + combine |
| `/home/johndpope/Documents/GitHub/ltx2-castlehill/packages/ltx-trainer/.../scd_strategy.py` | Train loop |
| `/media/2TB/ComfyUI/comfy/ldm/minimax/model.py` | H3 DiT |
| `/media/2TB/Fizgig/src/fizgig/minimax/` | H3 train stack |
| `docs/FIZGIG_MINIMAX_H3_LOCAL.md` | 24 GB still-train ops |
