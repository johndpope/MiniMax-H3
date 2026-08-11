#!/usr/bin/env python3
"""Phase 0 hypothesis probe for the H3 x SCD port (docs/MINIMAX_H3_SCD_PORT_DESIGN.md §7).

SUPERSEDED BY phase0_validate.py FOR ALL THREE AXES — read that script's output, not this one's.
This one scores everything as plain cos-sim on the residual stream, and on the real H3 that
stream is 91-99.6% a single shared vector (||mean row|| / mean ||row||, vs 0.022 for
uncorrelated rows). Raw cos-sim then reports ~1.0 for two runs whose informative parts are
unrelated, which is how this script produced the backwards result that sigma-invariance RISES
with depth. phase0_validate.py re-measures with relative L2 and mean-subtracted cos-sim, scores
the context rows separately for axis (a), and adds a control proving the mask is not inert.
Kept because the packing, mask and attention plumbing below are still the reference
implementation, and --self-test still guards them.

Forward passes only — no training, no backward. Answers whether H3's weights tolerate being
split into a causal encoder + per-frame diffusion decoder, on three axes:

  (a) sigma-invariance      per-block feature cos-sim across a sigma grid vs the clean run.
                            High in early blocks => the encoder can run once at sigma=0.
  (b) causal-mask drift     per-block cos-sim between frame-causal and full bidirectional
                            attention. This is the axis most likely to kill the port: H3 was
                            trained fully bidirectional and SCD needs frame t blind to t+1.
  (c) intra-frame mass      fraction of attention mass a video row spends inside its own frame.
                            High in late blocks => the per-frame decoder is nearly free.

The split point should be read off (a) and (b) — the layer where invariance collapses or causal
drift rises — not assumed at 33/17.

Runs on the full bf16 checkpoint. That file is 66 GB, so the base is NF4-quantized on load
(~11 GB resident) with the tail blocks CPU-parked. Quantization error is common-mode here: every
comparison is between two runs through the *same* quantized weights, so it cancels in cos-sim.
Use --base-quant none to check that assumption if you have the RAM.

Inputs are cached tensors, not raw media — this script never loads the VAE or the 32 B text
encoder. Produce them with Fizgig's minimax_cache_latents / minimax_cache_text.

Usage:
    python3 scripts/scd/phase0_probe.py \
        --checkpoint /path/minimax_h3_fl2va_bf16.safetensors \
        --latents clip_latents.pt --text clip_te.safetensors \
        --out docs/phase0_results.json --wandb h3-scd
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


def import_fizgig(src_root):
    if src_root and src_root not in sys.path:
        sys.path.insert(0, src_root)
    from fizgig.minimax import model as mm
    from fizgig.minimax.loader import load_minimax_h3_dit
    return mm, load_minimax_h3_dit


# --- packing -----------------------------------------------------------------------------

def frame_causal_mask(seq_len, video_start, frame_rows, latent_t, device):
    """Video row in frame f may attend to text, audio, and video frames <= f. Everything else
    attends fully — text and audio are shared context, per D1 and open question 1.

    Dense [S, S] is fine at probe resolution (~6 MB at S=2.4k). It is NOT fine at 768p: the same
    mask at S=62k is 30 GB, which is why Phase 2 needs FlexAttention (§6.3, and Tier 0 hit it).
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    qf = torch.arange(seq_len, device=device).sub(video_start).div(frame_rows).floor().long()
    is_video = torch.arange(seq_len, device=device) >= video_start
    kf = qf.clone()
    future_key = is_video.unsqueeze(0) & (kf.unsqueeze(0) > qf.unsqueeze(1))
    mask &= ~(is_video.unsqueeze(1) & future_key)
    return mask


# --- block execution with an injectable mask ---------------------------------------------

def attn_with_mask(attn, x, cos, sin, mask):
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
    k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    v = v.view(s, attn.heads, attn.head_dim)
    q = mm_apply_rope(q, cos, sin)
    k = mm_apply_rope(k, cos, sin)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.squeeze(0).transpose(0, 1).reshape(s, attn.heads * attn.head_dim)
    return attn.out_proj(out)


def run_block(mm, block, x, t_emb, mod_row, cos, sin, mask=None):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    h = mm._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_row)
    x = mm._mod_gate(x, gate_msa, attn_with_mask(block.attn, h, cos, sin, mask), mod_row)
    h = mm._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_row)
    return mm._mod_gate(x, gate_mlp, block.mlp(h), mod_row)


@torch.no_grad()
def intra_frame_mass(mm, block, x, t_emb, mod_row, cos, sin, query_rows, video_start, frame_rows):
    """Fraction of attention mass a sampled set of video queries keeps inside its own frame."""
    shift_msa, scale_msa, _, _, _, _ = block.adaln_proj(t_emb)
    h = mm._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_row)
    attn = block.attn
    s = h.shape[0]
    q, k, _ = attn.qkv_proj(h).split(attn.heads * attn.head_dim, dim=-1)
    q = mm_apply_rope(attn.q_norm(q.view(s, attn.heads, attn.head_dim)), cos, sin)
    k = mm_apply_rope(attn.k_norm(k.view(s, attn.heads, attn.head_dim)), cos, sin)
    qs = q[query_rows].transpose(0, 1).float()
    ks = k.transpose(0, 1).float()
    logits = torch.matmul(qs, ks.transpose(-1, -2)) / math.sqrt(attn.head_dim)
    probs = logits.softmax(dim=-1)

    idx = torch.arange(s, device=h.device)
    key_frame = torch.where(idx >= video_start, (idx - video_start) // frame_rows, -1)
    q_frame = (query_rows - video_start) // frame_rows
    same = key_frame.unsqueeze(0) == q_frame.unsqueeze(1)
    return probs.mul(same.unsqueeze(0)).sum(-1).mean().item()


def mm_apply_rope(x, cos, sin):
    rot_half = cos.shape[-1]
    rot = 2 * rot_half
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., :rot_half], xr[..., rot_half:]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat([torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1), xp], dim=-1)


# --- inputs ------------------------------------------------------------------------------

def load_latents(path, latent_t, device):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
        key = next(k for k in sd if k.startswith("latent"))
        z = sd[key]
    else:
        obj = torch.load(path, map_location="cpu")
        z = obj if torch.is_tensor(obj) else obj[next(iter(obj))]
    if z.dim() == 3:
        z = z.unsqueeze(0).unsqueeze(2)
    elif z.dim() == 4:
        z = z.unsqueeze(0)
    if z.shape[2] < latent_t:
        raise SystemExit(f"latents have {z.shape[2]} latent frames, need {latent_t}. "
                         "Axes (b) and (c) are meaningless on a single frame.")
    return z[:, :, :latent_t].to(device, torch.float32)


def load_text(path, device, dtype):
    from safetensors.torch import load_file
    sd = load_file(path)
    return sd["hidden_states"].unsqueeze(0).to(device, dtype)


def synthetic_inputs(latent_t, lat_h, lat_w, text_len, device, dtype):
    z = torch.randn(1, 24, latent_t, lat_h, lat_w, device=device)
    # Smooth spatially and correlate across frames so intra-frame mass is not measured on noise.
    z = F.avg_pool3d(z, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    for f in range(1, latent_t):
        z[:, :, f] = 0.7 * z[:, :, f - 1] + 0.3 * z[:, :, f]
    return z, torch.randn(1, text_len, 5120, device=device, dtype=dtype) * 0.5


# --- sweep -------------------------------------------------------------------------------

@torch.no_grad()
def build_stream(mm, model, video_latent, text_embeds, sigma, device, dtype):
    """Packed [text | audio | video] embeddings plus everything a block needs."""
    _, _, latent_t, lat_h, lat_w = video_latent.shape
    text_len = text_embeds.shape[1]

    text_states = text_embeds[0]
    if text_states.shape[-1] != model.hidden_size:
        text_states = model.token_refiner(model.condition_proj(text_states))

    eps = torch.randn(video_latent.shape, device=device, generator=torch.Generator(device).manual_seed(0))
    noised = (1.0 - sigma) * video_latent + sigma * eps
    video_rows = mm.patchify_video(noised, model.patch_size)
    video_embed = model.video_patch_proj(video_rows.to(model.video_patch_proj.weight.dtype)).to(dtype)

    t_val = torch.tensor([1.0 - sigma], device=device, dtype=torch.float32)
    pixel_frames = (latent_t - 1) * 4 + 1
    n_audio_latents = mm.audio_latents_for_frames(pixel_frames) if model.pack_audio_rows else 0

    audio_embed = None
    if n_audio_latents:
        sigma_a = mm.remap_sigma((1.0 - t_val).clamp(0.0, 1.0))
        t_audio = 1.0 - sigma_a
        a_eps = torch.randn(n_audio_latents * mm.AUDIO_CHANNELS, model.config.audio_latents_dim,
                            device=device, generator=torch.Generator(device).manual_seed(1))
        a_rows = (sigma_a * a_eps).to(model.audio_patch_proj.weight.dtype)
        audio_embed = model.audio_patch_proj(a_rows).to(dtype)

    parts = [text_states.to(dtype)] + ([audio_embed] if audio_embed is not None else []) + [video_embed]
    h = torch.cat(parts, dim=0)
    seq_len = h.shape[0]
    n_audio = 0 if audio_embed is None else audio_embed.shape[0]
    audio_start = text_len
    video_start = audio_start + n_audio

    t_parts = [t_val] + ([t_audio] if audio_embed is not None else [])
    t_all = torch.cat(t_parts) if len(t_parts) > 1 else t_val
    uniq, inverse = torch.unique(t_all, sorted=True, return_inverse=True)
    t_emb = model._time_embedding(uniq).to(dtype)

    tags = torch.full((seq_len,), mm.VIDEO_TAG, dtype=torch.long, device=device)
    tags[:text_len] = mm.TEXT_TAG
    row_t_index = torch.full((seq_len,), int(inverse[0]), dtype=torch.long, device=device)
    if audio_embed is not None:
        tags[audio_start:video_start] = mm.AUDIO_TAG
        row_t_index[audio_start:video_start] = int(inverse[1])
    mod_row = row_t_index * mm.MODALITY_NUM + tags

    pos = mm.image_position_ids(text_len, lat_h, lat_w, n_audio_latents,
                                latent_t=latent_t).to(device)
    cos, sin = mm.rope_cos_sin(pos, model.rope.inv_freq.to(device))
    return {"h": h, "t_emb": t_emb, "mod_row": mod_row, "cos": cos.to(dtype), "sin": sin.to(dtype),
            "seq_len": seq_len, "video_start": video_start, "audio_start": audio_start,
            "frame_rows": (lat_h // 2) * (lat_w // 2), "latent_t": latent_t,
            # the packer's own t-axis, on which audio and video are already commensurate —
            # scd_attention.row_time reads the AV clock straight off it
            "pos": pos,
            # eps and the video timestep row are needed to score a real flow-matching loss
            # through final_layer (phase0_leaveout.py); the cos-sim probes ignore them.
            "eps": eps, "video_t_index": int(inverse[0])}


@torch.no_grad()
def sweep_stack(model, mm, stream, mask, device, keep_full, collect_mass=None):
    """Run all blocks; return per-block outputs (CPU bf16) and optional intra-frame mass.

    keep_full=False stores from audio_start onward, so audio and video rows can be scored
    separately — AdaLN modulation is per-modality and the two need not degrade together.
    """
    h = stream["h"]
    outs, mass = [], []
    for i, block in enumerate(model.blocks):
        swapped = i >= model._swap_from
        if swapped:
            block.to(device)
        if collect_mass is not None:
            mass.append(intra_frame_mass(mm, block, h, stream["t_emb"], stream["mod_row"],
                                         stream["cos"], stream["sin"], collect_mass,
                                         stream["video_start"], stream["frame_rows"]))
        h = run_block(mm, block, h, stream["t_emb"], stream["mod_row"],
                      stream["cos"], stream["sin"], mask)
        if swapped:
            block.to("cpu")
        outs.append(h.to("cpu", torch.bfloat16) if keep_full
                    else h[stream["audio_start"]:].to("cpu", torch.bfloat16))
    return outs, mass


def cos_rows(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


@torch.no_grad()
def self_test(mm, device):
    """Validate the mask and packing on a tiny random H3 — no checkpoint needed.

    A silently non-causal mask would make axis (b) meaningless while still producing a plausible
    curve, so this asserts the property directly: under the mask, frame 0's rows must be bit-wise
    what they would be if later frames did not exist.
    """
    from fizgig.minimax.model import MiniMaxH3DiT, MiniMaxH3Config
    dtype = torch.float32
    torch.manual_seed(0)      # the reported drift is quoted in the design doc; keep it comparable
    cfg = MiniMaxH3Config(hidden_size=256, num_layers=2, num_attention_heads=2,
                          attention_head_dim=128, ffn_hidden_size=512,
                          time_embed_hidden_size=128, time_embed_dim=64, text_dim=64)
    model = MiniMaxH3DiT(cfg).to(device, dtype).eval()
    model._swap_from = len(model.blocks)

    latent_t, lat_h, lat_w = 4, 8, 8
    z = torch.randn(1, 24, latent_t, lat_h, lat_w, device=device)
    txt = torch.randn(1, 16, cfg.text_dim, device=device, dtype=dtype)
    st = build_stream(mm, model, z, txt, 0.0, device, dtype)
    vs, fr, s = st["video_start"], st["frame_rows"], st["seq_len"]
    assert fr == (lat_h // 2) * (lat_w // 2)
    assert s - vs == latent_t * fr

    mask = frame_causal_mask(s, vs, fr, latent_t, device)
    assert mask[vs + 2 * fr, vs:vs + fr].all(), "frame 2 must see frame 0"
    assert not mask[vs, vs + fr:].any(), "frame 0 must not see later frames"
    assert mask[vs, :vs].all(), "video must see text and audio"
    assert mask[0, :].all(), "text queries attend fully"

    blk = model.blocks[0]
    n = vs + fr
    full = run_block(mm, blk, st["h"], st["t_emb"], st["mod_row"], st["cos"], st["sin"], mask)
    pre = run_block(mm, blk, st["h"][:n], st["t_emb"], st["mod_row"][:n],
                    st["cos"][:n], st["sin"][:n], None)
    un = run_block(mm, blk, st["h"], st["t_emb"], st["mod_row"], st["cos"], st["sin"], None)
    same = (full[vs:n] - pre[vs:]).abs().max().item()
    diff = (full[vs:n] - un[vs:n]).abs().max().item()
    assert same < 1e-4, f"mask is not causal: frame 0 differs from prefix-only by {same:.3e}"
    assert diff > 1e-3, f"mask has no effect (delta {diff:.3e}) — it is not being applied"
    print(f"self-test passed  S={s} video_start={vs} frame_rows={fr}  "
          f"prefix-equivalence {same:.1e}, mask effect {diff:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="validate mask/packing on a tiny random model and exit")
    ap.add_argument("--checkpoint", help="minimax_h3_fl2va_bf16.safetensors")
    ap.add_argument("--latents", help="cached [1,24,T,h,w] normalized video latents")
    ap.add_argument("--text", help="Fizgig text cache (.safetensors with hidden_states)")
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke-test the harness with structured noise — NOT a valid measurement")
    ap.add_argument("--latent-t", type=int, default=8)
    ap.add_argument("--latent-hw", type=int, nargs=2, default=[32, 32], help="synthetic only")
    ap.add_argument("--text-len", type=int, default=128, help="synthetic only")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--attn-sample", type=int, default=192)
    ap.add_argument("--blocks-to-swap", type=int, default=42)
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--out", default="docs/phase0_results.json")
    ap.add_argument("--wandb", default=None, help="wandb project name")
    args = ap.parse_args()

    mm, load_dit = import_fizgig(args.fizgig_src)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if args.self_test:
        self_test(mm, device)
        return
    if not args.checkpoint:
        raise SystemExit("need --checkpoint (or --self-test)")
    if not args.synthetic and not (args.latents and args.text):
        raise SystemExit("need --latents and --text (or --synthetic to smoke-test the harness)")

    model = load_dit(args.checkpoint, device=device, compute_dtype=dtype,
                     quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    model.enable_block_swap(args.blocks_to_swap)

    if args.synthetic:
        video_latent, text_embeds = synthetic_inputs(args.latent_t, *args.latent_hw,
                                                     args.text_len, device, dtype)
        print("!! --synthetic: structured noise, not real latents. Axes (b) and (c) are "
              "harness checks only.", flush=True)
    else:
        video_latent = load_latents(args.latents, args.latent_t, device)
        text_embeds = load_text(args.text, device, dtype)

    n_blocks = len(model.blocks)
    ref = build_stream(mm, model, video_latent, text_embeds, 0.0, device, dtype)
    seq_len, video_start = ref["seq_len"], ref["video_start"]
    frame_rows, latent_t = ref["frame_rows"], ref["latent_t"]
    print(f"S={seq_len}  video_start={video_start}  frame_rows={frame_rows}  "
          f"latent_t={latent_t}  blocks={n_blocks}", flush=True)

    g = torch.Generator(device).manual_seed(7)
    n_video = seq_len - video_start
    query_rows = video_start + torch.randperm(n_video, generator=g, device=device)[:args.attn_sample]
    query_rows = query_rows.sort().values

    n_audio = video_start - ref["audio_start"]

    def score(outs, refs):
        v = [cos_rows(a[n_audio:], b[n_audio:]) for a, b in zip(outs, refs)]
        a_ = ([cos_rows(a[:n_audio], b[:n_audio]) for a, b in zip(outs, refs)]
              if n_audio else [float("nan")] * len(outs))
        return v, a_

    t0 = time.time()
    print("[1/4] clean pass (sigma=0) + intra-frame attention mass", flush=True)
    clean_full, mass = sweep_stack(model, mm, ref, None, device, keep_full=True,
                                   collect_mass=query_rows)
    clean_media = [o[ref["audio_start"]:].to("cpu", torch.bfloat16) for o in clean_full]

    print("[2/4] sigma sweep", flush=True)
    sigma_cos, sigma_cos_audio = {}, {}
    for sigma in args.sigmas:
        if sigma == 0.0:
            sigma_cos[sigma] = [1.0] * n_blocks
            sigma_cos_audio[sigma] = [1.0] * n_blocks
            continue
        stream = build_stream(mm, model, video_latent, text_embeds, sigma, device, dtype)
        outs, _ = sweep_stack(model, mm, stream, None, device, keep_full=False)
        sigma_cos[sigma], sigma_cos_audio[sigma] = score(outs, clean_media)
        del outs
        torch.cuda.empty_cache()

    print("[3/4] causal mask, cumulative", flush=True)
    mask = frame_causal_mask(seq_len, video_start, frame_rows, latent_t, device)
    causal_out, _ = sweep_stack(model, mm, ref, mask, device, keep_full=False)
    causal_cum, causal_cum_audio = score(causal_out, clean_media)
    del causal_out
    torch.cuda.empty_cache()

    print("[4/4] causal mask, isolated per block", flush=True)
    causal_iso, causal_iso_audio = [], []
    for i, block in enumerate(model.blocks):
        x = (ref["h"] if i == 0 else clean_full[i - 1].to(device, dtype))
        swapped = i >= model._swap_from
        if swapped:
            block.to(device)
        masked = run_block(mm, block, x, ref["t_emb"], ref["mod_row"], ref["cos"], ref["sin"], mask)
        if swapped:
            block.to("cpu")
        v, a_ = score([masked[ref["audio_start"]:].cpu()], [clean_media[i]])
        causal_iso.append(v[0])
        causal_iso_audio.append(a_[0])
        del masked, x
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    payload = {
        "checkpoint": os.path.basename(args.checkpoint), "base_quant": args.base_quant,
        "synthetic": args.synthetic, "seq_len": seq_len, "video_start": video_start,
        "frame_rows": frame_rows, "latent_t": latent_t, "n_blocks": n_blocks,
        "sigmas": args.sigmas, "elapsed_s": elapsed, "n_audio_rows": n_audio,
        "sigma_cos": {str(k): v for k, v in sigma_cos.items()},
        "sigma_cos_audio": {str(k): v for k, v in sigma_cos_audio.items()},
        "causal_cos_cumulative": causal_cum, "causal_cos_isolated": causal_iso,
        "causal_cos_cumulative_audio": causal_cum_audio,
        "causal_cos_isolated_audio": causal_iso_audio,
        "intra_frame_mass": mass,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    hi = args.sigmas[-1]
    print(f"\n{'blk':>4} {'cos@sig=' + str(hi):>13} {'causal iso':>11} {'causal cum':>11} "
          f"{'intra-frame':>12} {'aud sig':>9} {'aud iso':>9}")
    print("-" * 76)
    for i in range(n_blocks):
        print(f"{i:>4} {sigma_cos[hi][i]:>13.4f} {causal_iso[i]:>11.4f} "
              f"{causal_cum[i]:>11.4f} {mass[i]:>12.4f} "
              f"{sigma_cos_audio[hi][i]:>9.4f} {causal_iso_audio[i]:>9.4f}")

    drift = [1.0 - c for c in causal_iso]
    knee = max(range(1, n_blocks), key=lambda i: drift[i] - drift[i - 1])
    tail = 2 * n_blocks // 3
    payload["suggested_split"] = knee
    payload["tail_intra_frame_mass"] = sum(mass[tail:]) / max(1, len(mass[tail:]))
    # The CONTIGUOUS PREFIX that is sigma-invariant, which is the only thing SCD can use: the
    # encoder runs once at sigma=0 and is reused at every step, so an invariant block sitting
    # behind a non-invariant one is worthless. Reporting the last block anywhere above the
    # threshold instead would read "through block 49" off a curve that fails from block 0.
    prefix = next((i for i in range(n_blocks) if sigma_cos[hi][i] <= 0.9), n_blocks) - 1
    payload["sigma_invariant_prefix"] = prefix
    print(f"\nsigma-invariance (cos>0.9 at sigma={hi}) holds for blocks 0..{prefix}"
          f"{'  — NO invariant prefix' if prefix < 0 else ''}; "
          f"cos at blocks 0/mid/last: {sigma_cos[hi][0]:.3f} / "
          f"{sigma_cos[hi][n_blocks // 2]:.3f} / {sigma_cos[hi][-1]:.3f}")
    print(f"largest jump in causal drift at block {knee} -> candidate encoder/decoder split")
    print(f"mean intra-frame mass, blocks {tail}-{n_blocks - 1}: "
          f"{payload['tail_intra_frame_mass']:.4f}")
    print(f"elapsed {elapsed / 60:.1f} min")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb, job_type="phase0-probe", config=vars(args))
        table = wandb.Table(columns=["block", "sigma", "cos_video", "cos_audio"])
        for sigma, vals in sigma_cos.items():
            for i, c in enumerate(vals):
                table.add_data(i, sigma, c, sigma_cos_audio[sigma][i])
        run.log({"sigma_invariance": table})
        for i in range(n_blocks):
            run.log({"block": i, "causal_cos_isolated": causal_iso[i],
                     "causal_cos_cumulative": causal_cum[i], "intra_frame_mass": mass[i],
                     "causal_cos_isolated_audio": causal_iso_audio[i]})
        run.summary["suggested_split"] = knee
        run.finish()


if __name__ == "__main__":
    main()
