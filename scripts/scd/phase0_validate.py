#!/usr/bin/env python3
"""Controls for the Phase 0 probe (docs/MINIMAX_H3_SCD_PORT_DESIGN.md §7).

phase0_probe.py returned three results that all contradict the SCD premise, so before any of
them is written into the design doc they need to survive a control. Each check below exists to
kill one specific way the probe could be lying:

  1. mask reach       The probe's frame-causal mask barely moved the output (cos ~0.9999). The
                      self-test proves the mask is causal on a 2-block random model; it does not
                      prove it reaches the real forward path. So re-run with a deliberately
                      brutal mask (video sees text+audio+its OWN frame only). If drift is still
                      nil, the mask is inert and axis (b) is meaningless. If drift is large, the
                      plumbing works and the small frame-causal drift is a real property.

  2. where mass goes  If drift is real, it should be explainable: decompose video-query attention
                      into text / audio / past / own / future. Masking the future can only matter
                      in proportion to the mass actually sitting there.

  3. cos-sim inflation
                      Everything is measured as cos-sim on the RESIDUAL STREAM, which accumulates.
                      If late blocks add a large shared component, cos-sim rises toward 1 for two
                      runs that still differ a lot in the part that carries information. That
                      alone could manufacture the sigma-invariance curve (low early, high late).
                      So report, per block, relative L2 ||a-b||/||a|| and cos after subtracting
                      the clean run's mean row.

  4. on-distribution sigma
                      The first run used sigma=0 as the reference and sigma=0.9 as the test. H3
                      draws training sigma as shift_sigma(u, 12) with u uniform, which puts the
                      median at ~0.92 and ~3%% of steps below 0.3 -- so sigma=0 is a point the
                      model has essentially never seen, and at low sigma its output is
                      ANTICORRELATED with the flow-matching target (cos -0.44 at 0.1, see
                      phase0_leaveout.py). Both endpoints are now drawn on H3's own density: the
                      reference is u=0.1 (sigma 0.571) and the test is u=0.9 (sigma 0.991), so
                      the pair still spans the schedule but from inside the training distribution.
                      Both passes share one noise draw (build_stream seeds eps), so the delta is
                      sigma sensitivity, not a different sample.

Usage:
    python3 scripts/scd/phase0_validate.py \
        --checkpoint /path/to/FL2VA/transformer \
        --latents scripts/scd/clip_latents.safetensors --text scripts/scd/clip_te.safetensors \
        --out docs/phase0_validation.json
"""

import argparse
import json
import math
import os
import time

import torch

from phase0_probe import (build_stream, cos_rows, frame_causal_mask, import_fizgig, load_latents,
                          load_text, mm_apply_rope, run_block)


def own_frame_mask(seq_len, video_start, frame_rows, device):
    """Control mask: a video row sees text, audio, and only its own frame.

    This is the most aggressive restriction the SCD decoder could ever impose. If the model
    shrugs this off too, the mask is not being applied.
    """
    idx = torch.arange(seq_len, device=device)
    is_video = idx >= video_start
    f = torch.where(is_video, (idx - video_start) // frame_rows, -1)
    cross_frame = is_video.unsqueeze(0) & (f.unsqueeze(0) != f.unsqueeze(1))
    return ~(is_video.unsqueeze(1) & cross_frame)


@torch.no_grad()
def attention_breakdown(mm, block, x, t_emb, mod_row, cos, sin, query_rows,
                        text_len, video_start, frame_rows):
    """Where a video query actually spends its attention mass, as five fractions summing to 1."""
    shift, scale, _, _, _, _ = block.adaln_proj(t_emb)
    h = mm._mod_scale_shift(block.norm1(x), shift, scale, mod_row)
    attn = block.attn
    s = h.shape[0]
    q, k, _ = attn.qkv_proj(h).split(attn.heads * attn.head_dim, dim=-1)
    q = mm_apply_rope(attn.q_norm(q.view(s, attn.heads, attn.head_dim)), cos, sin)
    k = mm_apply_rope(attn.k_norm(k.view(s, attn.heads, attn.head_dim)), cos, sin)
    qs = q[query_rows].transpose(0, 1).float()
    ks = k.transpose(0, 1).float()
    probs = torch.matmul(qs, ks.transpose(-1, -2)).div(math.sqrt(attn.head_dim)).softmax(-1)

    idx = torch.arange(s, device=h.device)
    kf = torch.where(idx >= video_start, (idx - video_start) // frame_rows, -1)
    qf = ((query_rows - video_start) // frame_rows).unsqueeze(1)
    is_v = (idx >= video_start).unsqueeze(0)

    def frac(sel):
        return probs.mul(sel.unsqueeze(0)).sum(-1).mean().item()

    return {
        "text": frac((idx < text_len).unsqueeze(0).expand(len(query_rows), -1)),
        "audio": frac(((idx >= text_len) & (idx < video_start)).unsqueeze(0)
                      .expand(len(query_rows), -1)),
        "past": frac(is_v & (kf.unsqueeze(0) < qf)),
        "own": frac(is_v & (kf.unsqueeze(0) == qf)),
        "future": frac(is_v & (kf.unsqueeze(0) > qf)),
    }


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm(dim=-1).div(a.norm(dim=-1).clamp_min(1e-9)).mean().item()


def centered_cos(a, b, mu):
    return cos_rows(a.float() - mu, b.float() - mu)


def common_mode(a):
    """||mean row|| / mean ||row||. Near 1 means the rows are dominated by a shared vector, which
    is exactly the condition under which cos-sim stops distinguishing anything."""
    a = a.float()
    return (a.mean(0).norm() / a.norm(dim=-1).mean().clamp_min(1e-9)).item()


def score_video(got, want):
    """All three metrics on video rows only, centered on the clean video-row mean.

    Centering must use the same row set being scored: the audio and video rows sit in different
    parts of the space (separate AdaLN modulation), so a mean over both leaves a bias in each.
    """
    mu = want.float().mean(0)
    return cos_rows(got, want), rel_l2(want, got), centered_cos(got, want, mu)


@torch.no_grad()
def isolated_pass(model, mm, ref, clean_full, clean_video, mask, device, dtype):
    """Per-block: feed the clean input, apply `mask` in that block only, score against clean."""
    vs = ref["video_start"]
    cos_v, rl2, ccos = [], [], []
    for i, block in enumerate(model.blocks):
        x = ref["h"] if i == 0 else clean_full[i - 1].to(device, dtype)
        swapped = i >= model._swap_from
        if swapped:
            block.to(device)
        out = run_block(mm, block, x, ref["t_emb"], ref["mod_row"], ref["cos"], ref["sin"], mask)
        if swapped:
            block.to("cpu")
        c, r, cc = score_video(out[vs:].cpu(), clean_video[i])
        cos_v.append(c)
        rl2.append(r)
        ccos.append(cc)
        del out, x
        torch.cuda.empty_cache()
    return cos_v, rl2, ccos


@torch.no_grad()
def measure(mm, model, video_latent, text_embeds, sigma_ref, sigma_test, device, dtype,
            attn_sample=192, quiet=False):
    """Run all four passes and return the result payload. Model stays loaded — see run_phase0.py."""
    def say(*a):
        if not quiet:
            print(*a, flush=True)

    ref = build_stream(mm, model, video_latent, text_embeds, sigma_ref, device, dtype)
    S, vs, fr = ref["seq_len"], ref["video_start"], ref["frame_rows"]
    text_len = ref["audio_start"]
    n_blocks = len(model.blocks)
    say(f"S={S} video_start={vs} frame_rows={fr} latent_t={ref['latent_t']} blocks={n_blocks}")

    g = torch.Generator(device).manual_seed(7)
    query_rows = (vs + torch.randperm(S - vs, generator=g, device=device)[:attn_sample])
    query_rows = query_rows.sort().values

    t0 = time.time()

    # --- reference pass, keeping every block output, plus the attention decomposition ---------
    say(f"[1/4] reference pass (sigma={sigma_ref:.4f}) + attention decomposition")
    clean_full, breakdown, cm = [], [], []
    h = ref["h"]
    for i, block in enumerate(model.blocks):
        swapped = i >= model._swap_from
        if swapped:
            block.to(device)
        breakdown.append(attention_breakdown(mm, block, h, ref["t_emb"], ref["mod_row"],
                                             ref["cos"], ref["sin"], query_rows,
                                             text_len, vs, fr))
        h = run_block(mm, block, h, ref["t_emb"], ref["mod_row"], ref["cos"], ref["sin"], None)
        if swapped:
            block.to("cpu")
        clean_full.append(h.to("cpu", torch.bfloat16))
        cm.append(common_mode(clean_full[-1][vs:]))
    clean_video = [o[vs:] for o in clean_full]

    # --- control 1: does the mask reach the real forward path? --------------------------------
    say("[2/4] own-frame-only mask (control)")
    ofm = own_frame_mask(S, vs, fr, device)
    own_cos, own_rl2, own_ccos = isolated_pass(model, mm, ref, clean_full, clean_video,
                                               ofm, device, dtype)

    say("[3/4] frame-causal mask, L2 metrics")
    fcm = frame_causal_mask(S, vs, fr, ref["latent_t"], device)
    cau_cos, cau_rl2, cau_ccos = isolated_pass(model, mm, ref, clean_full, clean_video,
                                               fcm, device, dtype)

    # --- control 3: is the sigma curve an artefact of residual-stream accumulation? ------------
    # The video rows literally carry the noise, so their sigma-drift is expected and says nothing
    # about SCD. What an SCD encoder would actually cache is the CONTEXT — the text and audio
    # rows. Those are only indirectly noised, via attention to the video rows, so scoring them
    # separately is the fair test of the premise.
    say(f"[4/4] test pass (sigma={sigma_test:.4f}) with L2 metrics")
    st = build_stream(mm, model, video_latent, text_embeds, sigma_test, device, dtype)
    clean_text = [o[:text_len] for o in clean_full]
    clean_audio = [o[text_len:vs] for o in clean_full]
    sig_cos, sig_rl2, sig_ccos = [], [], []
    txt_ccos, aud_ccos = [], []
    h = st["h"]
    for i, block in enumerate(model.blocks):
        swapped = i >= model._swap_from
        if swapped:
            block.to(device)
        h = run_block(mm, block, h, st["t_emb"], st["mod_row"], st["cos"], st["sin"], None)
        if swapped:
            block.to("cpu")
        hc = h.to("cpu", torch.bfloat16)
        c, r, cc = score_video(hc[vs:], clean_video[i])
        sig_cos.append(c)
        sig_rl2.append(r)
        sig_ccos.append(cc)
        txt_ccos.append(score_video(hc[:text_len], clean_text[i])[2])
        aud_ccos.append(score_video(hc[text_len:vs], clean_audio[i])[2])

    return {
        "sigma_ref": sigma_ref, "sigma_test": sigma_test, "latent_t": ref["latent_t"],
        "seq_len": S, "video_start": vs, "frame_rows": fr, "n_blocks": n_blocks,
        "own_frame_cos": own_cos, "own_frame_rel_l2": own_rl2, "own_frame_centered_cos": own_ccos,
        "causal_cos": cau_cos, "causal_rel_l2": cau_rl2, "causal_centered_cos": cau_ccos,
        "sigma_cos": sig_cos, "sigma_rel_l2": sig_rl2, "sigma_centered_cos": sig_ccos,
        "sigma_centered_cos_text": txt_ccos, "sigma_centered_cos_audio": aud_ccos,
        "common_mode_ratio": cm, "attention_breakdown": breakdown,
        "elapsed_s": time.time() - t0, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def knee(curve, frac=0.85):
    """First block that falls below `frac` of the plateau — i.e. where the plateau ends.

    Returned as the index of the first block OFF the plateau, so the encoder that keeps it is
    blocks 0..knee-1.

    Deliberately not argmax-of-single-block-drop: on the text curve the 31->32 drop (0.134) edges
    out the 29->30 drop (0.133) by a thousandth and moves the answer two blocks, which is a
    coin-flip dressed up as a measurement. "First real departure from the plateau" is the thing
    the design doc actually claims, and it is stable under that kind of tie.
    """
    top = max(range(len(curve)), key=lambda i: curve[i])
    for i in range(top + 1, len(curve)):
        if curve[i] < frac * curve[top]:
            return i
    return len(curve)


def report(p):
    """Print the per-block table and the control verdicts for one `measure()` payload."""
    n_blocks, S, vs, cm = p["n_blocks"], p["seq_len"], p["video_start"], p["common_mode_ratio"]
    own_cos, own_rl2, own_ccos = p["own_frame_cos"], p["own_frame_rel_l2"], p["own_frame_centered_cos"]
    cau_cos, cau_rl2, cau_ccos = p["causal_cos"], p["causal_rel_l2"], p["causal_centered_cos"]
    sig_cos, sig_rl2, sig_ccos = p["sigma_cos"], p["sigma_rel_l2"], p["sigma_centered_cos"]
    txt_ccos, aud_ccos, breakdown = (p["sigma_centered_cos_text"], p["sigma_centered_cos_audio"],
                                     p["attention_breakdown"])

    s_ref, s_test = p["sigma_ref"], p["sigma_test"]
    print(f"\n{'blk':>4} | {'own-frame mask':^24} | {'frame-causal mask':^24} | "
          f"{f'sigma {s_ref:.3f}->{s_test:.3f}':^24} | {'cm':>5}")
    print(f"{'':>4} | {'cos':>7} {'relL2':>7} {'ccos':>7} | {'cos':>7} {'relL2':>7} {'ccos':>7} | "
          f"{'cos':>7} {'relL2':>7} {'ccos':>7} | {'txt':>6} {'aud':>6} | {'cm':>5}")
    print("-" * 114)
    for i in range(n_blocks):
        print(f"{i:>4} | {own_cos[i]:>7.4f} {own_rl2[i]:>7.4f} {own_ccos[i]:>7.4f} | "
              f"{cau_cos[i]:>7.4f} {cau_rl2[i]:>7.4f} {cau_ccos[i]:>7.4f} | "
              f"{sig_cos[i]:>7.4f} {sig_rl2[i]:>7.4f} {sig_ccos[i]:>7.4f} | "
              f"{txt_ccos[i]:>6.3f} {aud_ccos[i]:>6.3f} | {cm[i]:>5.3f}")

    print("\nattention mass for video queries (mean over blocks and sampled rows):")
    keys = ["text", "audio", "past", "own", "future"]
    print("  " + "  ".join(f"{k:>8}" for k in keys))
    print("  " + "  ".join(f"{sum(b[k] for b in breakdown) / n_blocks:>8.4f}" for k in keys))
    print("  first block: " + "  ".join(f"{breakdown[0][k]:>8.4f}" for k in keys))
    print("  last block:  " + "  ".join(f"{breakdown[-1][k]:>8.4f}" for k in keys))

    verdict_reach = max(1.0 - c for c in own_cos)
    print(f"\nCONTROL 1 (mask reach): max own-frame drift = {verdict_reach:.2e}  "
          f"-> {'mask IS reaching the forward path' if verdict_reach > 1e-2 else 'MASK LOOKS INERT'}")
    # For rows with no shared component the ratio is ~1/sqrt(n_rows); anything near 1 means the
    # residual stream is essentially one vector plus small per-row deltas, and raw cos-sim on it
    # is measuring the shared vector.
    print(f"CONTROL 3 (cos inflation): common-mode ratio blocks 0/mid/last = "
          f"{cm[0]:.3f} / {cm[n_blocks // 2]:.3f} / {cm[-1]:.3f}  "
          f"(uncorrelated-rows baseline {1 / math.sqrt(S - vs):.3f})")
    kt, kv = knee(txt_ccos), knee(sig_ccos)
    # Reported separately on purpose. The text plateau is flat and its departure is unambiguous;
    # the video plateau erodes for a few blocks before its cliff, so the two can differ by one and
    # collapsing them to a single "encoder = 0..N" hides which row set the number came from.
    print(f"KNEE (plateau ends): text rows block {kt}, video rows block {kv}"
          + ("" if abs(kt - kv) <= 1 else
             "  [ROWS DISAGREE BY >1 BLOCK — no split should be read off this]"))
    print(f"elapsed {p['elapsed_s'] / 60:.1f} min")


def add_args(ap):
    """Shared with run_phase0.py so the two entry points cannot drift apart."""
    ap.add_argument("--latent-t", type=int, default=7,
                    help="must sit on the 5n+2 grid (2, 7, 12, ...) — see pixel_frames_for_latent")
    ap.add_argument("--u-ref", type=float, default=0.1,
                    help="reference noise level, as a quantile of H3's OWN training density: "
                         "sigma = shift_sigma(u, 12). u=0 gives the old sigma=0 reference, which "
                         "is off-distribution — see the module docstring.")
    ap.add_argument("--u", type=float, default=0.9, help="test noise level, same density as --u-ref")
    ap.add_argument("--raw-sigmas", type=float, nargs=2, default=None,
                    metavar=("REF", "TEST"), help="bypass the shift map and give sigmas directly")
    ap.add_argument("--attn-sample", type=int, default=192)
    ap.add_argument("--blocks-to-swap", type=int, default=0)
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    return ap


def resolve_sigmas(mm, args):
    if args.raw_sigmas:
        return tuple(args.raw_sigmas)
    return (mm.shift_sigma(args.u_ref, mm.VIDEO_SIGMA_SHIFT),
            mm.shift_sigma(args.u, mm.VIDEO_SIGMA_SHIFT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="docs/phase0_validation.json")
    add_args(ap)
    args = ap.parse_args()

    mm, load_dit = import_fizgig(args.fizgig_src)
    device, dtype = torch.device("cuda"), torch.bfloat16

    model = load_dit(args.checkpoint, device=device, compute_dtype=dtype,
                     quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    model.enable_block_swap(args.blocks_to_swap)

    mm.pixel_frames_for_latent(args.latent_t)   # raises if off the 5n+2 grid the DiT requires
    video_latent = load_latents(args.latents, args.latent_t, device)
    text_embeds = load_text(args.text, device, dtype)

    sigma_ref, sigma_test = resolve_sigmas(mm, args)
    print(f"reference sigma {sigma_ref:.4f}, test sigma {sigma_test:.4f}"
          f"{'' if args.raw_sigmas else f' (u={args.u_ref}/{args.u} through shift {mm.VIDEO_SIGMA_SHIFT})'}",
          flush=True)

    payload = measure(mm, model, video_latent, text_embeds, sigma_ref, sigma_test, device, dtype,
                      attn_sample=args.attn_sample)
    payload.update(checkpoint=os.path.basename(args.checkpoint.rstrip("/")),
                   clip=os.path.basename(args.latents),
                   u_ref=None if args.raw_sigmas else args.u_ref,
                   u_test=None if args.raw_sigmas else args.u)
    report(payload)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
