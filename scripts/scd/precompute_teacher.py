#!/usr/bin/env python3
"""Offline stock-H3 teacher cache for SCD distill races (dLLM-castlehill playbook).

Teacher = full MiniMax H3 DiT (no SCD split). For each (clip, sigma, noise) we store:

  * v_teacher  — model velocity in latent space [1, 24, T, H, W]  (trajectory / path target)
  * h29, h49   — mean residual over VIDEO rows after blocks 29 and 49  (feature anchors)

Student training never co-loads the 66 GB base with the SCD graph: read this cache instead.

Usage (from scripts/scd/):
    python3 precompute_teacher.py --checkpoint /path/to/FL2VA/transformer \\
        --clips-list pixelgraph,pixelnews,isostreet,rigjump \\
        --n-per-clip 48 --out runs/teacher_cache.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scd_data import ClipSet  # noqa: E402


def import_fizgig(src):
    if src not in sys.path:
        sys.path.insert(0, src)
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.trainer import sample_sigmas
    return load_minimax_h3_dit, sample_sigmas


@torch.no_grad()
def teacher_forward(model, noised, t, text, capture_blocks=(29, 49)):
    """One stock forward. Returns (v_latent, {block: mean_video_h}).

    Hooks capture residual stream AFTER the named block, then we mean over video rows only so
    the cache stays small (~few KB per sample instead of tens of MB).
    """
    captured = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            # DiT blocks return the residual stream tensor [S, C].
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h.detach()
        return hook

    for i in capture_blocks:
        handles.append(model.blocks[i].register_forward_hook(make_hook(i)))

    try:
        v = model(noised, t, text)  # unpatchified velocity, same shape as latent
    finally:
        for h in handles:
            h.remove()

    # Video is always the last segment in H3 packing: [text | refs | audio | video].
    lat_t, lat_h, lat_w = noised.shape[2], noised.shape[3], noised.shape[4]
    ph, pw = model.patch_size[1], model.patch_size[2]
    n_video = lat_t * (lat_h // ph) * (lat_w // pw)

    anchors = {}
    for i, h in captured.items():
        vid = h[-n_video:]  # [n_video, C]
        anchors[i] = vid.float().mean(0).cpu()  # [C]
    return v.float().cpu(), anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clips", default=os.path.join(os.path.dirname(__file__), "clips"))
    ap.add_argument("--clips-list", default="pixelgraph,pixelnews,isostreet,rigjump",
                    help="comma names; keep small for a 1h race")
    ap.add_argument("--n-per-clip", type=int, default=48)
    ap.add_argument("--sigma-shift", type=float, default=3.0,
                    help="match v2/v3 train density (shift 3 denser low-noise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/teacher_cache.pt")
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    args = ap.parse_args()

    load_dit, sample_sigmas = import_fizgig(args.fizgig_src)
    device = torch.device("cuda")
    names = [n.strip() for n in args.clips_list.split(",") if n.strip()]
    clips = ClipSet(args.clips)
    for n in names:
        if n not in clips.names:
            raise SystemExit(f"clip {n!r} not in set ({clips.names})")

    print(f"loading stock teacher on {torch.cuda.get_device_name(0)}…", flush=True)
    model = load_dit(args.checkpoint, device=device, compute_dtype=torch.bfloat16,
                     quantize=args.base_quant != "none",
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    model.eval()

    gen = torch.Generator(device=device).manual_seed(args.seed)
    cache = {"meta": {
        "clips": names, "n_per_clip": args.n_per_clip, "sigma_shift": args.sigma_shift,
        "seed": args.seed, "capture_blocks": [29, 49],
    }, "samples": []}

    t0 = time.time()
    for name in names:
        clip = clips.load(name, device=device, dtype=torch.bfloat16)
        x0 = clip["video_latent"].float()
        text = clip["text_embeds"]
        print(f"  {name}  shape {tuple(x0.shape)}", flush=True)
        for k in range(args.n_per_clip):
            sigma = sample_sigmas(1, device, shift=args.sigma_shift, generator=gen)
            s = float(sigma.reshape(-1)[0])
            noise = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=gen)
            noised = ((1.0 - s) * x0 + s * noise).to(torch.bfloat16)
            t = (1.0 - sigma).to(device)
            v, anchors = teacher_forward(model, noised, t, text)
            cache["samples"].append({
                "clip": name,
                "sigma": s,
                "noise": noise.half().cpu(),
                "v": v.half(),
                "h29": anchors.get(29),
                "h49": anchors.get(49),
            })
            if (k + 1) % 8 == 0 or k == 0:
                print(f"    {k + 1}/{args.n_per_clip}  sigma={s:.3f}  "
                      f"elapsed {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(cache, args.out)
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({len(cache['samples'])} samples, {mb:.1f} MB) "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
