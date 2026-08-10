#!/usr/bin/env python3
"""What the SCD encoder's mask costs on real weights: strict frame-causal vs Phase 0's loose one.

Phase 0 axis (b) measured a mask that restricts only VIDEO queries and reported that frame-causal
masking is nearly free. For a single block that is exactly right, and for the encoder it is the
wrong mask: context rows that can see every frame carry the future backward, so after two blocks
the prefix is no longer a function of the prefix and an encoder KV cache is silently wrong
(`test_scd_attention.py::test_loose_mask_leaks_across_blocks`).

The shipping mask therefore also blinds the context rows to video. That is a STRICTLY harsher
constraint, and the video rows' isolated per-block drift cannot see the difference — for a video
query the two masks are identical row for row. The cost shows up cumulatively, through the
context rows, which is what this measures: every block's output under each mask against the
unmasked bidirectional pass at the same inputs.

Uses `scd_attention.causal_mask` — the mask the encoder will actually run — rather than a second
copy written for the measurement.

Usage:
    python3 scripts/scd/phase2_mask_cost.py \
        --checkpoint /run/media/johndpope/2TB/Fizgig/models/MiniMax-H3-FL2VA/FL2VA/transformer \
        --latents scripts/scd/clips/isodiorama_latents.safetensors \
        --text scripts/scd/clips/isodiorama_te.safetensors --clip isodiorama
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase0_probe import (build_stream, cos_rows, import_fizgig,  # noqa: E402
                          load_latents, load_text, sweep_stack)
from phase0_validate import centered_cos, common_mode  # noqa: E402
from scd_attention import FrameSpans, causal_mask  # noqa: E402


def rel_l2(a, b):
    return (a.float() - b.float()).norm().div(b.float().norm().clamp_min(1e-12)).item()


def score(outs, refs, n_audio):
    """Video and audio rows separately — AdaLN is per-modality and they need not move together.
    `outs` rows start at audio_start, so the first n_audio rows are audio.

    Both raw and centered cosine, because §7 was already burned once by the difference: the
    residual stream accumulates a large shared component (common mode 0.80-0.997 there), and raw
    cos on rows dominated by a shared vector rises toward 1 for two runs that still differ in the
    part carrying information. `ccos_*` is the comparable number; `common_mode_*` says how much
    the raw one was inflated. Centering uses the UNMASKED run's own mean, over the same row set
    being scored — video and audio sit in different parts of the space (separate AdaLN), so a
    mean over both would leave a bias in each.
    """
    vid = [(a[n_audio:], b[n_audio:]) for a, b in zip(outs, refs)]
    aud = [(a[:n_audio], b[:n_audio]) for a, b in zip(outs, refs)]
    return {
        "cos_video": [cos_rows(a, b) for a, b in vid],
        "cos_audio": [cos_rows(a, b) for a, b in aud],
        "rel_l2_video": [rel_l2(a, b) for a, b in vid],
        "ccos_video": [centered_cos(a, b, b.float().mean(0)) for a, b in vid],
        "ccos_audio": [centered_cos(a, b, b.float().mean(0)) for a, b in aud],
        "common_mode_video": [common_mode(b) for _, b in vid],
        "common_mode_audio": [common_mode(b) for _, b in aud],
    }


FLAT_METRICS = ("cos_video", "ccos_video", "rel_l2_video", "cos_audio", "ccos_audio",
                "common_mode_video")


def derive_flat(payload):
    """Add the numbers the design doc quotes as flat `key -> [one per sigma]` lists.

    The claims registry addresses a value as `{file, key, index}`, and `by_sigma` is three levels
    deep, so without this every Phase 2 number in the prose would be hand-transcribed — the exact
    failure the registry exists to prevent. Derived here rather than in a separate script so a
    re-run and a re-flatten cannot disagree; `--flatten-only` reapplies it to an existing file.
    """
    e = payload["encoder_depth"] - 1
    last = payload["n_blocks"] - 1
    order = [f"{s:.10f}" for s in payload["sigmas"]]
    for mask in ("strict", "loose"):
        for metric in FLAT_METRICS:
            curves = [payload["by_sigma"][k][mask][metric] for k in order]
            payload[f"{mask}_{metric}_at_encoder"] = [c[e] for c in curves]
        for metric in ("cos_video", "ccos_video"):
            payload[f"{mask}_{metric}_at_last"] = [
                payload["by_sigma"][k][mask][metric][last] for k in order]
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flatten-only", metavar="PATH",
                    help="re-derive the flat keys on an existing result file and exit (no GPU)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--latents")
    ap.add_argument("--text")
    ap.add_argument("--clip", help="name recorded in the result file")
    ap.add_argument("--latent-t", type=int, default=7)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.5714285714285715, 0.9230769230769231])
    ap.add_argument("--encoder-depth", type=int, default=30)
    ap.add_argument("--blocks-to-swap", type=int, default=42)
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--out", default="docs/phase2_mask_cost.json")
    args = ap.parse_args()

    if args.flatten_only:
        with open(args.flatten_only) as f:
            payload = derive_flat(json.load(f))
        with open(args.flatten_only, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"re-derived flat keys in {args.flatten_only}")
        return

    missing = [n for n in ("checkpoint", "latents", "text", "clip") if not getattr(args, n)]
    if missing:
        raise SystemExit(f"need {', '.join('--' + m for m in missing)} (or --flatten-only)")

    mm, load_dit = import_fizgig(args.fizgig_src)
    device, dtype = torch.device("cuda"), torch.bfloat16

    model = load_dit(args.checkpoint, device=device, compute_dtype=dtype,
                     quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    model.enable_block_swap(args.blocks_to_swap)

    video_latent = load_latents(args.latents, args.latent_t, device)
    text_embeds = load_text(args.text, device, dtype)

    t0 = time.time()
    results = {}
    for sigma in args.sigmas:
        stream = build_stream(mm, model, video_latent, text_embeds, sigma, device, dtype)
        sp = FrameSpans(stream["seq_len"], stream["latent_t"], stream["frame_rows"])
        assert sp.video_start == stream["video_start"], \
            f"FrameSpans says {sp.video_start}, the packer says {stream['video_start']}"
        frames = sp.frame_index(device)
        n_audio = stream["video_start"] - stream["audio_start"]
        print(f"sigma={sigma:.4f}  S={sp.seq_len}  video_start={sp.video_start}  "
              f"frames={sp.latent_t}x{sp.frame_rows}", flush=True)

        ref, _ = sweep_stack(model, mm, stream, None, device, keep_full=False)
        out = {}
        for name, loose in (("strict", False), ("loose", True)):
            mask = causal_mask(frames, frames, context_sees_video=loose)
            masked, _ = sweep_stack(model, mm, stream, mask, device, keep_full=False)
            out[name] = score(masked, ref, n_audio)
            e = args.encoder_depth - 1
            print(f"  {name:6s} @block{e}: cos_video {out[name]['cos_video'][e]:.4f}  "
                  f"ccos_video {out[name]['ccos_video'][e]:.4f}  "
                  f"rel_l2 {out[name]['rel_l2_video'][e]:.4f}  "
                  f"ccos_audio {out[name]['ccos_audio'][e]:.4f}  "
                  f"cm_video {out[name]['common_mode_video'][e]:.4f}", flush=True)
            del masked, mask
            torch.cuda.empty_cache()
        results[f"{sigma:.10f}"] = out
        del ref, stream
        torch.cuda.empty_cache()

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    payload = {
        "clip": args.clip, "latent_t": args.latent_t, "sigmas": args.sigmas,
        "latent_hw": list(video_latent.shape[-2:]), "encoder_depth": args.encoder_depth,
        "n_blocks": len(model.blocks), "base_quant": args.base_quant,
        "checkpoint": os.path.basename(args.checkpoint.rstrip("/")),
        "elapsed_s": time.time() - t0, "git_sha": sha, "by_sigma": results,
    }
    derive_flat(payload)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out} in {payload['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
