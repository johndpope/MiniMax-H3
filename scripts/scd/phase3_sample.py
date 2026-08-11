#!/usr/bin/env python3
"""Phase 3 kill criterion: sample through the SCD split and score it against the clip it came from.

The criterion is "after 1-2k steps, an AR or multi-step sample is **not pure noise** and beats the
unadapted split". Both halves of that need a number, so this reports latent MSE and correlation
against the ground-truth latent, per frame, and the same run with `--lora` omitted is the baseline.
Omitting it is a stronger baseline than the design doc's "decoder-only random init": it is the
composed split with every adapter at zero, i.e. exactly what the surgery gives you for free. If the
trained adapters do not beat THAT, the LoRA learned nothing worth the split.

Two modes, and the gap between them is the point:

`--mode oracle` denoises every frame with the encoder reading the REAL clean latents. It is the
regime training ran in, teacher-forced end to end, so it measures whether the decoder learned the
one-step conditional at all. It is a ceiling, not a result: no sampler at inference has the clean
latents of the frames it has not generated yet.

`--mode ar` is the real thing. The encoder only ever sees frames the decoder itself produced, so
errors compound the way they do in a real rollout. Training had no scheduled sampling (see
`phase3_train`'s module docstring), so a large oracle/AR gap is the expected failure and is the
evidence for adding the §6.4 curriculum -- not a reason to doubt the split.

Frame 0 is generated with a zeroed context half in both modes, because that is what
`decoder_frame_input` does and conditioning it on its own encoder feature is the leak the whole
shift exists to prevent. It is also only trained on ~50% of steps (`--first-frame-cond-p`), so it
is the weakest frame by construction and is reported separately rather than averaged in silently.

Usage:
    python3 scripts/scd/phase3_sample.py --checkpoint /path/to/FL2VA/transformer \
        --lora runs/scd_v0/scd_lora_002000.safetensors --mode ar --steps 20
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scd_data import ClipSet                                             # noqa: E402
from scd_lora import add_lora                                            # noqa: E402
from scd_model import DEFAULT_DECODER_SOURCE, DEFAULT_ENCODER_DEPTH, MiniMaxH3SCD  # noqa: E402
from phase3_train import import_fizgig                                   # noqa: E402


def load_lora(scd, path):
    """Install adapters shaped by the checkpoint's own metadata, then fill them.

    Rank and alpha come from the file rather than from a flag: they are properties of the thing
    being loaded, and a mismatch is a silent quality regression (`scale = alpha/rank`) rather than
    an error, because the shapes still line up whenever rank happens to agree.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}
    rank = int(meta.get("rank") or sd[next(k for k in sd if k.endswith("lora_down.weight"))].shape[0])
    made = add_lora(scd, rank=rank, alpha=float(meta["alpha"]) if "alpha" in meta else None)

    missing = set(made) - {k.rsplit(".", 2)[0] for k in sd}
    if missing:
        raise SystemExit(f"{path} has no weights for {len(missing)} installed adapters, "
                         f"e.g. {sorted(missing)[0]} — the split does not match the one trained")
    for name, mod in made.items():
        for part in ("lora_down", "lora_up"):
            w = sd[f"{name}.{part}.weight"]
            getattr(mod, part).weight.data.copy_(w.to(getattr(mod, part).weight.dtype))
    return meta


@torch.no_grad()
def sample(scd, mm, clip, sigmas, mode, *, seed, window, chunk_frames, media_start_on,
           duplicate_pos, seed_frames=0):
    """Denoise the clip one frame at a time. Returns the sampled latent, same shape as the truth.

    `seed_frames` takes the first N frames from the truth instead of generating them, and is the
    difference between measuring the rollout and measuring recall. Frame 0's context half is zeros
    by construction and `media_start_on` defaults off, so with N=0 frame 0 is generated from
    NOTHING — no context, no text — and the only way it can score is by having memorised the
    training clips. Every later frame then conditions on whatever it invented. N=1 is what FL2VA
    and I2VA actually do, and it is the honest autoregressive test for a model conditioned this
    way: frames 1.. are still generated, still fed back, still compounding.

    N>0 deliberately puts ground truth into an `ar` rollout, which is exactly what
    `test_ar_never_reads_the_ground_truth` forbids at the default N=0. Keep it 0 to compare
    against runs that predate this.

    The encoder is re-run once per FRAME, not once per denoising step: it reads only clean latents,
    and under both modes those stop changing as soon as frame f-1 is final. Inside a frame the
    context is therefore constant and only the noisy rows move, which is also what makes 20 steps
    per frame affordable — the 30-block half runs 7 times for the clip, not 140.

    The full noisy buffer is repacked every step even though `decode_frame` reads only frame f's
    rows from it. Slicing the pack instead would mean rebuilding the packer's RoPE positions and
    modulation table for a partial clip, which is the drift `preamble` exists to prevent, and the
    packer is not the expensive half.
    """
    x0 = clip["video_latent"].float()
    text = clip["text_embeds"]
    dev = x0.device
    g = torch.Generator(device=dev).manual_seed(seed)
    latent_t = x0.shape[2]

    # Frames the ENCODER reads. Under oracle they are the truth; under ar they are filled in as the
    # decoder produces them, and start at zero — which is what the encoder sees for frames that do
    # not exist yet, and is the same thing `decoder_frame_input` gives frame 0.
    ctx_buf = x0.clone() if mode == "oracle" else torch.zeros_like(x0)
    out = torch.zeros_like(x0)

    for f in range(latent_t):
        if f < seed_frames:
            # Into `ctx_buf` as well as `out`: under `ar` the buffer starts at zeros, and a seeded
            # frame that is not written back would condition frame f+1 on nothing at all.
            out[:, :, f] = x0[:, :, f]
            ctx_buf[:, :, f] = x0[:, :, f]
            continue
        enc, clean_ctx, _ = scd.encode_chunked(
            chunk_frames, window=window, layer_major=True, keep_audio=True,
            video_latent=ctx_buf, t=torch.tensor([1.0], device=dev), text_embeds=text)
        spans = scd.spans(x0, enc.shape[0])
        h, w = x0.shape[3] // scd.base.patch_size[1], x0.shape[4] // scd.base.patch_size[2]

        z = torch.randn(x0[:, :, f:f + 1].shape, device=dev, dtype=torch.float32, generator=g)
        buf = ctx_buf.clone()
        for i in range(len(sigmas) - 1):
            s, s_next = sigmas[i], sigmas[i + 1]
            buf[:, :, f:f + 1] = z
            noisy_h, noisy_ctx, pack = scd.preamble(
                video_latent=buf, t=torch.tensor([1.0 - s], device=dev), text_embeds=text)
            v = scd.decode_frame(enc, clean_ctx, noisy_h, noisy_ctx, spans, f, velocity=True,
                                 media_start=pack["media_start"] if media_start_on else None,
                                 duplicate_pos=duplicate_pos)
            # The head predicts x0 - noise, so Euler from s to s_next is x + (s - s_next)*v.
            z = z + (s - s_next) * mm.unpatchify_video(v.float(), 1, h, w,
                                                       c=x0.shape[1],
                                                       patch_size=scd.base.patch_size).to(dev)
        out[:, :, f:f + 1] = z
        if mode == "ar":
            ctx_buf[:, :, f:f + 1] = z
    return out


def score(pred, truth):
    """Per-frame MSE and correlation against the truth, plus the noise floor.

    Correlation, not MSE alone. A decoder that collapses to the dataset mean posts a LOWER MSE than
    the truth's own variance while carrying no frame-specific information at all, and `corr` is the
    number that separates those: it is ~0 for anything uncorrelated with the target, including a
    confident constant, and MSE cannot say so.

    `mse_noise` is what pure Gaussian noise of the truth's scale would score, so "not pure noise"
    is a comparison rather than an impression.
    """
    p, t = pred.float(), truth.float()
    per = []
    for f in range(t.shape[2]):
        a, b = p[:, :, f].flatten(), t[:, :, f].flatten()
        ac, bc = a - a.mean(), b - b.mean()
        denom = float(ac.norm() * bc.norm())
        per.append({"frame": f,
                    "mse": round(float((a - b).square().mean()), 5),
                    "corr": round(float(ac.dot(bc) / denom) if denom > 0 else 0.0, 4),
                    "std": round(float(a.std()), 4)})
    return {"per_frame": per,
            "mse": round(sum(d["mse"] for d in per) / len(per), 5),
            "corr": round(sum(d["corr"] for d in per) / len(per), 4),
            # frames 1.. only: frame 0 has no context half and is trained half as often
            "corr_ctx": round(sum(d["corr"] for d in per[1:]) / max(1, len(per) - 1), 4),
            "mse_noise": round(float((torch.randn_like(t) * t.std() - t).square().mean()), 5),
            "truth_var": round(float(t.var()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lora", help="omit for the unadapted-split baseline")
    ap.add_argument("--clips", default=os.path.join(os.path.dirname(__file__), "clips"))
    ap.add_argument("--mode", default="ar", choices=["ar", "oracle"])
    ap.add_argument("--steps", type=int, default=20, help="denoising steps per frame")
    ap.add_argument("--shift", type=float, default=12.0)
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--chunk-frames", type=int, default=1)
    ap.add_argument("--decoder-text", action="store_true")
    ap.add_argument("--own-context-pos", action="store_true")
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--out", help="write per-clip scores here as JSON")
    ap.add_argument("--seed-frames", type=int, default=0,
                    help="take the first N frames from the truth instead of generating them. "
                         "N=1 is what FL2VA/I2VA do and makes `ar` measure the rollout rather "
                         "than whether frame 0 was memorised; N=0 reproduces the earlier runs")
    ap.add_argument("--save-latents", metavar="DIR",
                    help="also write each clip's sampled latent and its ground truth here, for "
                         "phase3_decode.py. Separate steps because the 2.4 B VAE decoder does not "
                         "fit alongside the resident base")
    args = ap.parse_args()

    mm, load_dit, _ = import_fizgig(args.fizgig_src)
    from fizgig.minimax.sampling import sample_schedule

    device = torch.device("cuda")
    clips = ClipSet(args.clips)
    base = load_dit(args.checkpoint, device=device, compute_dtype=torch.bfloat16,
                    quantize=args.base_quant != "none", base_quant=args.base_quant)
    scd = MiniMaxH3SCD(base, encoder_depth=DEFAULT_ENCODER_DEPTH,
                       decoder_source=DEFAULT_DECODER_SOURCE)
    scd.eval()

    if args.lora:
        meta = load_lora(scd, args.lora)
        print(f"lora        : {args.lora} (step {meta.get('step', '?')}, "
              f"rank {meta.get('rank', '?')})", flush=True)
    else:
        # Not "no adapters" — adapters at zero. Same module tree, same numerics, same code path, so
        # the only difference from the trained run is the weights. A baseline that skipped
        # `add_lora` would also be testing the wrapper.
        add_lora(scd, rank=4)
        print("lora        : none (unadapted split baseline)", flush=True)

    sigmas = sample_schedule(args.steps, shift=args.shift)
    results = []
    for name in clips.names[:args.n_clips]:
        clip = clips.load(name, device=device, dtype=torch.bfloat16)
        pred = sample(scd, mm, clip, sigmas, args.mode, seed=args.seed, window=args.window,
                      chunk_frames=args.chunk_frames, media_start_on=args.decoder_text,
                      duplicate_pos=not args.own_context_pos, seed_frames=args.seed_frames)
        s = score(pred, clip["video_latent"].float())
        results.append({"clip": name, **s})
        if args.save_latents:
            from safetensors.torch import save_file
            os.makedirs(args.save_latents, exist_ok=True)
            path = os.path.join(args.save_latents, f"{name}_{args.mode}.safetensors")
            # The truth travels with the sample so the decode is a comparison rather than a
            # picture: the VAE's own round-trip loss is not small, and a blurry sample is not
            # evidence against the split unless the truth through the same path is sharp.
            save_file({"pred": pred.float().cpu().contiguous(),
                       "truth": clip["video_latent"].float().cpu().contiguous()}, path,
                      metadata={"clip": name, "mode": args.mode, "lora": args.lora or "none",
                                "steps": str(args.steps), "corr_ctx": str(s["corr_ctx"])})
        print(f"{name:>16}  mse {s['mse']:.4f} (noise {s['mse_noise']:.4f})  "
              f"corr {s['corr']:+.4f}  corr[1:] {s['corr_ctx']:+.4f}  "
              f"std {s['per_frame'][-1]['std']:.3f}", flush=True)

    n = len(results)
    print(f"\n{args.mode:>6} mean  mse {sum(r['mse'] for r in results) / n:.4f}  "
          f"corr {sum(r['corr'] for r in results) / n:+.4f}  "
          f"corr[1:] {sum(r['corr_ctx'] for r in results) / n:+.4f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"mode": args.mode, "lora": args.lora, "steps": args.steps,
                       "seed_frames": args.seed_frames,
                       "clips": results}, fh, indent=2)
        print(f"wrote       : {args.out}")


if __name__ == "__main__":
    main()
