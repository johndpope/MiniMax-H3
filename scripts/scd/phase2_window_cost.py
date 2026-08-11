#!/usr/bin/env python3
"""What §8.1's sliding window costs the encoder on real weights.

The window is the only APPROXIMATION in the Phase 2 encoder. Everything else — chunking, the KV
cache, the gather/scatter under the AV clock — is one masked pass reassociated, and is asserted
bit-exact by `test_scd_attention.py`. Dropping cache rows is not: a video frame outside the window
is genuinely unavailable to later queries, and no test can pin what that costs. Only a measurement
can, and until this ran there was none at any size.

It is also not optional. Thirty encoder blocks hold 840 KiB per row, so an unbounded cache is
53 GB at 768p/15s — an order of magnitude past the dense `[S, S]` mask that FlexAttention exists
to avoid. The question is therefore not whether to window but how narrow it can be.

Two sources, because the clips cannot answer the question alone:

    clip        real latents, latent_t=7 — the honest number, on the same clip §7 quotes
    synthetic   smoothed correlated noise, latent_t=17 — the only way to reach a regime where a
                window bites at all, since at 7 frames a window of 6 IS unbounded

The synthetic run's *content* is not real, and a window's cost plausibly depends on how much
frame-to-frame novelty there is, so it is reported beside the clip rather than instead of it. The
encoded clips top out at 8 latent frames and the packer's grid admits only 7 or 12.

That split applies to VIDEO only. There is no real audio anywhere in this tree: `build_stream`
gives video `(1-sigma)*latent + sigma*eps` but gives audio `sigma_a * eps` with no latent term, so
the audio rows are pure noise in the clip run exactly as much as in the synthetic one. Encoding a
longer or higher-resolution clip does not change that; only the audio VAE would, and it is not
downloaded. So `ccos_audio` is not "the window's cost on real audio" — it is its cost on IID noise,
which is the least autocorrelated input the audio axis can be handed and therefore an upper bound.
`--audio-corr` sweeps the correlation to find out whether that bound is tight or vacuous.

Scored per rule 10: centered cosine with the common mode beside it, against the SAME chunked path
at `window=None`, so the window's cost is isolated from the mask's (that is `phase2_mask_cost.py`).

Usage:
    python3 scripts/scd/phase2_window_cost.py \
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

from phase0_probe import (build_stream, import_fizgig, load_latents,  # noqa: E402
                          load_text, synthetic_inputs)
from phase0_validate import centered_cos, common_mode  # noqa: E402
from scd_attention import FrameSpans, encode_chunks, row_time  # noqa: E402


def rel_l2(a, b):
    return (a.float() - b.float()).norm().div(b.float().norm().clamp_min(1e-12)).item()


def by_frame(out, ref, bucket, n_frames):
    """ccos per video-frame span, for rows bucketed by `bucket` — the pooled score's missing axis.

    A window is supposed to make quality independent of clip length, and a POOLED score cannot
    tell you whether it does. Under a window of W, the first W frames' queries see everything and
    only later ones are evicted against, so lengthening the clip raises the share of rows that are
    degraded even if each degraded row is no worse than before. The pooled mean then falls with
    length and looks exactly like a window that stops working — which is the reading that would
    kill M1. The per-frame curve separates the two: if it plateaus after frame W, the window is
    doing its job and the pooled drop was an averaging artefact; if it keeps sliding, it is not.
    """
    return [centered_cos(out[m], ref[m], ref[m].float().mean(0)) if (m := bucket == f).any() else
            None for f in range(n_frames)]


def score(out, ref, n_audio, aud_bucket=None, vid_bucket=None, n_frames=0):
    """Video and audio separately — AdaLN is per-modality and they need not degrade together.

    Both slices start at `audio_start`, so the first `n_audio` rows are audio. The reference is
    the unbounded chunked run, and centering uses its mean over the same row set being scored.
    """
    vid = (out[n_audio:], ref[n_audio:])
    aud = (out[:n_audio], ref[:n_audio])
    s = {
        "ccos_video": centered_cos(*vid, vid[1].float().mean(0)),
        "ccos_audio": centered_cos(*aud, aud[1].float().mean(0)),
        "rel_l2_video": rel_l2(*vid),
        "rel_l2_audio": rel_l2(*aud),
        "common_mode_video": common_mode(vid[1]),
        "common_mode_audio": common_mode(aud[1]),
    }
    if aud_bucket is not None:
        s["ccos_audio_by_frame"] = by_frame(*aud, aud_bucket, n_frames)
        s["ccos_video_by_frame"] = by_frame(*vid, vid_bucket, n_frames)
    return s


@torch.no_grad()
def run(model, blocks, stream, window, chunk_frames, device, layer_major=False,
        keep_audio=False):
    """One chunked encode. Returns (rows from audio_start on, cache rows, cache bytes).

    Encoder blocks are pinned to the device for the duration rather than swapped per block: this
    loop touches every block once per CHUNK, so leaving `blocks_to_swap` in charge would move the
    same 22 blocks across PCIe once per frame and measure the interconnect.

    `layer_major` is what makes this affordable at 768p, and it changes what `cache` means. Chunk-
    major holds all `encoder_depth` caches at once, so at 1008 rows/frame the UNBOUNDED reference
    run — the thing every window is scored against — is tens of GB and simply cannot be taken on
    this card. Block-major holds one, and is bit-exact against chunk-major
    (`test_layer_major_matches_chunk_major`), so the scores are unaffected. The returned byte count
    is then one block's rather than all of them; the ratios reported against the reference are
    taken in the same mode and stay comparable, and `layer_major` is recorded in the payload so a
    reader cannot mistake a one-block figure for a whole-encoder one.
    """
    sp = FrameSpans(stream["seq_len"], stream["latent_t"], stream["frame_rows"])
    t = row_time(stream["pos"][:, 0], stream["audio_start"]).to(device)
    ctx = (stream["t_emb"], stream["mod_row"], stream["cos"], stream["sin"])
    out, cache = encode_chunks(blocks, stream["h"], ctx, t, sp, chunk_frames, window=window,
                               layer_major=layer_major, keep_audio=keep_audio)
    return out[stream["audio_start"]:].to("cpu", torch.bfloat16), len(cache), cache.bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents")
    ap.add_argument("--text")
    ap.add_argument("--clip", default="isodiorama")
    ap.add_argument("--latent-t", type=int, default=7)
    ap.add_argument("--synthetic-latent-t", type=int, default=17,
                    help="0 to skip the synthetic long run")
    ap.add_argument("--synthetic-lat-hw", type=int, nargs=2, default=[32, 32], metavar=("H", "W"),
                    help="latent H W for the synthetic source; 48 84 is 768p (1008 rows/frame)")
    ap.add_argument("--audio-corr", type=float, default=0.0,
                    help="AR(1) coefficient on the audio noise time axis. Every run so far is 0.0 "
                         "(IID), which is the least autocorrelated audio possible and so the "
                         "loosest upper bound on what a window costs it. Sweep to see if it binds")
    ap.add_argument("--keep-audio", action="store_true",
                    help="window video only; audio rows are never evicted. ~1.3%% of rows at 768p")
    ap.add_argument("--layer-major", action="store_true",
                    help="block-outer encode: one block's cache live instead of encoder_depth. "
                         "Required at 768p, where the unbounded reference is otherwise tens of GB")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--chunk-frames", type=int, default=1)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.5714285714285715, 0.9230769230769231])
    ap.add_argument("--encoder-depth", type=int, default=30)
    ap.add_argument("--blocks-to-swap", type=int, default=42)
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--out", default="docs/phase2_window_cost.json")
    args = ap.parse_args()

    mm, load_dit = import_fizgig(args.fizgig_src)
    device, dtype = torch.device("cuda"), torch.bfloat16

    model = load_dit(args.checkpoint, device=device, compute_dtype=dtype,
                     quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    model.enable_block_swap(args.blocks_to_swap)
    blocks = list(model.blocks)[:args.encoder_depth]
    for b in blocks:
        b.to(device)

    text_embeds = load_text(args.text, device, dtype)
    sources = {}
    if args.latents:
        sources["clip"] = (load_latents(args.latents, args.latent_t, device), text_embeds)
    if args.synthetic_latent_t:
        z, _ = synthetic_inputs(args.synthetic_latent_t, *args.synthetic_lat_hw,
                                text_embeds.shape[1], device, dtype)
        sources["synthetic"] = (z, text_embeds)

    t0 = time.time()
    results = {}
    for name, (video_latent, te) in sources.items():
        per_sigma = {}
        for sigma in args.sigmas:
            stream = build_stream(mm, model, video_latent, te, sigma, device, dtype,
                                  audio_corr=args.audio_corr)
            n_audio = stream["video_start"] - stream["audio_start"]
            print(f"{name} sigma={sigma:.4f}  S={stream['seq_len']}  "
                  f"frames={stream['latent_t']}x{stream['frame_rows']}  audio={n_audio}",
                  flush=True)

            # Bucket every scored row by the video frame whose span it falls in. Audio sits on its
            # own 40 Hz grid, so this is a searchsorted against the frame times rather than a
            # rows-per-frame division — same reason `row_time` reads `pos_t` instead of
            # reconstructing it. Video rows are frame-major, so theirs is exact by construction.
            t_all = row_time(stream["pos"][:, 0], stream["audio_start"]).to(device)
            frame_t = t_all[stream["video_start"]::stream["frame_rows"]]
            aud_bucket = torch.searchsorted(frame_t.contiguous(),
                                            t_all[stream["audio_start"]:stream["video_start"]]
                                            .contiguous()).clamp_max(stream["latent_t"] - 1).cpu()
            vid_bucket = torch.arange(stream["seq_len"] - stream["video_start"]).div(
                stream["frame_rows"], rounding_mode="floor")

            ref, ref_rows, ref_bytes = run(model, blocks, stream, None, args.chunk_frames, device,
                                           args.layer_major, args.keep_audio)
            entry = {"seq_len": stream["seq_len"], "latent_t": stream["latent_t"],
                     "frame_rows": stream["frame_rows"], "n_audio": n_audio,
                     "unbounded_cache_rows": ref_rows, "unbounded_cache_bytes": ref_bytes,
                     "windows": {}}
            for w in args.windows:
                out, rows, nbytes = run(model, blocks, stream, w, args.chunk_frames, device,
                                        args.layer_major, args.keep_audio)
                s = score(out, ref, n_audio, aud_bucket, vid_bucket, stream["latent_t"])
                s["cache_rows"], s["cache_bytes"] = rows, nbytes
                entry["windows"][str(w)] = s
                tail = [x for x in s["ccos_audio_by_frame"][w:] if x is not None]
                print(f"  window {w:2d}: ccos_video {s['ccos_video']:7.4f}  "
                      f"ccos_audio {s['ccos_audio']:7.4f}  rel_l2 {s['rel_l2_video']:.4f}  "
                      f"cache {rows:5d}/{ref_rows} rows "
                      f"({nbytes / ref_bytes:.2f}x)", flush=True)
                if tail:
                    print(f"            audio past frame {w}: "
                          f"{' '.join(f'{x:.3f}' for x in tail)}", flush=True)
                del out
                torch.cuda.empty_cache()
            per_sigma[f"{sigma:.10f}"] = entry
            del ref, stream
            torch.cuda.empty_cache()
        results[name] = per_sigma

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    payload = {
        "clip": args.clip, "sigmas": args.sigmas, "windows": args.windows,
        "chunk_frames": args.chunk_frames, "encoder_depth": args.encoder_depth,
        "layer_major": args.layer_major, "synthetic_lat_hw": args.synthetic_lat_hw,
        "keep_audio": args.keep_audio, "audio_corr": args.audio_corr,
        "base_quant": args.base_quant, "n_blocks": len(model.blocks),
        "checkpoint": os.path.basename(args.checkpoint.rstrip("/")),
        "elapsed_s": time.time() - t0, "git_sha": sha, "by_source": results,
    }
    for source, per_sigma in results.items():
        order = [f"{s:.10f}" for s in args.sigmas]
        for w in args.windows:
            for metric in ("ccos_video", "ccos_audio", "rel_l2_video"):
                payload[f"{source}_w{w}_{metric}"] = [
                    per_sigma[k]["windows"][str(w)][metric] for k in order]
            payload[f"{source}_w{w}_cache_rows"] = [
                per_sigma[k]["windows"][str(w)]["cache_rows"] for k in order]
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out} in {payload['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
