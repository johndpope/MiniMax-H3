#!/usr/bin/env python3
"""Phase 2.5 Tier 1 — the untrained SCD graph on real weights (§2.5 of the design doc).

Tier 0 timed four primitives at real dims and composed a cost model from them. This times the
GRAPH: the same blocks Phase 1 split, the same masked/cached attention Phase 2 wrote, the same
768p row geometry. Output is garbage — the weights are untrained for this split and nothing here
decodes to pixels — and the wall clock is real, which is the entire point of benchmarking before
spending on training.

Three things Tier 0's model got wrong, all in the same direction:

    encoder layers   33 assumed, 30 shipped
    decoder layers   17 assumed, 7 shipped  (blocks {0, 1, 45..49})
    encoder attn     modelled as block-causal over the FULL S; the real encoder is chunked
                     against a windowed cache, which is O(frames * R * window * R), not O(S^2/2)

The decoder count is the one that matters: it multiplies `n_steps * frames`, so Tier 0's 3.90x
at 768p/10s was built on a decoder 2.4x more expensive than the one that exists. Expect this to
come out ABOVE Tier 0, and treat it as a red flag if it does not — that would mean the graph is
paying for something the FLOP model cannot see.

Weights are pinned resident for whichever path is being timed, and the paths are timed one at a
time with the other freed. That isolates the graph from PCIe: under block swapping the stock path
moves 50 blocks per STEP and the layer-major encoder moves 30 for the whole clip, which is a real
and large effect but a property of a 24 GB card rather than of SCD, and mixing the two into one
number answers neither question.

Usage:
    python3 scripts/scd/phase25_tier1.py \
        --checkpoint /run/media/johndpope/2TB/Fizgig/models/MiniMax-H3-FL2VA/FL2VA/transformer \
        --latent-t 12 22 32 --steps 8 16 30
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase0_probe import build_stream, import_fizgig, synthetic_inputs  # noqa: E402
from scd_attention import FrameSpans, encode_chunks, row_time, run_block  # noqa: E402

# 768p 16:9 -> 1344x768 pixels -> 84x48 latent -> (84//2)*(48//2) = 1008 rows per latent frame.
LAT_H, LAT_W = 48, 84
ROWS_768P = (LAT_H // 2) * (LAT_W // 2)
ENCODER_DEPTH = 30
DECODER_SOURCE = (0, 1, 45, 46, 47, 48, 49)


class OOM(Exception):
    pass


def timed(fn, warmup=1, iters=3):
    """Median wall clock in ms, or OOM. CUDA events, so this measures the device and not the
    launch queue — at 92k tokens a single block is milliseconds and the queue is not."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            samples.append(a.elapsed_time(b))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise OOM
    samples.sort()
    return samples[len(samples) // 2]


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def resident(blocks, device):
    for b in blocks:
        b.to(device)


def evict(blocks):
    for b in blocks:
        b.to("cpu")
    torch.cuda.empty_cache()


@torch.no_grad()
def time_stock(blocks, stream, device):
    """One full bidirectional pass over the whole packed sequence — one denoise step of stock H3.

    No mask at all, which is what stock H3 does: every row sees every row. Passing an all-true
    mask instead would measure SDPA's masked kernel and quietly hand SCD a head start.
    """
    ctx = (stream["t_emb"], stream["mod_row"], stream["cos"], stream["sin"])

    def once():
        x = stream["h"]
        for blk in blocks:
            x = run_block(blk, x, ctx)
        return x

    torch.cuda.reset_peak_memory_stats()
    ms = timed(once)
    return ms, peak_gb()


@torch.no_grad()
def time_encoder(blocks, stream, device, window, chunk_frames):
    """The SCD encoder: once per clip, chunked, windowed, layer-major, dense per-chunk mask.

    `layer_major=True` because the clip exists up front here — that is what makes the encoder's
    live cache one block's worth instead of thirty. Phase 5's AR driver cannot use it.

    `block=False` despite the design doc's Tier 1 brief asking for the block mask. Chunking has
    already made `Q` one frame, so the dense mask is ~13 MB here; the block mask instead gives
    every chunk a distinct `(Q, K)` shape, blows Dynamo's recompile limit at chunk 9, falls back
    to eager `flex_attention` and materializes a 2.4 GB scores matrix per block. Measured: OOM at
    latent_t=12 with the block mask, 12.1 GB peak with the dense one.
    """
    sp = FrameSpans(stream["seq_len"], stream["latent_t"], stream["frame_rows"])
    t = row_time(stream["pos"][:, 0], stream["audio_start"]).to(device)
    ctx = (stream["t_emb"], stream["mod_row"], stream["cos"], stream["sin"])
    held = {}

    def once():
        out, cache = encode_chunks(blocks, stream["h"], ctx, t, sp, chunk_frames,
                                   block=False, window=window, layer_major=True)
        held["rows"], held["bytes"] = len(cache), cache.bytes()
        return out

    torch.cuda.reset_peak_memory_stats()
    ms = timed(once, warmup=1, iters=2)
    return ms, peak_gb(), held["rows"], held["bytes"]


@torch.no_grad()
def time_decoder_frame(blocks, stream, device):
    """One frame through the decoder re-composition, token_concat style: 2R rows.

    The conditioning half is the encoder's rows for this frame and the other half is the noisy
    latent for it, so the row count is 2R and the RoPE positions repeat — which is what
    token_concat means and is why Tier 0 modelled `attn(2R, 2R)`. The CONTENT is meaningless
    (nothing is trained), and content does not change the shapes this times.
    """
    r = stream["frame_rows"]
    lo = stream["video_start"]
    rows = torch.arange(lo, lo + r, device=device)
    rows = torch.cat([rows, rows])
    x = stream["h"][rows]
    ctx = (stream["t_emb"], stream["mod_row"][rows], stream["cos"][rows], stream["sin"][rows])

    def once():
        y = x
        for blk in blocks:
            y = run_block(blk, y, ctx)
        return y

    torch.cuda.reset_peak_memory_stats()
    ms = timed(once, warmup=2, iters=5)
    return ms, peak_gb()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latent-t", type=int, nargs="+", default=[12, 22, 32],
                    help="on the packer's 5n+2 grid; 32/62/92 are 768p 5s/10s/15s")
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 16, 30])
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--chunk-frames", type=int, default=1)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.5714285714285715)
    ap.add_argument("--blocks-to-swap", type=int, default=48,
                    help="parked on CPU at load; residency is then managed per path here")
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--out", default="docs/phase25_tier1.json")
    args = ap.parse_args()

    mm, load_dit = import_fizgig(args.fizgig_src)
    device, dtype = torch.device("cuda"), torch.bfloat16
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device {name} ({total_gb:.0f} GB), torch {torch.__version__}, "
          f"768p rows/frame {ROWS_768P}, window {args.window}", flush=True)

    model = load_dit(args.checkpoint, device=device, compute_dtype=dtype,
                     quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    stock_blocks = list(model.blocks)
    enc_blocks = stock_blocks[:ENCODER_DEPTH]
    dec_blocks = [stock_blocks[i] for i in DECODER_SOURCE]

    t0 = time.time()
    rows = []
    for latent_t in args.latent_t:
        z, te = synthetic_inputs(latent_t, LAT_H, LAT_W, args.text_len, device, dtype)
        stream = build_stream(mm, model, z, te, args.sigma, device, dtype)
        s = stream["seq_len"]
        n_audio = stream["video_start"] - stream["audio_start"]
        seconds = ((latent_t - 1) * 4 + 1) / 24.0
        print(f"\nlatent_t={latent_t} ({seconds:.1f}s)  S={s}  "
              f"video={latent_t * ROWS_768P}  audio={n_audio}", flush=True)

        row = {"latent_t": latent_t, "seconds": seconds, "seq_len": s, "n_audio": n_audio,
               "frame_rows": stream["frame_rows"], "steps": {}}

        resident(stock_blocks, device)
        try:
            row["stock_step_ms"], row["stock_peak_gb"] = time_stock(stock_blocks, stream, device)
            print(f"  stock   : {row['stock_step_ms'] / 1e3:7.3f} s/step   "
                  f"peak {row['stock_peak_gb']:5.1f} GB", flush=True)
        except OOM:
            row["stock_step_ms"] = None
            print("  stock   : OOM — a Tier 1 result in itself, but no speedup from this row",
                  flush=True)
        evict(stock_blocks)

        resident(enc_blocks, device)
        try:
            ms, gb, c_rows, c_bytes = time_encoder(enc_blocks, stream, device,
                                                   args.window, args.chunk_frames)
            row.update(encoder_ms=ms, encoder_peak_gb=gb,
                       cache_rows=c_rows, cache_bytes=c_bytes)
            print(f"  encoder : {ms / 1e3:7.3f} s once  peak {gb:5.1f} GB  "
                  f"cache {c_rows} rows / {c_bytes / 1e9:.2f} GB (one block)", flush=True)
        except OOM:
            row["encoder_ms"] = None
            print("  encoder : OOM", flush=True)
        evict(enc_blocks)

        resident(dec_blocks, device)
        try:
            ms, gb = time_decoder_frame(dec_blocks, stream, device)
            row.update(decoder_frame_ms=ms, decoder_peak_gb=gb)
            print(f"  decoder : {ms:7.3f} ms/frame  peak {gb:5.1f} GB", flush=True)
        except OOM:
            row["decoder_frame_ms"] = None
            print("  decoder : OOM", flush=True)
        evict(dec_blocks)

        if row.get("stock_step_ms") and row.get("encoder_ms") and row.get("decoder_frame_ms"):
            for n in args.steps:
                stock = n * row["stock_step_ms"] / 1e3
                scd = (row["encoder_ms"] + n * latent_t * row["decoder_frame_ms"]) / 1e3
                row["steps"][str(n)] = {"stock_s": stock, "scd_s": scd, "speedup": stock / scd}
                print(f"    N={n:<3d} stock {stock:8.1f} s   scd {scd:7.1f} s   "
                      f"{stock / scd:5.2f}x", flush=True)
        rows.append(row)
        del stream, z
        torch.cuda.empty_cache()

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    payload = {"device": name, "total_gb": total_gb, "torch": torch.__version__,
               "rows_per_frame": ROWS_768P, "encoder_depth": ENCODER_DEPTH,
               "decoder_source": list(DECODER_SOURCE), "window": args.window,
               "chunk_frames": args.chunk_frames, "sigma": args.sigma,
               "base_quant": args.base_quant, "steps": args.steps,
               "checkpoint": os.path.basename(args.checkpoint.rstrip("/")),
               "elapsed_s": time.time() - t0, "git_sha": sha, "by_length": rows}
    for r in rows:
        for n, v in r["steps"].items():
            payload[f"t{r['latent_t']}_n{n}_speedup"] = v["speedup"]
        if r.get("cache_rows"):
            payload[f"t{r['latent_t']}_cache_rows"] = r["cache_rows"]
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out} in {payload['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
