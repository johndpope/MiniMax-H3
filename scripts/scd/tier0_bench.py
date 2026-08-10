#!/usr/bin/env python3
"""Tier 0 speed POC for the H3 x SCD port (docs/MINIMAX_H3_SCD_PORT_DESIGN.md, Phase 2.5).

No weights, no checkpoint, no rental. Times the four primitives that dominate an H3 step at
real dims with torch.randn, then composes stock-H3 and SCD cost models from those timings.

Purpose is to confirm or refute the §2.2 speedup table on this silicon before any training
spend. Kill criterion: if SCD is not materially faster at 768p/15s, the port is not worth it.

Cost models, per generated clip:

  stock = N_steps * 50 * [ dense_attn(S) + linear(S) ]
  scd   = 33 * [ block_causal_attn(S) + linear(S) ]                    encoder, once
        + N_steps * frames * 17 * [ attn(2R, 2R) + linear(2R) ]        decoder, per frame

where R = rows per latent frame and token_concat doubles the decoder's token count.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM          # 7168
FFN = 14336
NUM_LAYERS = 50
ENCODER_LAYERS = 33
DECODER_LAYERS = NUM_LAYERS - ENCODER_LAYERS

# (name, in_features, out_features) — fc1 is gated (swiglu), hence 2 * FFN.
PROJECTIONS = [
    ("qkv_proj", HIDDEN, INNER * 3),
    ("out_proj", INNER, HIDDEN),
    ("fc1", HIDDEN, FFN * 2),
    ("fc2", FFN, HIDDEN),
]
LINEAR_MACS_PER_TOKEN = sum(i * o for _, i, o in PROJECTIONS)

# 768p 16:9 -> 1344x768 -> 84x48 latent -> (84//2)*(48//2) rows per latent frame.
ROWS_768P = 1008
# 512^2 -> 32x32 latent -> 16*16 rows.
ROWS_512 = 256

CONFIGS = [
    ("512^2, T=4 (Phase 3 train)", ROWS_512, 4),
    ("768p, 5s", ROWS_768P, 31),
    ("768p, 10s", ROWS_768P, 61),
    ("768p, 15s", ROWS_768P, 91),
]


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


class OOM(Exception):
    pass


def guard(fn):
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise OOM


def attn_dense(seq, device, dtype, iters):
    q, k, v = (torch.randn(1, HEADS, seq, HEAD_DIM, device=device, dtype=dtype) for _ in range(3))
    ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v), iters=iters)
    del q, k, v
    torch.cuda.empty_cache()
    return ms


def attn_cross(q_len, kv_len, device, dtype, iters):
    q = torch.randn(1, HEADS, q_len, HEAD_DIM, device=device, dtype=dtype)
    k, v = (torch.randn(1, HEADS, kv_len, HEAD_DIM, device=device, dtype=dtype) for _ in range(2))
    ms = timeit(lambda: F.scaled_dot_product_attention(q, k, v), iters=iters)
    del q, k, v
    torch.cuda.empty_cache()
    return ms


def attn_block_causal(seq, rows_per_frame, device, dtype, iters):
    """Encoder pattern: a frame attends to every earlier frame and fully within itself."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx // rows_per_frame) <= (q_idx // rows_per_frame)

    # Compiling create_block_mask is mandatory: the eager path materializes a dense
    # int64 [S, S] intermediate (30 GB at S=62k) and OOMs.
    build_mask = torch.compile(create_block_mask, dynamic=False)
    block_mask = build_mask(mask_mod, None, None, seq, seq, device=device)
    q, k, v = (torch.randn(1, HEADS, seq, HEAD_DIM, device=device, dtype=dtype) for _ in range(3))
    compiled = torch.compile(flex_attention, dynamic=False)
    ms = timeit(lambda: compiled(q, k, v, block_mask=block_mask), warmup=2, iters=iters)
    del q, k, v, block_mask
    torch.cuda.empty_cache()
    return ms


def linear_tile_ms(tile, device, dtype, iters):
    """Per-token linear cost, measured on a tile and reported as ms per token."""
    total = 0.0
    for _, fan_in, fan_out in PROJECTIONS:
        x = torch.randn(tile, fan_in, device=device, dtype=dtype)
        w = torch.randn(fan_out, fan_in, device=device, dtype=dtype)
        total += timeit(lambda: F.linear(x, w), iters=iters)
        del x, w
        torch.cuda.empty_cache()
    return total / tile


def flops_attn(seq_q, seq_kv):
    return 4.0 * seq_q * seq_kv * INNER


def flops_linear(tokens):
    return 2.0 * tokens * LINEAR_MACS_PER_TOKEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 16, 30])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--tile", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-flex", action="store_true", help="skip FlexAttention block-causal")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"device      : {name} ({total_gb:.0f} GB), torch {torch.__version__}, {args.dtype}")
    print(f"dims        : hidden {HIDDEN}, {HEADS}x{HEAD_DIM} (inner {INNER}), ffn {FFN}, {NUM_LAYERS} layers")
    print(f"split       : encoder {ENCODER_LAYERS} / decoder {DECODER_LAYERS}")
    print(f"linear MACs : {LINEAR_MACS_PER_TOKEN / 1e6:.1f}M per token per layer")
    print(f"crossover   : attention overtakes linear at S = {LINEAR_MACS_PER_TOKEN / (2 * INNER):,.0f} tokens\n")

    lin_per_token = linear_tile_ms(args.tile, device, dtype, args.iters)
    lin_check = linear_tile_ms(args.tile * 2, device, dtype, args.iters)
    print(f"linear      : {lin_per_token * 1e3:.3f} us/token/layer "
          f"(linearity check at 2x tile: {lin_check / lin_per_token:.3f}x, want ~1.00)\n")

    results = []
    print(f"{'config':<28} {'S':>8} {'attn share':>11} {'stock s/clip':>13} {'scd s/clip':>11} {'speedup':>8}")
    print("-" * 84)

    for label, rows, frames in CONFIGS:
        seq = rows * frames
        dec_tokens = 2 * rows

        try:
            dense_ms = guard(lambda: attn_dense(seq, device, dtype, args.iters))
        except OOM:
            print(f"{label:<28} {seq:>8,}   OOM on dense attention — skipped")
            continue

        if args.no_flex:
            enc_ms = guard(lambda: attn_dense(seq, device, dtype, args.iters)) / 2.0
            enc_src = "half-dense estimate"
        else:
            try:
                enc_ms = guard(lambda: attn_block_causal(seq, rows, device, dtype, args.iters))
                enc_src = "flex block-causal"
            except Exception as exc:  # flex can fail to compile on some shapes
                torch.cuda.empty_cache()
                enc_ms = dense_ms / 2.0
                enc_src = f"half-dense estimate ({type(exc).__name__})"

        dec_ms = attn_cross(dec_tokens, dec_tokens, device, dtype, args.iters)

        lin_full = lin_per_token * seq
        lin_dec = lin_per_token * dec_tokens

        row = {"config": label, "rows": rows, "frames": frames, "seq": seq,
               "encoder_attn_source": enc_src,
               "ms": {"dense_attn": dense_ms, "encoder_attn": enc_ms, "decoder_attn": dec_ms,
                      "linear_full": lin_full, "linear_decoder": lin_dec},
               "steps": {}}

        attn_share = flops_attn(seq, seq) / (flops_attn(seq, seq) + flops_linear(seq))

        for n_steps in args.steps:
            stock = n_steps * NUM_LAYERS * (dense_ms + lin_full) / 1e3
            scd = (ENCODER_LAYERS * (enc_ms + lin_full)
                   + n_steps * frames * DECODER_LAYERS * (dec_ms + lin_dec)) / 1e3

            stock_flops = n_steps * NUM_LAYERS * (flops_attn(seq, seq) + flops_linear(seq))
            scd_flops = (ENCODER_LAYERS * (flops_attn(seq, seq) / 2 + flops_linear(seq))
                         + n_steps * frames * DECODER_LAYERS
                         * (flops_attn(dec_tokens, dec_tokens) + flops_linear(dec_tokens)))

            row["steps"][n_steps] = {"stock_s": stock, "scd_s": scd, "speedup": stock / scd,
                                     "flop_model_speedup": stock_flops / scd_flops}

        results.append(row)
        mid = args.steps[len(args.steps) // 2]
        s = row["steps"][mid]
        print(f"{label:<28} {seq:>8,} {attn_share * 100:>10.1f}% "
              f"{s['stock_s']:>13.1f} {s['scd_s']:>11.1f} {s['speedup']:>7.2f}x")

    print("\nper-step-count detail (measured / FLOP model):")
    for row in results:
        cells = "  ".join(
            f"N={n}: {row['steps'][n]['speedup']:.2f}x/{row['steps'][n]['flop_model_speedup']:.2f}x"
            for n in args.steps)
        print(f"  {row['config']:<28} {cells}")

    print("\nencoder attention source per config:")
    for row in results:
        print(f"  {row['config']:<28} {row['encoder_attn_source']}")

    if args.json:
        payload = {"device": name, "torch": torch.__version__, "dtype": args.dtype,
                   "linear_ms_per_token": lin_per_token, "results": results,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
