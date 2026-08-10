#!/usr/bin/env python3
"""Leave-one-out layer importance for the H3 x SCD split (docs/MINIMAX_H3_SCD_PORT_DESIGN.md §7).

This is the criterion the SCD paper actually used to pick its encoder/decoder split, reproduced
on H3. arXiv:2602.10095 Fig. 7: "We separately remove each layer in WAN2.1 T2V-1.3B and calculate
the validation diffusion loss averaged across 5 noise levels." They found the earliest and latest
layers cost the most to remove and the middle costs least, which is why their decoder is built
from BOTH ends (layers 0-4 plus 25-29 of 30) rather than being the encoder's tail.

Unlike phase0_probe.py / phase0_validate.py, which compare block features, this scores the real
flow-matching objective through final_layer: MSE of video_out against (x0 - noise), which is
Fizgig's training target. So it measures what a layer is worth to the model's actual output.

Skipping a block is the identity on the residual stream, which is what "remove" means for a
pre-norm residual transformer and is how the layer's own contribution gets zeroed without
disturbing anything downstream.

Cost note: naively this is n_blocks^2 block executions per sigma. Skipping block i only changes
the stack from i onward, so caching each block's INPUT during the baseline pass lets the
leave-one-out run resume at i+1 and costs (n_blocks - 1 - i) instead of n_blocks -- ~2x overall.
Run with --blocks-to-swap 0 if it fits (NF4 is ~9.6 GB for all 50): the resume loop revisits late
blocks constantly, so CPU-parking them turns a compute-bound job into a PCIe-bound one.

Usage:
    python3 scripts/scd/phase0_leaveout.py \
        --checkpoint /path/to/FL2VA/transformer \
        --latents scripts/scd/clip_latents.safetensors --text scripts/scd/clip_te.safetensors \
        --out docs/phase0_leaveout.json
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from phase0_probe import build_stream, import_fizgig, load_latents, load_text, run_block


@torch.no_grad()
def video_loss(model, stream, h, target_rows):
    """Flow-matching MSE at the model's real output head, video rows only."""
    v = model.final_layer(h[stream["video_start"]:], stream["t_emb"], stream["video_t_index"])
    return F.mse_loss(v.float(), target_rows.float()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--latent-t", type=int, default=7,
                    help="must sit on the 5n+2 grid (2, 7, 12, ...) — see pixel_frames_for_latent")
    ap.add_argument("--u-grid", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="the paper's 5 noise levels, drawn on H3's OWN density: "
                         "sigma = shift_sigma(u, 12), so u is uniform where training is uniform")
    ap.add_argument("--sigmas", type=float, nargs="+", default=None,
                    help="raw sigmas, bypassing the shift map. Off-distribution below ~0.5 — "
                         "H3 trains ~3%% of steps under sigma 0.3 and the model is anticorrelated "
                         "with the target there, so a flat 0.1..0.9 grid scores mostly noise.")
    ap.add_argument("--blocks-to-swap", type=int, default=0)
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--out", default="docs/phase0_leaveout.json")
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
    n_blocks = len(model.blocks)

    sigmas = args.sigmas or [mm.shift_sigma(u, mm.VIDEO_SIGMA_SHIFT) for u in args.u_grid]
    print(f"sigmas: {[round(s, 4) for s in sigmas]}"
          f"{'' if args.sigmas else f' (from u={args.u_grid} through shift {mm.VIDEO_SIGMA_SHIFT})'}",
          flush=True)

    t0 = time.time()
    base_by_sigma, loo_by_sigma = {}, {}
    for sigma in sigmas:
        st = build_stream(mm, model, video_latent, text_embeds, sigma, device, dtype)
        target = mm.patchify_video(video_latent - st["eps"], model.patch_size)

        # Baseline pass, caching each block's input so the leave-one-out runs can resume.
        h = st["h"]
        h_in = []
        for i, block in enumerate(model.blocks):
            h_in.append(h)
            swapped = i >= model._swap_from
            if swapped:
                block.to(device)
            h = run_block(mm, block, h, st["t_emb"], st["mod_row"], st["cos"], st["sin"], None)
            if swapped:
                block.to("cpu")
        base = video_loss(model, st, h, target)

        losses = []
        for i in range(n_blocks):
            h = h_in[i]                                  # skip block i: residual passes through
            for j in range(i + 1, n_blocks):
                block = model.blocks[j]
                swapped = j >= model._swap_from
                if swapped:
                    block.to(device)
                h = run_block(mm, block, h, st["t_emb"], st["mod_row"], st["cos"], st["sin"], None)
                if swapped:
                    block.to("cpu")
            losses.append(video_loss(model, st, h, target))
        base_by_sigma[sigma] = base
        loo_by_sigma[sigma] = losses
        del h_in, st
        torch.cuda.empty_cache()
        print(f"sigma={sigma}: baseline {base:.5f}  worst block {max(range(n_blocks), key=lambda k: losses[k])}"
              f" ({max(losses):.5f})  best {min(range(n_blocks), key=lambda k: losses[k])}"
              f" ({min(losses):.5f})  [{(time.time() - t0) / 60:.1f} min]", flush=True)

    ns = len(sigmas)
    base_mean = sum(base_by_sigma.values()) / ns
    # Relative cost of removing a layer, averaged over noise levels: the paper's y-axis is the
    # validation loss itself, but normalising by the baseline makes the sigmas commensurate
    # (loss magnitude varies several-fold across the sigma grid).
    rel = [sum(loo_by_sigma[s][i] / base_by_sigma[s] - 1.0 for s in sigmas) / ns
           for i in range(n_blocks)]

    order = sorted(range(n_blocks), key=lambda i: rel[i], reverse=True)
    print(f"\nbaseline loss (mean over sigmas): {base_mean:.5f}")
    print(f"\n{'blk':>4} {'rel cost':>10}   {'blk':>4} {'rel cost':>10}   {'blk':>4} {'rel cost':>10}")
    half = (n_blocks + 2) // 3
    for r in range(half):
        cells = []
        for c in range(3):
            i = r + c * half
            cells.append(f"{i:>4} {rel[i]:>10.4f}" if i < n_blocks else " " * 15)
        print("   ".join(cells))

    print(f"\nmost load-bearing (top 10): {order[:10]}")
    print(f"least load-bearing (bottom 10): {sorted(order[-10:])}")
    thirds = [sum(rel[a:b]) / (b - a) for a, b in
              [(0, n_blocks // 3), (n_blocks // 3, 2 * n_blocks // 3), (2 * n_blocks // 3, n_blocks)]]
    print(f"mean rel cost by third: early {thirds[0]:.4f}  middle {thirds[1]:.4f}  late {thirds[2]:.4f}")
    print(f"paper's shape is early-high, middle-low, late-high -> "
          f"{'MATCHES' if thirds[1] < thirds[0] and thirds[1] < thirds[2] else 'DOES NOT MATCH'}")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")

    payload = {
        "checkpoint": os.path.basename(args.checkpoint.rstrip("/")), "sigmas": sigmas,
        "u_grid": None if args.sigmas else args.u_grid, "latent_t": args.latent_t,
        "n_blocks": n_blocks, "baseline_loss_by_sigma": {str(k): v for k, v in base_by_sigma.items()},
        "leaveout_loss_by_sigma": {str(k): v for k, v in loo_by_sigma.items()},
        "relative_cost": rel, "ranked_most_important": order,
        "mean_rel_cost_thirds": thirds, "elapsed_s": time.time() - t0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
