#!/usr/bin/env python3
"""Validate LoT timing on your GPU: one DiT forward, dense vs each LoT layout.

    python3 scripts/lot/time_lot_h3.py --shape still                    # 768x1152, 1 frame
    python3 scripts/lot/time_lot_h3.py --shape clip --swap 4            # 512x768, 22 frames
    python3 scripts/lot/time_lot_h3.py --shape long --swap 48           # 768x1344, 124 frames
    python3 scripts/lot/time_lot_h3.py --trained runs/lot_hf/run4_nikki_distill --shape still

Inputs are random tensors: compute does not depend on content. ``--trained`` loads
``adapter.safetensors`` / ``adapter.pt`` plus ``lora.safetensors`` (an HF download
works) so the LoRA's small extra cost is included; without it an untrained adapter
is built from the checkpoint's own head weights (same cost). Prints one
``LOT_H3 kind=lot_timing`` line per layout with the median of ``--repeats`` forwards
after a warmup, and the ratio to dense. Writes nothing.

Swap streams blocks from CPU and costs ~0.4 s per block per forward, which hides
the LoT gain on small inputs: use ``--swap 0`` when the whole DiT fits (a still
does on 24 GB), ``--swap 4`` for 22-frame clips, ``--swap 48`` for the long clip.
Math SDPA is turned off (it is quadratic in memory at these lengths).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import FIZGIG_SRC, LOT_H3_CHECKPOINT  # noqa: E402

sys.path.insert(0, FIZGIG_SRC)

from gpu_guard import refuse_if_busy  # noqa: E402
from h3 import GRID_LAYOUTS, grid_layout, make_h3_adapter  # noqa: E402
from h3_splice import LotSplice  # noqa: E402

SHAPES = {  # name: (width, height, pixel frames)
    "still": (768, 1152, 1),
    "clip": (512, 768, 22),
    "long": (1344, 768, 124),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shape", choices=list(SHAPES), default="still")
    parser.add_argument("--layouts", default="center,bands,uniform2", help=f"comma list from {GRID_LAYOUTS}")
    parser.add_argument("--trained", type=Path, default=None, help="run dir or HF download folder")
    parser.add_argument("--swap", type=int, default=0, help="DiT blocks streamed from CPU")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--text", type=int, default=128, help="text rows (cost of the prompt)")
    args = parser.parse_args()

    refuse_if_busy("time_lot_h3.py")
    torch.backends.cuda.enable_math_sdp(False)
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.model import audio_latents_for_frames, latent_frames_for_pixels

    width, height, frames = SHAPES[args.shape]
    latent_t = 1 if frames == 1 else latent_frames_for_pixels(frames)
    model = load_minimax_h3_dit(str(LOT_H3_CHECKPOINT), device="cuda", compute_dtype=torch.bfloat16,
                                base_quant="int8", blocks_to_swap=args.swap)
    if args.swap:
        model.enable_block_swap(args.swap, h2d_only=True)
    model.eval()
    model._tread = None
    model._lot = None

    adapter = make_h3_adapter()
    network = None
    if args.trained is not None:
        from fizgig.networks.lora import create_network
        from train_h3 import LORA_PATTERNS, load_adapter_state

        adapter.load_state_dict(load_adapter_state(args.trained))
        network = create_network(None, "lora_unet", 1.0, 16, 16.0, None, [], model,
                                 include_patterns=LORA_PATTERNS)
        network.apply_to(text_encoders=None, unet=model, apply_text_encoder=False, apply_unet=True)
        network.load_weights(str(args.trained / "lora.safetensors"))
        network.to(device="cuda", dtype=torch.bfloat16).eval()
    else:
        from procrustes_h3 import pretrained_maps

        adapter.init_from_pretrained(*pretrained_maps(LOT_H3_CHECKPOINT))
    adapter.float().cuda().eval()

    video = torch.randn(1, 24, latent_t, height // 16, width // 16, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, args.text, model.hidden_size, device="cuda", dtype=torch.bfloat16)
    n_audio = audio_latents_for_frames(frames)
    audio = torch.randn(n_audio * 2, model.config.audio_latents_dim, device="cuda")
    t = torch.tensor(0.6, device="cuda")
    seen = {}

    def remember(_module, inputs, _output) -> None:
        seen["seq"] = int(inputs[0].shape[0])

    hook = model.blocks[0].register_forward_hook(remember)

    def timed(splice) -> tuple[float, int]:
        model._lot = splice
        try:
            with torch.inference_mode():
                model(video, t, text, audio_rows=audio)
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    model(video, t, text, audio_rows=audio)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - start)
        finally:
            model._lot = None
        samples.sort()
        return samples[len(samples) // 2] * 1000.0, seen["seq"]

    shape_text = f"{width}x{height}x{frames}"
    dense_ms, dense_seq = timed(None)
    print(f"LOT_H3 kind=lot_timing shape={shape_text} latent_t={latent_t} layout=dense seq={dense_seq} "
          f"ms={dense_ms:.0f} ratio=1.00 swap={args.swap} trained={int(network is not None)}", flush=True)
    for name in [n for n in args.layouts.split(",") if n]:
        layout = grid_layout(name, latent_t, height // 32, width // 32)
        ms, seq = timed(LotSplice(adapter, layout, y_space=True))
        print(f"LOT_H3 kind=lot_timing shape={shape_text} latent_t={latent_t} layout={name} seq={seq} "
              f"ms={ms:.0f} ratio={dense_ms / ms:.2f} swap={args.swap} trained={int(network is not None)}",
              flush=True)
    hook.remove()
    print(f"LOT_H3 kind=lot_timing_done peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
