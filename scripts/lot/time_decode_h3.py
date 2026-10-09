#!/usr/bin/env python3
"""Time one H3 VAE clip decode at the LoT timing shape. Writes nothing.

    python3 scripts/lot/time_decode_h3.py [--latent-t 37] [--height 48] [--width 84]

LoT shortens the DiT sequence only; the VAE decodes the full latent either way.
This measures that fixed cost on the same latent shape the DiT timing used
(768×1344, ``latent_t=37`` -> 124 pixel frames). Content does not change the
cost, so the latent is random. fp16 decoder, as Fizgig and ComfyUI load it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/media/2TB/Fizgig/src")

from gpu_guard import refuse_if_busy  # noqa: E402

VAE = Path("/media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--latent-t", type=int, default=37)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=84)
    args = parser.parse_args()

    refuse_if_busy("time_decode_h3.py")
    from safetensors import safe_open
    from fizgig.minimax.model import pixel_frames_for_latent
    from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder

    decoder = MiniMaxH3VideoVAEDecoder()
    with safe_open(str(VAE), framework="pt", device="cpu") as handle:
        decoder.load_state_dict({key: handle.get_tensor(key) for key in handle.keys()}, strict=False)
    decoder = decoder.to("cuda", torch.float16).eval()
    latent = torch.randn(1, 24, args.latent_t, args.height, args.width, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        pixels = decoder.decode_clip(latent)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        f"LOT_H3 kind=vae_decode latent=({args.latent_t},{args.height},{args.width}) "
        f"pixels={tuple(pixels.shape)} frames={pixel_frames_for_latent(args.latent_t)} "
        f"decode_ms={seconds * 1000:.1f} peak_mb={peak_mb:.1f} dtype=fp16 ok=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
