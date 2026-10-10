#!/usr/bin/env python3
"""Closed 1x1 parity against Fizgig on one small latent.

Loads the pruned int8 checkpoint with block streaming. Writes no file.
Refuses while another LoT GPU script is alive. Does not call empty_cache.

    python3 scripts/lot/parity_h3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import FIZGIG_SRC, LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401
sys.path.insert(0, FIZGIG_SRC)

from h3 import make_h3_adapter  # noqa: E402
from gpu_guard import refuse_if_busy  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from layout import dense_layout  # noqa: E402

CHECKPOINT = LOT_H3_CHECKPOINT



def rel_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).pow(2).mean().sqrt()
    scale = right.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return float(delta / scale)


def main() -> None:
    refuse_if_busy("parity_h3.py")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {CHECKPOINT}")
    from fizgig.minimax.loader import load_minimax_h3_dit

    model = load_minimax_h3_dit(
        str(CHECKPOINT),
        device="cuda",
        compute_dtype=torch.bfloat16,
        base_quant="int8",
        blocks_to_swap=32,
    )
    model.enable_block_swap(32, h2d_only=True)
    model.eval()
    model._lot = None
    model._tread = None

    height = width = 16
    video = torch.randn(1, 24, 1, height, width, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 4, model.hidden_size, device="cuda", dtype=torch.bfloat16)
    audio = torch.randn(4, model.config.audio_latents_dim, device="cuda", dtype=torch.float32)
    time = torch.tensor(0.6, device="cuda")

    with torch.inference_mode():
        off_a = model(video, time, text, audio_rows=audio)
        off_b = model(video, time, text, audio_rows=audio)
    floor = rel_rms(off_a, off_b)

    adapter = make_h3_adapter()
    adapter.init_from_pretrained(
        model.video_patch_proj.weight.detach().float().cpu(),
        model.final_layer.video_out.weight.detach().float().cpu(),
        model.video_patch_proj.bias.detach().float().cpu(),
        model.final_layer.video_out.bias.detach().float().cpu(),
    )
    adapter.cuda()
    layout = dense_layout(1, height // 2, width // 2)
    model._lot = LotSplice(adapter, layout)
    with torch.inference_mode():
        on = model(video, time, text, audio_rows=audio)
    gap = rel_rms(on, off_a)
    max_abs = float((on.float() - off_a.float()).abs().max())
    limit = max(1e-4, 2.0 * floor)
    ok = int(gap <= limit and max_abs <= 1e-2 and floor <= 1e-3)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        "LOT_H3 kind=parity task=t2va "
        f"dense_tokens={layout.dense_count} lot_tokens={layout.count} "
        f"rel_rms={gap:.6g} rel_rms_floor={floor:.6g} max_abs={max_abs:.6g} "
        f"peak_mb={peak_mb:.1f} dtype=bf16 ok={ok}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
