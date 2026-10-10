#!/usr/bin/env python3
"""Time one mixed LoT tail against the dense Fizgig forward.

Writes nothing. Does not call empty_cache. Refuses while another LoT GPU script is alive.

    python3 scripts/lot/bench_h3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import FIZGIG_SRC, LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401
sys.path.insert(0, FIZGIG_SRC)

from h3 import clip_layout, make_h3_adapter  # noqa: E402
from gpu_guard import refuse_if_busy  # noqa: E402
from h3_splice import LotSplice  # noqa: E402

CHECKPOINT = LOT_H3_CHECKPOINT
# 32 leaves the eager int8 MLP cast (S, 28672) fp32 with about 1.8 GB free on this
# 24 GB card, and the 37-frame dense pack needs 4.1 GB. 48 is the loader maximum
# (keep two blocks resident) and is what makes that allocation fit.
BLOCKS_TO_SWAP = 48


def refuse_math_sdpa() -> None:
    """Math SDPA is quadratic in sequence length. The 37-frame pack is ~38k tokens."""
    torch.backends.cuda.enable_math_sdp(False)
    flash = torch.backends.cuda.flash_sdp_enabled()
    mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    print(
        f"LOT_H3 kind=attn flash={int(flash)} mem_efficient={int(mem)} math=0",
        flush=True,
    )
    if not flash and not mem:
        raise SystemExit("no flash or mem-efficient SDPA; refusing the math kernel")



def alloc_retries() -> int:
    return int(torch.cuda.memory_stats().get("num_alloc_retries", 0))


def timed(fn, label: str, repeats: int = 3) -> tuple[float, int]:
    """Median ms over ``repeats``, and allocator retries inside the timed steps only.

    A retry frees the cache and reallocates, so a nonzero count inflates that side.
    """
    print(f"LOT_H3 kind=progress phase={label} step=warmup", flush=True)
    fn()
    torch.cuda.synchronize()
    retries_before = alloc_retries()
    samples = []
    for index in range(repeats):
        print(f"LOT_H3 kind=progress phase={label} step={index}", flush=True)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1000.0, alloc_retries() - retries_before


def cuda_bytes(module) -> int:
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.is_cuda:
            total += tensor.numel() * tensor.element_size()
    return total


def run_clip(model, frames: int, text_rows: int) -> None:
    from fizgig.minimax.model import audio_latents_for_frames, pixel_frames_for_latent

    layout = clip_layout(frames)
    height, width = 48, 84
    video = torch.randn(1, 24, frames, height, width, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, text_rows, model.hidden_size, device="cuda", dtype=torch.bfloat16)
    n_audio = audio_latents_for_frames(pixel_frames_for_latent(frames))
    audio = torch.randn(n_audio * 2, model.config.audio_latents_dim, device="cuda", dtype=torch.float32)
    clock = torch.tensor(0.6, device="cuda")
    adapter = make_h3_adapter()
    adapter.init_from_pretrained(
        model.video_patch_proj.weight.detach().float().cpu(),
        model.final_layer.video_out.weight.detach().float().cpu(),
        model.video_patch_proj.bias.detach().float().cpu(),
        model.final_layer.video_out.bias.detach().float().cpu(),
    )
    adapter.cuda()
    splice = LotSplice(adapter, layout)
    seen: dict[str, tuple[int, ...]] = {}

    def remember(_module, inputs, _output):
        key = "lot" if model._lot is not None else "dense"
        seen.setdefault(key, tuple(inputs[0].shape))

    hook = model.blocks[0].register_forward_hook(remember)

    def dense():
        model._lot = None
        return model(video, clock, text, audio_rows=audio)

    def mixed():
        model._lot = splice
        return model(video, clock, text, audio_rows=audio)

    try:
        with torch.inference_mode():
            dense_ms, dense_retries = timed(dense, "dense")
            lot_ms, lot_retries = timed(mixed, "lot")
    finally:
        hook.remove()
        model._lot = None
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    packed_dense = text_rows + n_audio * 2 + layout.dense_count
    packed_lot = text_rows + n_audio * 2 + layout.count
    print(
        "LOT_H3 kind=timing task=t2va "
        f"latent_t={frames} swap={BLOCKS_TO_SWAP} "
        f"dense_tokens={packed_dense} lot_tokens={packed_lot} "
        f"dense_seq={seen.get('dense')} lot_seq={seen.get('lot')} "
        f"dense_ms={dense_ms:.1f} lot_ms={lot_ms:.1f} "
        f"ratio={dense_ms / lot_ms:.3f} dense_retries={dense_retries} lot_retries={lot_retries} "
        f"peak_mb={peak_mb:.1f} dtype=bf16 ok=1",
        flush=True,
    )


def main() -> None:
    refuse_if_busy("bench_h3.py")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {CHECKPOINT}")
    refuse_math_sdpa()
    from fizgig.minimax.loader import load_minimax_h3_dit

    model = load_minimax_h3_dit(
        str(CHECKPOINT),
        device="cuda",
        compute_dtype=torch.bfloat16,
        base_quant="int8",
        blocks_to_swap=BLOCKS_TO_SWAP,
    )
    model.enable_block_swap(BLOCKS_TO_SWAP, h2d_only=True)
    model.eval()
    model._tread = None
    head = cuda_bytes(model.blocks[0]) + cuda_bytes(model.blocks[1])
    tail = sum(cuda_bytes(block) for block in model.blocks[2:])
    alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    print(
        f"LOT_H3 kind=resident swap={BLOCKS_TO_SWAP} "
        f"head_mb={head / (1024 * 1024):.1f} tail_mb={tail / (1024 * 1024):.1f} "
        f"allocated_mb={alloc:.1f}",
        flush=True,
    )
    try:
        run_clip(model, 37, 512)
    except torch.cuda.OutOfMemoryError:
        print(
            f"LOT_H3 kind=timing task=t2va latent_t=37 swap={BLOCKS_TO_SWAP} ok=0 error=oom",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
