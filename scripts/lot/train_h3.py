#!/usr/bin/env python3
"""Phase 5: fine-tune H3 for Level-of-Token layouts. Stills first.

    LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs DIR [--steps 2000] [--swap 4]

What trains: the LoT adapter (per-extent ``in_proj`` / ``out_proj``, shape MLP)
and a rank-16 LoRA on ``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``,
``mlp.fc2`` in every block. AdaLN, the token refiner, the int8 base, and the
extent bank (``A``, ``s`` from ``--pairs``) stay frozen.

One step: sample a layout, move the clean latent to y-space (``to_y``), draw
sigma with H3's own training density (Fizgig ``sample_sigmas``), noise, run the
DiT with the LoT tail, and take ``lot_h3_clean_loss`` (eq. 17 in H3's
``x0 - eps`` convention: velocity MSE in y-space). About 20% of steps use the
dense 1×1 layout, which is the ordinary H3 LoRA loss and anchors the base.

Data: a Fizgig H3 cache directory of still latents ``*_minimaxh3.safetensors``
with their ``*_minimaxh3_te.safetensors`` text, so Qwen is never loaded. Each
latent is randomly cropped to a token grid divisible by 4. Clips come later,
once cached at 384×640, ``latent_t = 7``.

Writes only under ``--out`` (default ``scripts/lot/runs/train_h3``, gitignored):
``log.jsonl``, and at ``--save-every`` and the end ``adapter.pt`` +
``lora.safetensors``. Refuses to start unless ``LOT_H3_TRAIN=1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/media/2TB/Fizgig/src")

from flow import lot_h3_clean_loss, sample_noisy  # noqa: E402
from h3 import H3_EXTENTS, make_h3_adapter, patchify  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from layout import LotLayout, TokenRect, dense_layout, layout_from_rects  # noqa: E402

CACHE = Path("/media/2TB/lora-data/fizgig_minimax_h3/cache_iso3d")
LORA_PATTERNS = [
    r"blocks\.\d+\.attn\.qkv_proj",
    r"blocks\.\d+\.attn\.out_proj",
    r"blocks\.\d+\.mlp\.fc1",
    r"blocks\.\d+\.mlp\.fc2",
]
SUPER = 4  # layout super-cell, in tokens: every extent tiles a 4×4 cell
FINE = [(1, 1, 1)]
MID = [(1, 1, 2), (1, 2, 1), (1, 2, 2)]
COARSE = [(1, 2, 4), (1, 4, 2), (1, 4, 4), (1, 1, 4), (1, 4, 1)]


# ---------------------------------------------------------------------------------- data

def list_items(cache: Path, holdout: int) -> tuple[list, list]:
    """``(latent_path, te_path)`` pairs, split deterministically into train / held out."""
    items = []
    for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
        stem = re.sub(r"_\d{4}x\d{4}_minimaxh3\.safetensors$", "", latent.name)
        te = cache / f"{stem}_minimaxh3_te.safetensors"
        if te.is_file():
            items.append((latent, te))
    if len(items) <= holdout:
        raise SystemExit(f"{cache}: {len(items)} items with text, need more than {holdout}")
    rng = random.Random(0)
    rng.shuffle(items)
    return items[holdout:], items[:holdout]


def load_item(item, rng: random.Random, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean latent ``(1, 24, 1, H, W)`` cropped to a 4-token multiple, and text ``(1, L, 5120)``."""
    from safetensors import safe_open

    latent_path, te_path = item
    with safe_open(str(latent_path), "pt") as handle:
        key = next(k for k in handle.keys() if k.startswith("latent"))
        latent = handle.get_tensor(key).float()
    with safe_open(str(te_path), "pt") as handle:
        hidden = handle.get_tensor("hidden_states")
        mask = handle.get_tensor("attention_mask")
    if latent.ndim == 3:
        latent = latent.unsqueeze(1)                      # (24, 1, H, W)
    step = 2 * SUPER                                       # latent pixels per super-cell
    height, width = latent.shape[-2] // step * step, latent.shape[-1] // step * step
    top = rng.randint(0, latent.shape[-2] - height)
    left = rng.randint(0, latent.shape[-1] - width)
    latent = latent[..., top:top + height, left:left + width].unsqueeze(0)
    return latent.to(device), hidden[mask].unsqueeze(0).to(device, torch.bfloat16)


# -------------------------------------------------------------------------------- layout

def token_detail(latent: torch.Tensor) -> torch.Tensor:
    """``(T, H, W)`` detail per token: distance from its 4×4-cell mean, on the clean latent."""
    tokens = patchify(latent.float())[0]                   # (T, H, W, 96)
    grid = tokens.permute(0, 3, 1, 2)                       # (T, 96, H, W)
    coarse = F.avg_pool2d(grid, SUPER)
    coarse = F.interpolate(coarse, scale_factor=SUPER, mode="nearest")
    return (grid - coarse).norm(dim=1)


def sample_layout(latent: torch.Tensor, rng: random.Random, dense_p: float = 0.2) -> tuple[LotLayout, str]:
    """A layout for one example and its kind.

    ``dense``: all 1×1. ``uniform``: one extent everywhere. ``mosaic``: each 4×4
    super-cell takes one extent, finer where the clean latent has more detail,
    with jittered thresholds, so mixed boundaries and rectangles both appear.
    """
    _b, _c, time, lat_h, lat_w = latent.shape
    height, width = lat_h // 2, lat_w // 2
    roll = rng.random()
    if roll < dense_p:
        return dense_layout(time, height, width), "dense"
    if roll < dense_p + 0.15:
        extent = rng.choice(MID + COARSE)
        return _tile(time, height, width, lambda _t, _u, _v: extent), "uniform"
    detail = token_detail(latent).cpu()
    cells = F.avg_pool2d(detail.unsqueeze(1), SUPER).squeeze(1)    # (T, H/4, W/4)
    flat = cells.flatten()
    low_q, high_q = sorted((rng.uniform(0.2, 0.6), rng.uniform(0.5, 0.9)))
    low, high = float(flat.quantile(low_q)), float(flat.quantile(high_q))

    def pick(t: int, u: int, v: int):
        value = float(cells[t, u // SUPER, v // SUPER])
        if value >= high:
            return FINE[0]
        if value >= low:
            return rng.choice(MID)
        return rng.choice(COARSE)

    return _tile(time, height, width, pick), "mosaic"


def _tile(time: int, height: int, width: int, choose) -> LotLayout:
    rects = []
    for t in range(time):
        for u0 in range(0, height, SUPER):
            for v0 in range(0, width, SUPER):
                _et, eh, ew = choose(t, u0, v0)
                for u in range(u0, u0 + SUPER, eh):
                    for v in range(v0, v0 + SUPER, ew):
                        rects.append(TokenRect(t, u, v, 1, eh, ew))
    return layout_from_rects(time, height, width, rects)


def extent_losses(out, y_t, y0, sigma, layout: LotLayout) -> dict[str, float]:
    """Held-out squared clean error per extent, on the token lattice."""
    sig = float(sigma)
    err = (y_t.float() + sig * out.float() - y0.float()).square() / max(sig, 0.05) ** 2
    tokens = patchify(err)[0].mean(dim=-1)                  # (T, H, W)
    sums: dict[str, list] = {}
    for rect in layout.rects:
        block = tokens[rect.t, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew]
        sums.setdefault("x".join(map(str, rect.extent)), []).append(float(block.mean()))
    return {key: sum(values) / len(values) for key, values in sums.items()}


# --------------------------------------------------------------------------------- train

def check(adapter, dit, network, groups, items, rng, device, args) -> None:
    """Forward + backward once per layout kind. No optimizer step, nothing written."""
    from fizgig.minimax.trainer import sample_sigmas

    params = [p for g in groups for p in g["params"]]
    for kind_wanted in ("dense", "mosaic", "uniform"):
        for _ in range(50):
            x0, text = load_item(rng.choice(items), rng, device)
            layout, kind = sample_layout(x0, rng, args.dense_p)
            if kind == kind_wanted:
                break
        splice = LotSplice(adapter, layout, y_space=True)
        y0 = splice.to_y(x0)
        sigma = sample_sigmas(1, device)
        y_t, _ = sample_noisy(y0, sigma)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        dit._lot = splice
        try:
            out = dit(y_t, 1.0 - sigma, text)
        finally:
            dit._lot = None
        loss = lot_h3_clean_loss(out, y_t, y0, sigma)
        loss.backward()
        torch.cuda.synchronize()
        lora_grad = sum(float(p.grad.abs().sum()) for p in groups[0]["params"] if p.grad is not None)
        adapter_grad = sum(float(p.grad.abs().sum()) for p in groups[-1]["params"] if p.grad is not None)
        for p in params:
            p.grad = None
        print(f"LOT_H3 kind=train_check layout={kind} tokens={layout.count}/{layout.dense_count} "
              f"loss={float(loss.detach()):.4f} sigma={float(sigma):.3f} lora_grad={lora_grad:.3g} "
              f"adapter_grad={adapter_grad:.3g} seconds={time.perf_counter() - start:.2f} "
              f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok={int(math.isfinite(float(loss.detach())))}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, required=True, help="Procrustes pair dir for the frozen bank")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--swap", type=int, default=4, help="blocks streamed from CPU; >0 turns on checkpointing")
    parser.add_argument("--dense-p", type=float, default=0.2)
    parser.add_argument("--holdout", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=0, help="0 = save only at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "train_h3")
    parser.add_argument("--check", action="store_true",
                        help="one forward + backward per layout kind, no optimizer step, writes nothing; "
                             "allowed without LOT_H3_TRAIN")
    args = parser.parse_args()

    if not args.check and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("phase 5 is gated: set LOT_H3_TRAIN=1 to train (or --check for a dry pass)")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_h3.py")
    from procrustes_h3 import CHECKPOINT, fit_bank, load_pairs, pretrained_maps
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.trainer import sample_sigmas
    from fizgig.networks.lora import create_network

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_items, held_items = list_items(args.cache, args.holdout)

    adapter = make_h3_adapter()
    adapter.init_from_pretrained(*pretrained_maps(CHECKPOINT))
    fit_bank(adapter, load_pairs(args.pairs))
    adapter.to(device)
    adapter.bank.requires_grad_(False)

    dit = load_minimax_h3_dit(str(CHECKPOINT), device="cuda", compute_dtype=torch.bfloat16,
                              base_quant="int8", blocks_to_swap=args.swap)
    dit.requires_grad_(False)
    if args.swap:
        dit.enable_block_swap(args.swap, h2d_only=True)
        dit.enable_gradient_checkpointing(True)
    dit._tread = None
    network = create_network(None, "lora_unet", 1.0, args.rank, float(args.rank), None, [], dit,
                             include_patterns=LORA_PATTERNS)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=torch.bfloat16)
    expected = len(LORA_PATTERNS) * len(dit.blocks)
    if len(network.unet_loras) != expected:
        raise RuntimeError(f"{len(network.unet_loras)} LoRA modules wrapped, expected {expected} "
                           f"(4 linears x {len(dit.blocks)} blocks)")

    lora_groups, _ = network.prepare_optimizer_params(args.lr)
    adapter_params = [p for name, p in adapter.named_parameters() if not name.startswith("bank.")]
    groups = lora_groups + [{"params": adapter_params, "lr": args.adapter_lr}]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
        opt_name = "adamw8bit"
    except ImportError:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        opt_name = "adamw"
    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"LOT_H3 kind=train_start items={len(train_items)} holdout={len(held_items)} "
          f"lora_modules={len(network.unet_loras)} trainable={trainable} optimizer={opt_name} "
          f"swap={args.swap} allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    if args.check:
        check(adapter, dit, network, groups, train_items, rng, device, args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")

    def step_loss(item, item_rng, *, layout=None, sigma=None, noise_seed=None):
        x0, text = load_item(item, item_rng, device)
        if layout is None:
            layout, kind = sample_layout(x0, item_rng, args.dense_p)
        else:
            kind = "fixed"
        splice = LotSplice(adapter, layout, y_space=True)
        y0 = splice.to_y(x0)
        if sigma is None:
            sigma = sample_sigmas(1, device)
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(1)
        gen = None if noise_seed is None else torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(y0.shape, device=device, generator=gen)
        y_t, _ = sample_noisy(y0, sigma, noise)
        dit._lot = splice
        try:
            out = dit(y_t, 1.0 - sigma, text)
        finally:
            dit._lot = None
        return lot_h3_clean_loss(out, y_t, y0, sigma), out, y_t, y0, sigma, layout, kind

    def evaluate(step: int) -> None:
        network.eval()
        results = {"dense": [], "uniform2": [], "mosaic": []}
        per_extent: dict[str, list] = {}
        with torch.no_grad():
            for index, item in enumerate(held_items):
                for kind in results:
                    item_rng = random.Random(1000 + index)
                    x0, _text = load_item(item, random.Random(1000 + index), device)
                    _b, _c, tt, hh, ww = x0.shape
                    if kind == "dense":
                        layout = dense_layout(tt, hh // 2, ww // 2)
                    elif kind == "uniform2":
                        layout = _tile(tt, hh // 2, ww // 2, lambda *_: (1, 2, 2))
                    else:
                        layout, _ = sample_layout(x0, random.Random(2000 + index), dense_p=0.0)
                    sigma = (0.3, 0.6, 0.9)[index % 3]
                    loss, out, y_t, y0, sig, lay, _ = step_loss(item, item_rng, layout=layout,
                                                                 sigma=sigma, noise_seed=index)
                    results[kind].append(float(loss))
                    if kind == "mosaic":
                        for key, value in extent_losses(out, y_t, y0, sig, lay).items():
                            per_extent.setdefault(key, []).append(value)
        network.train()
        record = {"step": step, "eval": {k: sum(v) / len(v) for k, v in results.items()},
                  "eval_extent": {k: sum(v) / len(v) for k, v in sorted(per_extent.items())}}
        log.write(json.dumps(record) + "\n")
        log.flush()
        line = " ".join(f"{k}={v:.4f}" for k, v in record["eval"].items())
        print(f"LOT_H3 kind=train_eval step={step} {line}", flush=True)

    def save(tag: str) -> None:
        torch.save({k: v for k, v in adapter.state_dict().items()}, args.out / f"adapter{tag}.pt")
        network.save_weights(str(args.out / f"lora{tag}.safetensors"), torch.bfloat16,
                             {"lot_h3": "phase5", "rank": str(args.rank)})
        print(f"LOT_H3 kind=train_saved tag={tag or 'final'} out={args.out}", flush=True)

    network.train()
    evaluate(0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        item = rng.choice(train_items)
        loss, _out, _y_t, _y0, sigma, layout, kind = step_loss(item, rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"step {step}: loss {float(loss)}")
        optimizer.step()
        # Free the grads now, not at the next step: evaluation and saving run between
        # steps, and ~350 MB of live grads on top of the Adam state OOMed the step-250 eval.
        optimizer.zero_grad(set_to_none=True)
        record = {"step": step, "loss": float(loss), "sigma": float(sigma), "kind": kind,
                  "tokens": layout.count, "dense": layout.dense_count, "grad_norm": float(grad_norm),
                  "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if step % 25 == 0:
            log.flush()
            print(f"LOT_H3 kind=train step={step} loss={record['loss']:.4f} sigma={record['sigma']:.3f} "
                  f"layout={kind} tokens={layout.count}/{layout.dense_count} "
                  f"s_per_step={(time.perf_counter() - started) / step:.2f} "
                  f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            save(f"_{step}")
    save("")
    print(f"LOT_H3 kind=train_done steps={args.steps} seconds={time.perf_counter() - started:.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
