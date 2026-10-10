#!/usr/bin/env python3
"""Phase 5: fine-tune H3 for Level-of-Token layouts. Stills first.

    LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs DIR [--steps 2000] [--swap 4]

What trains: the LoT adapter (per-extent ``in_proj`` / ``out_proj``, shape MLP)
and a rank-16 LoRA on ``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``,
``mlp.fc2`` in every block. AdaLN, the token refiner, the int8 base, and the
extent bank (``A``, ``s`` from ``--pairs``) stay frozen.

One step: sample a layout, move the clean latent to y-space (``to_y``), draw
sigma (Fizgig ``sample_sigmas``), noise, run the DiT with the LoT tail, and
take ``lot_h3_clean_loss`` (eq. 17 in H3's ``x0 - eps`` convention: velocity
MSE in y-space). About 20% of steps use the dense 1×1 layout, which is the
ordinary H3 LoRA loss and anchors the base.

Sigma: ``--dense-shift`` for the dense anchor steps (default 12, H3's own
density: median 0.92, ~3% of steps below 0.3) and ``--shift`` for LoT steps.
The first 2,000-step run used 12 for both and the coarse regions came out
soft: detail is decided at low sigma, which shift 12 almost never trains.
A float is the uniform-u shift map, ``sigmoid`` is logit-normal, ``lognorm:S``
is logit-normal under shift S (see Fizgig ``sample_sigmas``).

Data: one or more Fizgig H3 cache directories (``--cache``, repeatable) of
``*_minimaxh3.safetensors`` latents with ``*_minimaxh3_te.safetensors`` text,
so Qwen is never loaded here. Stills are ``(24, H, W)``; clips are
``(24, T, H, W)`` (cache 384×640, 22 frames for ``latent_t = 7``). Each latent
is randomly cropped to a token grid divisible by 4. ``--init DIR`` warm-starts
the adapter and LoRA from an earlier run's ``adapter.pt`` / ``lora.safetensors``
with a fresh optimizer.

Distillation (``--distill W``): on LoT steps, the frozen base (LoRA off, dense
layout) predicts the clean latent from the same ``x_t`` (same noise, same
sigma), and the student's clean estimate, mapped back with ``to_x``, is pulled
toward it with the same ``1 / max(sigma, 0.05)^2`` weight. The target is still
a one-step posterior mean, not a sample; what it adds is that LoT learns to
reproduce *dense H3's* prediction at every sigma instead of a noisy data
target, and the base teacher carries no style drift. ``--data-weight`` scales
the eq. 17 data term (0 = distillation only). Each LoT step costs one extra
no-grad dense forward. Evals also report ``gap_*``: the same weighted distance
between LoT and the base teacher, which is the number distillation drives.

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
    if not cache.is_dir():
        raise SystemExit(f"cache directory {cache} does not exist")
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


MAX_LATENT_HW: tuple[int, int] | None = None   # set from --max-latent-hw in main()


def load_item(item, rng: random.Random, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean latent ``(1, 24, T, H, W)`` cropped to a 4-token multiple, and text ``(1, L, 5120)``.

    A still is ``T = 1``; a cached clip keeps its latent frames. ``--max-latent-hw``
    caps the spatial crop (a random window), for clips too large to train whole.
    """
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
    if MAX_LATENT_HW is not None:
        height = min(height, MAX_LATENT_HW[0] // step * step)
        width = min(width, MAX_LATENT_HW[1] // step * step)
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

def parse_shift(value: str):
    """``12`` -> 12.0; ``sigmoid`` and ``lognorm:S`` pass through to ``sample_sigmas``."""
    if value == "sigmoid" or value.startswith("lognorm:"):
        return value
    return float(value)


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
        sigma = sample_sigmas(1, device, shift=args.dense_shift if kind == "dense" else args.shift)
        y_t, _ = sample_noisy(y0, sigma)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        dit._lot = splice
        try:
            out = dit(y_t, 1.0 - sigma, text)
        finally:
            dit._lot = None
        loss = lot_h3_clean_loss(out, y_t, y0, sigma)
        if args.distill > 0 and kind != "dense":
            # Teacher: same x_t, LoRA off, dense, no grad. Pulled toward with the same weight.
            x_t = (1.0 - sigma).reshape(1, 1, 1, 1, 1) * x0 + sigma.reshape(1, 1, 1, 1, 1) * (y_t - (1.0 - sigma).reshape(1, 1, 1, 1, 1) * y0) / sigma.reshape(1, 1, 1, 1, 1)
            for module in network.unet_loras:
                module.multiplier = 0.0
            with torch.no_grad():
                teacher_out = dit(x_t, 1.0 - sigma, text)
            for module in network.unet_loras:
                module.multiplier = 1.0
            target = x_t + sigma.reshape(1, 1, 1, 1, 1) * teacher_out.float()
            student = splice.to_x(y_t.float() + sigma.reshape(1, 1, 1, 1, 1) * out.float())
            loss = loss + args.distill * (student - target).square().mean() / float(sigma.clamp(min=0.05)) ** 2
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
    parser.add_argument("--cache", type=Path, action="append", default=None,
                        help=f"Fizgig H3 cache dir; repeat to mix stills and clips (default {CACHE})")
    parser.add_argument("--shift", type=parse_shift, default=12.0,
                        help="sigma density for LoT steps: float shift, 'sigmoid', or 'lognorm:S'")
    parser.add_argument("--dense-shift", type=parse_shift, default=12.0,
                        help="sigma density for the dense anchor steps (12 = H3's own)")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="weight of the frozen-dense-teacher term on LoT steps (0 = off)")
    parser.add_argument("--data-weight", type=float, default=1.0,
                        help="weight of the eq. 17 data term (0 = distillation only)")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="random spatial crop cap in latent pixels (multiples of 8), e.g. 40 32")
    parser.add_argument("--init", type=Path, default=None,
                        help="warm-start adapter.pt + lora.safetensors from an earlier run dir")
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

    global MAX_LATENT_HW
    MAX_LATENT_HW = tuple(args.max_latent_hw) if args.max_latent_hw else None
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
    caches = args.cache or [CACHE]
    train_items, held_items = [], []
    for cache in caches:
        train_part, held_part = list_items(cache, args.holdout)
        train_items += train_part
        held_items += held_part

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
    if args.init is not None:
        state = torch.load(args.init / "adapter.pt", map_location="cpu", weights_only=True)
        adapter.load_state_dict(state)
        info = network.load_weights(str(args.init / "lora.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"--init LoRA is missing {len(info.missing_keys)} keys")
        print(f"LOT_H3 kind=train_init from={args.init}", flush=True)
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
          f"swap={args.swap} shift={args.shift} dense_shift={args.dense_shift} caches={len(caches)} "
          f"distill={args.distill} data_weight={args.data_weight} "
          f"allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    if args.check:
        check(adapter, dit, network, groups, train_items, rng, device, args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")

    def set_lora(on: bool) -> None:
        for module in network.unet_loras:
            module.multiplier = 1.0 if on else 0.0

    def teacher_clean(x0, noise, sigma, text) -> torch.Tensor:
        """Frozen base (LoRA off, dense) clean estimate from the same x_t; H3's head is x0 - eps."""
        x_t = (1.0 - sigma).reshape(1, 1, 1, 1, 1) * x0 + sigma.reshape(1, 1, 1, 1, 1) * noise
        set_lora(False)
        try:
            with torch.no_grad():
                out = dit(x_t, 1.0 - sigma, text)
        finally:
            set_lora(True)
        return (x_t + sigma.reshape(1, 1, 1, 1, 1) * out.float()).detach()

    def teacher_gap(splice, out, y_t, sigma, target) -> torch.Tensor:
        student = splice.to_x(y_t.float() + sigma.reshape(1, 1, 1, 1, 1) * out.float())
        weight = 1.0 / float(sigma.clamp(min=0.05)) ** 2
        return (student - target).square().mean() * weight

    def step_loss(item, item_rng, *, layout=None, sigma=None, noise_seed=None, with_teacher=False):
        x0, text = load_item(item, item_rng, device)
        if layout is None:
            layout, kind = sample_layout(x0, item_rng, args.dense_p)
        else:
            kind = "fixed"
        splice = LotSplice(adapter, layout, y_space=True)
        y0 = splice.to_y(x0)
        if sigma is None:
            sigma = sample_sigmas(1, device, shift=args.dense_shift if kind == "dense" else args.shift)
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(1)
        gen = None if noise_seed is None else torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(y0.shape, device=device, generator=gen)
        y_t, _ = sample_noisy(y0, sigma, noise)
        dit._lot = splice
        try:
            out = dit(y_t, 1.0 - sigma, text)
        finally:
            dit._lot = None
        data = lot_h3_clean_loss(out, y_t, y0, sigma)
        gap = None
        # Training draws its own layout (kind != "fixed"); evals pass one and opt in explicitly.
        if with_teacher or (args.distill > 0 and kind not in ("dense", "fixed")):
            gap = teacher_gap(splice, out, y_t, sigma, teacher_clean(x0, noise, sigma, text))
        loss = args.data_weight * data + (args.distill * gap if gap is not None and kind != "dense"
                                          and not with_teacher else 0.0)
        return loss, out, y_t, y0, sigma, layout, kind, data, gap

    def evaluate(step: int) -> None:
        network.eval()
        results = {"dense": [], "uniform2": [], "mosaic": [], "gap_uniform2": [], "gap_mosaic": []}
        per_extent: dict[str, list] = {}
        with torch.no_grad():
            for index, item in enumerate(held_items):
                for kind in ("dense", "uniform2", "mosaic"):
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
                    _loss, out, y_t, y0, sig, lay, _, data, gap = step_loss(
                        item, item_rng, layout=layout, sigma=sigma, noise_seed=index,
                        with_teacher=(kind != "dense"))
                    results[kind].append(float(data))
                    if gap is not None:
                        results[f"gap_{kind}"].append(float(gap))
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
        loss, _out, _y_t, _y0, sigma, layout, kind, data, gap = step_loss(item, rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"step {step}: loss {float(loss)}")
        optimizer.step()
        # Free the grads now, not at the next step: evaluation and saving run between
        # steps, and ~350 MB of live grads on top of the Adam state OOMed the step-250 eval.
        optimizer.zero_grad(set_to_none=True)
        record = {"step": step, "loss": float(loss.detach()), "data": float(data.detach()),
                  "gap": None if gap is None else float(gap.detach()), "sigma": float(sigma), "kind": kind,
                  "tokens": layout.count, "dense": layout.dense_count, "grad_norm": float(grad_norm),
                  "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if step % 25 == 0:
            log.flush()
            gap_text = "" if record["gap"] is None else f" gap={record['gap']:.4f}"
            print(f"LOT_H3 kind=train step={step} loss={record['loss']:.4f}{gap_text} sigma={record['sigma']:.3f} "
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
