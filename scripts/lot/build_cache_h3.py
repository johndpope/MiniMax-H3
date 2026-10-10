#!/usr/bin/env python3
"""Turn h3-atlas H3 latents (Nikki talk + pose sets) into a train_h3 cache.

    python3 scripts/lot/build_cache_h3.py latents [--out DIR]   # CPU: latents + captions
    python3 scripts/lot/build_cache_h3.py text    [--out DIR]   # GPU: Qwen3-VL captions -> _te

``latents`` writes ``<stem>_0512x0768_minimaxh3.safetensors`` with key
``latent_7x48x32`` (the first 7 latent frames of each ``z.pt``: 22 pixel frames,
the start of a clip, so the causal first frame stays a keyframe) and
``captions.json``. No VAE: these are H3's own latents.

``text`` loads the Qwen3-VL-32B text encoder once, encodes every caption, writes
``<stem>_minimaxh3_te.safetensors`` (``hidden_states``, ``attention_mask``) and
exits. Run it alone on the GPU. It never runs beside a DiT.

Sources:
- ``set_t_talk`` items whose ``meta.json`` id is ``nikki``: talking-head clips with
  full H3 prompts.
- ``set_a_pose_ext``: pose transitions. ``meta.json`` has no prompt, so the caption
  is built from ``from_card`` -> ``to_card``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import FIZGIG_SRC, LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

ATLAS = Path("/home/johndpope/Documents/GitHub/h3-atlas/data")
TEXT_ENCODER = LOT_H3_TEXT_ENCODER
DEFAULT_OUT = Path(__file__).resolve().parent / "runs" / "cache_nikki"
LATENT_T = 7


def card_words(card: str) -> str:
    """``forward_lean_kneel_afa9fa9c`` -> ``forward lean kneel``: drop hash-like tokens."""
    words = [w for w in card.split("_") if not re.fullmatch(r"[0-9a-f]{6,}", w)]
    return " ".join(words)


def pose_caption(meta: dict) -> str:
    start = card_words(meta["from_card"])
    end = card_words(meta["to_card"])
    def article(words: str) -> str:
        return "an" if words[:1] in "aeiou" else "a"

    return (f"Full-body shot of the woman in a plain studio, moving smoothly from {article(start)} {start} "
            f"pose into {article(end)} {end} pose. Static locked-off camera, single continuous shot, "
            f"sharp focus.")


def sources() -> list[tuple[str, Path, str]]:
    """``(stem, z_path, caption)`` for every usable item."""
    items = []
    for meta_path in sorted((ATLAS / "set_t_talk").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("id") == "nikki" and (meta_path.parent / "z.pt").is_file():
            items.append((f"talk_{meta_path.parent.name}", meta_path.parent / "z.pt", meta["prompt"]))
    for meta_path in sorted((ATLAS / "set_a_pose_ext").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if (meta_path.parent / "z.pt").is_file():
            items.append((f"pose_{meta_path.parent.name}", meta_path.parent / "z.pt", pose_caption(meta)))
    return items


def build_latents(out: Path) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    captions = {}
    shapes = {}
    for stem, z_path, caption in sources():
        z = torch.load(z_path, map_location="cpu", weights_only=True)
        if z.ndim != 5 or z.shape[:2] != (1, 24) or z.shape[2] < LATENT_T:
            print(f"skip {stem}: z {tuple(z.shape)}", flush=True)
            continue
        latent = z[0, :, :LATENT_T].float().contiguous()             # (24, 7, H, W)
        _c, t, h, w = latent.shape
        key = f"latent_{t}x{h}x{w}"
        save_file({key: latent}, str(out / f"{stem}_{w * 16:04d}x{h * 16:04d}_minimaxh3.safetensors"))
        captions[stem] = caption
        shapes[key] = shapes.get(key, 0) + 1
    (out / "captions.json").write_text(json.dumps(captions, indent=1) + "\n")
    print(f"LOT_H3 kind=cache_latents out={out} items={len(captions)} shapes={shapes}", flush=True)


def build_text(out: Path) -> None:
    from safetensors.torch import save_file

    sys.path.insert(0, FIZGIG_SRC)
    from gpu_guard import refuse_if_busy
    from fizgig.minimax.embedder import load_minimax_h3_te_planned

    refuse_if_busy("build_cache_h3.py")
    captions = json.loads((out / "captions.json").read_text())
    todo = {stem: text for stem, text in captions.items()
            if not (out / f"{stem}_minimaxh3_te.safetensors").is_file()}
    print(f"LOT_H3 kind=cache_text todo={len(todo)} of {len(captions)}", flush=True)
    if not todo:
        return
    encoder = load_minimax_h3_te_planned(str(TEXT_ENCODER), device="cuda", compute_dtype=torch.bfloat16,
                                         quantize=True)
    for index, (stem, text) in enumerate(todo.items()):
        with torch.inference_mode():
            hidden = encoder.encode(text)
        hidden = hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16).cpu()
        save_file({"hidden_states": hidden, "attention_mask": torch.ones(hidden.shape[0], dtype=torch.bool)},
                  str(out / f"{stem}_minimaxh3_te.safetensors"))
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(todo):
            print(f"LOT_H3 kind=cache_text done={index + 1}/{len(todo)} rows={hidden.shape[0]} "
                  f"dim={hidden.shape[1]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("latents", "text"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.phase == "latents":
        build_latents(args.out)
    else:
        build_text(args.out)


if __name__ == "__main__":
    main()
