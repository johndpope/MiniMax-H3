#!/usr/bin/env python3
"""Encode a prompt to Qwen3-VL-32B layer-50 states for the Phase 0 probe.

The companion to encode_clip.py: produces the `--text` half of the probe's cached input, so the
probe itself never loads a 32B encoder alongside the DiT.

    python scripts/scd/encode_text.py --prompt-file prompt.txt --out clip_te.safetensors

Tokenizer defaults to this repo's `text_encoder/`, NOT the one Fizgig bundles. They share
tokenizer.json byte for byte but Fizgig's tokenizer_config.json omits seven H3 special tokens
(`<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_{start,end}|>`, `<|caption_{start,end}|>`), so it
silently shreds them into ordinary byte pairs -- `<d>` becomes [90707, 30768, ...] instead of
the single id 151669. Plain prose tokenizes identically, which is why this only shows up on
real H3 prompts, where dialogue is wrapped in `<d>[Language] ...</d>`.

The encoder emits the RAW layer-50 output (final norm is Identity), which is what H3 conditions
on -- see fizgig/minimax/embedder.py.
"""

import argparse
import os
import sys

import torch

DEFAULT_TE = "/media/2TB/Fizgig/models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_FIZGIG_SRC = "/media/2TB/Fizgig/src"
REPO_TOKENIZER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "text_encoder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="prompt text; or use --prompt-file")
    ap.add_argument("--prompt-file", help="file containing the prompt")
    ap.add_argument("--out", default="clip_te.safetensors")
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    ap.add_argument("--tokenizer", default=REPO_TOKENIZER)
    ap.add_argument("--fizgig-src", default=DEFAULT_FIZGIG_SRC)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-quantize", action="store_true", help="bf16, needs ~66 GB VRAM")
    args = ap.parse_args()

    if bool(args.prompt) == bool(args.prompt_file):
        raise SystemExit("give exactly one of --prompt or --prompt-file")
    prompt = args.prompt if args.prompt else open(args.prompt_file, encoding="utf-8").read().strip()
    if not prompt:
        raise SystemExit("prompt is empty")

    if args.fizgig_src not in sys.path:
        sys.path.insert(0, args.fizgig_src)
    from fizgig.minimax.embedder import load_minimax_h3_te

    encoder = load_minimax_h3_te(args.text_encoder, device=torch.device(args.device),
                                 compute_dtype=torch.bfloat16, quantize=not args.no_quantize,
                                 tokenizer_dir=args.tokenizer)

    ids = encoder.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    specials = [t for t in ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>",
                            "<|caption_start|>", "<|caption_end|>") if t in prompt]
    for tok in specials:
        if encoder.tokenizer.convert_tokens_to_ids(tok) is None:
            raise SystemExit(f"tokenizer at {args.tokenizer} does not know {tok} — it would be "
                             "split into byte pairs. Point --tokenizer at this repo's text_encoder/.")
    print(f"prompt      : {len(prompt)} chars -> {len(ids)} tokens"
          + (f", special: {' '.join(specials)}" if specials else ""))
    if len(ids) > args.max_length:
        print(f"!! truncated to --max-length {args.max_length}")

    with torch.no_grad():
        emb = encoder.encode(prompt, max_length=args.max_length)[0]     # [L, 5120]
    emb = emb.detach().cpu().contiguous()

    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({"hidden_states": emb,
               "attention_mask": torch.ones(emb.shape[0], dtype=torch.bool)},
              args.out, metadata={"tokens": str(emb.shape[0]),
                                  "tokenizer": os.path.abspath(args.tokenizer)})
    print(f"hidden      : {tuple(emb.shape)} {emb.dtype}")
    print(f"wrote       : {args.out}")


if __name__ == "__main__":
    main()
