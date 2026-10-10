"""Machine-specific paths for the LoT scripts, each overridable by an environment variable.

The defaults are the machine these scripts were developed on. On another machine,
export the variables instead of editing the scripts:

| Variable | What |
|---|---|
| ``FIZGIG_SRC`` | Fizgig ``src`` directory (branch ``immiscible-h3-noise`` has the LoT hooks) |
| ``LOT_H3_CHECKPOINT`` | pruned int8 ConvRot FL2VA DiT the adapters were trained on |
| ``LOT_H3_VAE`` | H3 video VAE (fp16) |
| ``LOT_H3_TEXT_ENCODER`` | Qwen3-VL-32B MiniMax text encoder (only for caching captions) |
| ``LOT_H3_STILLS_CACHE`` | Fizgig H3 still cache with ``_te`` text (default training/render prompts) |
| ``LOT_H3_STILLS_CAPTIONS`` | caption ``.txt`` files for that cache |
"""

from __future__ import annotations

import os
from pathlib import Path

FIZGIG_SRC = os.environ.get("FIZGIG_SRC", "/media/2TB/Fizgig/src")
LOT_H3_CHECKPOINT = Path(os.environ.get(
    "LOT_H3_CHECKPOINT",
    "/media/2TB/Fizgig/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"))
LOT_H3_VAE = Path(os.environ.get(
    "LOT_H3_VAE", "/media/2TB/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors"))
LOT_H3_TEXT_ENCODER = Path(os.environ.get(
    "LOT_H3_TEXT_ENCODER",
    "/media/2TB/minimax-h3-nvfp4/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"))
LOT_H3_STILLS_CACHE = Path(os.environ.get(
    "LOT_H3_STILLS_CACHE", "/media/2TB/lora-data/fizgig_minimax_h3/cache_iso3d"))
LOT_H3_STILLS_CAPTIONS = Path(os.environ.get(
    "LOT_H3_STILLS_CAPTIONS", "/media/2TB/lora-data/fizgig_minimax_h3/isometric_3d_stills"))
