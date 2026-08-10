#!/usr/bin/env python3
"""Encode a video clip to H3 video-VAE latents for the Phase 0 probe.

phase0_probe.py takes cached latents rather than encoding, so it never has to load the VAE or
the 32B text encoder. This produces that cache from an ordinary video file.

    python scripts/scd/encode_clip.py clip.mp4 --latent-t 8 --out clip_latents.safetensors

The VAE is causal with a 4x temporal stride, so T pixel frames give ceil(T/4) latent frames and
1+4k gives exactly 1+k. This script asks for 1+4*(latent_t-1) frames so the count is exact --
a partial trailing group would make the last latent frame see fewer pixel frames than its
siblings, which is precisely the asymmetry the probe's causal-mask axis is trying to measure.

No temporal tiling: the encoder's first level runs at full T x H x W, so peak activation grows
linearly in T. 512^2 x 29 frames fits comfortably; 768p x 61 does not. Raise --latent-t only as
far as the card allows.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

DEFAULT_VAE = "/media/2TB/Fizgig/models/minimax_h3_video_vae_fp16.safetensors"
DEFAULT_FIZGIG_SRC = "/media/2TB/Fizgig/src"

VAE_SPATIAL = 16
VAE_TEMPORAL = 4


def read_frames(path, n_frames, target_fps, start_sec):
    """[T, H, W, 3] uint8, sampled at ~target_fps from start_sec."""
    import av
    import numpy as np

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        src_fps = float(stream.average_rate) if stream.average_rate else float(target_fps)
        step = max(1, round(src_fps / target_fps)) if target_fps else 1
        if start_sec > 0:
            container.seek(int(start_sec / stream.time_base), stream=stream)

        frames, seen = [], 0
        for frame in container.decode(stream):
            if frame.time is not None and frame.time < start_sec:
                continue
            if seen % step == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == n_frames:
                    break
            seen += 1

    if len(frames) < n_frames:
        raise SystemExit(
            f"{path}: got {len(frames)} frames at {target_fps} fps from {start_sec}s, "
            f"need {n_frames}. Use a longer clip, a lower --latent-t, or an earlier --start.")
    return torch.from_numpy(np.stack(frames)), src_fps


def to_pixel_tensor(frames_u8, height, width):
    """[T,H,W,3] uint8 -> [1,3,T,height,width] float in [-1,1], center-cropped to aspect."""
    x = frames_u8.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)   # [T,3,H,W]
    _, _, src_h, src_w = x.shape

    want = width / height
    have = src_w / src_h
    if have > want:
        crop_w = int(round(src_h * want))
        left = (src_w - crop_w) // 2
        x = x[:, :, :, left:left + crop_w]
    elif have < want:
        crop_h = int(round(src_w / want))
        top = (src_h - crop_h) // 2
        x = x[:, :, top:top + crop_h, :]

    x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False).clamp_(-1, 1)
    return x.permute(1, 0, 2, 3).unsqueeze(0)                          # [1,3,T,height,width]


def load_vae(vae_path, fizgig_src, device, dtype):
    if fizgig_src not in sys.path:
        sys.path.insert(0, fizgig_src)
    # Not safetensors.safe_open: the released fp16 VAE has 61 stray bytes past its last tensor,
    # which the strict rust parser rejects ("file not fully covered"). Fizgig's reader seeks to
    # each tensor's own offsets and is unbothered.
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder

    vae = MiniMaxH3VideoVAEEncoder()
    wanted = ("encoder.", "quant_conv.", "latents_")
    with MemoryEfficientSafeOpen(vae_path) as f:
        sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(wanted)}
    missing, _ = vae.load_state_dict(sd, strict=False)
    encoder_missing = [k for k in missing if not k.startswith("decoder")]
    if encoder_missing:
        raise SystemExit(f"{vae_path}: missing encoder weights {encoder_missing[:5]}")
    return vae.to(device, dtype).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="clip_latents.safetensors")
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--fizgig-src", default=DEFAULT_FIZGIG_SRC)
    ap.add_argument("--latent-t", type=int, default=8, help="latent frames to produce")
    ap.add_argument("--size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    ap.add_argument("--fps", type=float, default=24.0, help="H3 native rate; 0 keeps source rate")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # fp16 by default: the released VAE weights are fp16, so fp32 only upcasts them while
    # tripling activation memory (512^2 x 29 needs ~12 GB single allocations and thrashes the
    # allocator on a 24 GB card). Measured cost of fp16: cosine 0.9999996, 0.23% relative L2.
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    args = ap.parse_args()

    height, width = args.size
    for label, v in (("height", height), ("width", width)):
        if v % VAE_SPATIAL:
            raise SystemExit(f"--size {label} must be a multiple of {VAE_SPATIAL}, got {v}")
    if args.latent_t < 1:
        raise SystemExit("--latent-t must be >= 1")

    n_pixel = 1 + VAE_TEMPORAL * (args.latent_t - 1)
    dtype = getattr(torch, args.dtype)

    frames, src_fps = read_frames(args.video, n_pixel, args.fps, args.start)
    print(f"read        : {n_pixel} frames @ {args.fps or src_fps:g} fps "
          f"(source {src_fps:g} fps, {frames.shape[2]}x{frames.shape[1]})")

    pixels = to_pixel_tensor(frames, height, width)
    vae = load_vae(args.vae, args.fizgig_src, args.device, dtype)

    with torch.no_grad():
        z = vae.encode(pixels.to(args.device, dtype))
    z = z.float().cpu()

    expected = (1, 24, args.latent_t, height // VAE_SPATIAL, width // VAE_SPATIAL)
    if tuple(z.shape) != expected:
        raise SystemExit(f"latent shape {tuple(z.shape)} != expected {expected}")

    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file({"latent": z.contiguous()}, args.out, metadata={
        "source": os.path.abspath(args.video),
        "pixel_frames": str(n_pixel),
        "fps": str(args.fps or src_fps),
        "start_sec": str(args.start),
        "resolution": f"{height}x{width}",
    })
    print(f"latent      : {tuple(z.shape)}  mean {z.mean():+.4f}  std {z.std():.4f}")
    print(f"wrote       : {args.out}")


if __name__ == "__main__":
    main()
