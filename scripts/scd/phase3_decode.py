#!/usr/bin/env python3
"""Decode `phase3_sample --save-latents` output through the video VAE and look at it.

Everything Phase 3 has measured so far lives in latent space. `corr[1:] = +0.61` says the sampled
latents point the same way as the real ones; it does not say the thing decodes to video, and a
number that high is exactly where a latent-space metric stops being informative. This is the
cheapest measurement that can still kill the port.

It decodes the TRUTH alongside the sample from the same file, because the VAE's own round trip is
not lossless at 512 and a soft sample is only evidence against the split if the truth through the
same path is sharp. Read the pair, never the sample alone.

Runs on CPU by default: the 2.4 B decoder is ~4.8 GB in bf16 and a training run holds ~15.5 GB of
a 24 GB card, so `--device cuda` is for when the GPU is idle.

    python3 scripts/scd/phase3_decode.py runs/scd_v0/latents/pixelgraph_ar.safetensors --out /tmp
"""

import argparse
import os
import sys

import torch

DEFAULT_VAE = "/media/2TB/Fizgig/models/minimax_h3_video_vae_fp16.safetensors"
DEFAULT_FIZGIG_SRC = "/media/2TB/Fizgig/src"


def load_decoder(vae_path, fizgig_src, device, dtype):
    if fizgig_src not in sys.path:
        sys.path.insert(0, fizgig_src)
    # Fizgig's reader, not safetensors.safe_open: the released fp16 VAE has 61 stray bytes past
    # its last tensor and the strict rust parser rejects the file outright.
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder

    vae = MiniMaxH3VideoVAEDecoder()
    wanted = ("decoder.", "post_quant_conv.", "latents_")
    with MemoryEfficientSafeOpen(vae_path) as f:
        sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(wanted)}
    missing, _ = vae.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("encoder")]
    if bad:
        raise SystemExit(f"{vae_path}: missing decoder weights {bad[:5]}")
    return vae.to(device, dtype).eval()


def write_video(pixels, path, fps):
    """pixels [3, T, H, W] in [0, 1] -> an mp4, or a PNG contact sheet if PyAV is unavailable."""
    import numpy as np

    frames = (pixels.permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    try:
        import av
    except ImportError:
        from PIL import Image
        sheet = np.concatenate(list(frames), axis=1)
        Image.fromarray(sheet).save(path := path.rsplit(".", 1)[0] + ".png")
        return path

    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        # High quality: this is a diagnostic, and codec mush is not a result about the model.
        stream.options = {"crf": "12"}
        for f in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        container.mux(stream.encode())
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latents", nargs="+", help="*.safetensors from --save-latents")
    ap.add_argument("--out", default=".", help="directory for the mp4s")
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--fizgig-src", default=DEFAULT_FIZGIG_SRC)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    ap.add_argument("--fps", type=float, default=24.0, help="H3's native rate")
    ap.add_argument("--which", default="both", choices=["both", "pred", "truth"])
    args = ap.parse_args()

    from safetensors.torch import load_file

    dtype = getattr(torch, "float32" if args.device == "cpu" else args.dtype)
    vae = load_decoder(args.vae, args.fizgig_src, args.device, dtype)
    os.makedirs(args.out, exist_ok=True)

    for path in args.latents:
        tensors = load_file(path)
        # Include the parent folder in the output name. Different sampling runs
        # (e.g. lat_s50sh12/ vs latents/) all write files named pixelgraph_oracle.safetensors;
        # if we only used that basename, later decodes would overwrite earlier mp4s.
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        stem = f"{parent}_{os.path.basename(path).rsplit('.', 1)[0]}"
        keys = ["pred", "truth"] if args.which == "both" else [args.which]
        for key in keys:
            z = tensors[key].to(args.device, dtype)
            pixels = vae.decode(z)[0].float().cpu()
            out = write_video(pixels, os.path.join(args.out, f"{stem}_{key}.mp4"), args.fps)
            print(f"{stem:>28} {key:>5}  {tuple(pixels.shape)}  "
                  f"mean {pixels.mean():.3f}  std {pixels.std():.3f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
