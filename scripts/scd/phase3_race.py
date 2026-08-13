#!/usr/bin/env python3
"""1-hour SCD distill race: velocity (trajectory) vs feature anchors.

Both arms warm-start from the same LoRA, train on a shared offline teacher cache
(see precompute_teacher.py), log losses + reconstructed frames to wandb.

  arm=velocity  — L = MSE(v_scd, v_teacher)          # path distill
  arm=anchors   — L = (1-cos) on h29/h49 + small FM  # repr-align style

Usage:
    python3 phase3_race.py --arm velocity --minutes 55 --wandb h3-scd-race \\
        --cache runs/teacher_cache.pt --init-lora runs/scd_v2/scd_lora_002500.safetensors \\
        --checkpoint /path/to/FL2VA/transformer --out runs/race_velocity
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3_sample import sample, score  # noqa: E402
from phase3_train import import_fizgig  # noqa: E402
from scd_data import ClipSet  # noqa: E402
from scd_lora import add_lora, lora_param_groups, lora_parameters, lora_report, lora_state_dict  # noqa: E402
from scd_model import DEFAULT_DECODER_SOURCE, DEFAULT_ENCODER_DEPTH, MiniMaxH3SCD  # noqa: E402


def student_one_frame(scd, mm, x0, text, noised, sigma, frame, *, window, chunk_frames,
                      need_feats=False):
    """SCD velocity for ONE frame with two-stage backward (encode leaf cut).

    Same memory pattern as phase3_train.step_backward: encode → detach leaf → decode →
    backward on leaf → replay encode. Peak holds one decoder graph, not encode+decode forever.
    """
    device = x0.device
    s = float(sigma)
    enc, clean_ctx, _ = scd.encode_chunked(
        chunk_frames, window=window, layer_major=True, keep_audio=True, checkpoint=True,
        video_latent=x0, t=torch.tensor([1.0], device=device), text_embeds=text)
    spans = scd.spans(x0, enc.shape[0])
    leaf = enc.detach().requires_grad_(True)

    noisy_h, noisy_ctx, _pack = scd.preamble(
        video_latent=noised, t=torch.tensor([1.0 - s], device=device), text_embeds=text)
    ph, pw = scd.base.patch_size[1], scd.base.patch_size[2]
    h, w = x0.shape[3] // ph, x0.shape[4] // pw

    x_in, ctx = scd.decoder_frame_input(leaf, clean_ctx, noisy_h, noisy_ctx, spans, frame,
                                        media_start=None, duplicate_pos=True)
    x = x_in
    for block in scd.decoder_blocks:
        x = block(x, *ctx)
    r = spans.frame_rows
    out = x[-r:]
    t_emb, mod = ctx[0], ctx[1]
    import sys as _sys
    mm_mod = _sys.modules[type(scd.base).__module__]
    v_patch = scd.base.final_layer(out, t_emb, int(mod[-1]) // mm_mod.MODALITY_NUM)
    v = mm.unpatchify_video(v_patch.float(), 1, h, w, c=x0.shape[1],
                            patch_size=scd.base.patch_size)

    enc_mean = dec_mean = None
    if need_feats:
        lo = spans.video_start
        n = spans.latent_t * spans.frame_rows
        # leaf for encoder align so grads reach the cut; enc path restored in backward
        enc_mean = leaf[lo:lo + n].float().mean(0)
        dec_mean = out.float().mean(0)
    return v, enc_mean, dec_mean, enc, leaf


def load_init(scd, path, rank):
    if not path:
        return
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}
    file_rank = int(meta.get("rank") or next(
        sd[k].shape[0] for k in sd if k.endswith("lora_down.weight")))
    if file_rank != rank:
        raise SystemExit(f"init rank {file_rank} != {rank}")
    made = {n: m for n, m in scd.named_modules() if hasattr(m, "lora_down")}
    # walk via add_lora result instead
    from scd_lora import add_lora as _al
    # adapters already installed; fill weights
    for name, mod in scd.named_modules():
        if not hasattr(mod, "lora_down"):
            continue
        # lora_name attribute
        key = getattr(mod, "lora_name", None)
        if key is None:
            continue
        for part in ("lora_down", "lora_up"):
            wkey = f"{key}.{part}.weight"
            if wkey in sd:
                getattr(mod, part).weight.data.copy_(
                    sd[wkey].to(device=getattr(mod, part).weight.device,
                                dtype=getattr(mod, part).weight.dtype))
    print(f"init-lora   : {path} step={meta.get('step')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["velocity", "anchors"])
    ap.add_argument("--cache", default="runs/teacher_cache.pt")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--init-lora", default="",
                    help="optional warm-start; rank must match --rank")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=55.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16,
                    help="16 fits the race on 24 GB; 32 climbs into OOM by ~step 30")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--chunk-frames", type=int, default=1)
    ap.add_argument("--fm-weight", type=float, default=0.15,
                    help="anchors arm only: mix of GT flow-matching")
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--clips", default=os.path.join(os.path.dirname(__file__), "clips"))
    ap.add_argument("--wandb", default="h3-scd-race", help="project name; empty to disable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--preview-every", type=int, default=100,
                    help="steps between latent-score previews (full decode at end)")
    args = ap.parse_args()

    mm, load_dit, sample_sigmas = import_fizgig(args.fizgig_src)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    samples = cache["samples"]
    print(f"cache       : {len(samples)} teacher samples from {args.cache}", flush=True)

    clips = ClipSet(args.clips)
    # keep clip tensors on CPU; move per step
    clip_bank = {}
    for s in samples:
        n = s["clip"]
        if n not in clip_bank:
            clip_bank[n] = clips.load(n, device="cpu", dtype=torch.bfloat16)

    print(f"loading SCD student…", flush=True)
    base = load_dit(args.checkpoint, device=device, compute_dtype=torch.bfloat16,
                    quantize=True, base_quant="nf4")
    scd = MiniMaxH3SCD(base, encoder_depth=DEFAULT_ENCODER_DEPTH,
                       decoder_source=DEFAULT_DECODER_SOURCE)
    scd.eval()
    made = add_lora(scd, rank=args.rank, alpha=args.rank)
    n_mod, n_par = lora_report(made)
    print(f"lora        : {n_mod} modules, {n_par/1e6:.2f}M rank {args.rank}", flush=True)
    if args.init_lora and os.path.isfile(args.init_lora):
        # fill from file
        from safetensors import safe_open
        with safe_open(args.init_lora, framework="pt") as f:
            meta = f.metadata() or {}
            sd = {k: f.get_tensor(k) for k in f.keys()}
        for name, mod in made.items():
            for part in ("lora_down", "lora_up"):
                getattr(mod, part).weight.data.copy_(
                    sd[f"{name}.{part}.weight"].to(
                        device=getattr(mod, part).weight.device,
                        dtype=getattr(mod, part).weight.dtype))
        print(f"init-lora   : {args.init_lora} step={meta.get('step')}", flush=True)

    groups = lora_param_groups(scd, args.lr, decoder_ratio=2.0)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, name=f"{args.arm}-{int(time.time())}",
                        config=vars(args))
        print(f"wandb       : {wb.url}", flush=True)

    deadline = time.time() + args.minutes * 60.0
    step = 0
    t0 = time.time()
    losses = []
    rng = torch.Generator().manual_seed(args.seed)

    while time.time() < deadline:
        step += 1
        idx = int(torch.randint(0, len(samples), (1,), generator=rng).item())
        rec = samples[idx]
        clip = clip_bank[rec["clip"]]
        x0 = clip["video_latent"].float().to(device)
        text = clip["text_embeds"].to(device)
        noise = rec["noise"].float().to(device)
        sigma = float(rec["sigma"])
        noised = ((1.0 - sigma) * x0 + sigma * noise).to(torch.bfloat16)
        v_t = rec["v"].float().to(device)
        h29 = rec["h29"].float().to(device) if rec.get("h29") is not None else None
        h49 = rec["h49"].float().to(device) if rec.get("h49") is not None else None

        # One random frame — full-T graphs OOM on 24 GB with rank-32.
        latent_t = x0.shape[2]
        frame = int(torch.randint(0, latent_t, (1,), generator=rng).item())

        opt.zero_grad(set_to_none=True)
        v_s, enc_mean, dec_mean, enc, leaf = student_one_frame(
            scd, mm, x0, text, noised, sigma, frame,
            window=args.window, chunk_frames=args.chunk_frames,
            need_feats=(args.arm == "anchors"))

        parts = {}
        v_t_f = v_t[:, :, frame:frame + 1]
        if args.arm == "velocity":
            loss_v = F.mse_loss(v_s.float(), v_t_f)
            loss = loss_v
            parts = {"loss_v": float(loss_v.detach()), "frame": frame}
        else:
            loss_a = v_s.new_zeros(())
            if h29 is not None and enc_mean is not None:
                c = F.cosine_similarity(enc_mean.unsqueeze(0), h29.unsqueeze(0))
                loss_a = loss_a + (1.0 - c).mean()
            if h49 is not None and dec_mean is not None:
                c = F.cosine_similarity(dec_mean.unsqueeze(0), h49.unsqueeze(0))
                loss_a = loss_a + (1.0 - c).mean()
            target_f = (x0 - noise).float()[:, :, frame:frame + 1]
            loss_fm = F.mse_loss(v_s.float(), target_f)
            w = float(args.fm_weight)
            loss = (1.0 - w) * loss_a + w * loss_fm
            parts = {"loss_a": float(loss_a.detach()), "loss_fm": float(loss_fm.detach()),
                     "frame": frame}

        # Decoder (+ leaf) first, then re-run encoder grads — frees the decoder graph early.
        loss.backward()
        if leaf.grad is not None:
            enc.backward(leaf.grad)
        gnorm = torch.nn.utils.clip_grad_norm_(lora_parameters(scd), 1.0)
        opt.step()

        lv = float(loss.detach())
        losses.append(lv)
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        # Drop step tensors; fragmentation climbed 16.7→18.1 GB then OOMed without this.
        del v_s, enc, leaf, loss, noised, x0, text, noise, v_t
        if step % 5 == 0:
            torch.cuda.empty_cache()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            msg = (f"step {step:>5}  loss {lv:.4f}  |g| {float(gnorm):.2e}  "
                   f"{peak:.1f}GB  {elapsed:.0f}s  {rec['clip']} σ={sigma:.3f}")
            print(msg, flush=True)
            if wb:
                wb.log({"loss": lv, "grad_norm": float(gnorm), "peak_gb": peak,
                        "sigma": sigma, "step": step, **parts}, step=step)

        if step % args.preview_every == 0 and wb:
            with torch.no_grad():
                pg = clip_bank.get("pixelgraph") or clip_bank[rec["clip"]]
                x = pg["video_latent"].float().to(device)
                te = pg["text_embeds"].to(device)
                noise_p = torch.randn_like(x)
                s_p = 0.5
                no = ((1 - s_p) * x + s_p * noise_p).to(torch.bfloat16)
                v_p, _, _, _, _ = student_one_frame(
                    scd, mm, x, te, no, s_p, frame=1,
                    window=args.window, chunk_frames=args.chunk_frames, need_feats=False)
                mse = F.mse_loss(v_p, (x - noise_p).float()[:, :, 1:2]).item()
                wb.log({"preview/mse_to_gt_v": mse}, step=step)

    # save
    from safetensors.torch import save_file
    ckpt = os.path.join(args.out, f"scd_lora_{args.arm}_final.safetensors")
    save_file(lora_state_dict(scd), ckpt, metadata={
        "arm": args.arm, "steps": str(step), "rank": str(args.rank),
    })
    print(f"saved       : {ckpt} after {step} steps", flush=True)

    # sample + decode pixelgraph for reconstruction
    print("sampling pixelgraph oracle…", flush=True)
    from fizgig.minimax.sampling import sample_schedule
    sigmas = sample_schedule(20, shift=12.0)
    pg = clips.load("pixelgraph", device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        pred = sample(scd, mm, pg, sigmas, "oracle", seed=0, window=args.window,
                      chunk_frames=args.chunk_frames, media_start_on=False,
                      duplicate_pos=True, seed_frames=0)
    sc = score(pred, pg["video_latent"].float())
    print(f"oracle score: corr_ctx={sc['corr_ctx']:+.4f} mse={sc['mse']:.4f}", flush=True)
    lat_path = os.path.join(args.out, "pixelgraph_oracle.safetensors")
    from safetensors.torch import save_file as sf
    sf({"pred": pred.float().cpu().contiguous(),
        "truth": pg["video_latent"].float().cpu().contiguous()}, lat_path)

    # free student for VAE decode
    del scd, base, opt
    torch.cuda.empty_cache()

    print("decoding…", flush=True)
    import subprocess
    dec_out = os.path.join(args.out, "decoded")
    os.makedirs(dec_out, exist_ok=True)
    # unique parent for stem
    src = os.path.join(args.out, "lat")
    os.makedirs(src, exist_ok=True)
    link = os.path.join(src, "pixelgraph_oracle.safetensors")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(os.path.abspath(lat_path), link)
    subprocess.check_call([
        sys.executable, "phase3_decode.py", link, "--out", dec_out,
        "--fizgig-src", args.fizgig_src, "--which", "both",
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    # wandb reconstructions (truth | pred contact sheet)
    if wb:
        import wandb
        from PIL import Image
        import subprocess as sp
        mp4 = os.path.join(dec_out, "lat_pixelgraph_oracle_pred.mp4")
        png = os.path.join(dec_out, "preview.png")
        sp.check_call(["ffmpeg", "-y", "-i", mp4, "-vf", "select=eq(n\\,11)", "-vframes", "1",
                       png], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        truth_mp4 = os.path.join(dec_out, "lat_pixelgraph_oracle_truth.mp4")
        truth_png = os.path.join(dec_out, "truth.png")
        if os.path.isfile(truth_mp4):
            sp.check_call(["ffmpeg", "-y", "-i", truth_mp4, "-vf", "select=eq(n\\,11)",
                           "-vframes", "1", truth_png],
                          stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        log_imgs = {
            "final/corr_ctx": sc["corr_ctx"],
            "final/mse": sc["mse"],
            "final/steps": step,
        }
        if os.path.isfile(png):
            log_imgs["recon/pred"] = wandb.Image(
                png, caption=f"{args.arm} corr_ctx={sc['corr_ctx']:.4f}")
        if os.path.isfile(truth_png):
            log_imgs["recon/truth"] = wandb.Image(truth_png)
        if os.path.isfile(png) and os.path.isfile(truth_png):
            a = Image.open(truth_png).convert("RGB")
            b = Image.open(png).convert("RGB")
            sheet = Image.new("RGB", (a.width * 2, a.height))
            sheet.paste(a, (0, 0))
            sheet.paste(b, (a.width, 0))
            sheet_path = os.path.join(dec_out, "sheet.png")
            sheet.save(sheet_path)
            log_imgs["recon/truth_vs_pred"] = wandb.Image(
                sheet_path, caption=f"{args.arm} | left=truth right=pred")
        wb.log(log_imgs)
        wb.summary["corr_ctx"] = sc["corr_ctx"]
        wb.summary["mse"] = sc["mse"]
        wb.summary["steps"] = step
        wb.finish()

    summary = {"arm": args.arm, "steps": step, "corr_ctx": sc["corr_ctx"], "mse": sc["mse"],
               "mean_loss": sum(losses) / max(1, len(losses)), "ckpt": ckpt}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
