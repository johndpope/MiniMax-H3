#!/usr/bin/env python3
"""Phase 3: LoRA-train the SCD split on the cached clip set, flow-matching loss per frame.

One step is: encode the CLEAN clip frame-causally, pack the NOISY clip at one sigma, then run
`decode_frame(velocity=True)` for each frame against `patchify(x0 - noise)` for that frame. The
loss is the base's own objective (`fizgig.minimax.trainer` — the DiT's `video_out` predicts
`x0 - noise`); §6.4 says do not invent a second one, and this does not.

The encoder runs through `encode_chunked`, never `encode`, and that is load-bearing rather than a
performance choice. `encode` is bidirectional, so its frame `f-1` rows have already attended to
frame `f` — the decoder's context half would then carry the clean target it is being asked to
predict, one hop round, and the loss would fall to near zero while teaching nothing. §6.2's
"shift features by 1 frame" and the frame-causal mask are one mechanism with two halves;
`test_decoder_context_cannot_see_its_own_frame` is what pins it.

What this v0 does NOT do, so a result from it is read correctly
---------------------------------------------------------------
**No scheduled sampling.** §6.4's curriculum ramps 0 -> 50% AR after warmup, and CastleHill needed
it. Every step here is teacher-forced: the context half is always the encoder's output over real
clean latents, never over the decoder's own predictions. So this trains the one-step conditional
and says nothing about whether errors compound over an AR rollout — which is the failure mode AR
video actually has. If the kill criterion is met, exposure bias is the next thing to add; if it is
missed, scheduled sampling is the first thing to try, not the last.

**No audio in the loss.** The clips carry video latents only (`scd_data`), so the audio rows are
the base's own noise and are packed but never scored. The Phase 2 audio window drift is untouched
by anything measured here.

**512x512, 12 clips.** 256 rows per latent frame against 768p's 1008, and §6.4 asks for 20-50
clips. A positive result is a positive result at quarter scale.

Usage:
    python3 scripts/scd/phase3_train.py --checkpoint /path/to/FL2VA/transformer \
        --steps 2000 --rank 32 --out runs/scd_v0

    python3 scripts/scd/phase3_train.py --dry-run          # tiny CPU model, no weights, ~20 s
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scd_data import ClipSet, epoch_order                              # noqa: E402
from scd_lora import (  # noqa: E402
    add_lora, lora_param_groups, lora_parameters, lora_report, lora_state_dict,
)
from scd_model import DEFAULT_DECODER_SOURCE, DEFAULT_ENCODER_DEPTH, MiniMaxH3SCD  # noqa: E402


def import_fizgig(src):
    if src not in sys.path:
        sys.path.insert(0, src)
    from fizgig.minimax import model as mm
    from fizgig.minimax.loader import load_minimax_h3_dit
    from fizgig.minimax.trainer import sample_sigmas
    return mm, load_minimax_h3_dit, sample_sigmas


class Batch:
    """One clip's clean latent, noise, sigma(s) and text — everything a step's packs need.

    Holds `noise` rather than only the noised latent because the flow target is `x0 - noise` and
    recovering it from `(noised, sigma, x0)` divides by sigma, which is unbounded as sigma -> 0.

    `sigma` is a scalar OR one value per latent frame. Per-frame is what the SCD reference
    implementation does — it draws a timestep per (batch, frame), giving `latent_t` sigma samples
    per step where a clip-wide draw gives one. The target does not depend on sigma, so this costs
    nothing in the loss; what it costs is one `preamble` per DISTINCT sigma, because the base packs
    a single video timestep per forward. A scalar collapses to exactly one pack, which is what
    keeps `evaluate`'s fixed grid comparable with the runs that predate this.
    """

    def __init__(self, clip, sigma, generator=None):
        self.name = clip["name"]
        self.x0 = clip["video_latent"].float()
        self.text = clip["text_embeds"]
        latent_t = self.x0.shape[2]
        s = torch.as_tensor(sigma, dtype=torch.float32, device=self.x0.device).reshape(-1)
        if s.numel() == 1:
            s = s.expand(latent_t)
        elif s.numel() != latent_t:
            raise ValueError(f"{s.numel()} sigmas for {latent_t} frames")
        self.sigmas = s
        self.noise = torch.randn(self.x0.shape, device=self.x0.device, dtype=torch.float32,
                                 generator=generator)
        b = s.view(1, 1, -1, 1, 1)
        self.noised = (1.0 - b) * self.x0 + b * self.noise

    @property
    def t(self):
        """The single timestep `preamble` packs at, shape (1,). Only defined for a clip-wide sigma.

        Raising rather than returning the mean is the point: `preamble` takes ONE video timestep and
        broadcasts silently against a longer tensor, so a per-frame batch handed here would pack
        every frame at frame 0's noise level and still run. `step_backward` builds its own `t` per
        distinct sigma and does not come through here.
        """
        if not bool((self.sigmas == self.sigmas[0]).all()):
            raise ValueError("this batch has per-frame sigmas; there is no single timestep to pack "
                             "at — build one per distinct sigma as step_backward does")
        return 1.0 - self.sigmas[:1]

    @property
    def sigma(self):
        """The clip-wide value when there is one, else the mean — for logging only."""
        return float(self.sigmas.mean())

    @property
    def target(self):
        return self.x0 - self.noise


def step_backward(scd, mm, batches, *, window, chunk_frames, context_noise, score_first_frame,
                  media_start_on, duplicate_pos, checkpoint=True):
    """Flow-matching loss over the frames this step scores, backward included. Returns (loss, stats).

    `batches` is one `Batch` or several drawn from the SAME clip. Several is the reference
    implementation's `decoder_multi_batch`: the encoder reads only the clean latent, so it is
    independent of sigma and noise and one encoder pass can serve K draws. That trade is far better
    here than in the reference — our encoder is 30 blocks against a 7-block decoder, so a second
    draw is ~23% more compute for twice the decoder's gradient signal. It also composes with the
    two-stage backward for free, since the extra frames simply accumulate onto the same leaf.

    The backward is here rather than at the call site because it runs in two stages, and the split
    is the only reason a 7-frame clip fits. Every frame reads the same `enc`, so scoring them all
    and calling `backward` once holds seven decoder graphs at peak; backing up each frame as it is
    scored would instead free the ENCODER graph on the first one. Cutting the graph at `enc` gets
    both: each frame's decoder graph dies when that frame's backward returns, its gradient lands on
    the detached leaf, and the encoder is replayed once at the end with the accumulated sum. The
    gradients are identical to the one-shot version -- it is the same sum, associated differently.

    `context_noise` is the paper's `c~ = c + eta*zeta` (§2, "context corruption during training"),
    applied to the encoder features the decoder conditions on. It is the only place noise is added
    that the base does not add itself, and it is what makes the decoder tolerate a context that
    drifts — which at inference it will, because the encoder's window evicts.

    `score_first_frame=False` is §6.4's `first_frame_cond_p` as applied HERE, which is a reading
    and not a transcription. The hyper is described as aligning with fl2va, i.e. the first frame is
    given. Frame 0 is the one frame with no context half (it gets zeros, because conditioning it on
    its own encoder feature is the leak), so scoring it trains unconditional generation through a
    decoder that was not split to do that. Dropping it on a coin flip is the cheap version of
    "the first frame is given"; substituting clean rows into the noisy pack would be the faithful
    one, and cannot be done without also retiming those rows, which the packer owns.
    """
    if isinstance(batches, Batch):
        batches = (batches,)
    b0 = batches[0]
    clean_kwargs = dict(video_latent=b0.x0, t=torch.tensor([1.0], device=b0.x0.device),
                        text_embeds=b0.text)
    enc, clean_ctx, _ = scd.encode_chunked(chunk_frames, window=window, layer_major=True,
                                           keep_audio=True, checkpoint=checkpoint, **clean_kwargs)
    if context_noise:
        enc = enc + context_noise * torch.randn_like(enc)

    spans = scd.spans(b0.x0, enc.shape[0])
    r = spans.frame_rows
    first = 0 if score_first_frame else 1
    n = (spans.latent_t - first) * len(batches)
    leaf = enc.detach().requires_grad_(enc.requires_grad)

    total, v_sq = 0.0, 0.0
    for batch in batches:
        target = mm.patchify_video(batch.target, scd.base.patch_size)
        # One pack per DISTINCT sigma, keyed by value: a clip-wide sigma packs once, which is what
        # every run before per-frame sigma did and what `evaluate` still does.
        packed = {}

        for f in range(first, spans.latent_t):
            key = round(float(batch.sigmas[f]), 6)
            if key not in packed:
                packed[key] = scd.preamble(
                    video_latent=batch.noised,
                    t=torch.tensor([1.0 - key], device=batch.x0.device),
                    text_embeds=batch.text)
            noisy_h, noisy_ctx, pack = packed[key]

            pred = scd.decode_frame(leaf, clean_ctx, noisy_h, noisy_ctx, spans, f, velocity=True,
                                    media_start=pack["media_start"] if media_start_on else None,
                                    duplicate_pos=duplicate_pos)
            want = target[f * r:(f + 1) * r].to(pred.dtype)
            part = F.mse_loss(pred.float(), want.float()) / n
            if leaf.requires_grad:
                part.backward()
            total += float(part.detach())
            v_sq += float(pred.detach().float().var())

    if leaf.grad is not None:
        enc.backward(leaf.grad)

    # v_std is CastleHill's diagnostic: a decoder that collapses reports a velocity variance that
    # falls toward zero while the loss also falls, because predicting the mean beats predicting
    # nothing. Loss alone cannot tell those apart.
    return total, {"frames": n, "draws": len(batches), "v_std": math.sqrt(v_sq / n),
                   "seq_len": enc.shape[0]}


EVAL_SIGMAS = (0.25, 0.5, 0.9)


def evaluate(scd, mm, clips, names, seed, **kw):
    """Mean loss over a FIXED (clip, sigma, noise) grid. Returns (mean, per-sigma dict).

    The training loss is unreadable as a curve: sigma is redrawn every step and the flow-matching
    loss scales hard with it, so consecutive steps differ by 4x for reasons that have nothing to do
    with learning. Holding sigma and the noise draw fixed is what makes two numbers comparable.

    This is TRAIN loss on TRAIN clips -- 12 of them over 2000 steps is 166 epochs, so it measures
    that the decoder can fit at all, which is exactly the Phase 3 question. It is not a
    generalisation number and must not be reported as one.

    `context_noise` is forced to zero here rather than left to the caller. It is a training
    regulariser, and a fresh `randn` inside a number whose only job is to be comparable across
    steps would put noise on the one axis being read.
    """
    kw = dict(kw, context_noise=0.0)
    out = {}
    with torch.no_grad():
        for s in EVAL_SIGMAS:
            tot = 0.0
            for j, name in enumerate(names):
                clip = clips.load(name, device=next(scd.base.parameters()).device,
                                  dtype=torch.bfloat16)
                g = torch.Generator(device=clip["video_latent"].device)
                g.manual_seed(seed * 7919 + j)
                tot += step_backward(scd, mm, Batch(clip, s, generator=g),
                                     score_first_frame=False, **kw)[0]
            out[f"eval_s{s}"] = round(tot / len(names), 5)
    return round(sum(out.values()) / len(out), 5), out


def build_dry_run():
    """The tiny 6-block model from the unit suites, with a synthetic 2-frame clip.

    A dry run exists because everything above this line is arithmetic that a 66 GB checkpoint does
    not make more or less correct, and the failures it catches — a shape mismatch between the
    velocity head and the patchified target, an optimizer holding parameters that never move — are
    the ones that otherwise surface twenty minutes into a GPU run.
    """
    from test_scd_model import DECODER_SOURCE, ENCODER_DEPTH, build, sample_inputs

    stock, mm = build(os.environ.get("FIZGIG_SRC", "/media/2TB/Fizgig/src"))
    inputs = sample_inputs(mm)
    scd = MiniMaxH3SCD(stock, encoder_depth=ENCODER_DEPTH, decoder_source=DECODER_SOURCE)
    clip = {"name": "dry", "video_latent": inputs["video_latent"].float(),
            "text_embeds": inputs["text_embeds"]}
    return mm, scd, clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="FL2VA/transformer directory; omit with --dry-run")
    ap.add_argument("--clips", default=os.path.join(os.path.dirname(__file__), "clips"))
    ap.add_argument("--out", default="runs/scd_v0")
    ap.add_argument("--steps", type=int, default=None, help="default 2000, or 3 under --dry-run")
    ap.add_argument("--lr", type=float, default=1e-4)
    # 16, not sd-scripts' usual 32. Rank is charged three times on this card -- factors, gradients
    # and two AdamW moments, all fp32 -- so 32 is ~1.6 GB against a 12.5 GB resident base, and 155
    # adapters at rank 16 are still 66 M parameters against 12 clips. Capacity is not the binding
    # constraint here; memory is.
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None, help="default: rank (scale 1.0)")
    ap.add_argument("--window", type=int, default=12,
                    help="encoder KV window in latent frames; 12 is Phase 2's pick off the audio "
                         "curve, and is >= the 7-frame clips here, so it does not bind at 512")
    ap.add_argument("--chunk-frames", type=int, default=1)
    ap.add_argument("--context-noise", type=float, default=0.05,
                    help="eta in the paper's c~ = c + eta*zeta")
    ap.add_argument("--first-frame-cond-p", type=float, default=0.5)
    ap.add_argument("--decoder-text", action="store_true",
                    help="pack text rows into the decoder (1.26x/frame, ~18%% of the N=30 "
                         "speedup). Undecided by measurement — see the design doc")
    ap.add_argument("--own-context-pos", action="store_true",
                    help="give the context half frame f-1's real RoPE rows instead of duplicating "
                         "the target's, which is what CastleHill does and Tier 1 timed")
    ap.add_argument("--sigma-shift", default=None, help="passed to fizgig sample_sigmas")
    # The SCD reference draws a timestep per (batch, frame); v0 drew one per clip, which is 7x less
    # sigma coverage per step at the same cost. Per-frame is not free here the way it is there --
    # the base packs ONE video timestep per forward, so each distinct sigma costs a `preamble`.
    ap.add_argument("--per-frame-sigma", action="store_true", default=True)
    ap.add_argument("--clip-sigma", dest="per_frame_sigma", action="store_false",
                    help="one sigma for the whole clip, as the v0 run did")
    # `decoder_multi_batch` in the reference. The encoder reads only the clean latent, so one
    # encoder pass serves K noise draws; at 30 encoder blocks against 7 decoder blocks a second
    # draw is ~23%% more compute for twice the decoder's gradient signal.
    ap.add_argument("--draws", type=int, default=2,
                    help="decoder noise draws per encoder pass")
    ap.add_argument("--decoder-lr-ratio", type=float, default=2.0,
                    help="decoder adapters train at lr * this; the reference uses 2.0")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="hold the encoder's activations instead of recomputing them. Faster per "
                         "step and does not fit: 30 blocks over ~2100 rows is ~10 GB against "
                         "~9.5 GB of NF4 weights on a 24 GB card")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-clips", type=int, default=4,
                    help="first N clips of the fixed eval grid; 3 sigmas each, ~4 s per forward")
    # 0, not the probes' 48. Those drive the base's own forward, which swaps a block in and out
    # around `_run_block`; the SCD path calls `scd_attention.run_block`, which does not. Every
    # block instance has to be resident. 37 of 50 blocks at NF4 is ~8 GB, which leaves room.
    ap.add_argument("--blocks-to-swap", type=int, default=0)
    ap.add_argument("--base-quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--include-control", action="store_true",
                    help="train on isodiorama640 too — a near-duplicate at a different geometry")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny CPU model, 3 steps, no checkpoint and no clips")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if args.dry_run:
        mm, scd, clip = build_dry_run()
        device, clips, order = torch.device("cpu"), None, ["dry"]
        from fizgig.minimax.trainer import sample_sigmas
        if args.steps is None:
            args.steps, args.save_every, args.log_every = 3, 3, 1
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required without --dry-run")
        mm, load_dit, sample_sigmas = import_fizgig(args.fizgig_src)
        device = torch.device("cuda")
        clips = ClipSet(args.clips, include_control=args.include_control)
        print(f"device {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
              f"{len(clips)} clips", flush=True)
        base = load_dit(args.checkpoint, device=device, compute_dtype=torch.bfloat16,
                        quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                        base_quant="nf4" if args.base_quant == "nf4" else "auto")
        scd = MiniMaxH3SCD(base, encoder_depth=DEFAULT_ENCODER_DEPTH,
                           decoder_source=DEFAULT_DECODER_SOURCE)
        order = clips.names
    if args.steps is None:
        args.steps = 2000

    made = add_lora(scd, rank=args.rank, alpha=args.alpha)
    n_mod, n_par = lora_report(made)
    print(f"lora        : {n_mod} modules, {n_par / 1e6:.2f} M trainable "
          f"(rank {args.rank}, alpha {args.alpha or args.rank})", flush=True)
    scd.eval()                       # the base stays in eval; only the adapters learn

    groups = lora_param_groups(scd, args.lr, args.decoder_lr_ratio)
    print(f"opt         : {len(groups)} groups, lr " +
          " / ".join(f"{g['lr']:.2e}x{len(g['params'])}" for g in groups), flush=True)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    cpu_gen = torch.Generator().manual_seed(args.seed)

    step_kw = dict(window=args.window, chunk_frames=args.chunk_frames,
                   context_noise=args.context_noise, media_start_on=args.decoder_text,
                   duplicate_pos=not args.own_context_pos, checkpoint=not args.no_checkpoint)
    eval_names = [] if args.dry_run else order[:args.eval_clips]

    def run_eval(step, log):
        # eta=0 on the eval grid. Context corruption is a training regulariser; leaving it on would
        # put a fresh randn into a number whose whole job is to be comparable across steps.
        mean, per = evaluate(scd, mm, clips, eval_names, args.seed,
                             **step_kw)
        log.write(json.dumps({"step": step, "eval": mean, **per}) + "\n")
        log.flush()
        print(f"eval  {step:>5}  {mean:.4f}   " +
              "  ".join(f"{k[5:]} {v:.4f}" for k, v in per.items()), flush=True)

    t0 = time.time()
    with open(log_path, "a") as log:
        for step in range(1, args.steps + 1):
            epoch, i = divmod(step - 1, len(order))
            name = epoch_order(order, args.seed, epoch)[i]
            clip_data = clip if args.dry_run else clips.load(name, device=device,
                                                             dtype=torch.bfloat16)
            # Each draw gets its OWN sigmas as well as its own noise. The reference varies only the
            # noise across `decoder_multi_batch`, but the encoder pass is what the reuse is for and
            # it is independent of both, so redrawing sigma too costs nothing and widens the
            # coverage the extra draw is being bought for.
            n_sigma = clip_data["video_latent"].shape[2] if args.per_frame_sigma else 1
            batches = [Batch(clip_data,
                             sample_sigmas(n_sigma, device, shift=args.sigma_shift, generator=gen),
                             generator=gen)
                       for _ in range(args.draws)]

            if eval_names and (step == 1 or step % args.eval_every == 0):
                run_eval(step - 1, log)

            score_first = torch.rand(1, generator=cpu_gen).item() >= args.first_frame_cond_p
            opt.zero_grad(set_to_none=True)
            loss, stats = step_backward(scd, mm, batches, score_first_frame=score_first, **step_kw)

            gnorm = torch.nn.utils.clip_grad_norm_(lora_parameters(scd), 1.0)
            opt.step()
            if not args.dry_run:
                stats["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                torch.cuda.reset_peak_memory_stats()

            rec = {"step": step, "clip": batches[0].name,
                   "sigma": round(sum(b.sigma for b in batches) / len(batches), 4),
                   "loss": loss, "grad_norm": float(gnorm),
                   "elapsed_s": round(time.time() - t0, 1), **stats}
            log.write(json.dumps(rec) + "\n")
            if step % args.log_every == 0 or step == 1:
                log.flush()
                print(f"step {step:>5}  loss {rec['loss']:.4f}  v_std {rec['v_std']:.3f}  "
                      f"sigma {rec['sigma']:.3f}  |g| {rec['grad_norm']:.2e}  "
                      f"{rec.get('peak_gb', 0):.1f}GB  {rec['elapsed_s']:.0f}s  {rec['clip']}",
                      flush=True)

            if step % args.save_every == 0 or step == args.steps:
                from safetensors.torch import save_file
                path = os.path.join(args.out, f"scd_lora_{step:06d}.safetensors")
                save_file(lora_state_dict(scd), path, metadata={
                    "step": str(step), "rank": str(args.rank),
                    "alpha": str(args.alpha or args.rank),
                    "encoder_depth": str(scd.encoder_depth),
                    "decoder_source": ",".join(str(i) for i in scd.decoder_source),
                    "window": str(args.window), "clips": ",".join(order),
                    "decoder_text": str(bool(args.decoder_text)),
                    "duplicate_pos": str(not args.own_context_pos),
                    "draws": str(args.draws),
                    "per_frame_sigma": str(bool(args.per_frame_sigma)),
                    "decoder_lr_ratio": str(args.decoder_lr_ratio),
                })
                print(f"saved       : {path}", flush=True)

        if eval_names:
            run_eval(args.steps, log)

    print(f"done        : {args.steps} steps in {time.time() - t0:.0f}s, log {log_path}")


if __name__ == "__main__":
    main()
