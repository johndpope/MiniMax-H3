#!/usr/bin/env python3
"""Invariants for `phase3_sample`. CPU, seconds, no weights and no GPU.

The sampler's job is to produce a number that decides whether the split works, so the failure that
matters is not a crash — it is a sampler that scores well for the wrong reason. Both cases here are
about that: an `ar` rollout that can see the ground truth would post near-perfect correlation and
be reported as a pass, and a `score` that reads only MSE would call a collapsed decoder a success.

Needs fizgig on the path (`--fizgig-src`), which CI does not have; CI lints and byte-compiles.

Usage:
    python3 scripts/scd/test_phase3_sample.py
"""

import argparse
import sys

import torch

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_ar_never_reads_the_ground_truth(mm, scd, clip):
    """Two different truths, same seed, `mode="ar"` — the samples must be bit-identical.

    A sampler handed the answer would post high correlation everywhere and the kill criterion would
    be met by nothing at all, so this is the case that protects the result rather than the code.

    Worth knowing what it does and does not catch. Seeding `ctx_buf` from `x0` alone is NOT caught,
    and that is a fact about the model rather than a hole: the encoder is frame-causal and every
    frame below `f` has already been overwritten by generated content, so ground truth in the
    frames above `f` cannot reach it. The dangerous version is that change TOGETHER with a missing
    write-back — the combination that turns `ar` into `oracle` — and that is caught here. The
    write-back on its own is `test_ar_frames_condition_on_the_generated_ones`.

    The `oracle` half is not decoration: without it this passes for a model that ignores its
    context entirely, which is also what a broken `decoder_frame_input` looks like.
    """
    from phase3_sample import sample

    kw = dict(seed=3, window=None, chunk_frames=1, media_start_on=False, duplicate_pos=True)
    sigmas = [0.9, 0.5, 0.0]
    torch.manual_seed(11)
    other = dict(clip, video_latent=torch.randn_like(clip["video_latent"]))

    a = sample(scd, mm, clip, sigmas, "ar", **kw)
    b = sample(scd, mm, other, sigmas, "ar", **kw)
    assert torch.equal(a, b), \
        f"the AR sample moved by {(a - b).abs().max():.3e} when only the GROUND TRUTH changed — " \
        "the rollout is conditioning on the frames it is supposed to be generating"

    c = sample(scd, mm, clip, sigmas, "oracle", **kw)
    d = sample(scd, mm, other, sigmas, "oracle", **kw)
    assert not torch.equal(c, d), \
        "the oracle sample ignored the ground truth too — the context half is not reaching the " \
        "decoder, so the AR case above passes for the wrong reason"


@case
def test_ar_frames_condition_on_the_generated_ones(mm, scd, clip):
    """The write-back is load-bearing: without it every frame is decoded against a zero context.

    Comparing two seeds cannot show this — a different seed redraws every frame's own noise, so all
    frames move either way. The comparison that can is `ar` against `oracle` ON A ZERO CLIP: oracle
    then holds a zero context for the whole clip, which is exactly what `ar` degenerates to if the
    generated frames are never written back. Frame 0 must agree (it has no context in either) and
    every later frame must not.

    A rollout that skips the write-back still produces plausible-looking output and still scores,
    it just never compounds — which is to say it answers a question nobody asked.
    """
    from phase3_sample import sample

    kw = dict(seed=1, window=None, chunk_frames=1, media_start_on=False, duplicate_pos=True)
    sigmas = [0.9, 0.5, 0.0]
    zeros = dict(clip, video_latent=torch.zeros_like(clip["video_latent"]))

    roll = sample(scd, mm, zeros, sigmas, "ar", **kw)
    flat = sample(scd, mm, zeros, sigmas, "oracle", **kw)
    assert torch.equal(roll[:, :, 0], flat[:, :, 0]), \
        "frame 0 differs between ar and a zero context, but it has no context half in either"
    same = [f for f in range(1, roll.shape[2]) if torch.equal(roll[:, :, f], flat[:, :, f])]
    assert not same, \
        f"frames {same} are identical to a zero-context decode — generated frames are not being " \
        "written back into the context the next frame reads"


@case
def test_a_perfect_velocity_recovers_the_clip(mm, scd, clip):
    """Feed the sampler the TRUE velocity and it must return the true latent, to ~1e-5.

    This is the sampler's agreement with the trainer, written as arithmetic. Training's target is
    `x0 - noise` at `noised = (1-s)x0 + s*noise`; at s=1 the noisy latent IS the noise, so one
    Euler step to 0 gives `noise + 1.0*(x0 - noise) = x0` exactly. Any of the three things that
    could be wrong independently — the sign of the step, `(s - s_next)` vs `(s_next - s)`, or an
    `unpatchify` that disagrees with the `patchify` the loss was scored through — breaks this and
    breaks nothing else that is checkable without a trained model.

    It matters because a sign-flipped sampler is not a crash. It runs, it produces a latent with a
    sane standard deviation, and it scores near zero correlation — which is indistinguishable from
    "the split did not learn", and would be reported as a kill.
    """
    from phase3_sample import sample

    x0 = clip["video_latent"].float()
    r = None

    def oracle_velocity(_enc, _cctx, _nh, _nctx, spans, frame, **_kw):
        nonlocal r
        r = spans.frame_rows
        rows = mm.patchify_video(x0[:, :, frame:frame + 1], scd.base.patch_size)
        # what the sampler subtracts is `noise`, and at s=1 the frame it holds IS the noise
        return rows - mm.patchify_video(noise_seen[frame], scd.base.patch_size)

    # The sampler draws its own noise, so capture it through the same generator and seed.
    g = torch.Generator(device=x0.device).manual_seed(4)
    noise_seen = [torch.randn(x0[:, :, f:f + 1].shape, dtype=torch.float32, generator=g)
                  for f in range(x0.shape[2])]

    real = scd.decode_frame
    scd.decode_frame = oracle_velocity
    try:
        got = sample(scd, mm, clip, [1.0, 0.0], "oracle", seed=4, window=None, chunk_frames=1,
                     media_start_on=False, duplicate_pos=True)
    finally:
        scd.decode_frame = real

    assert r is not None, "decode_frame was never called"
    err = float((got - x0).abs().max())
    assert err < 1e-5, \
        f"a perfect velocity landed {err:.3e} from the truth — the Euler step or the unpatchify " \
        "disagrees with the convention the loss is scored in"


@case
def test_score_separates_a_collapse_from_a_fit(_mm, _scd, clip):
    """`corr` is ~1 for the truth, ~0 for noise, and ~0 for a confident constant that beats the
    truth's own MSE. The third is the one MSE alone gets wrong, and is what a collapsed decoder
    looks like: predicting the mean is a lower error than predicting anything, and carries nothing.
    """
    from phase3_sample import score

    # Offset, because a zero-mean fixture makes the uncentered dot product agree with the centered
    # one by accident and real latents are not zero-mean.
    t = clip["video_latent"].float() + 3.0
    assert score(t, t)["corr"] > 0.999 and score(t, t)["mse"] < 1e-9

    torch.manual_seed(5)
    noise = score(torch.randn_like(t) * t.std(), t)
    assert abs(noise["corr"]) < 0.1, f"noise scored corr {noise['corr']}"

    flat = score(torch.full_like(t, float(t.mean())), t)
    assert abs(flat["corr"]) < 0.1, f"a constant scored corr {flat['corr']}"
    assert flat["mse"] < noise["mse"], \
        "the constant did not beat noise on MSE, so this fixture cannot show the trap corr exists" \
        " to catch"


@case
def test_frame_0_is_excluded_from_corr_ctx(_mm, _scd, clip):
    """`corr_ctx` averages frames 1.. only. Frame 0 has a zeroed context half and is trained on
    roughly half of steps, so folding it into the headline number moves it for a reason that has
    nothing to do with whether the split works."""
    from phase3_sample import score

    t = clip["video_latent"].float()
    pred = t.clone()
    torch.manual_seed(2)
    pred[:, :, 0] = torch.randn_like(pred[:, :, 0])          # ruin frame 0 only
    s = score(pred, t)
    assert s["corr_ctx"] > 0.999, f"corr_ctx {s['corr_ctx']} moved with frame 0"
    assert s["corr"] < 0.95, f"corr {s['corr']} did not move with frame 0; the two are the same"


@case
def test_seeding_frame_0_does_not_leak_the_later_frames(mm, scd, clip):
    """`seed_frames=1` hands the rollout frame 0 and NOTHING else.

    The seeded frame has to reach `ctx_buf`, not just the output, or frame 1 conditions on zeros —
    and the moment ground truth is written into an `ar` buffer, the obvious way to get it wrong is
    to write all of it and quietly recreate `oracle`. So: frame 0 must be the truth exactly, and
    perturbing frames 1.. of the truth must not move the sample at all.
    """
    from phase3_sample import sample

    kw = dict(seed=5, window=None, chunk_frames=1, media_start_on=False, duplicate_pos=True)
    sigmas = [0.9, 0.5, 0.0]
    x0 = clip["video_latent"].float()

    got = sample(scd, mm, clip, sigmas, "ar", seed_frames=1, **kw)
    assert torch.equal(got[:, :, 0], x0[:, :, 0].to(got.dtype)), \
        "frame 0 is not the truth, so the seed was not applied"
    assert not torch.equal(got[:, :, 1], x0[:, :, 1].to(got.dtype)), \
        "frame 1 came back as the truth too — the seed is copying more than it was asked for"

    torch.manual_seed(13)
    moved = x0.clone()
    moved[:, :, 1:] = torch.randn_like(moved[:, :, 1:])       # ruin every UNSEEDED frame
    other = sample(scd, mm, dict(clip, video_latent=moved), sigmas, "ar", seed_frames=1, **kw)
    assert torch.equal(got, other), \
        f"the sample moved by {(got - other).abs().max():.3e} when only the UNSEEDED truth " \
        "changed — seeding has turned the ar rollout into an oracle"

    # And the default is still the strict rollout the rest of the suite pins.
    plain = sample(scd, mm, clip, sigmas, "ar", **kw)
    assert not torch.equal(plain[:, :, 0], x0[:, :, 0].to(plain.dtype)), \
        "seed_frames defaults to something other than 0; earlier runs are not comparable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    args = ap.parse_args()

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import os

    os.environ.setdefault("FIZGIG_SRC", args.fizgig_src)
    from phase3_train import build_dry_run

    failures = 0
    for fn in CASES:
        mm, scd, clip = build_dry_run()
        # Seven frames because the grid is 5n+2 and two would leave `ar` with a single frame that
        # reads a context at all. NON-SQUARE on purpose: 8x12 latents give h=4, w=6, so an
        # `unpatchify` that transposes them is a shape error rather than a silent scramble. The
        # real clips are 32x32 and would not have caught it.
        torch.manual_seed(7)
        clip = dict(clip, video_latent=torch.randn(1, 24, 7, 8, 12))
        try:
            fn(mm, scd, clip)
            print(f"ok    {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
