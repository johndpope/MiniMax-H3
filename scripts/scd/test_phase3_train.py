#!/usr/bin/env python3
"""Invariants for `phase3_train`. CPU, seconds, no weights and no GPU.

The train loop's arithmetic is checkable without the 66 GB checkpoint: the flow-matching
convention is algebra, and the two-stage backward is an associativity claim about a sum. What the
tiny model cannot check is whether the result fits in 24 GB, which is the reason the two-stage
backward exists at all.

Needs fizgig on the path (`--fizgig-src`), which CI does not have; CI lints and byte-compiles.

Usage:
    python3 scripts/scd/test_phase3_train.py
"""

import argparse
import sys

import torch
import torch.nn.functional as F

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_noised_and_target_match_fizgigs_convention(_mm, _scd, clip):
    """`noised = (1-s)x0 + s*noise` and the target is `x0 - noise`, not `noise - x0`.

    A sign flip here trains a model that runs the ODE backwards: the loss falls exactly as fast,
    every diagnostic looks healthy, and sampling produces noise. It is only checkable against the
    base's own convention, which is why this is pinned as algebra rather than trusted as a comment.
    """
    from phase3_train import Batch

    b = Batch(clip, 0.25, generator=torch.Generator().manual_seed(0))
    x0, noise = b.x0, b.noise
    assert torch.allclose(b.noised, 0.75 * x0 + 0.25 * noise, atol=0), "noised is not (1-s)x0+s*eps"
    assert torch.allclose(b.target, x0 - noise, atol=0), "target is not x0 - noise"
    assert torch.allclose(b.t, torch.tensor([0.75])), f"t is {b.t}, not 1 - sigma"


@case
def test_sigma_0_is_the_clean_latent(_mm, _scd, clip):
    """The endpoints of the interpolation are the two things they are named after. Cheap, and it
    catches an off-by-one in the direction of `s` that the midpoint would hide."""
    from phase3_train import Batch

    g = torch.Generator().manual_seed(0)
    assert torch.allclose(Batch(clip, 0.0, generator=g).noised, clip["video_latent"].float())
    b1 = Batch(clip, 1.0, generator=g)
    assert torch.allclose(b1.noised, b1.noise)


@case
def test_two_stage_backward_equals_one_shot(mm, scd, clip):
    """Cutting the graph at `enc` changes nothing about the gradients.

    This is the memory fix's whole correctness claim, and it is a claim that fails silently: if a
    second tensor ever crosses from the encoder into the decoder -- a KV cache, a pooled feature --
    the cut stops covering the path, the encoder adapters quietly receive only part of their
    gradient, and the run still trains. Comparing against the un-cut sum is the only way to see it.
    """
    from phase3_train import Batch, step_backward
    from scd_lora import add_lora, lora_parameters

    made = add_lora(scd, rank=4)
    for m in made.values():
        m.lora_up.weight.data.normal_(0, 0.02)
    params = lora_parameters(scd)
    kw = dict(window=None, chunk_frames=1, context_noise=0.0, score_first_frame=True,
              media_start_on=False, duplicate_pos=True, checkpoint=False)

    def run(two_stage):
        b = Batch(clip, 0.7, generator=torch.Generator().manual_seed(1))
        for p in params:
            p.grad = None
        if two_stage:
            loss, _ = step_backward(scd, mm, b, **kw)
        else:
            enc, cctx, _ = scd.encode_chunked(1, window=None, layer_major=True, keep_audio=True,
                                              video_latent=b.x0, t=torch.tensor([1.0]),
                                              text_embeds=b.text)
            noisy_h, nctx, _ = scd.preamble(video_latent=b.noised, t=b.t, text_embeds=b.text)
            sp = scd.spans(b.x0, enc.shape[0])
            r = sp.frame_rows
            tgt = mm.patchify_video(b.target, scd.base.patch_size)
            total = b.x0.new_zeros(())
            for f in range(sp.latent_t):
                pred = scd.decode_frame(enc, cctx, noisy_h, nctx, sp, f, velocity=True,
                                        duplicate_pos=True)
                total = total + F.mse_loss(pred.float(), tgt[f * r:(f + 1) * r].float()) / sp.latent_t
            total.backward()
            loss = float(total.detach())
        return loss, [p.grad.clone() for p in params]

    cut_loss, cut_grads = run(True)
    ref_loss, ref_grads = run(False)

    assert abs(cut_loss - ref_loss) < 1e-6, f"loss {cut_loss} vs {ref_loss}"
    worst = max(float((a - b).abs().max()) for a, b in zip(cut_grads, ref_grads))
    assert worst < 1e-8, f"two-stage gradient differs from one-shot by {worst:.3e}"
    dead = [i for i, g in enumerate(cut_grads) if float(g.abs().sum()) == 0.0]
    assert not dead, f"{len(dead)} adapters got no gradient through the cut"


@case
def test_frames_are_freed_as_they_are_scored(mm, scd, clip):
    """The peak number of live saved tensors is well below the total the step packs.

    Gradient equality does not distinguish the cut from `part.backward(retain_graph=True)` -- that
    mutant produces bit-identical gradients and OOMs on the real card, which is the only failure
    mode this whole change exists to prevent. So this counts instead of comparing: every tensor
    autograd saves is wrapped, and a weakref fires when the graph holding it dies. If frames are
    freed as they are scored, peak lands near half of total on this fixture; if any are retained it
    goes to ~0.99, because nothing is released until the step ends. 0.75 sits in the gap.
    """
    import weakref

    from phase3_train import Batch, step_backward
    from scd_lora import add_lora

    made = add_lora(scd, rank=4)
    for m in made.values():
        m.lora_up.weight.data.normal_(0, 0.02)

    class Held:
        __slots__ = ("t", "__weakref__")

        def __init__(self, t):
            self.t = t

    n = {"live": 0, "peak": 0, "total": 0}

    def pack(t):
        h = Held(t)
        n["live"] += 1
        n["total"] += 1
        n["peak"] = max(n["peak"], n["live"])
        weakref.finalize(h, lambda: n.__setitem__("live", n["live"] - 1))
        return h

    b = Batch(clip, 0.7, generator=torch.Generator().manual_seed(1))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda h: h.t):
        step_backward(scd, mm, b, window=None, chunk_frames=1, context_noise=0.0,
                      score_first_frame=True, media_start_on=False, duplicate_pos=True,
                      checkpoint=False)

    ratio = n["peak"] / n["total"]
    assert ratio < 0.75, \
        f"peak {n['peak']} of {n['total']} saved tensors live at once ({ratio:.2f}) — the " \
        "decoder's per-frame graphs are being retained to the end of the step"


@case
def test_the_eval_grid_is_reproducible(mm, scd, clip):
    """Two evaluations of an unchanged model return the same number, to the bit.

    The eval exists so that step 0 and step 2000 are comparable, and everything that would break
    that is invisible in a single call: a redrawn sigma, a fresh noise tensor, context corruption
    left on. A stub `ClipSet` stands in for the cache so this needs no clips on disk -- what is
    under test is `evaluate`'s determinism, not the loader's.
    """
    from phase3_train import evaluate
    from scd_lora import add_lora

    made = add_lora(scd, rank=4)
    for m in made.values():
        m.lora_up.weight.data.normal_(0, 0.02)

    class Stub:
        def load(self, name, device=None, dtype=None):
            return clip

    # eta=2.0 on the way in: `evaluate` is supposed to override it, and passing zero here would
    # test the caller's discipline instead of the function's.
    kw = dict(window=None, chunk_frames=1, context_noise=2.0, media_start_on=False,
              duplicate_pos=True, checkpoint=False)
    first = evaluate(scd, mm, Stub(), ["a", "b"], 0, **kw)
    again = evaluate(scd, mm, Stub(), ["a", "b"], 0, **kw)
    assert first == again, f"eval is not reproducible: {first} then {again}"
    assert len(first[1]) == 3, f"eval reported {len(first[1])} sigmas, not 3"


@case
def test_skipping_the_first_frame_scores_the_rest(mm, scd, clip):
    """`score_first_frame=False` drops exactly one frame from the average, not the whole clip.

    Frame 0 is the one with a zeroed context half, so a bug that dropped every frame with a
    predecessor -- or none -- would leave a loop that still returns a number.
    """
    from phase3_train import Batch, step_backward
    from scd_lora import add_lora

    add_lora(scd, rank=4)
    kw = dict(window=None, chunk_frames=1, context_noise=0.0, media_start_on=False,
              duplicate_pos=True, checkpoint=False)
    b = Batch(clip, 0.7, generator=torch.Generator().manual_seed(1))
    _, all_f = step_backward(scd, mm, b, score_first_frame=True, **kw)
    _, tail = step_backward(scd, mm, b, score_first_frame=False, **kw)
    assert tail["frames"] == all_f["frames"] - 1, \
        f"dropping frame 0 went from {all_f['frames']} frames to {tail['frames']}"


@case
def test_context_noise_perturbs_the_loss(mm, scd, clip):
    """`context_noise` reaches the encoder features. It is a float that multiplies a `randn_like`,
    which is exactly the kind of line that gets wired to nothing and never complains."""
    from phase3_train import Batch, step_backward
    from scd_lora import add_lora

    made = add_lora(scd, rank=4)
    for m in made.values():
        m.lora_up.weight.data.normal_(0, 0.02)
    kw = dict(window=None, chunk_frames=1, score_first_frame=True, media_start_on=False,
              duplicate_pos=True, checkpoint=False)

    def loss_at(eta):
        torch.manual_seed(3)
        b = Batch(clip, 0.7, generator=torch.Generator().manual_seed(1))
        return step_backward(scd, mm, b, context_noise=eta, **kw)[0]

    assert loss_at(0.0) == loss_at(0.0), "eta=0 is not deterministic; something else draws noise"
    assert loss_at(2.0) != loss_at(0.0), "eta=2 left the loss unchanged; context noise is not wired"


@case
def test_per_frame_sigma_noises_each_frame_at_its_own_level(_mm, _scd, clip):
    """A vector sigma must noise frame f at sigma[f], and must refuse to name one timestep.

    The failure this catches is the quiet one: `s.view(1,1,-1,1,1)` broadcasting against the wrong
    axis noises every frame at sigma[0] and leaves the loss looking entirely healthy, because the
    target does not depend on sigma at all. Only the noised latent knows.
    """
    from phase3_train import Batch

    sig = torch.linspace(0.1, 0.9, 7)
    b = Batch(clip, sig, generator=torch.Generator().manual_seed(0))
    for f in range(7):
        want = (1 - sig[f]) * b.x0[:, :, f] + sig[f] * b.noise[:, :, f]
        assert torch.allclose(b.noised[:, :, f], want, atol=1e-6), \
            f"frame {f} was not noised at sigma {sig[f]:.2f} — the per-frame axis is wrong"

    try:
        b.t
    except ValueError:
        pass
    else:
        raise AssertionError("a per-frame batch handed back a single timestep; `preamble` would "
                             "broadcast it and pack every frame at frame 0's noise level")
    assert torch.allclose(Batch(clip, 0.4).t, torch.tensor([0.6])), "a scalar sigma lost its t"


@case
def test_repeated_draws_average_rather_than_accumulate(mm, scd, clip):
    """K identical draws must give the SAME loss and gradient as one, not K times either.

    `decoder_multi_batch` divides by `n = frames * draws`, and getting that denominator wrong is
    invisible in the loss curve's shape — it rescales every step equally — while silently making
    the effective learning rate K times what the flag says.
    """
    from phase3_train import Batch, step_backward
    from scd_lora import add_lora, lora_parameters

    made = add_lora(scd, rank=4)
    for m in made.values():
        m.lora_up.weight.data.normal_(0, 0.02)
    params = lora_parameters(scd)
    kw = dict(window=None, chunk_frames=1, context_noise=0.0, score_first_frame=True,
              media_start_on=False, duplicate_pos=True, checkpoint=False)

    def run(k):
        for p in params:
            p.grad = None
        bs = [Batch(clip, 0.7, generator=torch.Generator().manual_seed(1)) for _ in range(k)]
        loss, stats = step_backward(scd, mm, bs, **kw)
        return loss, stats, [p.grad.clone() for p in params]

    one, s1, g1 = run(1)
    three, s3, g3 = run(3)
    assert s1["draws"] == 1 and s3["draws"] == 3, f"draws not reported: {s1}, {s3}"
    assert s3["frames"] == 3 * s1["frames"], f"{s3['frames']} scored frames, not 3x {s1['frames']}"
    assert abs(one - three) < 1e-6, f"three identical draws moved the loss {one} -> {three}"
    worst = max(float((a - b).abs().max()) for a, b in zip(g1, g3))
    assert worst < 1e-8, f"three identical draws changed the gradient by {worst:.3e}"


@case
def test_distinct_sigmas_cost_one_encoder_pass_and_one_pack_each(mm, scd, clip):
    """The whole point of the reuse: draws multiply `preamble`, never `encode_chunked`.

    And a clip-wide sigma must still pack exactly ONCE per draw — that is what keeps `evaluate`'s
    fixed grid, and every number the v0 run produced, comparable with runs that pack per frame.
    """
    from phase3_train import Batch, step_backward

    kw = dict(window=None, chunk_frames=1, context_noise=0.0, score_first_frame=True,
              media_start_on=False, duplicate_pos=True, checkpoint=False)
    calls = {"enc": 0, "pack": 0}
    real_enc, real_pre = scd.encode_chunked, scd.preamble

    def counted_enc(*a, **k):
        calls["enc"] += 1
        return real_enc(*a, **k)

    def counted_pre(*a, **k):
        calls["pack"] += 1
        return real_pre(*a, **k)

    scd.encode_chunked, scd.preamble = counted_enc, counted_pre
    try:
        # No grad: this counts calls, and the un-adapted model has every base parameter trainable,
        # so a real backward would free the `clean_ctx` graph the next frame still needs.
        with torch.no_grad():
            flat = [Batch(clip, 0.7, generator=torch.Generator().manual_seed(i)) for i in range(2)]
            step_backward(scd, mm, flat, **kw)
            # 1 + K, not K: `encode_chunked` packs the CLEAN sequence through `preamble` too, and
            # that one is the pass being shared.
            assert calls == {"enc": 1, "pack": 3}, \
                f"two clip-wide draws cost {calls}, not 1 enc / 1 clean + 2 noisy packs"

            calls.update(enc=0, pack=0)
            varied = [Batch(clip, torch.linspace(0.1, 0.9, 7),
                            generator=torch.Generator().manual_seed(i)) for i in range(2)]
            step_backward(scd, mm, varied, **kw)
            assert calls["enc"] == 1, f"per-frame sigma re-ran the encoder {calls['enc']} times; " \
                                      "it reads only the CLEAN latent and cannot depend on sigma"
            assert calls["pack"] == 15, \
                f"{calls['pack']} packs, not 1 clean + 2 draws x 7 distinct sigmas"
    finally:
        scd.encode_chunked, scd.preamble = real_enc, real_pre


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
        # A fresh model per case: `add_lora` mutates the tree in place and several cases write to
        # parameters, so a shared one would make the order significant.
        mm, scd, clip = build_dry_run()
        # Seven latent frames, not the shared fixture's two, and 7 because the grid is 5n+2 so
        # there is nothing between. Frame 0's context half is zeros, so a 2-frame clip has exactly
        # ONE frame reading `enc` -- and the two-stage backward, whose whole point is that several
        # frames share that root, would be trivially correct on it. It is also the real clips'
        # length, which is the geometry the memory fix was sized against.
        torch.manual_seed(7)
        clip = dict(clip, video_latent=torch.randn(1, 24, 7, 8, 8))
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
