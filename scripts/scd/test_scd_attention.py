#!/usr/bin/env python3
"""Phase 2 invariants: frame spans, the causal mask, and the encoder KV cache. CPU, seconds.

Shares `TINY` and `build` with `test_scd_model.py` so there is one tiny-model definition; runs a
7-frame latent because everything here is vacuous on a single frame.

The load-bearing case is `test_loose_mask_leaks_across_blocks`. It asserts a FAILURE — that
Phase 0's mask, which restricts only video queries, stops being frame-causal the moment you
stack two blocks. That mask is what axis (b)'s drift numbers were measured with, and it is the
mask anyone would write first, so the leak is pinned here rather than left as a comment.

Usage:
    python3 scripts/scd/test_scd_attention.py
"""

import argparse
import os
import sys

import torch

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def sample_inputs(mm, latent_t=7):
    """Same fixed-audio-rows discipline as Phase 1: left to itself the base redraws its silence
    rows every call, and any bit-exact comparison across two forwards then fails by ~0.6."""
    from test_scd_model import TINY

    torch.manual_seed(1)
    n_audio = mm.audio_latents_for_frames(mm.pixel_frames_for_latent(latent_t))
    return dict(video_latent=torch.randn(1, 24, latent_t, 8, 8), t=torch.tensor([0.5]),
                text_embeds=torch.randn(1, 5, TINY["text_dim"]),
                audio_noise=torch.randn(n_audio * mm.AUDIO_CHANNELS, 32))


def prefix_delta(scd, h, ctx, t, keep_t, context_sees_video, depth):
    """Disagreement between the rows at time <= `keep_t` of a masked full-clip run and a run in
    which the later rows DO NOT EXIST. Zero is what "causal" has to mean for a cache to be sound;
    anything else is future information reaching the prefix.

    Takes the row-time vector rather than a frame count, so the same check covers the video-only
    clock and the AV clock — under the latter the prefix is a gather, not a row slice, and a
    version written around `rows_for_frames` would simply not be able to express the question.

    Returns (ordered, context) separately because that is the mechanism: the loose mask lets
    future frames into the CONTEXT rows immediately, and they reach the video rows one block later.
    """
    from scd_attention import CONTEXT, causal_mask, run_block

    keep = (t <= keep_t).nonzero().squeeze(1)
    full_mask = causal_mask(t, t, context_sees_video)
    pre_mask = causal_mask(t[keep], t[keep], context_sees_video)
    t_emb, mod_row, cos, sin = ctx
    pre_ctx = (t_emb, mod_row[keep], cos[keep], sin[keep])

    x, xp = h, h[keep]
    with torch.no_grad():
        for block in list(scd.encoder_blocks)[:depth]:
            x = run_block(block, x, ctx, mask=full_mask)
            xp = run_block(block, xp, pre_ctx, mask=pre_mask)
    d = (x[keep] - xp).abs()
    is_context = t[keep] == CONTEXT
    return d[~is_context].max().item(), d[is_context].max().item()


@case
def test_spans_agree_with_the_packed_sequence(scd, h, ctx, sp, mm, _pack):
    """`FrameSpans` locates the video segment by subtraction; the base tags every row by modality.
    They must agree, or the mask is causal over the wrong rows and nothing downstream notices."""
    tags = ctx[1] % mm.MODALITY_NUM
    is_video = tags == mm.VIDEO_TAG
    assert is_video[sp.video_start:].all(), "rows at/after video_start are not all video-tagged"
    assert not is_video[:sp.video_start].any(), "a context row is video-tagged"
    assert sp.seq_len == h.shape[0]
    _ = scd


@case
def test_mask_is_frame_causal(scd, _h, _ctx, sp, _mm, _pack):
    from scd_attention import causal_mask

    frames = sp.frame_index()
    m = causal_mask(frames, frames)
    q2 = sp.rows_for_frames(2, 3).start
    assert m[q2, sp.rows_for_frames(1, 2)].all(), "frame 2 cannot see frame 1"
    assert m[q2, sp.rows_for_frames(2, 3)].all(), "frame 2 cannot see itself"
    assert not m[q2, sp.rows_for_frames(3, sp.latent_t)].any(), "frame 2 sees a later frame"
    assert m[q2, :sp.video_start].all(), "frame 2 cannot see the context rows"
    assert m[0, :sp.video_start].all(), "context rows cannot see each other"
    assert not m[0, sp.video_start:].any(), "a context row sees video under the strict mask"
    _ = scd


@case
def test_loose_mask_lets_context_see_video(_scd, _h, _ctx, sp, _mm, _pack):
    """The one behavioural difference between the two masks, isolated from its consequences."""
    from scd_attention import causal_mask

    frames = sp.frame_index()
    loose = causal_mask(frames, frames, context_sees_video=True)
    assert loose[0, sp.video_start:].all(), "loose mask should leave context queries unrestricted"
    q1 = sp.rows_for_frames(1, 2).start
    assert not loose[q1, sp.rows_for_frames(2, sp.latent_t)].any(), \
        "loose mask must still be causal for VIDEO queries — that part was never in doubt"


@case
def test_unmasked_matches_base(scd, h, ctx, _sp, _mm, _pack):
    """`attention()` with no mask and no cache is the base's own attention, bit for bit.

    Everything else here compares a masked run against an unmasked one, so a masked path that
    quietly differed from the base by a reordering would show up as a drift number and be read
    as a property of the model. `torch.equal`, not `allclose`.
    """
    from scd_attention import attention

    block = scd.encoder_blocks[0]
    _, _, cos, sin = ctx
    x = block.norm1(h)
    with torch.no_grad():
        assert torch.equal(attention(block.attn, x, cos, sin), block.attn(x, cos=cos, sin=sin))


@case
def test_strict_mask_composes_across_blocks(scd, h, ctx, sp, _mm, _pack):
    """Prefix equivalence through the WHOLE encoder, not one block.

    float32 roundoff only — the residual stream is ~1e0, so 1e-5 is four orders below signal and
    still far under the 3.7e-3 the loose mask produces at depth 2.
    """
    vid, ctxd = prefix_delta(scd, h, ctx, sp.frame_index(), 0, False, len(scd.encoder_blocks))
    assert max(vid, ctxd) < 1e-5, \
        f"strict mask is not prefix-exact through the stack: video {vid:.3e}, context {ctxd:.3e}"


@case
def test_loose_mask_leaks_across_blocks(scd, h, ctx, sp, _mm, _pack):
    """Phase 0's mask is frame-causal for one block and NOT for a stack, and this shows the path.

    At depth 1 the video rows are clean — which is why Phase 0's single-block self-test passed
    and why axis (b)'s per-block drift numbers stand — but the CONTEXT rows have already absorbed
    every future frame. One block later that contamination is in the video rows too. Asserted in
    both directions: if the video number at depth 2 ever comes back clean, the encoder could use
    the cheaper mask and §7 would need rewriting.
    """
    v1, c1 = prefix_delta(scd, h, ctx, sp.frame_index(), 0, True, 1)
    v2, _ = prefix_delta(scd, h, ctx, sp.frame_index(), 0, True, 2)
    assert v1 < 1e-5, f"loose mask should leave VIDEO rows causal for a single block; got {v1:.3e}"
    assert c1 > 1e-3, f"context rows should already be contaminated at depth 1; got {c1:.3e}"
    assert v2 > 1e-4, f"loose mask no longer leaks into video at depth 2 (got {v2:.3e})"


@case
def test_chunked_matches_full(scd, _h, _ctx, sp, mm, pack):
    """The cache is not an approximation: chunked encoding is one masked pass, reassociated.

    Run on BOTH clocks. Under the AV clock a chunk is a gather over two non-adjacent slabs of the
    sequence and the cache fills in time order rather than row order, so this is the case that
    fails if `encode_chunked` reverts to slicing — the video-only clock cannot tell the two apart
    because there the packed order already IS the causal order.

    Compared against `encode(mask=...)` on the same inputs rather than against a stored number,
    so this stays honest if the split or the tiny config changes.
    """
    from scd_attention import causal_mask

    inputs = sample_inputs(mm, sp.latent_t)
    for audio_is_context in (True, False):
        t = scd.clock(pack, sp.video_start, audio_is_context)
        with torch.no_grad():
            full, _ = scd.encode(mask=causal_mask(t, t), **inputs)
            for chunk in (1, 2, 3, sp.latent_t):
                got, _, cache = scd.encode_chunked(chunk, audio_is_context, **inputs)
                d = (got - full).abs().max().item()
                assert got.shape == full.shape, f"chunk {chunk}: {got.shape} vs {full.shape}"
                assert d < 1e-5, (f"audio_is_context={audio_is_context}, chunk {chunk}: differs "
                                  f"from a single masked pass by {d:.3e}")
                assert len(cache) == sp.seq_len, \
                    f"chunk {chunk}: cache holds {len(cache)} rows, sequence has {sp.seq_len}"


@case
def test_block_mask_matches_dense(scd, _h, _ctx, sp, mm, _pack):
    """The FlexAttention block mask is the same mask, not a cheaper approximation of it.

    Dense is what every Phase 2 number was measured with and what the other cases here reason
    about; block is the only one that can exist at 768p/15s, where a bool `[S, S]` is 3.8 GB per
    chunk. So the two have to be pinned together, and through `encode_chunked` rather than on a
    synthetic q/k/v: the rectangular chunk masks — new rows as queries, the whole cache as keys —
    are the shapes the encoder actually builds, and they are where a block mask that quietly
    padded or transposed would show up.

    Tolerance is 1e-4, looser than the 1e-5 elsewhere, because this compares two attention
    KERNELS rather than two orderings of one. Anything near the 1e-1 a wrong mask produces is
    still caught by three orders of magnitude.
    """
    inputs = sample_inputs(mm, sp.latent_t)
    with torch.no_grad():
        for chunk in (2, sp.latent_t):
            dense, _, _ = scd.encode_chunked(chunk, False, block=False, **inputs)
            flex, _, _ = scd.encode_chunked(chunk, False, block=True, **inputs)
            d = (dense - flex).abs().max().item()
            assert d < 1e-4, f"chunk {chunk}: block mask differs from dense by {d:.3e}"


@case
def test_clocks_agree_on_video_only(scd, _h, _ctx, sp, _mm, pack):
    """`FrameSpans.frame_index` and `row_time(..., video_start)` are two routes to one mask.

    The first is shape-derived and needs no packer positions; the second reads the packer's real
    rotary axis. They must produce the same mask or one of them is wrong about where the video
    segment starts, and the AV clock — which only the second can express — inherits that error.
    """
    from scd_attention import causal_mask

    a = sp.frame_index()
    b = scd.clock(pack, sp.video_start, audio_is_context=True)
    assert torch.equal(causal_mask(a, a), causal_mask(b, b)), \
        "the shape-derived clock and the packer's clock disagree on the video-only mask"


@case
def test_av_clock_orders_audio(scd, _h, ctx, sp, mm, pack):
    """Under the AV clock audio is ORDERED, not exempt: causal in both directions, and no longer
    video-blind. Video-blindness is the whole reason this clock exists — the encoder's audio rows
    measured centered cos 0.047 against a bidirectional pass when audio was context."""
    from scd_attention import CONTEXT, causal_mask

    t = scd.clock(pack, sp.video_start, audio_is_context=False)
    audio = slice(int((t == CONTEXT).sum()), sp.video_start)
    assert audio.stop > audio.start, "no audio rows in this fixture — the case proves nothing"
    assert (t[audio] > CONTEXT).all(), "audio is still context under the AV clock"

    m = causal_mask(t, t)
    last = sp.rows_for_frames(sp.latent_t - 1, sp.latent_t).start
    early = audio.start
    assert m[last, early], "the last video frame cannot see the earliest audio row"
    assert not m[early, last], \
        "the earliest audio row sees the last video frame — the future flows backward"
    assert m[early, early], "an audio row cannot see itself"
    assert m[early, :audio.start].all(), "audio lost sight of the text/reference rows"
    _ = ctx, mm


@case
def test_av_clock_composes_across_blocks(scd, h, ctx, sp, _mm, pack):
    """Ordering audio does not reopen the leak the strict mask was written to close.

    Both modalities causal on one shared axis still means nothing carries the future backward, so
    the prefix-in-time stays a function of the prefix-in-time through the whole encoder — the
    property a KV cache needs. Worth asserting rather than arguing: the tempting cheap variant,
    letting audio queries see all video while video stays causal, satisfies every intuition about
    "audio is bidirectional anyway" and is the loose mask under another name.
    """
    t = scd.clock(pack, sp.video_start, audio_is_context=False)
    keep_t = t[sp.rows_for_frames(0, 1).start]
    ordered, context = prefix_delta(scd, h, ctx, t, keep_t, False, len(scd.encoder_blocks))
    assert max(ordered, context) < 1e-5, \
        f"AV clock is not prefix-exact through the stack: ordered {ordered:.3e}, " \
        f"context {context:.3e}"


@case
def test_cache_holds_post_rope_keys(scd, _h, ctx, sp, mm, _pack):
    """Cached K must already carry its rows' RoPE.

    A cache written before rotation would need every entry re-rotated on each read, and the
    version that forgets to would still produce plausible video — wrong positions, not garbage.
    Detected by giving chunk 2 the same content at a different position and requiring a change.
    """
    from scd_attention import LayerCache, attention

    _, _, cos, sin = ctx
    block = scd.encoder_blocks[0]
    torch.manual_seed(3)
    x = torch.randn(sp.frame_rows, block.attn.qkv_proj.in_features)
    a, b = LayerCache(), LayerCache()
    with torch.no_grad():
        attention(block.attn, x, cos[:sp.frame_rows], sin[:sp.frame_rows], cache=a)
        r = sp.rows_for_frames(3, 4)
        attention(block.attn, x, cos[r], sin[r], cache=b)
    assert not torch.equal(a.k, b.k), "cached keys are position-independent — RoPE is not applied"
    assert torch.equal(a.v, b.v), "values must not depend on position"
    _ = mm


@case
def test_cache_bytes_scale_with_rows(scd, _h, _ctx, sp, mm, _pack):
    """Cache size is linear in rows and blocks — the arithmetic §8.1's 0.95 GB/frame rests on."""
    from scd_attention import KVCache

    inputs = sample_inputs(mm, sp.latent_t)
    with torch.no_grad():
        _, _, cache = scd.encode_chunked(2, **inputs)
    attn = scd.encoder_blocks[0].attn
    expect = (2 * sp.seq_len * attn.heads * attn.head_dim * 4 * len(scd.encoder_blocks))
    assert cache.bytes() == expect, f"cache is {cache.bytes()} B, expected {expect} B"
    assert isinstance(cache, KVCache)


@case
def test_window_holds_the_cache_flat_vs_length(scd, _h, _ctx, _sp, mm, _pack):
    """§8.1's claim, which is the whole reason SCD's duration is unbounded: with a window the
    cache stops growing with the clip. Asserted as EQUALITY across three lengths rather than as a
    ceiling, because a ceiling is satisfied by something that still grows, just slower.

    Also asserts the context rows survive. They are the failure mode a plausible implementation
    walks into: context sits at -inf, which is behind every horizon, so a window that evicts on
    time alone drops the text and reference rows first and the encoder quietly loses its
    conditioning one chunk in — with no shape change to notice.

    Lengths are on the 5n+2 latent grid the packer enforces; off-grid values raise rather than
    round, which is how this notices if the grid ever moves.
    """
    from scd_attention import CONTEXT

    seen = {}
    for latent_t in (7, 12, 17):
        inputs = sample_inputs(mm, latent_t)
        with torch.no_grad():
            h, _, pack = scd.preamble(**inputs)
            spans = scd.spans(inputs["video_latent"], h.shape[0])
            t = scd.clock(pack, spans.video_start, audio_is_context=False)
            _, _, cache = scd.encode_chunked(1, window=2, **inputs)
            _, _, unbounded = scd.encode_chunked(1, **inputs)
        assert int((t == CONTEXT).sum()) > 0, \
            "the AV clock left no context rows — nothing pins the conditioning"
        # Exactly the rows inside the window, plus context. Equality rather than a bound: a
        # version that evicts on time alone leaves a cache that is still flat, still smaller than
        # the sequence, and still big enough to clear any count-based floor — it is simply missing
        # the text rows, and only the exact number says so.
        horizon = t[spans.video_start::spans.frame_rows][spans.latent_t - 2]
        expect = int(((t >= horizon) | (t == CONTEXT)).sum())
        assert len(cache) == expect, \
            f"latent_t={latent_t}: cache holds {len(cache)} rows, the window plus context is " \
            f"{expect} — off by {len(cache) - expect}"
        assert len(unbounded) == spans.seq_len > len(cache), \
            f"latent_t={latent_t}: window kept {len(cache)} of {spans.seq_len} rows"
        seen[latent_t] = len(cache)
    assert len(set(seen.values())) == 1, \
        f"windowed cache grew with clip length: {seen} — §8.1's flat-VRAM claim does not hold"


@case
def test_wide_window_is_the_unbounded_cache(scd, _h, _ctx, sp, mm, _pack):
    """A window at least as wide as the clip evicts nothing, so it must reproduce `window=None`
    bit for bit. This is what pins the window as a restriction of the exact path rather than a
    second, subtly different encoder — if the two disagree here, the difference is a bug in the
    bookkeeping and not the approximation the window is allowed to make."""
    inputs = sample_inputs(mm, sp.latent_t)
    with torch.no_grad():
        ref, _, _ = scd.encode_chunked(2, **inputs)
        got, _, _ = scd.encode_chunked(2, window=sp.latent_t, **inputs)
    assert torch.equal(got, ref), \
        f"a full-width window differs from no window by {(got - ref).abs().max().item():.3e}"


@case
def test_layer_major_matches_chunk_major(scd, _h, _ctx, sp, mm, _pack):
    """Blocks-outer and chunks-outer are the same arithmetic, and the whole memory argument for
    layer-major rests on that being exact rather than close.

    `torch.equal`, not `allclose`: the two orders issue identical ops on identical tensors, so any
    difference at all is a scheduling bug — a cache filled in the wrong order, or a mask reused
    across blocks after the window moved under it. Something that merely rounds differently would
    mean "the same arithmetic reassociated" is false and the tolerance is hiding it.

    Swept across both clocks and with the window on and off, because the two orders can only
    diverge where those flags act: the AV clock makes a chunk a gather, and the window makes
    `held` stop being a prefix. Layer-major builds its masks once on the first block and reuses
    them, so a window whose eviction schedule does not match that mask list shows up only here.
    """
    inputs = sample_inputs(mm, sp.latent_t)
    with torch.no_grad():
        for audio_is_context in (False, True):
            for window in (None, 2):
                kw = dict(audio_is_context=audio_is_context, window=window, **inputs)
                ref, _, full = scd.encode_chunked(1, **kw)
                got, _, one = scd.encode_chunked(1, layer_major=True, **kw)
                tag = f"clock={'video' if audio_is_context else 'av'} window={window}"
                assert torch.equal(got, ref), \
                    f"{tag}: layer-major differs by {(got - ref).abs().max().item():.3e}"
                # Same rows retained, a thirtieth of the bytes — the point of the reorder. Row
                # equality alone would pass for a cache that kept every layer and returned one.
                n = len(scd.encoder_blocks)
                assert len(one) == len(full), f"{tag}: {len(one)} rows vs {len(full)}"
                assert one.bytes() * n == full.bytes(), \
                    f"{tag}: layer-major held {one.bytes()} B, not a {n}th of {full.bytes()} B " \
                    "— more than one block's cache is live"


@case
def test_encode_unmasked_is_untouched(scd, _h, _ctx, _sp, mm, _pack):
    """`encode()` with no mask and no cache must still be the plain base prefix — Phase 1's
    prefix-parity guarantee is what the whole split rests on, and Phase 2 must not erode it."""
    inputs = sample_inputs(mm, 2)
    ref = {}
    handle = scd.base.blocks[scd.encoder_depth - 1].register_forward_hook(
        lambda _m, _a, out: ref.update(h=out))
    try:
        with torch.no_grad():
            scd.base(**inputs)
    finally:
        handle.remove()
    with torch.no_grad():
        h, _ = scd.encode(**inputs)
    assert torch.equal(h, ref["h"]), "encode() no longer reproduces the base's prefix exactly"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    ap.add_argument("--allow-cuda", action="store_true",
                    help="do not hide the GPU; only useful for debugging the flex path")
    args = ap.parse_args()

    # Hide the GPU by default. This suite is CPU-only by design, but `block_mask` goes through
    # torch.compile, which initialises CUDA when a device is visible and then fails with a CUDA
    # OOM if something else is using it — so running the suite while a measurement is in flight
    # reports `test_block_mask_matches_dense` as a failure of the block mask, which it is not.
    if not args.allow_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    here = __file__.rsplit("/", 1)[0]
    sys.path.insert(0, here)
    from test_scd_model import DECODER_SOURCE, ENCODER_DEPTH, build

    _, mm = build(args.fizgig_src)
    inputs = sample_inputs(mm)
    failures = 0
    for fn in CASES:
        # A fresh composition per case: MiniMaxH3SCD consumes its base, and the block-level
        # attention override would be visible to a later case if one ever failed to restore it.
        import copy

        from scd_model import MiniMaxH3SCD
        stock, _ = build(args.fizgig_src)
        scd = MiniMaxH3SCD(copy.deepcopy(stock), encoder_depth=ENCODER_DEPTH,
                           decoder_source=DECODER_SOURCE)
        with torch.no_grad():
            h, ctx, pack = scd.preamble(**inputs)
        sp = scd.spans(inputs["video_latent"], h.shape[0])
        try:
            fn(scd, h, ctx, sp, mm, pack)
            print(f"ok    {fn.__name__}")
        except Exception as e:
            # Not just AssertionError: a broken cache raises a shape mismatch, and letting that
            # abort the run hides the nine cases that would have told you where it broke.
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
