#!/usr/bin/env python3
"""Phase 1 invariants for MiniMaxH3SCD. CPU, seconds, no weights and no GPU.

The real model is 50 blocks at hidden 5376; these run a 6-block, hidden-64 config with the same
code path, ~775K parameters. Composition, weight copying and aliasing are shape-independent, so a
tiny model tests them exactly — and a test that needs a 62 GB checkpoint is a test nobody runs.

The design doc's original identity test (`concat(encoder, decoder)` reproduces the base) is
IMPOSSIBLE and is deliberately not here: the decoder re-uses early blocks, so the composition has
more layer instances than the base. `test_more_instances_than_base` pins that fact so nobody
reintroduces the test later. What replaces it:

  * weight parity   — every instance loads bit-identical to its source block in stock H3
  * prefix parity   — the encoder reproduces stock's first L_e block outputs exactly
  * independence    — blocks sharing an init are distinct storage and diverge under a write

Needs fizgig on the path (`--fizgig-src`), which CI does not have; CI lints and byte-compiles.

Usage:
    python3 scripts/scd/test_scd_model.py
"""

import argparse
import copy
import sys

import torch

TINY = dict(hidden_size=64, num_layers=6, token_refiner_num_layers=1, num_attention_heads=2,
            attention_head_dim=128, ffn_hidden_size=32, text_dim=48,
            time_embed_hidden_size=64, time_embed_dim=32)

ENCODER_DEPTH = 3
DECODER_SOURCE = (0, 1, 4, 5)


def build(fizgig_src):
    if fizgig_src not in sys.path:
        sys.path.insert(0, fizgig_src)
    from fizgig.minimax import model as mm

    torch.manual_seed(0)
    return mm.MiniMaxH3DiT(mm.MiniMaxH3Config(**TINY)).eval(), mm


def sample_inputs(mm):
    """Fixed inputs INCLUDING the audio rows.

    Left to itself the base draws fresh `torch.randn` silence rows on every call, so two forwards
    with identical arguments pack different sequences and any bit-exact comparison across calls
    fails by ~0.6. Passing them explicitly is what makes prefix parity a test of the composition
    rather than of the RNG.
    """
    torch.manual_seed(1)
    latent_t = 2
    n_audio = mm.audio_latents_for_frames(mm.pixel_frames_for_latent(latent_t))
    return dict(video_latent=torch.randn(1, 24, latent_t, 8, 8), t=torch.tensor([0.5]),
                text_embeds=torch.randn(1, 5, TINY["text_dim"]),
                audio_noise=torch.randn(n_audio * mm.AUDIO_CHANNELS, 32))


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_weight_parity(stock, scd, _inputs, _consumed):
    """Every block instance is bit-identical to the stock block it came from."""
    from scd_model import block_parity

    for i, block in enumerate(scd.encoder_blocks):
        assert block_parity(block, stock.blocks[i]), f"encoder block {i} differs from stock"
    for slot, src in enumerate(scd.decoder_source):
        assert block_parity(scd.decoder_blocks[slot], stock.blocks[src]), \
            f"decoder slot {slot} (source block {src}) differs from stock"


@case
def test_prefix_parity(stock, scd, inputs, _consumed):
    """The encoder reproduces stock's first L_e block outputs EXACTLY.

    Exactly, not approximately: same weights, same inputs and the same block code should give
    bit-identical activations on CPU. An allclose here would hide a genuinely different graph.
    """
    ref = {}
    handle = stock.blocks[scd.encoder_depth - 1].register_forward_hook(
        lambda _m, _a, out: ref.update(h=out))
    try:
        with torch.no_grad():
            stock(**inputs)
    finally:
        handle.remove()

    with torch.no_grad():
        h, _ctx = scd.encode(**inputs)
    assert torch.equal(h, ref["h"]), \
        f"encoder output differs from stock block {scd.encoder_depth - 1}; " \
        f"max abs delta {(h - ref['h']).abs().max().item():.3e}"


@case
def test_shared_init_blocks_are_independent(stock, scd, _inputs, _consumed):
    """Decoder blocks that re-use an encoder block's index must be separate parameters.

    This is the failure the whole re-composition rests on: identical weights that share storage
    look correct in every state_dict comparison and then train as one tensor for 55K steps.
    """
    from scd_model import aliases

    overlap = [(s, i) for i, s in enumerate(scd.decoder_source) if s < scd.encoder_depth]
    assert overlap, "test is vacuous unless the decoder re-uses an encoder block"

    for src, slot in overlap:
        enc, dec = scd.encoder_blocks[src], scd.decoder_blocks[slot]
        assert enc is not dec, f"decoder slot {slot} IS encoder block {src}"
        assert not aliases(enc, dec), f"decoder slot {slot} shares storage with encoder block {src}"

        with torch.no_grad():
            p_enc = next(enc.parameters())
            before = next(dec.parameters()).clone()
            p_enc.add_(1.0)
            assert torch.equal(next(dec.parameters()), before), \
                f"writing encoder block {src} changed decoder slot {slot}"
            p_enc.sub_(1.0)
    _ = stock


@case
def test_tail_blocks_are_moved_not_copied(_stock, scd, _inputs, consumed):
    """Tail blocks past the prefix are moved into the decoder, not duplicated.

    Not a style preference: a spare copy of five 5376-wide blocks does not fit beside the base on
    a 24 GB card, so this is the assumption the VRAM plan in §8 rests on. Compared against the
    blocks of the model that was CONSUMED, not the pristine stock model — those are different
    objects by construction, so `stock` here would make the test pass vacuously.
    """
    for slot, src in enumerate(scd.decoder_source):
        if src >= scd.encoder_depth:
            assert scd.decoder_blocks[slot] is consumed[src], \
                f"decoder slot {slot} (source {src}) was copied; it should be the base's own block"


@case
def test_more_instances_than_base(_stock, scd, _inputs, _consumed):
    """The composition has MORE layer instances than the base — why the identity test is dropped."""
    expected = ENCODER_DEPTH + len(DECODER_SOURCE)
    assert scd.num_layer_instances == expected, \
        f"{scd.num_layer_instances} instances, expected {expected}"
    assert scd.num_layer_instances > scd.source_num_layers, \
        "composition is not a re-composition — concat(enc, dec) would be a valid identity test"


@case
def test_encoder_truncated(_stock, scd, _inputs, _consumed):
    assert len(scd.encoder_blocks) == ENCODER_DEPTH, \
        f"encoder has {len(scd.encoder_blocks)} blocks, expected {ENCODER_DEPTH}"


@case
def test_decode_runs_and_moves_activations(_stock, scd, inputs, _consumed):
    """The decoder accepts the encoder's context unchanged and actually does something."""
    with torch.no_grad():
        h, ctx = scd.encode(**inputs)
        out = scd.decode(h, ctx)
    assert out.shape == h.shape, f"decoder changed shape {h.shape} -> {out.shape}"
    assert torch.isfinite(out).all(), "decoder produced non-finite activations"
    assert not torch.equal(out, h), "decoder was a no-op"


def _two_preambles(scd, inputs):
    """(enc, clean_ctx, noisy_h, noisy_ctx, spans, media_start) — the token_concat decoder's inputs.

    Clean is the same latent at t=1.0, which is sigma=0: the encoder's whole premise is that it
    sees the video unnoised, and passing `inputs` unchanged would make every shift test vacuous
    by feeding both halves the same packing.

    The encoder is `encode_chunked`, not `encode`. `encode` runs the blocks unmasked, so its
    frame f-1 rows have attended to frame f and the shift buys nothing — the context half hands
    the decoder the clean target anyway, one hop further round. The shift and the frame-causal
    mask are one mechanism and neither is sufficient alone; `test_decoder_context_cannot_see_its
    _own_frame` fails against `encode` for exactly this reason.
    """
    clean = dict(inputs, t=torch.tensor([1.0]))
    with torch.no_grad():
        enc, clean_ctx, _ = scd.encode_chunked(1, **clean)
        noisy_h, noisy_ctx, pack = scd.preamble(**inputs)
    spans = scd.spans(inputs["video_latent"], enc.shape[0])
    return enc, clean_ctx, noisy_h, noisy_ctx, spans, pack["media_start"]


@case
def test_decoder_frame_input_is_shifted(_stock, scd, inputs, _consumed):
    """Frame f's context half is the encoder's frame f-1, and frame 0's is zeros.

    This is the whole reason the shift exists. The encoder ran on CLEAN latents, so conditioning
    frame f on its own encoder rows would hand the decoder the answer it is being trained to
    predict — the leak is total, and it would show up as a loss that falls to zero and a sampler
    that produces noise. Pinning the exact row slice is the only cheap way to catch an off-by-one
    that would otherwise look like unusually fast convergence.
    """
    enc, cctx, nh, nctx, sp, _ = _two_preambles(scd, inputs)
    r, lo = sp.frame_rows, sp.video_start
    assert sp.latent_t >= 2, "needs at least two frames for a shift to mean anything"

    x, _ = scd.decoder_frame_input(enc, cctx, nh, nctx, sp, 1)
    assert x.shape[0] == 2 * r, f"token_concat gave {x.shape[0]} rows, expected 2R = {2 * r}"
    cond, tgt = x[:r], x[r:]
    assert torch.equal(cond, enc[lo:lo + r]), "context half is not the encoder's frame 0"
    assert not torch.equal(cond, enc[lo + r:lo + 2 * r]), \
        "context half IS the encoder's own frame 1 — the shift is missing and the clip leaks"
    assert torch.equal(tgt, nh[lo + r:lo + 2 * r]), "target half is not the noisy frame 1"

    x0, _ = scd.decoder_frame_input(enc, cctx, nh, nctx, sp, 0)
    assert torch.equal(x0[:r], torch.zeros_like(x0[:r])), \
        "frame 0 has no predecessor, so its context half must be zeros, not encoder rows"
    assert torch.equal(x0[r:], nh[lo:lo + r]), "frame 0's target half is not the noisy frame 0"


@case
def test_decoder_halves_get_different_timesteps(_stock, scd, inputs, _consumed):
    """Context and target must not share an AdaLN row: one is clean, the other is at sigma.

    `mod_row = t_index * MODALITY_NUM + tag`, so equal indices would modulate the clean context as
    though it were noise. That failure is invisible in shapes and in the loss curve's first
    thousand steps, and it silently removes the conditioning signal the decoder exists to use.
    """
    mm = sys.modules[type(scd.base).__module__]
    enc, cctx, nh, nctx, sp, media_start = _two_preambles(scd, inputs)
    r = sp.frame_rows
    _, (t_emb, mod, _, _) = scd.decoder_frame_input(enc, cctx, nh, nctx, sp, 1,
                                                    media_start=media_start)

    cond_t = (mod[media_start:media_start + r] // mm.MODALITY_NUM).unique()
    tgt_t = (mod[media_start + r:] // mm.MODALITY_NUM).unique()
    assert cond_t.numel() == 1 and tgt_t.numel() == 1, \
        f"each half must sit at one timestep, got {cond_t.tolist()} and {tgt_t.tolist()}"
    assert cond_t.item() != tgt_t.item(), \
        f"both halves modulate at timestep {cond_t.item()} — the context is being treated as noisy"
    assert int(mod.max()) < t_emb.shape[0] * mm.MODALITY_NUM, \
        f"mod_row {int(mod.max())} indexes past the {t_emb.shape[0]}-timestep table"
    # Tags survive the retime: the context half is video rows and must still say so, or AdaLN
    # picks the text/audio modulation group for them.
    tags = mod[media_start:media_start + r] % mm.MODALITY_NUM
    assert (tags == mm.VIDEO_TAG).all(), f"context half lost its video tag: {tags.unique().tolist()}"


@case
def test_decoder_text_rows_are_a_prefix(_stock, scd, inputs, _consumed):
    """`media_start` prepends exactly the text/ref rows and perturbs nothing else.

    The 2R and 2R+text packs differ by 1.26x on the decoder frame, so both will be trained and
    compared; that only means anything if the two are otherwise the same forward.
    """
    enc, cctx, nh, nctx, sp, media_start = _two_preambles(scd, inputs)
    assert media_start > 0, "no text rows in this fixture — the test would pass vacuously"

    bare, _ = scd.decoder_frame_input(enc, cctx, nh, nctx, sp, 1)
    full, _ = scd.decoder_frame_input(enc, cctx, nh, nctx, sp, 1, media_start=media_start)
    assert full.shape[0] == bare.shape[0] + media_start, \
        f"text pack is {full.shape[0]} rows, expected {bare.shape[0]} + {media_start}"
    assert torch.equal(full[:media_start], nh[:media_start]), "prefix is not the packed text rows"
    assert torch.equal(full[media_start:], bare), "adding text changed the token_concat rows"


@case
def test_decoder_context_cannot_see_its_own_frame(_stock, scd, inputs, _consumed):
    """Perturbing the clean latent at frame f leaves frame f's decoder output untouched.

    The end-to-end statement of the shift, through the frame-causal encoder rather than through
    row indices — the version that still fails if the encoder's causality and the decoder's shift
    are each individually right but disagree about which frame is which.
    """
    enc, cctx, nh, nctx, sp, _ = _two_preambles(scd, inputs)
    with torch.no_grad():
        base = scd.decode_frame(enc, cctx, nh, nctx, sp, 1)

    bumped = dict(inputs)
    bumped["video_latent"] = inputs["video_latent"].clone()
    bumped["video_latent"][:, :, 1] += 1.0          # the frame being denoised, clean side
    enc2, cctx2, _, _, _, _ = _two_preambles(scd, bumped)
    with torch.no_grad():
        after = scd.decode_frame(enc2, cctx2, nh, nctx, sp, 1)
    assert torch.equal(base, after), \
        "changing frame 1's CLEAN latent moved frame 1's decoder output — the encoder feature " \
        "the decoder conditions on is not blind to the frame it is predicting"

    bumped0 = dict(inputs)
    bumped0["video_latent"] = inputs["video_latent"].clone()
    bumped0["video_latent"][:, :, 0] += 1.0
    enc3, cctx3, _, _, _, _ = _two_preambles(scd, bumped0)
    with torch.no_grad():
        after0 = scd.decode_frame(enc3, cctx3, nh, nctx, sp, 1)
    assert not torch.equal(base, after0), \
        "changing frame 0 did nothing to frame 1 — the context half is not reaching the decoder"


@case
def test_production_split(_stock, _scd, inputs, _consumed):
    """The module's real defaults, on a 50-block model — the split Phase 2 will inherit.

    Everything above runs a 3/4 stand-in split, which would keep passing if DEFAULT_ENCODER_DEPTH
    or DEFAULT_DECODER_SOURCE were edited to something incoherent. Width is still tiny; depth is
    the only thing that has to be real for the composition to mean anything.
    """
    from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT
    from scd_model import (DEFAULT_DECODER_SOURCE, DEFAULT_ENCODER_DEPTH, MiniMaxH3SCD,
                           block_parity)

    torch.manual_seed(0)
    stock = MiniMaxH3DiT(MiniMaxH3Config(**{**TINY, "num_layers": 50})).eval()
    scd = MiniMaxH3SCD(copy.deepcopy(stock))

    assert scd.num_layer_instances == 37, f"{scd.num_layer_instances} instances, expected 37"
    assert len(scd.encoder_blocks) == DEFAULT_ENCODER_DEPTH
    assert max(DEFAULT_DECODER_SOURCE) < 50 and DEFAULT_ENCODER_DEPTH <= 50
    # Phase 0's load-bearing set. The decoder may hold more, but never less.
    assert {0, 1, 48, 49} <= set(DEFAULT_DECODER_SOURCE), \
        f"decoder {DEFAULT_DECODER_SOURCE} drops a load-bearing block from {{0, 1, 48, 49}}"
    for slot, src in enumerate(scd.decoder_source):
        assert block_parity(scd.decoder_blocks[slot], stock.blocks[src])

    with torch.no_grad():
        h, ctx = scd.encode(**inputs)
        assert torch.isfinite(scd.decode(h, ctx)).all()


@case
def test_rejects_out_of_range_split(_stock, _scd, _inputs, _consumed):
    from scd_model import MiniMaxH3SCD

    for kwargs in ({"encoder_depth": 99}, {"encoder_depth": 0}, {"decoder_source": (0, 99)}):
        try:
            MiniMaxH3SCD(copy.deepcopy(_stock), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid split {kwargs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    args = ap.parse_args()

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from scd_model import MiniMaxH3SCD

    _, mm = build(args.fizgig_src)
    inputs = sample_inputs(mm)
    failures = 0
    for fn in CASES:
        # Each case gets a pristine stock model and its own composition: MiniMaxH3SCD consumes
        # the base, and test_shared_init_blocks_are_independent writes to parameters.
        stock, _ = build(args.fizgig_src)
        victim = copy.deepcopy(stock)
        consumed = list(victim.blocks)          # snapshot before the composition truncates it
        scd = MiniMaxH3SCD(victim, encoder_depth=ENCODER_DEPTH, decoder_source=DECODER_SOURCE)
        try:
            fn(stock, scd, inputs, consumed)
            print(f"ok    {fn.__name__}")
        except Exception as e:
            # Not just AssertionError: a shape or attribute break should be reported as one
            # failing case, not abort the run and hide the others.
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
