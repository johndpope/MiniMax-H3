#!/usr/bin/env python3
"""Compose H3's 50 DiT blocks into an SCD encoder prefix + decoder (Phase 1), and run the encoder
frame-causally with a KV cache (Phase 2 — the mask and cache themselves live in `scd_attention`).

SCD's decoder is a RE-COMPOSITION, not the encoder's tail (§2 of the design doc). The paper takes
WAN's 30 layers to encoder 0-24 plus decoder {0-4} u {25-29} — 35 layer instances from a 30-layer
model. So the two halves overlap at the front, and the overlapping blocks are separate parameter
sets that share an initialisation and then diverge over 55K steps of full fine-tuning. Anything
that aliases them is silently training one tensor through two paths.

The H3 split starts from Phase 0's two probes: encoder 0-29 (the sigma-invariance knee under the
widest on-distribution contrast), decoder {0, 1} u {45-49}. Only {0, 1, 48, 49} is actually
load-bearing by leave-one-out cost; 45-47 are the design's margin, not a measurement.

What this module deliberately does NOT do
-----------------------------------------
No training, and no re-implementation of the base's ~100-line input packing. That preamble decides
segment order, modulation rows, audio row count and RoPE positions, and a second copy of it would
drift from the real one exactly the way Phase 0's hand-transcribed numbers drifted from their JSON.
`preamble()` instead captures the base's own block arguments through a forward hook, so whatever
the preamble does, the encoder and decoder see the same thing.

There is also no pixel output path. The final layer needs `video_t_index`, which the preamble
derives from a sorted-unique over the distinct timesteps; reproducing that here would be the same
drift risk for no benefit until there is something to decode into pixels.

Usage:
    from scd_model import MiniMaxH3SCD
    scd = MiniMaxH3SCD(base)                     # consumes `base`
    h, ctx = scd.encode(video_latent=z, t=t, text_embeds=te)
    h = scd.decode(h, ctx)

    # frame-causal, chunked, carrying an encoder KV cache
    h, ctx, cache = scd.encode_chunked(4, video_latent=z, t=t, text_embeds=te)
"""

import copy
import inspect
import sys

import torch
from torch import nn

from scd_attention import FrameSpans, encode_chunks, row_time, run_block


class _PreambleDone(Exception):
    """Raised from the block-0 hook to unwind out of the base forward once packing is done."""

# Encoder is a prefix; the knee sits at block 30, so 0..29.
DEFAULT_ENCODER_DEPTH = 30
# Decoder source blocks, in the order they run. {0, 1} are the load-bearing front pair,
# {45..49} the tail. See the module docstring on which of these Phase 0 actually justifies.
DEFAULT_DECODER_SOURCE = (0, 1, 45, 46, 47, 48, 49)


class MiniMaxH3SCD(nn.Module):
    """Encoder prefix + decoder re-composition over one stock `MiniMaxH3DiT`.

    Consumes the base model: its `blocks` list is truncated to the encoder prefix in place, and
    tail blocks the encoder no longer needs are MOVED into the decoder rather than copied. On a
    24 GB card a spare copy of five 5376-wide blocks is not affordable, and the base is not
    reusable afterwards anyway. Blocks the encoder still holds (0 and 1) are deep-copied, which
    is the point: those two must be independent parameters.
    """

    def __init__(self, base, encoder_depth=DEFAULT_ENCODER_DEPTH,
                 decoder_source=DEFAULT_DECODER_SOURCE):
        super().__init__()
        n = len(base.blocks)
        if not 0 < encoder_depth <= n:
            raise ValueError(f"encoder_depth {encoder_depth} outside 1..{n}")
        bad = [i for i in decoder_source if not 0 <= i < n]
        if bad:
            raise ValueError(f"decoder_source blocks {bad} outside 0..{n - 1}")

        self.base = base
        self.encoder_depth = encoder_depth
        self.decoder_source = tuple(decoder_source)
        self.source_num_layers = n

        # Take the tail blocks out of the base BEFORE truncating, so a decoder block at or past
        # the prefix is moved rather than copied. Order follows decoder_source, not block index.
        stock = list(base.blocks)
        dec = []
        for i in self.decoder_source:
            dec.append(stock[i] if i >= encoder_depth else copy.deepcopy(stock[i]))
        self.decoder_blocks = nn.ModuleList(dec)
        base.blocks = nn.ModuleList(stock[:encoder_depth])

    @property
    def encoder_blocks(self):
        return self.base.blocks

    @property
    def num_layer_instances(self):
        """Total DiT block instances. More than the base has, which is the whole point of §2 and
        the reason the original identity test (`concat(enc, dec) == base`) is impossible."""
        return len(self.base.blocks) + len(self.decoder_blocks)

    def preamble(self, **forward_kwargs):
        """Run the base's input packing and stop at the first block. Returns (h, ctx, pack).

        `ctx` is `(t_emb, mod_row, cos, sin)` — the per-block arguments the base built, captured
        from its own first block rather than recomputed, so this cannot drift from the preamble.
        The hook raises to abort: the preamble is what we want and the 50 blocks behind it are
        not, and at 768p running them to throw the result away is minutes of GPU time.

        `pack` is `{"pos", "media_start"}`, captured the same way and for the same reason. The
        mask needs a time per row, and the packer is the only thing that knows how the 40 Hz
        audio spans line up against `_video_t_grid`'s fractional video spans; the alternative is
        a rows-per-frame ratio computed here, which would be a rounding error with a plausible
        story. `media_start` comes out of the captured ARGUMENTS (`text_len + ref_row_count`)
        rather than by pattern-matching the positions back into segments.
        """
        cap = {}

        def grab(_module, args):
            cap["h"], cap["ctx"] = args[0], tuple(args[1:])
            raise _PreambleDone

        mm = sys.modules[type(self.base).__module__]
        real_pos = mm.image_position_ids
        sig = inspect.signature(real_pos)

        def spy(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            cap["media_start"] = (bound.arguments["text_len"]
                                  + mm.ref_row_count(bound.arguments["refs"]))
            cap["pos"] = real_pos(*args, **kwargs)
            return cap["pos"]

        handle = self.base.blocks[0].register_forward_pre_hook(grab)
        mm.image_position_ids = spy
        try:
            self.base(**forward_kwargs)
        except _PreambleDone:
            pass
        finally:
            handle.remove()
            mm.image_position_ids = real_pos
        return cap["h"], cap["ctx"], {"pos": cap["pos"], "media_start": cap["media_start"]}

    def spans(self, video_latent, seq_len):
        """`FrameSpans` for a packed sequence of `seq_len` rows holding this video latent."""
        _, _, latent_t, lat_h, lat_w = video_latent.shape
        ph, pw = self.base.patch_size[1], self.base.patch_size[2]
        return FrameSpans(seq_len, latent_t, (lat_h // ph) * (lat_w // pw))

    def clock(self, pack, video_start, audio_is_context):
        """Per-row time for the mask. `audio_is_context` picks v0's video-only clock over the AV
        clock; see `scd_attention.row_time` for what that costs the audio rows."""
        return row_time(pack["pos"][:, 0], pack["media_start"],
                        video_start if audio_is_context else None)

    def encode(self, *, mask=None, cache=None, **forward_kwargs):
        """Run the base preamble and the encoder prefix. Returns (h, ctx).

        With `mask=None, cache=None` every block runs its own unmodified forward, so the encoder
        is the base's first `encoder_depth` blocks exactly — `test_prefix_parity` holds by
        construction rather than by the masked path happening to be equivalent.
        """
        h, ctx, _ = self.preamble(**forward_kwargs)
        for i, block in enumerate(self.encoder_blocks):
            h = run_block(block, h, ctx, mask=mask, cache=None if cache is None else cache[i])
        return h, ctx

    def encode_chunked(self, chunk_frames, audio_is_context=False, block=False, window=None,
                       layer_major=False, keep_audio=False, **forward_kwargs):
        """Encode causally in chunks of `chunk_frames` video frames, carrying a KV cache. Returns
        (h, ctx, cache), with `h` back in PACKED row order.

        Chunks are cut on TIME, not on rows. Under the video-only clock those coincide, because
        the packed order `[context | audio | video]` is already the causal order. Under the AV
        clock they do not: audio and video interleave in time while staying in separate slabs of
        the sequence, so a chunk is a gather, the cache fills in time order, and the result is
        scattered back at the end. Doing it by row slice instead would silently cache rows in an
        order the mask does not describe.

        With `window=None` this is not an approximation of a single masked pass, it is the same
        arithmetic in a different order — `test_chunked_matches_full` asserts that for both clocks.

        `window` is §8.1's policy, and it IS an approximation: keep only the last `window` latent
        frames of history (plus the context rows, which are never evicted), so the cache stops
        growing with clip length. It is not optional at scale — 30 encoder blocks hold 840 KiB per
        row, which is 53 GB at 768p/15s, an order of magnitude past the dense `[S, S]` mask that
        FlexAttention exists to avoid. Duration is bounded by the window, not by the card.

        This evicts rather than CastleHill's reset-and-re-encode-the-overlap. The two differ in
        what the retained rows know: an evicted-window row keeps the K/V it was computed with, over
        the full history it actually saw, while a re-encoded overlap row is recomputed against the
        overlap alone and so knows strictly less. Eviction is also the cheaper of the two. What it
        gives up is the reset's one real property — that a chunk boundary is a clean restart — which
        matters if drift accumulates, and is a thing to measure rather than assume.

        The preamble runs ONCE, over the whole clip. Real AR inference cannot do that — later
        frames do not exist yet — so this is the correctness harness for the cache, not the
        inference driver. That driver is Phase 5's, and it must build positions incrementally.

        `layer_major=True` runs blocks outer / chunks inner instead of the reverse. Same output,
        bit for bit, but only one block's KV cache is ever live — 28 KiB per row against 840 —
        and each block's weights are touched once for the whole clip rather than once per chunk.
        It requires the whole clip up front, so it is right for this offline harness and for
        training and wrong for Phase 5's AR driver. See `encode_chunks` for why the two agree.

        `block=True` builds each chunk's mask as a FlexAttention `BlockMask` instead of a dense
        `[Q, K]`, and in the CHUNKED path it is a trap — kept so the two masks can be compared,
        not because it should be used. Chunking already solves what the block mask is for: `Q` is
        one chunk, so a chunk's dense mask at 768p/15s is ~13 MB against the full pass's 3.8 GB.
        What `block=True` adds is a different `(Q, K)` shape per chunk, which blows
        `torch._dynamo`'s recompile limit after 8 chunks; the fallback is eager `flex_attention`,
        which materializes the full `[H, Q, K]` scores matrix — 2.4 GB per block at 768p — and
        OOMs. It degrades, with only a warning, into the thing it was chosen to avoid. Making it
        viable needs a fixed-capacity cache so every chunk is one shape. Until then: chunked means
        dense, and `block_mask` is for a single full-sequence pass.
        `test_block_mask_matches_dense` still runs both, which is what keeps them interchangeable.
        """
        h, ctx, pack = self.preamble(**forward_kwargs)
        sp = self.spans(forward_kwargs["video_latent"], h.shape[0])
        t = self.clock(pack, sp.video_start, audio_is_context).to(h.device)
        out, cache = encode_chunks(self.encoder_blocks, h, ctx, t, sp, chunk_frames,
                                   block=block, window=window, layer_major=layer_major,
                                   keep_audio=keep_audio)
        return out, ctx, cache

    def decode(self, h, ctx):
        """Run the decoder re-composition over encoder output. Same block signature as the base."""
        for block in self.decoder_blocks:
            h = block(h, *ctx)
        return h

    def decoder_frame_input(self, enc, clean_ctx, noisy_h, noisy_ctx, spans, frame,
                            media_start=None, duplicate_pos=True):
        """Build one frame's token_concat decoder input: `[text? | enc_feat_{f-1} | noisy_f]`.

        Two preambles are needed because the two halves sit at DIFFERENT timesteps — the context
        is the encoder's output over clean latents, the target is noise at sigma — and the base
        packs one video timestep per forward. Running its own packer twice and gathering rows is
        the only way to get both without re-deriving the (timestep, modality) table by hand, which
        is the drift `preamble` exists to prevent.

        AdaLN indexes `shift[mod_row]` with `mod_row = t_index * MODALITY_NUM + tag`, so retiming
        a row to a different timestep is `mod_row % MODALITY_NUM + new_index * MODALITY_NUM`. The
        tag is read back out of the captured `mod_row` rather than rebuilt, so the segment layout
        still comes from the base and only the timestep axis is ours.

        `frame` 0 has no predecessor and its context half is zeros. The alternative that suggests
        itself -- let frame 0 condition on its own encoder feature -- is the leak this shift exists
        to prevent, since that feature saw frame 0's CLEAN latent. Dropping the half instead would
        make frame 0 a different shape from every other frame and recompile the decoder for it.

        `media_start` is `pack["media_start"]` to prepend the text and ref rows, or None/0 to
        leave them out. It is a parameter rather than a default because the two cost 1.26x apart
        on the decoder frame (~18% of the N=30 speedup) and nothing measured picks between them:
        the context half is encoder output that already attended to text, so the text is
        compressed rather than absent. Only a post-training quality comparison decides it.

        `duplicate_pos` gives the context half the target frame's RoPE rows rather than its own,
        which is what CastleHill's `_duplicate_pe` does and what Tier 1 timed. False gives it
        frame f-1's real positions, so the decoder sees a genuine two-frame layout instead of two
        overlaid ones. Untested against quality either way -- it is a knob, not a default.
        """
        mm = sys.modules[type(self.base).__module__]
        r, lo = spans.frame_rows, spans.video_start
        if not 0 <= frame < spans.latent_t:
            raise ValueError(f"frame {frame} outside 0..{spans.latent_t - 1}")

        c_t_emb, c_mod, _, _ = clean_ctx
        n_t_emb, n_mod, n_cos, n_sin = noisy_ctx
        # One table holding both timesteps: clean rows index the first half, noisy the second.
        # `_time_embedding` already ran for each, so this is a concat rather than a third call --
        # and it keeps whatever the pruned-AdaLN path did to those rows intact.
        t_emb = torch.cat([c_t_emb, n_t_emb])
        shift = c_t_emb.shape[0]

        tgt = torch.arange(lo + frame * r, lo + (frame + 1) * r, device=noisy_h.device)
        x = [noisy_h[tgt]]
        mod = [n_mod[tgt] + shift * mm.MODALITY_NUM]
        cos, sin = [n_cos[tgt]], [n_sin[tgt]]

        if frame == 0:
            ctx_rows = tgt                              # positions/tags only; content is zeroed
            cond = torch.zeros_like(noisy_h[tgt])
        else:
            ctx_rows = tgt - r
            cond = enc[ctx_rows]
        pos_rows = tgt if duplicate_pos else ctx_rows
        x.insert(0, cond)
        mod.insert(0, c_mod[ctx_rows])
        cos.insert(0, n_cos[pos_rows])
        sin.insert(0, n_sin[pos_rows])

        if media_start:
            txt = torch.arange(media_start, device=noisy_h.device)
            x.insert(0, noisy_h[txt])
            # Not retimed: text already carries the video timestep in the base's own table, and
            # here the video being denoised is the noisy half, so its index is the right one.
            mod.insert(0, n_mod[txt] + shift * mm.MODALITY_NUM)
            cos.insert(0, n_cos[txt])
            sin.insert(0, n_sin[txt])

        return (torch.cat(x), (t_emb, torch.cat(mod), torch.cat(cos), torch.cat(sin)))

    def decode_frame(self, enc, clean_ctx, noisy_h, noisy_ctx, spans, frame, **kwargs):
        """`decoder_frame_input` through the decoder blocks. Returns the target frame's rows only.

        The context half and any text rows are inputs, not outputs: they exist to condition the
        `frame_rows` that get a velocity, and returning them would invite scoring a loss on rows
        the encoder already produced.
        """
        x, ctx = self.decoder_frame_input(enc, clean_ctx, noisy_h, noisy_ctx, spans, frame,
                                          **kwargs)
        for block in self.decoder_blocks:
            x = block(x, *ctx)
        return x[-spans.frame_rows:]

    def forward(self, **forward_kwargs):
        h, ctx = self.encode(**forward_kwargs)
        return self.decode(h, ctx)


def block_parity(a, b):
    """True when two DiT blocks hold bit-identical weights. `torch.equal`, not `allclose`: a
    freshly composed model has copied tensors, so anything short of exact means the copy is wrong.
    """
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def aliases(a, b):
    """True when two modules share any parameter STORAGE — the failure that survives a state_dict
    comparison, and the one that makes 55K steps train a single tensor through two paths."""
    sa = {p.data_ptr() for p in a.parameters()}
    return any(p.data_ptr() in sa for p in b.parameters())
