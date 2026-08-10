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

import torch
from torch import nn

from scd_attention import FrameSpans, KVCache, causal_mask, run_block


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
        """Run the base's input packing and stop at the first block. Returns (h, ctx).

        `ctx` is `(t_emb, mod_row, cos, sin)` — the per-block arguments the base built, captured
        from its own first block rather than recomputed, so this cannot drift from the preamble.
        The hook raises to abort: the preamble is what we want and the 50 blocks behind it are
        not, and at 768p running them to throw the result away is minutes of GPU time.
        """
        cap = {}

        def grab(_module, args):
            cap["h"], cap["ctx"] = args[0], tuple(args[1:])
            raise _PreambleDone

        handle = self.base.blocks[0].register_forward_pre_hook(grab)
        try:
            self.base(**forward_kwargs)
        except _PreambleDone:
            pass
        finally:
            handle.remove()
        return cap["h"], cap["ctx"]

    def spans(self, video_latent, seq_len):
        """`FrameSpans` for a packed sequence of `seq_len` rows holding this video latent."""
        _, _, latent_t, lat_h, lat_w = video_latent.shape
        ph, pw = self.base.patch_size[1], self.base.patch_size[2]
        return FrameSpans(seq_len, latent_t, (lat_h // ph) * (lat_w // pw))

    def encode(self, *, mask=None, cache=None, **forward_kwargs):
        """Run the base preamble and the encoder prefix. Returns (h, ctx).

        With `mask=None, cache=None` every block runs its own unmodified forward, so the encoder
        is the base's first `encoder_depth` blocks exactly — `test_prefix_parity` holds by
        construction rather than by the masked path happening to be equivalent.
        """
        h, ctx = self.preamble(**forward_kwargs)
        for i, block in enumerate(self.encoder_blocks):
            h = run_block(block, h, ctx, mask=mask, cache=None if cache is None else cache[i])
        return h, ctx

    def encode_chunked(self, chunk_frames, **forward_kwargs):
        """Encode frame-causally in chunks of `chunk_frames`, carrying a KV cache. Returns
        (h, ctx, cache), where `h` is the full sequence reassembled.

        The first chunk carries the context rows; later chunks are video rows only, and read the
        earlier rows out of the cache. Under the strict mask this is not an approximation of a
        single masked pass, it is the same arithmetic in a different order, which is what
        `test_chunked_matches_full` asserts.

        The preamble runs ONCE, over the whole clip. Real AR inference cannot do that — later
        frames do not exist yet — so this is the correctness harness for the cache, not the
        inference driver. That driver is Phase 5's, and it must build positions incrementally.
        """
        h, ctx = self.preamble(**forward_kwargs)
        sp = self.spans(forward_kwargs["video_latent"], h.shape[0])
        frames = sp.frame_index(h.device)
        cache = KVCache(len(self.encoder_blocks))
        t_emb, mod_row, cos, sin = ctx

        out = []
        start = 0
        while start < sp.latent_t:
            stop = min(start + chunk_frames, sp.latent_t)
            rows = slice(0, sp.rows_for_frames(0, stop).stop) if start == 0 \
                else sp.rows_for_frames(start, stop)
            chunk_ctx = (t_emb, mod_row[rows], cos[rows], sin[rows])
            m = causal_mask(frames[rows], frames[:rows.stop])
            x = h[rows]
            for i, block in enumerate(self.encoder_blocks):
                x = run_block(block, x, chunk_ctx, mask=m, cache=cache[i])
            out.append(x)
            start = stop
        return torch.cat(out, dim=0), ctx, cache

    def decode(self, h, ctx):
        """Run the decoder re-composition over encoder output. Same block signature as the base."""
        for block in self.decoder_blocks:
            h = block(h, *ctx)
        return h

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
