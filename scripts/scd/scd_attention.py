#!/usr/bin/env python3
"""Phase 2: frame spans, the encoder's causal mask, and a KV cache over H3's packed sequence.

H3 packs one flat sequence, `[text | refs | audio | video]`, and attends over all of it with
`mask=None`. SCD needs the video half to be frame-causal so the encoder can run once per frame
and be cached. Two things have to be true for that, and only the first is obvious:

1. A video row may attend to context rows and to video frames <= its own.
2. **Context rows must not attend to video at all.**

(2) is not decoration. Phase 0's probe masked only *video* queries, which is frame-causal for a
single block and measurably not for a stack: text rows absorb every frame at layer 1, and at
layer 2 frame 0 reads those rows back. Measured on the tiny model, frame 0's output against a
run where later frames do not exist:

    depth 1   1.2e-07     (roundoff — the Phase 0 self-test's number)
    depth 2   3.7e-03
    depth 6   9.0e-03

Under the strict mask the same comparison stays at 3.6e-07 through 6 blocks. So (2) is the
difference between a cache that is a cache and a cache that is a plausible-looking bug: with a
leak, chunk 2's context rows differ from chunk 1's and the cached K/V are stale in a way no
shape check can see. `test_scd_attention.py` pins both halves.

The mask here is DENSE. That is correct at probe and test sizes and hopeless at 768p, where
S~62k makes an [S, S] bool 3.8 GB (§6.3). FlexAttention's `create_block_mask` is the shipping
path; `frame_index()` is already the `mask_mod` this needs, so that swap is local to `attention`.
"""

import torch
import torch.nn.functional as F

# Frame index for rows that are not video: text, r2v references, audio. Shared context — visible
# to every query, and blind to video.
#
# Audio sits here by SCOPE, not by nature. §4 keeps v0's audio bidirectional (a second decoder
# pass over the same encoder cache), so audio rows are encoded once and every video frame sees
# all of them. Fully causal AV would give audio its own spans on the 40 Hz clock and is a v1
# item; the consequence to remember is that this encoder is frame-causal in VIDEO only.
CONTEXT = -1


class FrameSpans:
    """Which latent frame each row of the packed sequence belongs to.

    Deliberately derived from `seq_len` and the video latent's own shape rather than by redoing
    the base's audio-row arithmetic: video is the LAST segment and is contiguous and frame-major
    (`patchify_video` emits t-major rows), so `video_start = seq_len - latent_t * frame_rows` is
    exact whatever the preamble decided about refs and audio. Recomputing the audio row count
    here would be a second copy of a thing that already exists, which is how Phase 0's numbers
    drifted from their JSON.
    """

    def __init__(self, seq_len, latent_t, frame_rows):
        n_video = latent_t * frame_rows
        if not 0 < n_video <= seq_len:
            raise ValueError(f"{n_video} video rows do not fit in a sequence of {seq_len}")
        self.seq_len = seq_len
        self.latent_t = latent_t
        self.frame_rows = frame_rows
        self.video_start = seq_len - n_video

    def frame_index(self, device=None):
        """[seq_len] long: CONTEXT for context rows, 0..latent_t-1 for video rows."""
        idx = torch.arange(self.seq_len, device=device)
        vid = (idx - self.video_start).div(self.frame_rows, rounding_mode="floor")
        return torch.where(idx >= self.video_start, vid, torch.full_like(idx, CONTEXT))

    def rows_for_frames(self, start, stop):
        """Row slice of video frames [start, stop) — the unit a chunk is cut on."""
        return slice(self.video_start + start * self.frame_rows,
                     self.video_start + stop * self.frame_rows)

    def __repr__(self):
        return (f"FrameSpans(seq_len={self.seq_len}, video_start={self.video_start}, "
                f"latent_t={self.latent_t}, frame_rows={self.frame_rows})")


def causal_mask(q_frames, k_frames, context_sees_video=False):
    """[Q, K] bool, True = attend. Frame indices as returned by `FrameSpans.frame_index`.

    One rule covers both row kinds, because CONTEXT is -1 and every real frame is >= 0:
    a key is visible when it is context (`k < 0`) or not in the query's future (`k <= q`). A
    context query has q = -1, so no video key satisfies `k <= q` and it is blind to video for
    free.

    `context_sees_video=True` restores Phase 0's looser mask, where only video queries were
    restricted. It exists so the leak above can be measured and regression-tested, NOT as a
    supported encoder configuration — see the module docstring.
    """
    q, k = q_frames.unsqueeze(1), k_frames.unsqueeze(0)
    allowed = (k < 0) | (k <= q)
    if context_sees_video:
        allowed = allowed | (q < 0)
    return allowed


class LayerCache:
    """One block's K/V for rows already encoded. Post-RoPE, post-q/k-norm — cached at the point
    where the values stop depending on anything a later chunk can change."""

    def __init__(self):
        self.k = None
        self.v = None

    def append(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=0)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=0)
        return self.k, self.v

    def __len__(self):
        return 0 if self.k is None else self.k.shape[0]


class KVCache:
    """A `LayerCache` per encoder block. Every block holds the same rows, so `__len__` is the
    cache's row count and disagreement between layers is a bug worth catching loudly."""

    def __init__(self, num_blocks):
        self.layers = [LayerCache() for _ in range(num_blocks)]

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        n = {len(c) for c in self.layers}
        if len(n) != 1:
            raise RuntimeError(f"layers hold different row counts: {sorted(n)}")
        return n.pop()

    def bytes(self):
        return sum(t.numel() * t.element_size()
                   for c in self.layers for t in (c.k, c.v) if t is not None)


def attention(attn, x, cos, sin, mask=None, cache=None):
    """The base's own `Attention` module, with a mask and an optional KV cache.

    Takes the module rather than reimplementing it: same projection, same q/k norms, same RoPE,
    same SDPA call in the same order, so with `mask=None, cache=None` this is bit-identical to
    `attn(x, cos, sin)` — asserted, not assumed, by `test_unmasked_matches_base`.

    `cos`/`sin` are the NEW rows' RoPE only. Cached keys were rotated by their own positions when
    they were written, which is the reason the cache stores post-RoPE k: a position-dependent
    tensor cached before rotation would need re-rotating on every read.
    """
    from fizgig.minimax.model import apply_rope_split_half   # lazy: CI has no fizgig

    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
    k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    v = v.view(s, attn.heads, attn.head_dim)
    if cos is not None:
        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)
    if cache is not None:
        k, v = cache.append(k, v)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.squeeze(0).transpose(0, 1).reshape(s, attn.heads * attn.head_dim)
    return attn.out_proj(out)


def run_block(block, x, ctx, mask=None, cache=None):
    """One `DiTBlock` with its attention swapped for the masked/cached one.

    The block's own code runs — AdaLN modulation, the gated residuals, the MLP — because those
    are exactly what a hand-copied block loop gets subtly wrong and then reports as a finding.
    Only `attn.forward` is substituted, and it is restored in `finally`.
    """
    if mask is None and cache is None:
        return block(x, *ctx)

    def patched(h, cos=None, sin=None):
        return attention(block.attn, h, cos, sin, mask=mask, cache=cache)

    # An instance attribute shadows the class method for `__call__`; deleting it restores the
    # class lookup, which assigning the bound method back would not.
    block.attn.forward = patched
    try:
        return block(x, *ctx)
    finally:
        del block.attn.forward
