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

Which rows count as "context" is then a design choice, and it turned out to be a load-bearing one.
Putting AUDIO there — v0's plan, since §5 D1 decodes audio in a second bidirectional pass — makes
it blind to video by construction, and on real weights the encoder's audio rows come out at
centered cos 0.047 / -0.074 against a bidirectional pass: no video conditioning whatsoever, so the
encoder cache holds no audio-video fusion for that second pass to reuse. `row_time` therefore also
builds the AV clock, where audio is ORDERED on its own 40 Hz spans rather than exempt from order.
Both modalities causal on a shared time axis is still leak-free — nothing carries the future
backward — and audio comes back to 0.445 / 0.339. Not to the bidirectional 0.99, and it cannot:
causal audio genuinely cannot see future video, so that pass is a ceiling rather than a target.
Video pays about 0.03 of centered cos for it, having lost future audio. The rejected middle
option, letting audio queries see all video while video stays causal, is the loose mask above
wearing a hat — audio at the end of the clip has absorbed all of it, and any video row that can
read those rows has read the future.

The mask here is DENSE. That is correct at probe and test sizes and hopeless at 768p, where
S~62k makes an [S, S] bool 3.8 GB (§6.3). FlexAttention's `create_block_mask` is the shipping
path; `frame_index()` is already the `mask_mod` this needs, so that swap is local to `attention`.
"""

import torch
import torch.nn.functional as F

# Time for rows outside the causal order: text and r2v references, and — under the video-only
# clock — audio. Visible to every query, blind to everything ordered.
#
# -inf rather than a magic index, so the same sentinel works on the integer frame clock and on
# the packer's real-valued rotary clock, and so `k <= q` needs no special case at either end:
# a context KEY precedes everything, and a context QUERY is preceded by nothing but context.
CONTEXT = float("-inf")


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
        """[seq_len] float: CONTEXT for every non-video row, 0..latent_t-1 for video rows.

        The video-only clock, and shape-derived — it needs no packer positions, which is why the
        CPU tests can build it from a latent shape alone. `row_time(..., audio_is_context=True)`
        is the same mask from the packer's real positions; `test_clocks_agree_on_video_only`
        pins them equal so the two cannot drift apart.
        """
        idx = torch.arange(self.seq_len, device=device)
        vid = (idx - self.video_start).div(self.frame_rows, rounding_mode="floor").float()
        return torch.where(idx >= self.video_start, vid, torch.full_like(vid, CONTEXT))

    def rows_for_frames(self, start, stop):
        """Row slice of video frames [start, stop) — the unit a chunk is cut on."""
        return slice(self.video_start + start * self.frame_rows,
                     self.video_start + stop * self.frame_rows)

    def __repr__(self):
        return (f"FrameSpans(seq_len={self.seq_len}, video_start={self.video_start}, "
                f"latent_t={self.latent_t}, frame_rows={self.frame_rows})")


def row_time(pos_t, media_start, video_start=None):
    """[S] float: each row's time on the packer's own rotary t-axis, context rows at CONTEXT.

    `pos_t` is `image_position_ids(...)[:, 0]` — the packer's answer, not a reconstruction of it.
    This matters more than usual here, because the alignment between the two clocks is not
    something a caller could reasonably re-derive: audio latents advance 1.0 per row while video
    frames sit on `_video_t_grid`'s (1,4,4,4,4)x5/3 spans, chosen so that 17 pixel frames = 5
    latents = 28.33 rotary units ~ 28 audio latents. Reading `pos_t` gets that for free; a
    rows-per-frame ratio computed here would be a rounding error with a plausible story.

    `media_start` is where the ordered rows begin — `text_len + ref_row_count(refs)`. Everything
    before it (text, r2v references) is context under both clocks.

    Pass `video_start` for the **video-only clock**: audio joins the context, so the encoder is
    frame-causal in video alone and audio rows come out with no video conditioning at all
    (measured: centered cos 0.047 / -0.074, "Phase 2 mask cost"). Leave it None for the **AV
    clock**, where audio is ordered on its own 40 Hz spans, audio at time t sees video <= t and
    video at time t sees audio <= t. Both directions causal, so neither carries the future
    backward and the composition property the strict mask exists for is untouched.
    """
    t = pos_t.to(torch.float32).clone()
    t[:media_start] = CONTEXT
    if video_start is not None:
        t[media_start:video_start] = CONTEXT
    return t


def causal_mask(q_time, k_time, context_sees_video=False):
    """[Q, K] bool, True = attend. Times from `FrameSpans.frame_index` or from `row_time`.

    The whole rule is `k <= q`. Context is -inf, so a context KEY is visible to everyone and a
    context QUERY admits nothing but other context — "always visible" and "blind to the ordered
    rows" are the same inequality read from its two ends, and neither needs a branch.

    `context_sees_video=True` restores Phase 0's looser mask, where only video queries were
    restricted. It exists so the leak above can be measured and regression-tested, NOT as a
    supported encoder configuration — see the module docstring.
    """
    q, k = q_time.unsqueeze(1), k_time.unsqueeze(0)
    allowed = k <= q
    if context_sees_video:
        allowed = allowed | (q == CONTEXT)
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
