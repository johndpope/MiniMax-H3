#!/usr/bin/env python3
"""The Phase 3 clip set: cached video latents + cached text embeddings, keyed by MANIFEST.tsv.

This is a loader, not a `torch.utils.data.Dataset`. Batch size is 1 and will stay 1 — every clip
has a different text length (226..265 tokens across the 13 clips), so a batch would need padding
in the TEXT segment, and the base's packer builds one `[text | refs | audio | video]` sequence per
forward with `text_len` baked into the RoPE positions and the modulation rows. Padding that is a
change to the packer, which is the one thing `scd_model.preamble` exists to avoid re-deriving.
So: one clip per step, gradient accumulation if a larger effective batch is wanted.

MANIFEST.tsv is the source of truth, not a glob over the directory. The `.safetensors` are
gitignored, so the manifest plus the `.txt` prompts is the only thing that survives a clone, and a
glob would train on whatever files happened to be lying in `clips/` — including an earlier
geometry left over from a re-encode.

Three things about this set that a training run has to know and cannot see from the tensors:

  * It is 512x512, so 32x32 latents, patchified 1x2x2 -> 256 rows per latent frame. 768p is 1008.
    Nothing trained here validates the 768p geometry; it validates the mechanism at 1/4 the rows.
  * `isodiorama640` is the SAME source video and a byte-identical prompt as `isodiorama`, encoded
    at 640x640 as Phase 0's geometry control. As training data it is a near-duplicate at a
    different frame_rows, so it is excluded by default — `include_control=True` puts it back.
  * `*_w2` rows are a second temporal window of the base clip (start=2.5s). Same text embedding,
    different frames — doubles the set without reloading the 32B text encoder.
  * There is no audio. The clips carry video latents only; the base draws its own noise rows for
    the audio segment. Every audio-side claim in this repo is measured on noise (see the design
    doc's audio-correlation subsection) and a train loop here does not change that.

Usage:
    from scd_data import ClipSet
    clips = ClipSet("scripts/scd/clips")
    batch = clips.load("pixelgraph", device="cuda", dtype=torch.bfloat16)
    # batch["video_latent"] [1, 24, 7, 32, 32] fp32, batch["text_embeds"] [1, L, 5120]

    python3 scripts/scd/scd_data.py --check       # verify every manifest row loads
"""

import argparse
import os

import torch

# The packer accepts latent_t on a 5n+2 grid ({2, 7, 12, 17, ...}); the clips are stored at 8 so
# that the 29th pixel frame lands on a full temporal group. 7 is the largest valid value below it.
# Truncation is exact rather than approximate: the video VAE is temporally causal, so latent frame
# f is a function of pixel frames <= 4f and dropping frame 7 cannot change frames 0..6.
DEFAULT_LATENT_T = 7
GEOMETRY_CONTROL = "isodiorama640"


class MissingCache(SystemExit):
    """A manifest row whose tensors are not on disk. Carries the command that rebuilds them."""


def read_manifest(path):
    """[(name, source, style, camera, subjects)] in file order, comments and blanks dropped."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(f"{path}: expected 5 tab-separated fields, got "
                                 f"{len(parts)} in {line!r}")
            rows.append(tuple(parts))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


class ClipSet:
    """The manifest's clips, loaded on demand.

    Holds no tensors: 13 clips of latents plus bf16 text is ~40 MB and would fit, but the point of
    keeping it lazy is that `names` and `geometry` stay usable on a machine where the cache was
    never built — CI can check the manifest parses without the gitignored files existing.
    """

    def __init__(self, root, latent_t=DEFAULT_LATENT_T, include_control=False):
        self.root = root
        self.latent_t = latent_t
        self.rows = {r[0]: r for r in read_manifest(os.path.join(root, "MANIFEST.tsv"))}
        self.names = [n for n in self.rows
                      if include_control or n != GEOMETRY_CONTROL]

    def __len__(self):
        return len(self.names)

    def paths(self, name):
        if name not in self.rows:
            raise KeyError(f"{name} is not in the manifest ({', '.join(self.rows)})")
        return (os.path.join(self.root, f"{name}_latents.safetensors"),
                os.path.join(self.root, f"{name}_te.safetensors"))

    def present(self, name):
        return all(os.path.exists(p) for p in self.paths(name))

    def geometry(self, name):
        """(latent_h, latent_w) read from the latent file's header, without loading the tensor.

        Frame geometry decides `frame_rows`, which decides every decoder input shape, so a training
        loop that shuffles across geometries recompiles per clip. Reading it from the header rather
        than from the manifest's `resolution` metadata means the answer comes from the tensor that
        will actually be trained on.
        """
        from safetensors import safe_open
        latents, _ = self.paths(name)
        self._require(name)
        with safe_open(latents, framework="pt") as f:
            shape = f.get_slice("latent").get_shape()
        return tuple(shape[-2:])

    def _require(self, name):
        latents, te = self.paths(name)
        missing = [p for p in (latents, te) if not os.path.exists(p)]
        if missing:
            src = self.rows[name][1]
            raise MissingCache(
                f"{name}: missing {', '.join(os.path.basename(p) for p in missing)}. The clip "
                f"cache is gitignored; rebuild with\n"
                f"  python3 scripts/scd/encode_clip.py $ROOT/{src} --latent-t 8 --size 512 512 "
                f"--out {latents}\n"
                f"  python3 scripts/scd/encode_text.py {self.root}/{name}.txt --out {te}\n"
                f"($ROOT is recorded in {self.root}/MANIFEST.tsv)")

    def load(self, name, device="cpu", dtype=torch.bfloat16):
        """`dict(name, video_latent, text_embeds)` ready to hand to the base's forward.

        The latent stays fp32 while the text goes to `dtype`. That asymmetry is the base's, not a
        choice made here: `phase0_probe.build_stream` noises the latent in fp32 and only casts at
        the patch projection, so upcasting the cached bf16 text is the cheap half and downcasting
        the latent early would quantize the clean signal the encoder is supposed to see.

        `attention_mask` is in the cache files and is deliberately not returned: every clip's mask
        is all-ones (no padding, because each was encoded alone), so plumbing it would add a
        parameter that is constant across the whole set and untested against the padded case.
        """
        from safetensors.torch import load_file
        self._require(name)
        latents, te = self.paths(name)

        z = load_file(latents)["latent"]
        if z.shape[2] < self.latent_t:
            raise ValueError(f"{name}: {z.shape[2]} latent frames, need {self.latent_t}")
        z = z[:, :, :self.latent_t].to(device, torch.float32)

        h = load_file(te)["hidden_states"]
        return {"name": name, "video_latent": z,
                "text_embeds": h.unsqueeze(0).to(device, dtype)}

    def by_geometry(self, only=None):
        """{(h, w): [names]} — the recompile-free groups a shuffled loader has to respect."""
        groups = {}
        for name in (self.names if only is None else only):
            groups.setdefault(self.geometry(name), []).append(name)
        return groups


def epoch_order(names, seed, epoch):
    """A shuffled order over `names`, deterministic in (seed, epoch).

    A generator seeded per epoch rather than a long-lived one: a resumed run reconstructs the same
    order from the step count alone, so a crash at step 900 does not silently reshuffle the set on
    restart and turn a held-out clip into a seen one.
    """
    g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return [names[i] for i in torch.randperm(len(names), generator=g).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "clips"))
    ap.add_argument("--latent-t", type=int, default=DEFAULT_LATENT_T)
    ap.add_argument("--include-control", action="store_true")
    ap.add_argument("--check", action="store_true", help="load every row and print its shapes")
    args = ap.parse_args()

    clips = ClipSet(args.root, latent_t=args.latent_t, include_control=args.include_control)
    print(f"manifest    : {len(clips.rows)} rows, {len(clips)} selected"
          f"{'' if args.include_control else f' ({GEOMETRY_CONTROL} held out)'}")

    if not args.check:
        for name in clips.names:
            print(f"  {name:<16} {'cached' if clips.present(name) else 'MISSING'}")
        return

    for name in clips.names:
        b = clips.load(name)
        print(f"  {name:<16} latent {tuple(b['video_latent'].shape)}  "
              f"text {tuple(b['text_embeds'].shape)}")
    groups = clips.by_geometry()
    print(f"geometries  : {len(groups)} — " +
          "; ".join(f"{h}x{w}: {len(v)}" for (h, w), v in sorted(groups.items())))


if __name__ == "__main__":
    main()
