#!/usr/bin/env python3
"""Sweep both Phase 0 probes over many clips and append every run to a ledger.

Why this exists. Phase 0 produced two findings that later turned out to be artefacts of how the
measurement was configured, not properties of H3 — raw cos-sim on a residual stream that is 91-99%
one shared vector, and a sigma grid sitting where H3 has ~no training density. Both were caught by
hand, late, after the numbers had already been written into the design doc. The design doc's
numbers are also hand-transcribed, so they can drift from the JSON they cite without anything
complaining.

So: one entry point that (a) loads the 66 GB base ONCE and reuses it across clips, since the NF4
load is ~2 min and the probes themselves are ~25 s and ~3 min, (b) records the full config of every
run next to its results in an append-only JSONL, so a result can never be read without the
conditions that produced it, and (c) reports whether the findings are stable ACROSS clips, which is
the open question Phase 1 actually depends on — everything in the doc so far is n=1.

Clip discovery follows the naming convention `NAME_latents.safetensors` / `NAME_te.safetensors`.

Usage:
    python3 scripts/scd/run_phase0.py \
        --checkpoint /path/to/FL2VA/transformer \
        --clips 'scripts/scd/clips/*_latents.safetensors'

    python3 scripts/scd/run_phase0.py --summarize-only     # re-read the ledger, no GPU needed
"""

import argparse
import glob
import json
import os
import subprocess
import time
from collections import Counter

import torch

import phase0_leaveout as LO
import phase0_validate as VA
from phase0_probe import import_fizgig, load_latents, load_text

LEDGER = "docs/phase0_ledger.jsonl"
SUMMARY = "docs/phase0_summary.json"


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def text_for(latents_path):
    """`X_latents.safetensors` -> `X_te.safetensors`."""
    cand = latents_path.replace("_latents.safetensors", "_te.safetensors")
    if cand == latents_path or not os.path.exists(cand):
        raise SystemExit(f"no text embedding beside {latents_path} (expected {cand})")
    return cand


def append(row, path=LEDGER):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def summarize(path=LEDGER, summary_json=SUMMARY):
    """Across-clip stability. n=1 is where every Phase 0 claim currently sits; this says when
    that stops being true, and flags disagreement rather than averaging it away."""
    if not os.path.exists(path):
        raise SystemExit(f"no ledger at {path} — run a sweep first")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    print(f"{len(rows)} runs in {path}\n")

    # Runs are only comparable within one configuration. Grouping by it rather than pooling is the
    # whole point: the off-distribution sigma run and the on-distribution one must never be mixed.
    groups = {}
    for r in rows:
        key = (r["sigma_ref"], r["sigma_test"], tuple(r["loo_sigmas"]), r["latent_t"],
               r["base_quant"], r["checkpoint"])
        groups.setdefault(key, []).append(r)

    summaries = {}
    for key, g in groups.items():
        s_ref, s_test, loo_sig, lt, quant, ckpt = key
        print(f"--- {ckpt}  sigma {s_ref:.3f}->{s_test:.3f}  latent_t={lt}  {quant}  n={len(g)}")
        kt = Counter(r["knee_text"] for r in g)
        kv = Counter(r["knee_video"] for r in g)
        print(f"  knee text  {dict(sorted(kt.items()))}")
        print(f"  knee video {dict(sorted(kv.items()))}")
        lb = Counter(tuple(r["load_bearing"]) for r in g)
        print(f"  load-bearing set {dict(lb)}")
        thirds = [sum(r["thirds"][i] for r in g) / len(g) for i in range(3)]
        print(f"  mean rel cost by third: early {thirds[0]:.4f}  middle {thirds[1]:.4f}  "
              f"late {thirds[2]:.4f}")
        causal = max(max(r["causal_rel_l2"]) for r in g)
        print(f"  worst causal-mask rel L2 over all clips/blocks: {causal:.4f}")

        if len(g) < 5:
            verdict = "INSUFFICIENT"
            print("  VERDICT: n<5, not enough to call anything stable")
        elif len(kt) == 1 and len(kv) == 1 and len(lb) == 1:
            verdict = "STABLE"
            print(f"  VERDICT: STABLE — knee {kt.most_common(1)[0][0]}, "
                  f"decoder must contain {lb.most_common(1)[0][0]}")
        else:
            verdict = "UNSTABLE"
            print("  VERDICT: UNSTABLE across clips — the split is clip-dependent, do not "
                  "hardcode it in Phase 1")
        print()

        summaries[key] = {
            "config": {"sigma_ref": s_ref, "sigma_test": s_test, "loo_sigmas": list(loo_sig),
                       "latent_t": lt, "base_quant": quant, "checkpoint": ckpt},
            "clips": sorted(r["clip"] for r in g),
            "n": len(g),
            "verdict": verdict,
            "knee_text_mode": kt.most_common(1)[0][0],
            "knee_text_agree": kt.most_common(1)[0][1] / len(g),
            "knee_video_mode": kv.most_common(1)[0][0],
            "knee_video_agree": kv.most_common(1)[0][1] / len(g),
            "load_bearing_mode": list(lb.most_common(1)[0][0]),
            "load_bearing_agree": lb.most_common(1)[0][1] / len(g),
            "third_early": thirds[0], "third_middle": thirds[1], "third_late": thirds[2],
            "worst_causal_rel_l2": causal,
        }

    if summary_json:
        # Flat keys at the top level, because docs/phase0_claims.json can only cite `{file, key}`
        # on a flat JSON object and the ledger itself is JSONL. The group described is the one with
        # the most clips: a claim should quote the best-supported configuration, and `config` below
        # names which one that is so the choice can never be silent.
        best = max(summaries.values(), key=lambda s: s["n"])
        out = dict(best, n_groups=len(summaries), source=path)
        with open(summary_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {summary_json} (largest group, n={best['n']} of {len(summaries)} groups)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--clips", help="glob for *_latents.safetensors")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--skip-leaveout", action="store_true",
                    help="validation sweep only (~25 s/clip vs ~3 min for both)")
    ap.add_argument("--summarize-only", action="store_true")
    VA.add_args(ap)
    ap.add_argument("--u-grid", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--sigmas", type=float, nargs="+", default=None)
    args = ap.parse_args()

    if args.summarize_only:
        return summarize(args.ledger)
    if not args.checkpoint or not args.clips:
        raise SystemExit("--checkpoint and --clips are required unless --summarize-only")

    clips = sorted(glob.glob(args.clips))
    if not clips:
        raise SystemExit(f"no clips matched {args.clips!r}")

    mm, load_dit = import_fizgig(args.fizgig_src)
    device, dtype = torch.device("cuda"), torch.bfloat16
    mm.pixel_frames_for_latent(args.latent_t)   # fail before the 2-minute load, not after

    sigma_ref, sigma_test = VA.resolve_sigmas(mm, args)
    loo_sigmas = LO.resolve_sigmas(mm, args)
    print(f"{len(clips)} clips | validate sigma {sigma_ref:.4f}->{sigma_test:.4f} | "
          f"leaveout sigmas {[round(s, 4) for s in loo_sigmas]}", flush=True)

    model = load_dit(args.checkpoint, device=device, compute_dtype=dtype,
                     quantize=args.base_quant != "none", blocks_to_swap=args.blocks_to_swap,
                     base_quant="nf4" if args.base_quant == "nf4" else "auto")
    model.enable_block_swap(args.blocks_to_swap)

    sha, ckpt = git_sha(), os.path.basename(args.checkpoint.rstrip("/"))
    t0 = time.time()
    for n, lat in enumerate(clips, 1):
        name = os.path.basename(lat).replace("_latents.safetensors", "")
        print(f"\n[{n}/{len(clips)}] {name}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        video_latent = load_latents(lat, args.latent_t, device)
        text_embeds = load_text(text_for(lat), device, dtype)

        va = VA.measure(mm, model, video_latent, text_embeds, sigma_ref, sigma_test,
                        device, dtype, attn_sample=args.attn_sample, quiet=True)
        lo = ({} if args.skip_leaveout
              else LO.measure(mm, model, video_latent, text_embeds, loo_sigmas, device, dtype,
                              quiet=True))

        row = {
            "clip": name, "git_sha": sha, "checkpoint": ckpt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "latent_t": args.latent_t, "base_quant": args.base_quant,
            "seq_len": va["seq_len"], "n_blocks": va["n_blocks"],
            "sigma_ref": sigma_ref, "sigma_test": sigma_test,
            "loo_sigmas": [] if args.skip_leaveout else loo_sigmas,
            "knee_text": VA.knee(va["sigma_centered_cos_text"]),
            "knee_video": VA.knee(va["sigma_centered_cos"]),
            "sigma_centered_cos_text": va["sigma_centered_cos_text"],
            "sigma_centered_cos_video": va["sigma_centered_cos"],
            "causal_rel_l2": va["causal_rel_l2"],
            "own_frame_rel_l2": va["own_frame_rel_l2"],
            "common_mode_ratio": va["common_mode_ratio"],
            "attention_breakdown": va["attention_breakdown"],
            "relative_cost": lo.get("relative_cost", []),
            "load_bearing": LO.load_bearing(lo["relative_cost"]) if lo else [],
            "thirds": lo.get("mean_rel_cost_thirds", [0.0, 0.0, 0.0]),
        }
        append(row, args.ledger)
        print(f"  knee text {row['knee_text']} video {row['knee_video']} | "
              f"load-bearing {row['load_bearing']} | "
              f"worst causal relL2 {max(va['causal_rel_l2']):.4f}", flush=True)

        del video_latent, text_embeds, va, lo
        torch.cuda.empty_cache()

    print(f"\nappended {len(clips)} runs to {args.ledger} in {(time.time() - t0) / 60:.1f} min\n")
    summarize(args.ledger)


if __name__ == "__main__":
    main()
