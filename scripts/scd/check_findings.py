#!/usr/bin/env python3
"""Validate recorded Phase 0 results, and refuse the configurations that produced false findings.

Runs without a GPU, weights, or fizgig — it only reads JSON and Markdown, so CI can run it.

The two errors that made it into the design doc were both configuration errors that the result
files happily recorded without complaint:

  * sigma=0 as the invariance reference, and a flat 0.1..0.9 leave-one-out grid. H3 trains
    sigma = shift_sigma(u, 12), median ~0.92, ~3% of steps below 0.3, and below ~0.5 the model is
    anticorrelated with its own flow-matching target. Numbers from there are noise wearing a
    result's clothes.
  * latent_t off the DiT's 5n+2 grid. model.forward() raises on it; the block-by-block probe path
    does not, so it ran for hours and nobody noticed.

Neither is detectable by looking at the output — you have to look at the config. So they are
checked here, mechanically, on every result file and every ledger row.

The third check is transcription: docs/*.md quote numbers that live in the JSON, and prose does
not update itself when a probe is re-run. Any claim that matters is registered in
docs/phase0_claims.json as {file, key, index, expected, tol} and verified against its source.

Usage:
    python3 scripts/scd/check_findings.py            # from the repo root
"""

import argparse
import glob
import json
import os
import sys

# Below this the model is anticorrelated with the flow-matching target on H3's schedule, so a
# measurement taken there is not a measurement of the model.
MIN_ON_DIST_SIGMA = 0.5

REQUIRED_VALIDATE = ["sigma_ref", "sigma_test", "n_blocks", "sigma_centered_cos_text",
                     "sigma_centered_cos", "causal_rel_l2", "own_frame_rel_l2",
                     "common_mode_ratio", "attention_breakdown"]
REQUIRED_LEAVEOUT = ["sigmas", "n_blocks", "relative_cost", "mean_rel_cost_thirds",
                     "baseline_loss_by_sigma", "leaveout_loss_by_sigma"]


def on_grid(latent_t):
    """The DiT accepts 5n+2 latent frames (2, 7, 12, ...); 1 is the single-frame case."""
    return latent_t == 1 or (latent_t >= 2 and (latent_t - 2) % 5 == 0)


def check_config(tag, latent_t, sigmas, errs, warns, soft=False):
    """`soft` routes failures to warnings — used for runs already marked superseded, where a bad
    config is the documented reason the file is kept rather than a problem to fix."""
    bad = warns if soft else errs
    if latent_t is not None and not on_grid(latent_t):
        bad.append(f"{tag}: latent_t={latent_t} is off the DiT's 5n+2 grid (2, 7, 12, ...)")
    low = [s for s in sigmas if s < MIN_ON_DIST_SIGMA]
    if low:
        warns.append(f"{tag}: sigmas {[round(s, 4) for s in low]} sit below {MIN_ON_DIST_SIGMA}, "
                     f"where H3 has ~no training density — superseded run, or off-distribution")


def check_lengths(tag, payload, keys, errs):
    n = payload.get("n_blocks")
    for k in keys:
        v = payload.get(k)
        if isinstance(v, list) and n and len(v) != n:
            errs.append(f"{tag}: {k} has {len(v)} entries, expected n_blocks={n}")


def check_summary_fresh(path, p, errs):
    """The summary is derived from the ledger, and nothing regenerates it when a row is appended.

    A stale summary is the quiet failure this guards: claims cite it by flat key, so if a sweep adds
    clips and the summary is not rebuilt, every claim still passes while the doc quotes an n that no
    longer exists. Compared against the ledger rows sharing the summary's own configuration.
    """
    ledger = "docs/phase0_ledger.jsonl"
    if not os.path.exists(ledger):
        errs.append(f"{path}: summary present but no {ledger} to check it against")
        return
    cfg = p["config"]
    keys = ("sigma_ref", "sigma_test", "latent_t", "base_quant", "checkpoint")
    # latent_hw is compared through the same "legacy" placeholder summarize() groups by, so rows
    # written before the field existed keep matching the summary they produced.
    hw = cfg.get("latent_hw") or ["legacy"]
    matching = []
    for line in open(ledger):
        if not line.strip():
            continue
        r = json.loads(line)
        if (all(r.get(k) == cfg.get(k) for k in keys)
                and r.get("loo_sigmas") == cfg.get("loo_sigmas")
                and (r.get("latent_hw") or ["legacy"]) == hw):
            matching.append(r["clip"])

    if len(matching) != p.get("n"):
        errs.append(f"{path}: says n={p.get('n')} but {ledger} has {len(matching)} rows for that "
                    "configuration — re-run `run_phase0.py --summarize-only`")
    if sorted(matching) != sorted(p.get("clips", [])):
        only_ledger = sorted(set(matching) - set(p.get("clips", [])))
        errs.append(f"{path}: clip list disagrees with {ledger}; missing from summary: "
                    f"{only_ledger[:5]} — re-run `run_phase0.py --summarize-only`")


def check_result_files(errs, warns):
    seen = 0
    for path in sorted(glob.glob("docs/phase0_*.json")):
        p = json.load(open(path))
        if not isinstance(p, dict):
            continue
        # A superseded run is kept as the record of what a bad configuration produced, so it is
        # exempt from the schema but NOT from the config checks — the whole reason it is on disk
        # is that those checks fire on it.
        if "superseded_by" in p:
            warns.append(f"{path}: superseded by {p['superseded_by']} — not a live result")
            check_config(path, p.get("latent_t"),
                         [v for k, v in p.items() if k.startswith("sigma") and isinstance(v, float)]
                         + p.get("sigmas", []), errs, warns, soft=True)
            seen += 1
            continue
        if "relative_cost" in p:
            missing = [k for k in REQUIRED_LEAVEOUT if k not in p]
            check_config(path, p.get("latent_t"), p.get("sigmas", []), errs, warns)
            check_lengths(path, p, ["relative_cost"], errs)
        elif "sigma_centered_cos_text" in p:
            missing = [k for k in REQUIRED_VALIDATE if k not in p]
            sig = [p[k] for k in ("sigma_ref", "sigma_test") if k in p]
            check_config(path, p.get("latent_t"), sig, errs, warns)
            check_lengths(path, p, REQUIRED_VALIDATE, errs)
        elif "verdict" in p and "config" in p:
            missing = []
            cfg = p["config"]
            check_config(path, cfg.get("latent_t"),
                         [cfg[k] for k in ("sigma_ref", "sigma_test") if k in cfg]
                         + cfg.get("loo_sigmas", []), errs, warns)
            check_summary_fresh(path, p, errs)
        else:
            continue
        seen += 1
        if missing:
            errs.append(f"{path}: missing required keys {missing}")
    return seen


def check_ledger(errs, warns):
    path = "docs/phase0_ledger.jsonl"
    if not os.path.exists(path):
        return 0
    rows = 0
    for i, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        rows += 1
        r = json.loads(line)
        tag = f"{path}:{i}"
        for k in ("clip", "sigma_ref", "sigma_test", "latent_t", "n_blocks", "knee_text",
                  "knee_video", "git_sha"):
            if k not in r:
                errs.append(f"{tag}: ledger row missing {k!r}")
        check_config(tag, r.get("latent_t"),
                     [r[k] for k in ("sigma_ref", "sigma_test") if k in r] + r.get("loo_sigmas", []),
                     errs, warns)
    return rows


def field_eq(have, want):
    """Selector match. Floats compare with slack so a claim can name sigma as 0.5714285714 rather
    than carrying the full 17-digit repr, which differs in the last bit between the value a claim
    was written from and the one shift_sigma() recomputes."""
    if isinstance(have, float) or isinstance(want, float):
        try:
            return abs(float(have) - float(want)) < 1e-9
        except (TypeError, ValueError):
            return False
    return have == want


def load_claim_source(c, errs):
    """The object a claim reads its key from — a flat JSON file, or one row of the ledger.

    Only the largest configuration group reaches phase0_summary.json, so a number belonging to any
    other group (a geometry control, say) has nowhere flat to live. Rather than widen the summary
    until every group is in it, a claim may name a ledger row by `row: {field: value}`; the ledger
    is the primary record anyway, and this makes it quotable without copying numbers out of it.
    """
    src = c["file"]
    if "row" not in c:
        return json.load(open(src))
    match = [r for r in (json.loads(ln) for ln in open(src) if ln.strip())
             if all(field_eq(r.get(k), v) for k, v in c["row"].items())]
    if len(match) != 1:
        errs.append(f"claim {c['id']!r}: row selector {c['row']} matches {len(match)} rows in "
                    f"{src}, expected exactly 1")
        return None
    return match[0]


def check_claims(errs):
    """Every number the design doc leans on, checked against the file it came from."""
    path = "docs/phase0_claims.json"
    if not os.path.exists(path):
        return 0
    claims = json.load(open(path))
    for c in claims:
        src = c["file"]
        if not os.path.exists(src):
            errs.append(f"claim {c['id']!r}: source {src} does not exist")
            continue
        payload = load_claim_source(c, errs)
        if payload is None:
            continue
        v = payload.get(c["key"])
        if v is None:
            errs.append(f"claim {c['id']!r}: {src} has no key {c['key']!r}")
            continue
        if "index" in c:
            v = v[c["index"]]
        if abs(v - c["expected"]) > c["tol"]:
            errs.append(f"claim {c['id']!r}: doc says {c['expected']}, {src}:{c['key']}"
                        f"{'[' + str(c['index']) + ']' if 'index' in c else ''} is {v:.4f} "
                        f"(tol {c['tol']})")
    return len(claims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = ap.parse_args()

    if not os.path.isdir("docs"):
        raise SystemExit("run from the repo root (no docs/ here)")

    errs, warns = [], []
    n_files = check_result_files(errs, warns)
    n_rows = check_ledger(errs, warns)
    n_claims = check_claims(errs)
    print(f"checked {n_files} result files, {n_rows} ledger rows, {n_claims} registered claims")

    for w in warns:
        print(f"WARN  {w}")
    for e in errs:
        print(f"ERROR {e}")
    if errs or (args.strict and warns):
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
