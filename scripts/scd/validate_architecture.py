#!/usr/bin/env python3
"""One-shot architecture gate for the H3 × SCD port.

What this is for
----------------
When you change attention, the model split, LoRA, or the docs that quote numbers, run this.
It re-checks three layers of evidence and writes one JSON + one Markdown report under docs/:

  1. Code contracts (unit tests) — does the split / mask / cache / LoRA behave as designed?
  2. Recorded measurements (docs/*.json) — are config and claims still consistent?
  3. Headline numbers already measured (Tier 0 primitives, Tier 1 real-weights graph)
     — surface the speed and VRAM claims so a human can see them without digging JSON.

It does NOT re-train, re-sample, or re-decode pixels. Those are quality questions; this
script answers "is the architecture still what we think it is?"

Usage (from the repo root):
    python3 scripts/scd/validate_architecture.py
    python3 scripts/scd/validate_architecture.py --skip-tests          # claims + numbers only
    python3 scripts/scd/validate_architecture.py --bench               # also re-run Tier 0 (needs GPU)
    python3 scripts/scd/validate_architecture.py --out-dir docs

Exit code: 0 if everything required passed, 1 if any required check failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Paths (relative to the repo root). Keep these here so the report and the
# runner always agree on where results live.
# ---------------------------------------------------------------------------
REPO_ROOT_MARKER = "docs"
DEFAULT_OUT_DIR = "docs"
RESULTS_JSON = "arch_validation_results.json"
RESULTS_MD = "arch_validation_report.md"

# Tiny-model unit tests: no real H3 weights, seconds on CPU/GPU.
# Run as scripts (not pytest) — each file has its own CASES list and main().
ARCH_TEST_SCRIPTS = [
    "scripts/scd/test_scd_attention.py",  # frame spans, causal mask, KV cache (key-value memory)
    "scripts/scd/test_scd_model.py",      # encoder/decoder split and token concat
    "scripts/scd/test_scd_lora.py",       # LoRA adapters on the split graph
    "scripts/scd/test_phase3_train.py",   # training step math (noise, loss, free-as-you-go)
    "scripts/scd/test_phase3_sample.py",  # oracle vs AR (autoregressive) sampling rules
]

# Headline numbers we care about when reading Tier 0 / Tier 1 result files.
# Keys are human labels; values tell us which file and how to pull the number.
TIER0_PATH = "docs/tier0_results.json"
TIER1_PATH = "docs/phase25_tier1.json"


def repo_root() -> str:
    """Walk up until we find docs/, so the script works from repo root or scripts/scd/."""
    here = os.path.abspath(os.getcwd())
    for path in [here, os.path.dirname(os.path.abspath(__file__)),
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]:
        if os.path.isdir(os.path.join(path, REPO_ROOT_MARKER)):
            return path
    raise SystemExit(
        "Could not find the repo root (no docs/ directory nearby). "
        "Run from MiniMax-H3 or scripts/scd/."
    )


def git_sha(root: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_cmd(cmd: list[str], cwd: str, timeout_s: int = 600) -> dict[str, Any]:
    """Run a command, capture stdout/stderr, never raise — failures become a result row."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s,
        )
        return {
            "cmd": cmd,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "seconds": round(time.time() - t0, 3),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "exit_code": -1,
            "ok": False,
            "seconds": round(time.time() - t0, 3),
            "stdout_tail": ((e.stdout or b"") if isinstance(e.stdout, bytes)
                            else (e.stdout or ""))[-4000:],
            "stderr_tail": f"timed out after {timeout_s}s",
        }


def parse_test_summary(stdout: str) -> dict[str, Any]:
    """Native runners print 'N/M passed' on the last line. Recover that for the report."""
    passed = failed = total = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        # e.g. "19/19 passed" or "12/14 passed, 2 failed"
        if "passed" in line and "/" in line:
            head = line.split()[0]  # "19/19"
            try:
                a, b = head.split("/")
                passed, total = int(a), int(b)
                failed = total - passed
            except ValueError:
                pass
            break
    return {"passed": passed, "failed": failed, "total": total}


def run_unit_tests(root: str) -> list[dict[str, Any]]:
    # Tests live next to each other under scripts/scd/ and import siblings by name
    # (e.g. `from test_scd_model import TINY`). Run them with that folder as cwd
    # so those imports work without packaging the tree.
    scd_dir = os.path.join(root, "scripts/scd")
    rows = []
    for script in ARCH_TEST_SCRIPTS:
        name = os.path.basename(script)
        path = os.path.join(root, script)
        if not os.path.isfile(path):
            rows.append({
                "name": name,
                "path": script,
                "ok": False,
                "skipped": False,
                "error": "file missing",
            })
            continue
        # Pass only the basename: cwd is already scripts/scd/.
        result = run_cmd([sys.executable, name], cwd=scd_dir)
        summary = parse_test_summary(result["stdout_tail"])
        rows.append({
            "name": name,
            "path": script,
            "ok": result["ok"],
            "skipped": False,
            "seconds": result["seconds"],
            "exit_code": result["exit_code"],
            **summary,
            "stdout_tail": result["stdout_tail"][-800:],
            "stderr_tail": result["stderr_tail"][-400:] if not result["ok"] else "",
        })
        status = "ok" if result["ok"] else "FAIL"
        detail = ""
        if summary["total"] is not None:
            detail = f"  {summary['passed']}/{summary['total']} passed"
        print(f"  [{status}] {name}{detail}  ({result['seconds']:.1f}s)", flush=True)
    return rows


def run_check_findings(root: str, strict: bool) -> dict[str, Any]:
    """Docs/claims gate: configs, ledger freshness, numbers quoted in the design doc."""
    cmd = [sys.executable, "scripts/scd/check_findings.py"]
    if strict:
        cmd.append("--strict")
    result = run_cmd(cmd, cwd=root, timeout_s=120)
    warns = [ln[5:].strip() for ln in result["stdout_tail"].splitlines() if ln.startswith("WARN ")]
    errs = [ln[6:].strip() for ln in result["stdout_tail"].splitlines() if ln.startswith("ERROR ")]
    print(f"  [{'ok' if result['ok'] else 'FAIL'}] check_findings.py  "
          f"({len(errs)} errors, {len(warns)} warnings)", flush=True)
    return {
        "name": "check_findings",
        "ok": result["ok"],
        "seconds": result["seconds"],
        "exit_code": result["exit_code"],
        "errors": errs,
        "warnings": warns,
        "stdout_tail": result["stdout_tail"][-1200:],
    }


def summarize_tier0(path: str) -> dict[str, Any] | None:
    """Pull speedup table from an existing Tier 0 JSON (weights-free microbench)."""
    if not os.path.isfile(path):
        return None
    data = json.load(open(path))
    rows = []
    for r in data.get("results", []):
        steps = r.get("steps") or {}
        # Prefer N=16 as the design-doc midpoint; fall back to whatever is there.
        key = "16" if "16" in steps else (list(steps.keys())[0] if steps else None)
        if key is None:
            continue
        s = steps[key]
        rows.append({
            "config": r.get("config"),
            "tokens": r.get("seq"),
            "steps": int(key),
            "speedup": round(s["speedup"], 3),
            "flop_model_speedup": round(s.get("flop_model_speedup", 0), 3),
            # How far measured speedup sits from the pure FLOP (math ops) estimate.
            "model_error_pct": round(
                abs(s["speedup"] - s.get("flop_model_speedup", s["speedup"]))
                / max(s.get("flop_model_speedup", 1), 1e-9) * 100, 2,
            ),
        })
    return {
        "path": path,
        "device": data.get("device"),
        "torch": data.get("torch"),
        "dtype": data.get("dtype"),
        "timestamp": data.get("timestamp"),
        "rows": rows,
        # Architecture kill criterion: SCD should be faster at 768p/15s (N≈16).
        "pass_kill_criterion": any(
            r["config"] and "15s" in r["config"] and r["speedup"] > 1.2 for r in rows
        ),
    }


def summarize_tier1(path: str) -> dict[str, Any] | None:
    """Pull real-weights graph timings + flat KV-cache claim from Tier 1 JSON."""
    if not os.path.isfile(path):
        return None
    data = json.load(open(path))
    lengths = data.get("by_length") or []
    cache_rows = [row.get("cache_rows") for row in lengths if row.get("cache_rows") is not None]
    decoder_ms = [row.get("decoder_frame_ms") for row in lengths
                  if row.get("decoder_frame_ms") is not None]
    speedups = {}
    for row in lengths:
        for n, v in (row.get("steps") or {}).items():
            if "speedup" in v:
                speedups[f"latent_t={row.get('latent_t')}_N={n}"] = round(v["speedup"], 3)

    # Flat VRAM claim: windowed cache should not grow by a full frame when length grows.
    rows_per_frame = data.get("rows_per_frame") or 1008
    cache_spread = (max(cache_rows) - min(cache_rows)) if len(cache_rows) >= 2 else 0
    cache_is_flat = cache_spread < rows_per_frame if cache_rows else None

    # Decoder time should stay roughly constant across length (linear duration, not quadratic).
    decoder_spread_pct = None
    if len(decoder_ms) >= 2 and min(decoder_ms) > 0:
        decoder_spread_pct = round(
            (max(decoder_ms) - min(decoder_ms)) / min(decoder_ms) * 100, 1
        )

    return {
        "path": path,
        "device": data.get("device"),
        "encoder_depth": data.get("encoder_depth"),
        "decoder_source": data.get("decoder_source"),
        "window": data.get("window"),
        "git_sha": data.get("git_sha"),
        "speedups": speedups,
        "cache_rows_by_length": cache_rows,
        "cache_spread_rows": cache_spread,
        "cache_is_flat": cache_is_flat,
        "decoder_frame_ms": [round(x, 2) for x in decoder_ms],
        "decoder_spread_pct": decoder_spread_pct,
        # Flat cache + nearly-flat decoder time = "duration is linear, not quadratic".
        "pass_flat_vram_claim": bool(cache_is_flat),
    }


def maybe_run_tier0_bench(root: str, out_json: str) -> dict[str, Any] | None:
    """Optional live re-measure of Tier 0 primitives (needs CUDA). Writes a fresh JSON."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [skip] tier0_bench — no CUDA device", flush=True)
            return {"name": "tier0_bench", "ok": True, "skipped": True, "reason": "no CUDA"}
    except ImportError:
        print("  [skip] tier0_bench — torch not installed", flush=True)
        return {"name": "tier0_bench", "ok": True, "skipped": True, "reason": "no torch"}

    cmd = [
        sys.executable, "scripts/scd/tier0_bench.py",
        "--json", out_json,
        # Shorter configs keep CI-ish runs under a few minutes; still hits 768p.
        "--configs", "0", "1", "2", "3",
    ]
    # tier0_bench may not support --configs; fall back to defaults if it rejects them.
    # Prefer calling the script as shipped: just --json.
    cmd = [sys.executable, "scripts/scd/tier0_bench.py", "--json", out_json]
    print("  [run] tier0_bench.py (GPU microbench, a few minutes)…", flush=True)
    result = run_cmd(cmd, cwd=root, timeout_s=1800)
    print(f"  [{'ok' if result['ok'] else 'FAIL'}] tier0_bench  ({result['seconds']:.1f}s)",
          flush=True)
    return {
        "name": "tier0_bench",
        "ok": result["ok"],
        "skipped": False,
        "seconds": result["seconds"],
        "exit_code": result["exit_code"],
        "wrote": out_json if result["ok"] else None,
        "stdout_tail": result["stdout_tail"][-1500:],
        "stderr_tail": result["stderr_tail"][-500:] if not result["ok"] else "",
    }


def build_markdown(payload: dict[str, Any]) -> str:
    """Human report: plain language first, jargon only in parentheses."""
    lines = [
        "# SCD architecture validation report",
        "",
        f"- **When:** {payload['timestamp']}",
        f"- **Git:** `{payload.get('git_sha') or 'unknown'}`",
        f"- **Overall:** **{'PASS' if payload['ok'] else 'FAIL'}**",
        "",
        "This report answers: *is the SCD (Separable Causal Diffusion) split still "
        "correct, consistent, and as fast as we measured?* It is not a quality check "
        "for decoded video pixels.",
        "",
        "## 1. Code contracts (unit tests)",
        "",
        "| Suite | Result | Cases | Time |",
        "|-------|--------|-------|------|",
    ]
    for t in payload.get("unit_tests") or []:
        if t.get("skipped"):
            res = "skipped"
        else:
            res = "PASS" if t["ok"] else "FAIL"
        cases = "—"
        if t.get("total") is not None:
            cases = f"{t.get('passed')}/{t.get('total')}"
        lines.append(
            f"| `{t['name']}` | {res} | {cases} | {t.get('seconds', '—')}s |"
        )
    lines += ["", "## 2. Recorded measurements & claims", ""]
    cf = payload.get("check_findings") or {}
    lines.append(f"- **check_findings:** {'PASS' if cf.get('ok') else 'FAIL'} "
                 f"({len(cf.get('errors') or [])} errors, "
                 f"{len(cf.get('warnings') or [])} warnings)")
    for e in (cf.get("errors") or [])[:20]:
        lines.append(f"  - ERROR: {e}")
    for w in (cf.get("warnings") or [])[:12]:
        lines.append(f"  - WARN: {w}")

    lines += ["", "## 3. Headline numbers (from stored benches)", ""]
    t0 = payload.get("tier0")
    if t0:
        lines.append(f"**Tier 0** (weights-free primitives on `{t0.get('device')}`):")
        lines.append("")
        lines.append("| Config | Tokens | Steps (N) | Measured speedup | FLOP-model | Drift |")
        lines.append("|--------|--------|-----------|------------------|------------|-------|")
        for r in t0.get("rows") or []:
            lines.append(
                f"| {r['config']} | {r['tokens']:,} | {r['steps']} | "
                f"**{r['speedup']:.2f}×** | {r['flop_model_speedup']:.2f}× | "
                f"{r['model_error_pct']:.1f}% |"
            )
        kill = "yes" if t0.get("pass_kill_criterion") else "no"
        lines.append("")
        lines.append(f"- Kill criterion (SCD faster at 768p/15s): **{kill}**")
    else:
        lines.append("_No `docs/tier0_results.json` found._")

    t1 = payload.get("tier1")
    lines.append("")
    if t1:
        lines.append(
            f"**Tier 1** (real weights, untrained split — encoder depth "
            f"{t1.get('encoder_depth')}, window {t1.get('window')}):"
        )
        lines.append("")
        if t1.get("speedups"):
            lines.append("| Length / steps | Speedup |")
            lines.append("|----------------|---------|")
            for k, v in t1["speedups"].items():
                lines.append(f"| {k} | **{v:.2f}×** |")
        flat = t1.get("pass_flat_vram_claim")
        lines.append("")
        lines.append(
            f"- Flat KV cache (key-value memory pinned ~constant across length): "
            f"**{'yes' if flat else 'no / unknown'}** "
            f"(spread {t1.get('cache_spread_rows')} rows; "
            f"rows={t1.get('cache_rows_by_length')})"
        )
        if t1.get("decoder_frame_ms"):
            lines.append(
                f"- Decoder time per frame (should stay ~flat): "
                f"{t1['decoder_frame_ms']} ms "
                f"(spread {t1.get('decoder_spread_pct')}%)"
            )
    else:
        lines.append("_No `docs/phase25_tier1.json` found._")

    lines += [
        "",
        "## 4. What this does **not** claim",
        "",
        "- Pixel quality / blur (that is Phase 3 sampling + VAE decode).",
        "- Training loss or LoRA rank adequacy.",
        "- That re-running Tier 0 live is required every time "
        "(use `--bench` when silicon or dims change).",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate SCD architecture and write results under docs/.",
    )
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Where to write JSON + Markdown (default: docs/)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="Skip unit-test suites (claims + stored benches only)")
    ap.add_argument("--bench", action="store_true",
                    help="Also re-run Tier 0 GPU microbench and overwrite docs/tier0_results.json")
    ap.add_argument("--strict", action="store_true",
                    help="Treat check_findings warnings as failures")
    ap.add_argument("--json-only", action="store_true",
                    help="Write JSON but skip the Markdown report")
    args = ap.parse_args()

    root = repo_root()
    os.chdir(root)
    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("SCD architecture validation", flush=True)
    print(f"  repo: {root}", flush=True)
    print(f"  git:  {git_sha(root)}", flush=True)
    print(flush=True)

    unit_tests: list[dict[str, Any]] = []
    if args.skip_tests:
        print("## Unit tests (skipped)", flush=True)
    else:
        print("## 1. Unit tests", flush=True)
        unit_tests = run_unit_tests(root)

    print("\n## 2. Claims & result files", flush=True)
    findings = run_check_findings(root, strict=args.strict)

    bench_row = None
    if args.bench:
        print("\n## 3. Live Tier 0 bench", flush=True)
        tier0_out = os.path.join(out_dir, "tier0_results.json")
        bench_row = maybe_run_tier0_bench(root, tier0_out)

    print("\n## 4. Stored headline numbers", flush=True)
    tier0 = summarize_tier0(os.path.join(root, TIER0_PATH))
    tier1 = summarize_tier1(os.path.join(root, TIER1_PATH))
    if tier0:
        for r in tier0["rows"]:
            print(f"  Tier0  {r['config']:<28}  N={r['steps']:<3}  {r['speedup']:.2f}×",
                  flush=True)
        print(f"  Tier0  kill criterion (768p/15s faster): "
              f"{'yes' if tier0['pass_kill_criterion'] else 'no'}", flush=True)
    else:
        print("  Tier0  (missing)", flush=True)
    if tier1:
        for k, v in (tier1.get("speedups") or {}).items():
            print(f"  Tier1  {k:<28}  {v:.2f}×", flush=True)
        print(f"  Tier1  flat KV cache: "
              f"{'yes' if tier1.get('pass_flat_vram_claim') else 'no'}", flush=True)
    else:
        print("  Tier1  (missing)", flush=True)

    tests_ok = all(t["ok"] for t in unit_tests) if unit_tests else True
    findings_ok = bool(findings.get("ok"))
    bench_ok = True if bench_row is None or bench_row.get("skipped") else bool(bench_row.get("ok"))
    # Stored benches are informational if files are missing; only fail when present and broken.
    tier0_ok = True if tier0 is None else bool(tier0.get("pass_kill_criterion"))
    tier1_ok = True if tier1 is None else bool(tier1.get("pass_flat_vram_claim"))

    overall = tests_ok and findings_ok and bench_ok and tier0_ok and tier1_ok

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(root),
        "ok": overall,
        "unit_tests": unit_tests,
        "check_findings": findings,
        "tier0_bench_live": bench_row,
        "tier0": tier0,
        "tier1": tier1,
        "gates": {
            "unit_tests": tests_ok,
            "check_findings": findings_ok,
            "tier0_live_bench": bench_ok,
            "tier0_kill_criterion": tier0_ok,
            "tier1_flat_vram": tier1_ok,
        },
    }

    json_path = os.path.join(out_dir, RESULTS_JSON)
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(json_path, root)}", flush=True)

    if not args.json_only:
        md_path = os.path.join(out_dir, RESULTS_MD)
        with open(md_path, "w") as fh:
            fh.write(build_markdown(payload))
        print(f"wrote {os.path.relpath(md_path, root)}", flush=True)

    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}", flush=True)
    if not overall:
        failed = [k for k, v in payload["gates"].items() if not v]
        print(f"  failed gates: {', '.join(failed)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
