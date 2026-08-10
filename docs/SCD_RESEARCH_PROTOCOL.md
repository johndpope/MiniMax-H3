# SCD research protocol

How measurements get made and how findings get into `MINIMAX_H3_SCD_PORT_DESIGN.md`. Written after
Phase 0 produced two confident, wrong headlines, both of which survived review and reached the
design doc before anyone caught them. The rules below are each a direct response to one of those.

This covers the local SCD port research only (`docs/`, `scripts/scd/`). None of it is upstream
mirror content and none of it should be pushed to `MiniMax-AI/MiniMax-H3`.

## What went wrong, since the rules only make sense against it

**1. The metric was measuring the wrong thing.** The first probe scored every axis as plain cosine
similarity on the residual stream. H3's residual stream is 91–99.7% a single shared vector
(`‖mean row‖ / mean‖row‖`, against 0.024 for uncorrelated rows), so cos-sim between two runs
reports ~1.0 even when the informative parts are unrelated. This manufactured the headline that
σ-invariance *rises* with depth — backwards from what SCD needs, which should have been the tell.

**2. The measurement sat off-distribution.** σ-invariance used σ=0 as its reference and the
leave-one-out probe used a flat 0.1–0.9 grid. H3 draws training σ as `shift_sigma(u, 12)` with *u*
uniform: median ~0.92, ~3% of steps below 0.3. At σ=0.1 the model's output is *anticorrelated* with
its own flow-matching target (cos −0.44) and the baseline loss is worse than predicting zero. The
resulting "video rows carry no σ-invariant content" claim reversed on re-measurement — 0.061 mean
over blocks 1–25 became 0.779.

A third, smaller one: every early run used `latent_t=8`, off the DiT's `5n+2` grid. `model.forward()`
raises on it; the block-by-block probe path does not, so it ran for hours unnoticed.

The common factor is that **none of these were visible in the output**. Each produced a plausible
curve. You had to look at the configuration, and nothing forced anyone to.

## Rules

### 1. Every axis needs a control that could falsify it

Not a sanity check — a specific experiment that comes out differently if the measurement is
broken. `phase0_validate.py` carries three, and each maps to a way the probe could have been lying:

| Control | Kills |
|---|---|
| Own-frame-only mask (brutal ablation) | A causal mask that never reaches the forward path. If a savage mask also shows ~no drift, the plumbing is dead. |
| Attention mass decomposition | An unexplained result. Masking the future can only matter in proportion to the mass actually there. |
| Common-mode ratio + centered cos + relative L2 | Cos-sim inflation from a shared residual component. |

A new axis without a control does not go in the doc, however good the number looks.

### 2. Never measure off H3's own σ density

Use `shift_sigma(u, VIDEO_SIGMA_SHIFT)` with *u* as the parameter you sweep, so uniform steps in
*u* are uniform where training was. Raw σ below ~0.5 is off-distribution and the model is
anticorrelated with the target there. `--raw-sigmas` / `--sigmas` exist as escape hatches for
deliberately probing that region; they are not for normal use, and `check_findings.py` warns on any
recorded run that used them below 0.5.

Related: `latent_t` must sit on the `5n+2` grid. Both probes now call `pixel_frames_for_latent()`
as an assertion before the model loads.

### 3. A finding is n=1 until it is not

Every number currently in §7 comes from **one clip, one prompt, one resolution**. That is enough to
size an adapter and place scaffolding. It is not enough to hardcode a block index into Phase 1.

Before a split point or a layer set is frozen in code, run `run_phase0.py` over ≥10 clips and check
the ledger summary reports `STABLE`. If the knee or the load-bearing set moves with content, the
split is clip-dependent and the design has to accommodate that rather than pick a winner.

### 4. Results are recorded with their configuration, in an append-only ledger

`docs/phase0_ledger.jsonl` — one row per probe run, carrying the full config (σ values, `latent_t`,
quantization, checkpoint, git SHA) beside the results. Never edit or reorder it; append only.
`run_phase0.py --summarize-only` groups by configuration rather than pooling, so an
off-distribution run can never be averaged in with a good one.

Standalone `docs/phase0_*.json` files stay as the artifacts individual scripts write. When one is
replaced, mark the old file with `superseded_by` and `superseded_reason` instead of deleting it —
the record of what a bad configuration produced is worth more than the disk it costs.

### 5. Numbers quoted in the design doc are registered, not just typed

Prose does not update itself when a probe is re-run. Any number §7 leans on goes into
`docs/phase0_claims.json` as `{id, file, key, index, expected, tol}`, and `check_findings.py`
verifies each against the file it cites. CI runs this on every push touching `docs/` or
`scripts/scd/`.

If a re-run legitimately changes a number, update the claim in the same commit as the prose. A
failing claim check means the doc and the data disagree — it does not mean the tolerance is wrong.

Only the largest configuration group reaches `phase0_summary.json`, so a number from any other
group has no flat file to be cited from. Such a claim names a ledger row instead —
`{file: docs/phase0_ledger.jsonl, row: {clip: NAME}, key: seq_len}` — and the check fails unless the
selector matches exactly one row. Prefer this to widening the summary: the ledger is the record, and
copying numbers out of it is the drift this rule exists to prevent.

### 6. Superseded code keeps a header, and keeps working

`phase0_probe.py` opens with a `SUPERSEDED BY` block naming what replaced it and why. Deleting it
would erase the evidence for why the metric changed. Do not silently fix a superseded script's
numbers either — its output is the record of the mistake.

### 7. A free parameter inside the estimator is swept, or it is not a result

`knee(curve, frac=0.85)` reports "first block below `frac` of the plateau". `frac` was picked once,
never swept, and quoted for months as if it were not there. Sweeping it over 0.70–0.95 moves the
text knee 2 blocks and the video knee 5, monotonically — so the video knee was a restatement of the
threshold, and §7 had been reading it as independent corroboration of the text knee.

This is a different failure from rules 1 and 2. There the *measurement* was off; here the
measurement was fine and the *summary statistic* had a knob in it. The tell is the same though —
nothing in the output shows it, you have to go looking.

So: any estimator with a tunable constant reports its own sensitivity beside its value.
`run_phase0.py` emits `knee_*_by_frac` and `knee_*_frac_spread` into the summary and prints a NOTE
when the spread exceeds `FRAC_SPREAD_OK`. A statistic whose answer tracks its own knob is describing
the knob, and does not get quoted as a finding.

### 8. A contrast measurement is quoted with its contrast, and checked for having any

σ-invariance is a difference between two runs, so every number it produces is a function of how far
apart those two runs are. Sweeping the pair in *u* moves the text knee from 30 (Δu 0.8) to 37
(Δu 0.2) — a bigger swing than content and geometry combined.

The trap is that this reads like a finding: narrower separation, later knee, more of the stack
shareable. It is not. Over the same range the curve's total drop across all 50 blocks falls from
0.78 to 0.32 and the knee's `frac` sensitivity grows from 2 blocks to 13 — the narrow pairs have not
moved the knee, they have stopped resolving one, and a threshold applied to a curve with no plateau
returns the threshold. **Rule 7's sensitivity number is what tells the two apart**, which is the
argument for computing it always rather than when suspicious.

So: quote the contrast beside the result (`Δu`, not raw σ — raw σ is not uniform on H3's density),
and before reading any knee, check that its frac spread is small. A knee from a curve with no
plateau is not a late knee. Where a split has to be conservative, the widest on-distribution pair is
both the most demanding question and the best-resolved one; prefer it.

### 9. Distinguish "the frozen base does not do this" from "this cannot work"

SCD full fine-tunes for 55K steps to *teach* the model to rewire. Zero-shot measurements on frozen
H3 are budget estimates — how far the base sits from the target wiring and which half has to move —
not go/no-go gates. Phase 0's flat cross-frame attention is what the paper's own account predicts
for a bidirectional model that has had no causal training; the meaningful version of that
measurement is the same probe re-run *after* Phase 3.

## Running things

```bash
# Both probes over a set of clips, one model load, appended to the ledger
python3 scripts/scd/run_phase0.py \
    --checkpoint /path/to/FL2VA/transformer \
    --clips 'scripts/scd/clips/*_latents.safetensors'

# Re-read the ledger and report across-clip stability (no GPU needed)
python3 scripts/scd/run_phase0.py --summarize-only

# Validate every recorded result and every doc claim (no GPU, no weights — this is what CI runs)
python3 scripts/scd/check_findings.py
```

Clips follow the naming convention `NAME_latents.safetensors` / `NAME_te.safetensors`; use
`encode_clip.py` and `encode_text.py` to produce them. `.safetensors` is gitignored, so clips are
local — the ledger records the clip *name*, which is what makes a run reproducible on the same set.

`scripts/scd/clips/MANIFEST.tsv` maps each name to its source video and encoder settings, and
`scripts/scd/clips/NAME.txt` is the prompt that produced `NAME_te.safetensors`. Both are checked in,
so the set can be rebuilt from the corpus even though the tensors cannot be. Two things about the
prompts are load-bearing rather than cosmetic: they are Context-IR (the three-field form H3 was
trained on, not free-form captions — a caption is off-distribution in prompt space the same way
σ=0.1 is off-distribution in noise space), and they describe only the first 29 pixel frames, which
is the entire window the probe sees.

Individual scripts still run standalone and write their own JSON; use them for one-off
investigation, and the runner when the result is meant to count.
