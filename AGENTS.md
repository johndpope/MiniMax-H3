# AGENTS.md

HF mirror of `MiniMaxAI/MiniMax-H3` plus **local** research. Upstream model-card rules: [`CLAUDE.md`](CLAUDE.md).

## Local research tracks (not upstream)

| Track | Where | Doc |
|-------|--------|-----|
| **VFM 1-NFE flywheel** | sibling checkout `MiniMax-H3-vfm` | [`MiniMax-H3-vfm/scripts/vfm/README.md`](../MiniMax-H3-vfm/scripts/vfm/README.md) · [`MiniMax-H3-vfm/AGENTS.md`](../MiniMax-H3-vfm/AGENTS.md) |
| **SCD port** | this repo `docs/` + `scripts/scd/` | `docs/MINIMAX_H3_SCD_PORT_DESIGN.md` |
| **Level-of-Token (LoT)** | this repo `scripts/lot/` + Fizgig `immiscible-h3-noise` | [`scripts/lot/CHEATSHEET.md`](scripts/lot/CHEATSHEET.md) · `docs/LOT_H3_CLAUDE_HANDOFF.md` |

Do not push `docs/`, `scripts/scd/`, `.grok/`, or `.agents/skills/balls-flywheel/` upstream.

If the task is VFM / 1-NFE / adapter / LoRA mix / i2v / flywheel graph: **work in MiniMax-H3-vfm**. This tree has the clips (`scripts/scd/clips/`) and the balls-flywheel skill; the trainer is not here.

## VFM pointers (so you find recreations)

- Recreate every stage: `MiniMax-H3-vfm/scripts/vfm/README.md`
- Graph: `MiniMax-H3-vfm/scripts/vfm/runs/flywheel/graph.jsonl`
- Launch: `MiniMax-H3-vfm/scripts/vfm/launch_next.sh`
- Live portrait: `MiniMax-H3-vfm/scripts/vfm/runs/flywheel/i2v/live_photoportrait.png`
- Skill: `.agents/skills/balls-flywheel/SKILL.md`
- Workflows: `.grok/workflows/balls-flywheel.rhai`, `babysit-vfm.rhai`

## LoT pointers

- Start here: `scripts/lot/CHEATSHEET.md` (commands, flags, measured numbers, gotchas)
- Train from mp4s: `scripts/lot/README.md`, section "Training H3 for LoT from your own mp4s"
- Current state and rules: `docs/LOT_H3_CLAUDE_HANDOFF.md`; design: `docs/LOT_H3_COMPUTE_PLAN.md`
- Results log: GitHub issue 1 on `johndpope/MiniMax-H3` (`gh ... -R johndpope/MiniMax-H3`)
- Training is gated on `LOT_H3_TRAIN=1`; `train_h3.py --check` is the safe dry run

## Standing VFM rules (duplicate of the VFM AGENTS.md)

No Qwen-32B, no NVFP4. Fizgig pruned INT8 + NF4 + precached clips. Do not `empty_cache` on the display GPU. Do not relaunch `pin` / `v4` / `zfit` / `invert` / `nfe4` / `lora_dit` / `lora_mix`. Keep `lora_step00750.pt`. Visual PASS is a face/street in the middle preview panel.

## This mirror

`FL2VA/` and `Ref2VA/` are byte-identical except `model_index.json` — patch both. Do not mix original vs modular class names. Four `README*.md` files are translations of the same upstream document; local research banners are the exception and must stay in all four.
