#!/usr/bin/env bash
# Two-way SCD distill race (~1 h each) after teacher cache is ready.
# Prefer velocity (trajectory) arm first — our bet for better pixels.
set -euo pipefail
cd "$(dirname "$0")"
CKPT="${CKPT:-/media/2TB/Fizgig/models/MiniMax-H3-FL2VA/FL2VA/transformer}"
CACHE="${CACHE:-runs/teacher_cache.pt}"
INIT="${INIT:-runs/scd_v2/scd_lora_002500.safetensors}"
MINUTES="${MINUTES:-50}"
LOG=runs/race_master.log

echo "[$(date -Is)] waiting for $CACHE" | tee -a "$LOG"
while [[ ! -f "$CACHE" ]]; do sleep 15; done
# ensure precompute finished writing
sleep 5
echo "[$(date -Is)] cache ready ($(du -h "$CACHE" | awk '{print $1}'))" | tee -a "$LOG"

run_arm() {
  local arm=$1
  echo "[$(date -Is)] START arm=$arm minutes=$MINUTES" | tee -a "$LOG"
  # Rank 16: rank 32 climbed into OOM by ~step 30 on 24 GB even with one-frame steps.
  # Skip init if rank mismatch (v2 is rank 32).
  local init_args=()
  if [[ -f "$INIT" ]]; then
    init_args=(--init-lora "$INIT")
  fi
  # Only warm-start when ranks match (v2 is 32; race uses 16 for memory).
  python3 -u phase3_race.py \
    --arm "$arm" \
    --cache "$CACHE" \
    --checkpoint "$CKPT" \
    --rank 16 \
    --out "runs/race_${arm}" \
    --minutes "$MINUTES" \
    --wandb h3-scd-race \
    --fizgig-src /media/2TB/Fizgig/src \
    2>&1 | tee "runs/race_${arm}.log"
  echo "[$(date -Is)] DONE arm=$arm" | tee -a "$LOG"
  if [[ -f "runs/race_${arm}/summary.json" ]]; then
    echo "[$(date -Is)] summary $(cat runs/race_${arm}/summary.json)" | tee -a "$LOG"
  fi
}

# Velocity first (our preferred bet), then anchors
run_arm velocity
run_arm anchors

echo "[$(date -Is)] RACE COMPLETE" | tee -a "$LOG"
python3 - <<'PY' | tee -a runs/race_master.log
import json
from pathlib import Path
rows=[]
for arm in ("velocity","anchors"):
    p=Path(f"runs/race_{arm}/summary.json")
    if p.exists():
        d=json.loads(p.read_text()); rows.append(d); print(arm, d)
if len(rows)==2:
    # Prefer higher corr_ctx; break ties with lower mse. Pixels still decide offline.
    best=sorted(rows, key=lambda d: (d.get("corr_ctx") or -1, -(d.get("mse") or 9)))[-1]
    print("WINNER", best["arm"], "corr_ctx", best.get("corr_ctx"), "mse", best.get("mse"))
    Path("runs/race_winner.json").write_text(json.dumps(best, indent=2)+"\n")
PY
