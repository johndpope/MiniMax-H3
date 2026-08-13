#!/usr/bin/env bash
# Distill race: velocity (best bet) then anchors. Teacher cache required.
# Defaults: rank 8, decoder-only LoRA, 50 min each, offline teacher_cache.pt
set -euo pipefail
cd "$(dirname "$0")"
CKPT="${CKPT:-/media/2TB/Fizgig/models/MiniMax-H3-FL2VA/FL2VA/transformer}"
CACHE="${CACHE:-runs/teacher_cache.pt}"
MINUTES="${MINUTES:-50}"
RANK="${RANK:-8}"
LOG=runs/race_master.log

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[$(date -Is)] waiting for $CACHE" | tee -a "$LOG"
while [[ ! -f "$CACHE" ]]; do sleep 15; done
sleep 2
echo "[$(date -Is)] cache ready ($(du -h "$CACHE" | awk '{print $1}'))" | tee -a "$LOG"

run_arm() {
  local arm=$1
  echo "[$(date -Is)] START arm=$arm minutes=$MINUTES rank=$RANK decoder-only" | tee -a "$LOG"
  python3 -u phase3_race.py \
    --arm "$arm" \
    --cache "$CACHE" \
    --checkpoint "$CKPT" \
    --rank "$RANK" \
    --decoder-only \
    --out "runs/race_${arm}" \
    --minutes "$MINUTES" \
    --wandb h3-scd-race \
    --fizgig-src /media/2TB/Fizgig/src \
    2>&1 | tee "runs/race_${arm}.log"
  echo "[$(date -Is)] DONE arm=$arm" | tee -a "$LOG"
  if [[ -f "runs/race_${arm}/summary.json" ]]; then
    echo "[$(date -Is)] summary $(cat "runs/race_${arm}/summary.json")" | tee -a "$LOG"
  fi
}

# Velocity first (pixel bet), then anchors
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
if len(rows)>=1:
    best=sorted(rows, key=lambda d: (d.get("corr_ctx") or -1, -(d.get("mse") or 9)))[-1]
    print("WINNER", best["arm"], "corr_ctx", best.get("corr_ctx"), "mse", best.get("mse"))
    Path("runs/race_winner.json").write_text(json.dumps(best, indent=2)+"\n")
PY
