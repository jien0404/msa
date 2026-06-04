#!/usr/bin/env bash
# B2: run several seeds with early stopping + valid-MAE checkpoint selection,
# then aggregate the honest single-checkpoint test results (mean +/- std).
#
# Usage (run from the MSAmba/ directory):
#   bash tools/run_multiseed.sh [PROJECT] [DATASET] [DATAPATH]
# Env overrides:
#   SEEDS="1111 2222 3333"  EPOCHS=100  PATIENCE=20  SELECT=MAE  BATCH=32  GPU=2  PYTHON=.venv/bin/python
#
# Example:
#   GPU=2 BATCH=32 SEEDS="1111 2222 3333 4444 5555" \
#     bash tools/run_multiseed.sh MSAmba_ALMT mosi /mnt/disk2/chiendx/msa/MOSI/aligned_50.pkl
set -euo pipefail

PROJECT="${1:-MSAmba_ALMT}"
DATASET="${2:-mosi}"
DATAPATH="${3:-/mnt/disk2/chiendx/msa/MOSI/aligned_50.pkl}"
SEEDS="${SEEDS:-1111 2222 3333 4444 5555}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
SELECT="${SELECT:-MAE}"
BATCH="${BATCH:-32}"          # 10GB VRAM -> 32 an toàn; tăng nếu còn dư VRAM
GPU="${GPU:-2}"               # dùng GPU lành (0/1/2); TRÁNH GPU3 lỗi

# Ưu tiên python của venv nếu có, cho phép override bằng PYTHON=...
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON";
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python";
else PY="python"; fi
echo ">>> dùng interpreter: $PY"

OUT="runs/${PROJECT}_${DATASET}"
mkdir -p "$OUT"

for s in $SEEDS; do
  echo ">>> training seed=$s  project=$PROJECT  dataset=$DATASET  gpu=$GPU  batch=$BATCH"
  "$PY" train_msamba.py \
    --project_name "$PROJECT" \
    --datasetName "$DATASET" \
    --dataPath "$DATAPATH" \
    --seed "$s" \
    --n_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --select_metric "$SELECT" \
    --batch_size "$BATCH" \
    --CUDA_VISIBLE_DEVICES "$GPU" \
    2>&1 | tee "$OUT/seed_${s}.log"
done

echo
echo ">>> aggregating results"
"$PY" tools/aggregate_seeds.py "$OUT"/seed_*.log
