#!/usr/bin/env bash
# B2: run several seeds with early stopping + valid-MAE checkpoint selection,
# then aggregate the honest single-checkpoint test results (mean +/- std).
#
# Usage (run from the MSAmba/ directory):
#   bash tools/run_multiseed.sh [PROJECT] [DATASET] [DATAPATH]
# Env overrides:
#   SEEDS="1111 2222 3333"  EPOCHS=100  PATIENCE=20  SELECT=MAE
#
# Example:
#   SEEDS="1111 2222 3333 4444 5555" bash tools/run_multiseed.sh MSAmba_ALMT mosi /data/mosi/aligned_50.pkl
set -euo pipefail

PROJECT="${1:-MSAmba_ALMT}"
DATASET="${2:-mosi}"
DATAPATH="${3:-/mosi/aligned_50.pkl}"
SEEDS="${SEEDS:-1111 2222 3333 4444 5555}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
SELECT="${SELECT:-MAE}"

OUT="runs/${PROJECT}_${DATASET}"
mkdir -p "$OUT"

for s in $SEEDS; do
  echo ">>> training seed=$s  project=$PROJECT  dataset=$DATASET"
  python train_msamba.py \
    --project_name "$PROJECT" \
    --datasetName "$DATASET" \
    --dataPath "$DATAPATH" \
    --seed "$s" \
    --n_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --select_metric "$SELECT" \
    2>&1 | tee "$OUT/seed_${s}.log"
done

echo
echo ">>> aggregating results"
python tools/aggregate_seeds.py "$OUT"/seed_*.log
