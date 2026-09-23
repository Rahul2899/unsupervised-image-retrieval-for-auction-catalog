#!/bin/bash -l
#SBATCH --job-name=dino-features
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dino-%j.out
#SBATCH --error=logs/dino-%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to this checkout}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the crop directory}
FEATURES_BASE=${FEATURES_BASE:?Set FEATURES_BASE to the output directory}

cd "$PROJECT_ROOT"
mkdir -p logs "$FEATURES_BASE"
python pipeline/extract_features.py \
  --data-dir "$DATA_DIR" \
  --features-base "$FEATURES_BASE" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --device "${DEVICE:-cuda}"
