#!/bin/bash -l
#SBATCH --job-name=auction-yolo-ft
#SBATCH --time=04:00:00
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT to this checkout}
DATASET=${DATASET:?Set DATASET to the directory containing dataset.yaml}
WEIGHTS=${YOLO_WEIGHTS:?Set YOLO_WEIGHTS to the starting detector weights}
RUNS_DIR=${RUNS_DIR:-$PROJECT_ROOT/runs}

cd "$PROJECT_ROOT"
mkdir -p reports "$RUNS_DIR"
python - "$DATASET" "$WEIGHTS" "$RUNS_DIR" <<'PY'
import sys
from pathlib import Path
from ultralytics import YOLO

data, weights, runs = map(Path, sys.argv[1:])
model = YOLO(str(weights))
model.train(data=str(data / "dataset.yaml"), epochs=30, imgsz=640, batch=-1,
            device=0, workers=8, project=str(runs), name="yolo_artwork_v1",
            exist_ok=False, pretrained=True)
best = runs / "yolo_artwork_v1" / "weights" / "best.pt"
metrics = YOLO(str(best)).val(data=str(data / "dataset.yaml"), split="val", device=0)
print({"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)})
PY
