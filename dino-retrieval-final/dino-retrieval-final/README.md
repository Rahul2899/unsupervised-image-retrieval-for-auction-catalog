# DINO Image Retrieval

Production-ready pipeline for:
- extracting DINO features from your image dataset (`extractfull.py`)
- running a Gradio search app (`newapp.py`)
- evaluating retrieval metrics (`eval.py`)

## Files
- `extractfull.py`: builds feature files (`features_l2.pt`, `index.json`, `pca_model.pkl`)
- `newapp.py`: web app for query image retrieval
- `eval.py`: computes Top-1/5/10 and mAP metrics from query set + ground truth
- `jobfull.sh`: optional SLURM launcher

## Install
```bash
pip install torch torchvision numpy opencv-python pillow tqdm psutil scikit-learn gradio faiss-cpu
```

## 1) Extract Features
`extractfull.py` requires CUDA.

```bash
python extractfull.py \
  --data-dir /path/to/images \
  --features-base /path/to/features_output \
  --batch-size 32 \
  --num-workers 8
```

Optional overwrite:
```bash
python extractfull.py --overwrite --data-dir /path/to/images --features-base /path/to/features_output
```

## 2) Run Retrieval App
```bash
python newapp.py \
  --db-image-dir /path/to/images \
  --features-base /path/to/features_output \
  --host 0.0.0.0 \
  --port 7865
```

Open: `http://<DEPLOYMENT_HOST>:7865`

## 3) Run Evaluation
Set `QUERY_DIR` and `GT_JSON` in `eval.py`, then run:

```bash
python eval.py --feature_dirs C4_C5_max_mac_c1_pca512_whiten C5_mac_c1_pca512_whiten
```

Show summary:
```bash
python eval.py --summary
```

## Environment Variables (optional)
- `DINO_DATA_DIR`
- `DINO_DB_IMAGE_DIR`
- `DINO_FEATURES_BASE`
- `DINO_BATCH_SIZE`
- `DINO_NUM_WORKERS`
- `DINO_APP_HOST`
- `DINO_APP_PORT`

## Quick Demo
```bash
# 1) Set your paths
export DINO_DATA_DIR=/path/to/images
export DINO_FEATURES_BASE=/path/to/features_output

# 2) Extract features (CUDA required)
python extractfull.py --data-dir "$DINO_DATA_DIR" --features-base "$DINO_FEATURES_BASE"

# 3) Launch app
python newapp.py --db-image-dir "$DINO_DATA_DIR" --features-base "$DINO_FEATURES_BASE" --port 7865
```
