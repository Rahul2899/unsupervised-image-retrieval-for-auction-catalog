# Unsupervised Image Retrieval for Auction Catalogues

A reproducible visual search pipeline for historical auction catalogues. It turns catalogue pages into artwork crops, extracts self-supervised DINO descriptors, and serves an interactive Dash research interface with source links back to the Heidelberg catalogue.

Live demo: [auctionquery.duckdns.org](https://auctionquery.duckdns.org)

## What is included

- `app.py` — CPU/GPU compatible Dash retrieval application.
- `pipeline/render_pages.py` — resumable PDF-to-page conversion.
- `pipeline/crop_pages.py` — batched YOLO artwork cropping with provenance metadata.
- `pipeline/extract_features.py` — DINO C5 and C4+C5 extraction, PCA-512, and whitening.
- `pipeline/evaluate.py` — retrieval evaluation against a supplied ground-truth JSON file.
- `pipeline/verify_extraction.py` — memory and fusion smoke check.
- `training/` — weak-label dataset builder and the final YOLO training notes.
- `docs/` — research log, provenance alias map, and artifact manifest.

The final corpus contains **16,469 PDFs**, **1,215,291 rendered pages**, and **589,501 indexed crops**. The repository does not contain those large artifacts; see [`data/README.md`](data/README.md).

## Run the app

Requirements: Python 3.11+ and enough disk space for the external crop and feature artifacts.

```bash
git clone https://github.com/Rahul2899/unsupervised-image-retrieval-for-auction-catalog.git
cd <repository-directory>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Place the final artifacts at:

```text
data/crops_yolo_v2/crop_metadata.jsonl
data/features_yolo_v2/C5_mac_c1_pca512_whiten/660K/{features_l2.pt,index.json,pca_model.pkl}
data/features_yolo_v2/C4_C5_max_mac_c1_pca512_whiten/660K/{features_l2.pt,index.json,pca_model.pkl}
```

Start locally:

```bash
python app.py --host <DEPLOYMENT_HOST> --port 8050
```

Open <http://<DEPLOYMENT_HOST>:8050>. The app loads both indexes once at startup, uses **C5 Only** by default, and keeps similarity values internal to ranking. Each result exposes an **Image Source** link and opens the inspector only when selected.

For a container deployment, mount the external data at `/data`:

```bash
docker build -t auction-retrieval .
docker run --rm -p 8050:7865 -v /path/to/final-data:/data auction-retrieval
```

## Rebuild the pipeline

### 1. Download catalogue PDFs

The downloader is resumable and rate-limited. Supply a catalogue JSON file exported from the project data source:

```bash
AUCTION_RETRIEVAL_DATA_DIR="$PWD/data" \\
python pipeline/download_catalogues.py \\
  --catalogue-file /path/to/catalogues.json \\
  --output-dir "$PWD/data/pdfs"
```

### 2. Render pages

```bash
python pipeline/render_pages.py \\
  --pdf-dir data/pdfs \\
  --image-dir data/pages \\
  --workers 8
```

### 3. Crop artwork with YOLO

Set `YOLO_WEIGHTS` to the final fine-tuned weights. The cropper resumes existing work and writes metadata beside the crops.

```bash
YOLO_WEIGHTS=/path/to/best.pt \\
python pipeline/crop_pages.py \\
  --image-dir data/pages \\
  --crop-dir data/crops_yolo_v2
```

### 4. Extract DINO features

The extractor supports both CUDA and CPU. CPU is suitable for a small smoke run; the full corpus needs a GPU and substantial temporary storage.

```bash
python pipeline/extract_features.py \\
  --data-dir data/crops_yolo_v2 \\
  --features-base data/features_yolo_v2 \\
  --batch-size 32 \\
  --num-workers 8 \\
  --device cuda
```

Use `--overwrite` only when intentionally rebuilding an index.

### 5. Evaluate retrieval

Evaluation requires a ground-truth JSON file. It is deliberately not assumed to exist in a fresh clone.

```bash
python pipeline/evaluate.py \\
  --query-dir data/query_images \\
  --gt-json data/ground_truth.json \\
  --features-base data/features_yolo_v2 \\
  --results-dir evaluation_results \\
  --overwrite
python pipeline/evaluate.py --summary --results-dir evaluation_results
```

### Slurm

`jobs/extract_features.sbatch` is a portable template. Set `PROJECT_ROOT`, `DATA_DIR`, and `FEATURES_BASE` before submitting it; do not copy HPC paths into the Python source.

## Reproducibility and provenance

- Crop metadata is the source of catalogue, page, and image links.
- The app derives stable Heidelberg links for `diglit_*` identifiers and preserves recorded provenance when available.
- Feature indexes are generated from the same DINO preprocessing used for uploaded queries.
- Search uses exact inner-product ranking over normalized vectors. The displayed rank is the user-facing result; raw similarity is retained only for internal ordering.
- `docs/RESEARCH_LOG.md` records experiments and discarded candidates. Production data and experimental outputs stay separate.

## Project layout

```text
app.py                  Dash application
assets/                 UI styles and keyboard controls
data/                   examples plus artifact contract
pipeline/               download, render, crop, extract, evaluate
training/               YOLO dataset/training helpers
docs/                   research and provenance records
jobs/                   Slurm templates
```

## License

No license is declared yet. Add one before accepting external contributions or redistributing the corpus.

The thesis PDF and its source archive are retained under [`docs/thesis/`](docs/thesis/) for reference; they are separate from the runnable pipeline.
