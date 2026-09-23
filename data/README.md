# Data contract

The public repository contains only small example images. Large corpus artifacts stay in external storage and are intentionally ignored by Git:

- 16,469 source PDFs
- 1,215,291 rendered pages
- 589,501 final YOLO v2 crops
- Two 589,501-row PCA-512 feature indexes

To run the application, place the crop directory and both feature directories at the paths in `.env.example`, or pass the equivalent command-line flags. Each feature directory must contain `features_l2.pt`, `index.json`, and `pca_model.pkl` under its `660K/` folder.

`FINAL_PIPELINE_MANIFEST.csv` records the final artifact names, counts, and storage boundary. Do not commit PDFs, crops, feature tensors, PCA pickles, or model weights.
