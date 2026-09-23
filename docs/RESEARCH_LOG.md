# Auction Retrieval Research Log

This log records reproducible pipeline decisions and experiment outcomes. Paths are represented with portable placeholders; no machine-specific paths are stored.

## Current research question

Can the existing supervised YOLO detector isolate artwork/object regions from historical auction-catalogue pages while preserving page-level provenance links?

## Pipeline contract

`catalogues.json` → PDF download → page rendering → YOLO detection/cropping → crop metadata/provenance → feature extraction → nearest-neighbour retrieval.

PDF and page links are retained in crop metadata. Visual retrieval and future text/page retrieval remain separate experiments.

## Experiments

| Experiment | Data | Device | Parameters | Result | Decision |
|---|---:|---|---|---|---|
| `page_triage_v0/run_data_v2` | 1 catalogue, 58 pages | CPU | YOLO confidence 0.25 | 3 crops; all inspected outputs were library-logo false positives | Preserve as failure evidence |
| `page_triage_v0/run_data_v3` | 3 catalogues, 201 pages | CPU | YOLO confidence 0.25 | 60 crops, 0 download failures | Baseline for comparison |
| `page_triage_v0/conf_025` | 3 catalogues, 201 pages | CPU | confidence 0.25 | 60 crops | Baseline |
| `page_triage_v0/conf_050` | 3 catalogues, 201 pages | CPU | confidence 0.50 | 46 crops; some false positives remain | Not sufficient alone |
| `page_triage_v0/conf_075` | 3 catalogues, 201 pages | CPU | confidence 0.75 | 40 crops; removes more detections, but recall loss is too high | Not selected |
| `page_triage_v0/telemetry_v0` | 1 catalogue, 58 pages | CPU | confidence 0.25 | 3 crops with confidence and normalized bounding boxes | Use for deterministic filtering |

## Known failures and corrections

- Detection workers previously received the stop signal before download workers finished producing pages. Queue shutdown order was corrected: join download queue first, then stop the detector.
- Heidelberg IIIF manifest requests can time out on HPC. The pipeline falls back to deterministic catalogue/page URLs.
- The existing detector confuses footer/library logos and handwritten catalogue text with artwork.
- `processing_stats.json` has bookkeeping defects: catalogue attempt/completion counters and update timestamps are not always final. This does not change crop output, but must be fixed before production reporting.

## Current decision

Do not fine-tune from detector predictions alone. Build a candidate review set first, add bounding-box telemetry, apply deterministic filters, and require human review before training labels become ground truth.

## Pending work

1. Review `fine_tune_v0/review.csv`.
2. Test footer-region, minimum-size, and duplicate-box filters.
3. Freeze the downloaded PDF count.
4. Run cropping into a staging directory.
5. Extract embeddings only after crop filtering.
6. Correct statistics reporting and write final experiment cards.

## Downloader snapshot (2026-09-14)

The detached CPU downloader was active while the detector experiments ran. Latest observed snapshot: 11,640 attempted, 11,483 successful, 86 failed, 71 existing, approximately 302,876 MB downloaded. These values are a live snapshot and must be refreshed when the downloader finishes.

## Provisional fine-tune submission (2026-09-14)

`fine_tune_v0/dataset` was built from `telemetry_v1` and three PDFs. It contains 38 rendered pages (14 train, 24 validation), 55 auto-generated YOLO boxes, 17 positive candidates, 3 known negative candidates (footer/logo cases), and 18 weak negatives sampled from pages without detector boxes. Weak negatives may contain missed artwork; this dataset is therefore provisional and is not production ground truth.

GPU training job `1812477` was submitted on the `work` partition with one RTX 2080 Ti, 8 CPUs, 4-hour limit, 30 epochs, 640px images, auto batch sizing, and eight dataloader workers. Production weights remain unchanged until validation results are reviewed.

## Status refresh (2026-09-18)

- Live Slurm checks on Woody and TinyX found no active jobs.
- PDF conversion job `12499885` timed out at its 24-hour limit. Slurm requested 15,500 MB, recorded 15,077,220 KB maximum RSS, and reported 33 OOM kills. Its log ended at 13,654/15,705 catalogues: 3,813 successes and 9,841 failures.
- Conversion checkpoint contains 4,572 completed catalogues and 370,513 pages. The page directory contains 372,487 JPEG files; the 1,974-file difference is not reconciled. The PDF directory contains 16,470 files while the converter reported 16,469 inputs.
- The converter rendered pages at 200 DPI, then resized each raster to 640px. Failure reporting collapsed conversion exceptions to `Unknown error`. Direct rendering at 640px is a plausible fix for memory peaks; it has not been tested across the corpus.
- Local candidate now renders directly at target size and returns worker exceptions to the log. Its self-check passed on Woody's Python environment, including one page from previously failed PDF `diglit_21734` (440×640, 42,638 bytes). Candidate remains local; production converter was not replaced and no job was submitted.
- GPU job `1813899` failed four seconds after starting (exit code 53); no training metrics or weights were produced. `review.csv` has 60 candidates and zero reviewed labels. At this snapshot, detector target still needed clarification; it has since been confirmed as YOLO artwork detection.
- `data/features` and `data/rn50_features` are empty. No embeddings have been extracted from this expanded corpus.

## YOLO retry preparation (2026-09-18)

- The failed job's stdout path was inside `<HPC_DATA_ROOT>`, which is at its 400,000-file hard limit. The environment and Ultralytics import work on TinyX; the missing stdout plus full file quota make Slurm output setup the likely cause of exit `0:53`.
- Retry candidate sends logs, Ultralytics config/cache, dataset copy, and run artifacts to `<HPC_WORK_ROOT>/auction-retrieval-yolo-v0`; production source data and baseline weights stay in `<HPC_DATA_ROOT>`.
- Dataset has 14 train and 24 validation images, with 5 and 12 nonempty label files. Labels remain pseudo-labeled and unreviewed; any resulting metrics are provisional, not ground truth.
- Candidate uses class `artwork`; it does not detect arbitrary yellow-colored objects.
- Staged dataset copy matches source files apart from its YAML root path. TinyX loaded the dataset and baseline checkpoint with Ultralytics 8.3.204; all 55 boxes are structurally valid, with a maximum 5e-9 normalized edge-rounding overrun.
- `bash -n`, Python heredoc compilation, and `sbatch --test-only` passed. Test-only estimated start: 2026-09-19 03:55:39 on `tg069`; the check itself did not submit a job.
- Actual training job `1815796` was submitted on 2026-09-18 and is currently `PENDING (Resources)` on `work`; logs and run artifacts target `<HPC_WORK_ROOT>`.
- Storage plan authorized by user: Woody storage may be used if needed. After PDF-to-image conversion finishes and crop outputs are validated, archive the PDFs and verify the tar before the later batch cleanup removes source PDFs, PDF-to-image files, and PDF-to-pages files.
