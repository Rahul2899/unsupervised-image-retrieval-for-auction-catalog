# YOLO fine-tune candidate set v0

This is a review set, not ground truth. Candidates come from the three-catalogue detector audit.

`positive_candidate` means the detector produced a plausible artwork/object crop.
`negative_candidate` marks known page-furniture/text failures from manual inspection.

Review `review.csv` and replace `review_label` with `positive`, `negative`, or `uncertain`.
Do not train until every candidate has a reviewed label. Training from detector predictions alone would teach the existing false positives back to the model.

The JSONL records retain catalogue ID, PDF page, Heidelberg page URL, confidence, and normalized bounding box for provenance.
