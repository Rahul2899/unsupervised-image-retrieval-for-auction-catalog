#!/usr/bin/env python3
"""Build a reviewable YOLO candidate set from detector metadata."""
import argparse, json, random
from pathlib import Path

import fitz


KNOWN_NEGATIVES = {
    ("creutzer1930_04_10", 3),
    ("creutzer1930_04_10", 28),
    ("creutzer1930_05_22", 7),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    images = out / "images"
    labels = out / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    pages = {}
    for line in Path(args.metadata).read_text().splitlines():
        row = json.loads(line)
        key = (row["catalogue_id"], int(row["pdf_page_index"]))
        pages.setdefault(key, []).append(row)

    catalogue_ids = sorted({k[0] for k in pages})
    val_catalogue = catalogue_ids[-1]
    manifest = []
    for (catalogue, page_index), rows in sorted(pages.items()):
        pdf = Path(args.pdf_dir) / f"{catalogue}.pdf"
        if not pdf.exists():
            continue
        doc = fitz.open(pdf)
        if page_index >= len(doc):
            doc.close()
            continue
        pix = doc.load_page(page_index).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
        page_num = int(rows[0]["catalogue_page"])
        stem = f"{catalogue}__pdf-{page_num:04d}"
        split = "val" if catalogue == val_catalogue else "train"
        (images / split).mkdir(parents=True, exist_ok=True)
        (labels / split).mkdir(parents=True, exist_ok=True)
        image_path = images / split / f"{stem}.jpg"
        label_path = labels / split / f"{stem}.txt"
        pix.save(str(image_path))
        negative = (catalogue, int(rows[0]["catalogue_page"])) in KNOWN_NEGATIVES
        label_lines = []
        if not negative:
            for row in rows:
                x1, y1, x2, y2 = row["bbox_relative"]
                label_lines.append(f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        manifest.append({
            "image": str(image_path), "label": str(label_path),
            "catalogue_id": catalogue, "pdf_page_index": page_index,
            "catalogue_page": rows[0]["catalogue_page"],
            "source_page_url": rows[0]["source_page_url"],
            "auto_label": "negative_candidate" if negative else "positive_candidate",
            "split": split,
            "detections": len(label_lines),
        })
        doc.close()

    # Balance with deterministic weak-negative pages that had no detector box.
    for catalogue in catalogue_ids:
        pdf = Path(args.pdf_dir) / f"{catalogue}.pdf"
        if not pdf.exists():
            continue
        doc = fitz.open(pdf)
        detected = {page_index for cat, page_index in pages if cat == catalogue}
        positive_count = sum(1 for item in manifest if item["catalogue_id"] == catalogue and item["detections"] > 0)
        candidates = [i for i in range(len(doc)) if i not in detected]
        rng = random.Random(5273 + len(catalogue))
        for page_index in rng.sample(candidates, min(len(candidates), max(1, positive_count))):
            split = "val" if catalogue == val_catalogue else "train"
            (images / split).mkdir(parents=True, exist_ok=True)
            (labels / split).mkdir(parents=True, exist_ok=True)
            stem = f"{catalogue}__pdf-{page_index + 1:04d}__weak-negative"
            image_path = images / split / f"{stem}.jpg"
            label_path = labels / split / f"{stem}.txt"
            pix = doc.load_page(page_index).get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
            pix.save(str(image_path))
            label_path.write_text("")
            manifest.append({
                "image": str(image_path), "label": str(label_path),
                "catalogue_id": catalogue, "pdf_page_index": page_index,
                "catalogue_page": page_index,
                "source_page_url": f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue}/{page_index + 1:04d}",
                "auto_label": "weak_negative", "split": split, "detections": 0,
            })
        doc.close()

    random.Random(5273).shuffle(manifest)
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest) + "\n")
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: artwork\n"
    )
    print(f"pages={len(manifest)} train={sum(x['split']=='train' for x in manifest)} val={sum(x['split']=='val' for x in manifest)}")


if __name__ == "__main__":
    main()
