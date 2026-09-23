#detector.py
from ultralytics import YOLO
import os, gc, cv2, torch, json
import numpy as np
from pathlib import Path
from PIL import Image

class YoloDetector:
    def __init__(self, cache_dir=None, device=None, crop_dir=None, imgsz=640, max_det=10, verbose=False, conf=0.25):
        """
        GPU-aware YOLOv8 detector with immediate crop persistence.
        - Uses FP16 on CUDA.
        - Works with ndarray/PIL and ensures BGR for OpenCV.
        """
        if not cache_dir:
            cache_dir = os.getcwd()
        weights_path = self._get_yolo_weights(cache_dir)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.device_str = "cpu" if self.device.type == "cpu" else "cuda:0"
        self.use_half = (self.device.type == "cuda")

        self.model = YOLO(weights_path)
        try:
            self.model.to(self.device_str)
        except Exception:
            self.model.to(str(self.device))

        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.conf = float(conf)
        self.verbose = bool(verbose)

        self.crop_dir = crop_dir
        if self.crop_dir:
            os.makedirs(self.crop_dir, exist_ok=True)

    def _get_yolo_weights(self, cache_dir):
        print(f'[YoloDetector] Looking for cached weights in {cache_dir}')
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = Path(cache_dir) / 'detection_weights.pt'
        if weights_path.is_file():
            print('[YoloDetector] Weights found, loading from disk.')
            return weights_path
        print('[YoloDetector] No weights found. Downloading from Zenodo...')
        import requests
        url = 'https://zenodo.org/records/15322180/files/detection_weights.pt?download=1'
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(weights_path, 'wb') as f:
            f.write(resp.content)
        print('[YoloDetector] Download complete.')
        return weights_path

    def _page_has_existing_crops(self, page):
        if not self.crop_dir: return False
        catalogue = page.get('source_catalogue')
        page_num = page.get('catalogue_page')
        if not catalogue or page_num is None: return False
        pattern = f"{catalogue}__pdf-{int(page_num):04d}__crop-*.jpg"
        return len(list(Path(self.crop_dir).glob(pattern))) > 0

    def _load_existing_crops(self, page):
        if not self.crop_dir: return []
        catalogue = page.get('source_catalogue')
        page_num = page.get('catalogue_page')
        if not catalogue or page_num is None: return []
        crops, pattern = [], f"{catalogue}__pdf-{int(page_num):04d}__crop-*.jpg"
        for p in sorted(Path(self.crop_dir).glob(pattern)):
            try:
                img = cv2.imread(str(p))
                if img is not None:
                    crops.append(img)
            except Exception as e:
                print(f"[YoloDetector] Error loading crop {p}: {e}")
        return crops

    def _save_crops_to_disk(self, page, crops, catalogue, page_num, detections=None):
        if not self.crop_dir or not crops: return 0
        saved = 0
        metadata_path = Path(self.crop_dir) / 'crop_metadata.jsonl'
        for idx, crop in enumerate(crops):
            try:
                outp = Path(self.crop_dir) / f"{catalogue}__pdf-{int(page_num):04d}__crop-{idx:02d}.jpg"
                cv2.imwrite(str(outp), crop)
                metadata = {
                    'crop_path': str(outp),
                    'catalogue_id': page.get('catalogue_id', catalogue),
                    'catalogue_page': page_num,
                    'pdf_page_index': page.get('pdf_page_index', page_num),
                    'canvas_number': page.get('canvas_number'),
                    'printed_page_label': page.get('printed_page_label'),
                    'source_catalogue_url': page.get('source_catalogue_url'),
                    'source_page_url': page.get('source_page_url'),
                    'source_image_url': page.get('source_image_url'),
                    'crop_index': idx,
                    'legacy_stem': f"{catalogue}_page{page_num}_{idx}",
                }
                if detections and idx < len(detections):
                    metadata.update(detections[idx])
                with metadata_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                saved += 1
            except Exception as e:
                print(f"[YoloDetector] Error saving crop {outp}: {e}")
        if saved:
            print(f"[YoloDetector] Saved {saved} crops to disk for page {page_num}")
        return saved

    def _validate_image(self, img):
        """Ensure ndarray uint8 BGR (OpenCV) with 3 channels."""
        if img is None:
            raise ValueError("Image is None")
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            img = np.asarray(img, dtype=np.uint8)
        if img.size == 0:
            raise ValueError("Empty image array")
        if img.dtype != np.uint8:
            if img.dtype in (np.float32, np.float64) and getattr(img, "max", lambda:2)() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Final image must be 3-channel, got shape: {img.shape}")
        img = np.ascontiguousarray(img[:, :, ::-1])  # RGB->BGR
        return img

    def _get_crop_records(self, yolo_result):
        records = []
        try:
            img = self._validate_image(yolo_result.orig_img)
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                return records
            h, w = img.shape[:2]
            confidences = yolo_result.boxes.conf.detach().cpu().tolist()
            for i, xyxy in enumerate(yolo_result.boxes.xyxy):
                try:
                    x1, y1, x2, y2 = xyxy.int().cpu().numpy().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    crop = img[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        records.append((crop, {
                            'detection_confidence': float(confidences[i]),
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'bbox_relative': [x1 / w, y1 / h, x2 / w, y2 / h],
                            'crop_width': x2 - x1,
                            'crop_height': y2 - y1,
                        }))
                except Exception as e:
                    print(f"[YoloDetector] Error processing bbox {i}: {e}")
        except Exception as e:
            print(f"[YoloDetector] _get_crop_records error: {e}")
        return records

    def _get_crop(self, yolo_result):
        crops = []
        crops.extend(crop for crop, _ in self._get_crop_records(yolo_result))
        return crops

    def detect_and_crop(self, pdf_pages, chunksize=50, skip_existing=True):
        for page_idx, page in enumerate(pdf_pages):
            try:
                catalogue_page = page.get('catalogue_page', '?')
                catalogue = page.get('source_catalogue')

                if skip_existing and self._page_has_existing_crops(page):
                    existing = self._load_existing_crops(page)
                    page['crops'] = existing
                    page['page_image'] = None
                    print(f"[YoloDetector] Page {catalogue_page}: loaded {len(existing)} existing crops")
                    continue

                img = page.get('page_image')
                if img is None:
                    page['crops'] = []
                    continue

                img = self._validate_image(img).copy()

                results = None
                try:
                    from tempfile import NamedTemporaryFile
                    from PIL import Image as PILImage
                    with NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        pil_img.save(tmp.name, 'JPEG', quality=90)
                        with torch.inference_mode():
                            results = self.model.predict(
                                tmp.name,
                                device=self.device_str,
                                max_det=self.max_det,
                                conf=self.conf,
                                verbose=self.verbose,
                                save=False, show=False, augment=False,
                                half=self.use_half, imgsz=self.imgsz
                            )
                    try: os.unlink(tmp.name)
                    except Exception: pass
                except Exception as e:
                    print(f"[YoloDetector] Temp-file path failed ({e}); falling back to ndarray.")
                    with torch.inference_mode():
                        results = self.model.predict(
                            img,
                            device=self.device_str,
                            max_det=self.max_det,
                            conf=self.conf,
                            verbose=self.verbose,
                            save=False, show=False, augment=False,
                            half=self.use_half, imgsz=self.imgsz
                        )

                page_crops, page_detections = [], []
                for res in results or []:
                    for crop, detection in self._get_crop_records(res):
                        page_crops.append(crop)
                        page_detections.append(detection)
                page['crops'] = page_crops

                if catalogue and page_crops:
                    self._save_crops_to_disk(page, page_crops, catalogue, catalogue_page, page_detections)

                page['page_image'] = None  # free memory

            except Exception as e:
                print(f"[YoloDetector] Error on page {page_idx}: {e}")
                page['crops'] = []
                page['page_image'] = None
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[YoloDetector] detect_and_crop completed")
        return pdf_pages
