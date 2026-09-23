#!/usr/bin/env python3
"""
GPU-Optimized Cropping with Batched YOLO Inference
Key optimizations:
- Batched YOLO inference (process multiple images at once)
- Increased default batch size
- Better GPU memory utilization
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import warnings

import torch
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

warnings.filterwarnings('ignore')
os.environ['NNPACK_DISABLED'] = '1'


class YoloDetector:
    def __init__(self, cache_dir=None, device=None, crop_dir=None, imgsz=640, max_det=10, verbose=False, conf=0.25):
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
        self.min_crop_std = float(os.environ.get('MIN_CROP_STD', '0'))
        self.verbose = bool(verbose)
        self.crop_dir = crop_dir

        if self.crop_dir:
            os.makedirs(self.crop_dir, exist_ok=True)

    def _get_yolo_weights(self, cache_dir):
        print(f'[YoloDetector] Looking for cached weights in {cache_dir}')
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = Path(os.environ.get('YOLO_WEIGHTS', str(Path(cache_dir) / 'detection_weights.pt')))
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

    def _validate_image(self, img):
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

    def _get_crop(self, yolo_result):
        crops = []
        try:
            img = yolo_result.orig_img
            img = self._validate_image(img)
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                return crops
            h, w = img.shape[:2]
            for i, xyxy in enumerate(yolo_result.boxes.xyxy):
                try:
                    x1, y1, x2, y2 = xyxy.int().cpu().numpy().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue
                    crop = img[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        if self.min_crop_std and cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std() < self.min_crop_std:
                            continue
                        crops.append(crop)
                except Exception as e:
                    print(f"[YoloDetector] Error processing bbox {i}: {e}")
                    continue
        except Exception as e:
            print(f"[YoloDetector] _get_crop error: {e}")
        return crops

    def _save_crops_to_disk(self, catalogue, page_num, crops):
        if not self.crop_dir or not crops:
            return 0
        saved = 0
        for idx, crop in enumerate(crops):
            try:
                outp = Path(self.crop_dir) / f"{catalogue}_page{page_num}_{idx}.jpg"
                cv2.imwrite(str(outp), crop)
                saved += 1
            except Exception as e:
                print(f"[YoloDetector] Error saving crop {outp}: {e}")
        return saved

    def detect_and_crop_batched(self, image_batch, yolo_batch_size=16, skip_existing=False):
        """
        Process images with batched YOLO inference for better GPU utilization.

        Args:
            image_batch: List of image data dicts
            yolo_batch_size: Number of images to process in single YOLO call
        """
        all_processed = []

        for batch_start in range(0, len(image_batch), yolo_batch_size):
            batch_end = min(batch_start + yolo_batch_size, len(image_batch))
            mini_batch = image_batch[batch_start:batch_end]

            # Prepare images for batched inference
            imgs_to_process = []
            img_indices = []

            for idx, image_data in enumerate(mini_batch):
                try:
                    img = image_data.get('page_image')
                    if img is not None:
                        validated_img = self._validate_image(img).copy()
                        imgs_to_process.append(validated_img)
                        img_indices.append(idx)
                except Exception as e:
                    print(f"[YoloDetector] Error validating image {idx}: {e}")
                    mini_batch[idx]['crops'] = []
                    mini_batch[idx]['page_image'] = None

            if not imgs_to_process:
                all_processed.extend(mini_batch)
                continue

            # Run batched YOLO inference
            try:
                with torch.inference_mode():
                    results = self.model.predict(
                        imgs_to_process,
                        device=self.device_str,
                        max_det=self.max_det,
                        conf=self.conf,
                        verbose=False,
                        save=False,
                        show=False,
                        augment=False,
                        half=self.use_half,
                        imgsz=self.imgsz
                    )

                # Process results
                for result_idx, yolo_result in enumerate(results):
                    original_idx = img_indices[result_idx]
                    image_data = mini_batch[original_idx]

                    crops = self._get_crop(yolo_result)
                    image_data['crops'] = crops

                    # Save crops immediately
                    catalogue = image_data.get('source_catalogue')
                    page_num = image_data.get('catalogue_page')
                    if catalogue and crops:
                        self._save_crops_to_disk(catalogue, page_num, crops)

                    # Free memory
                    image_data['page_image'] = None

            except Exception as e:
                print(f"[YoloDetector] Batch inference error: {e}")
                for idx in img_indices:
                    mini_batch[idx]['crops'] = []
                    mini_batch[idx]['page_image'] = None

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_processed.extend(mini_batch)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return all_processed


class CropProgress:
    def __init__(self, crop_dir: Path):
        self.crop_dir = crop_dir
        self.progress_file = crop_dir / '.crop_progress.json'

    def load(self) -> dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {'completed': {}, 'last_catalogue': None}
        return {'completed': {}, 'last_catalogue': None}

    def save(self, data: dict):
        try:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.progress_file.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump({
                    **data,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
            tmp.replace(self.progress_file)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def mark_complete(self, catalogue_id: str, num_crops: int):
        data = self.load()
        data['completed'][catalogue_id] = num_crops
        data['last_catalogue'] = catalogue_id
        self.save(data)

    def is_complete(self, catalogue_id: str) -> bool:
        data = self.load()
        return catalogue_id in data.get('completed', {})

    def get_stats(self) -> dict:
        data = self.load()
        completed = data.get('completed', {})
        return {
            'catalogues': len(completed),
            'total_crops': sum(completed.values()),
            'last_catalogue': data.get('last_catalogue')
        }

    def get_last_processed(self) -> Optional[str]:
        data = self.load()
        return data.get('last_catalogue')


def find_all_catalogues(image_dir: Path) -> List[str]:
    catalogue_ids = set()
    for img_file in image_dir.glob("*_page*.jpg"):
        name = img_file.stem
        parts = name.split('_page')
        if len(parts) >= 2:
            catalogue_ids.add(parts[0])
    return sorted(list(catalogue_ids))


def get_catalogue_images(image_dir: Path, catalogue_id: str) -> List[Path]:
    pattern = f"{catalogue_id}_page*.jpg"
    images = list(image_dir.glob(pattern))

    def extract_page_num(path):
        name = path.stem
        parts = name.split('_page')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    return sorted(images, key=extract_page_num)


def load_image_batch(image_paths: List[Path], pin_memory: bool = False) -> List[Dict]:
    batch_data = []

    for img_path in image_paths:
        try:
            name = img_path.stem
            parts = name.split('_page')

            if len(parts) < 2:
                continue

            catalogue_id = parts[0]
            page_num = int(parts[1])

            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.uint8)

            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)

            if pin_memory and torch.cuda.is_available():
                img_tensor = torch.from_numpy(img_array).pin_memory()
                img_array = img_tensor.numpy()

            batch_data.append({
                'page_image': img_array,
                'source_catalogue': catalogue_id,
                'catalogue_page': page_num,
            })

        except Exception as e:
            print(f"  Warning: Failed to load {img_path.name}: {e}")
            continue

    return batch_data


def process_catalogue_gpu(
    image_dir: Path,
    catalogue_id: str,
    detector: YoloDetector,
    batch_size: int = 150,
    yolo_batch_size: int = 16,
    pin_memory: bool = False
) -> Optional[int]:
    image_files = get_catalogue_images(image_dir, catalogue_id)

    if not image_files:
        return None

    total_crops = 0

    try:
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_paths = image_files[batch_start:batch_end]

            batch_data = load_image_batch(batch_paths, pin_memory=pin_memory)

            if not batch_data:
                continue

            try:
                results = detector.detect_and_crop_batched(
                    batch_data,
                    yolo_batch_size=yolo_batch_size
                )

                for result in results:
                    if result:
                        total_crops += len(result.get('crops', []))

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  ⚠ GPU OOM! Try reducing --batch-size or --yolo-batch-size")
                    torch.cuda.empty_cache()
                    raise
                else:
                    print(f"  Error in batch {batch_start}-{batch_end}: {e}")

            finally:
                del batch_data
                if 'results' in locals():
                    del results
                gc.collect()

        return total_crops

    except Exception as e:
        print(f"  Error processing {catalogue_id}: {e}")
        return None


def process_batch_of_catalogues(
    image_dir: Path,
    catalogue_ids: List[str],
    detector: YoloDetector,
    progress: CropProgress,
    batch_size: int = 150,
    yolo_batch_size: int = 16,
    pin_memory: bool = False
) -> Dict[str, int]:
    results = {}

    for idx, catalogue_id in enumerate(catalogue_ids, 1):
        print(f"  [{idx}/{len(catalogue_ids)}] {catalogue_id}...", end=' ', flush=True)

        if progress.is_complete(catalogue_id):
            data = progress.load()
            num_crops = data['completed'].get(catalogue_id, 0)
            print(f"✓ Already done ({num_crops} crops)")
            results[catalogue_id] = num_crops
            continue

        num_crops = process_catalogue_gpu(
            image_dir, catalogue_id, detector,
            batch_size, yolo_batch_size, pin_memory
        )

        if num_crops is not None:
            print(f"✓ {num_crops} crops")
            progress.mark_complete(catalogue_id, num_crops)
            results[catalogue_id] = num_crops
        else:
            print("✗ Failed")
            results[catalogue_id] = 0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return results


def find_resume_point(all_catalogues: List[str], progress: CropProgress, start_from: Optional[str]) -> int:
    if not start_from:
        last = progress.get_last_processed()
        if last:
            print(f"🔄 Auto-resume: Last processed was {last}")
            for idx, cat_id in enumerate(all_catalogues):
                if cat_id == last:
                    resume_idx = idx + 1
                    if resume_idx < len(all_catalogues):
                        print(f"   Resuming from: {all_catalogues[resume_idx]}\n")
                        return resume_idx
            print(f"   All done after {last}!\n")
            return len(all_catalogues)
        return 0

    for idx, cat_id in enumerate(all_catalogues):
        if cat_id == start_from:
            print(f"📍 Starting from: {start_from}\n")
            return idx

    print(f"⚠ Warning: '{start_from}' not found, starting from beginning\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized Cropping with Batched Inference')

    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--crop-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=150,
                       help='Images to load at once (default: 150)')
    parser.add_argument('--yolo-batch-size', type=int, default=16,
                       help='Images per YOLO forward pass (default: 16)')
    parser.add_argument('--catalogue-batch', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--max-det', type=int, default=15)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--start-from', type=str, default=None)
    parser.add_argument('--max-catalogues', type=int, default=None)

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    crop_dir = Path(args.crop_dir)

    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return 1

    crop_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("🚀 GPU-OPTIMIZED IMAGE CROPPING (BATCHED INFERENCE)")
    print(f"{'='*80}")
    print(f"Image Directory:      {image_dir}")
    print(f"Crop Directory:       {crop_dir}")
    print(f"Image Batch Size:     {args.batch_size}")
    print(f"YOLO Batch Size:      {args.yolo_batch_size}")
    print(f"{'='*80}\n")

    progress = CropProgress(crop_dir)
    stats = progress.get_stats()

    if stats['catalogues'] > 0:
        print(f"📊 Existing: {stats['catalogues']} catalogues, {stats['total_crops']:,} crops\n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {total_mem_gb:.1f} GB\n")

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("⚠ NO GPU - Using CPU\n")
        total_mem_gb = 0

    print("Scanning...", end=' ', flush=True)
    all_catalogues = find_all_catalogues(image_dir)
    print(f"✓ Found {len(all_catalogues)} catalogues")

    if not all_catalogues:
        print("No images found!")
        return 1

    start_idx = find_resume_point(all_catalogues, progress, args.start_from)

    catalogues_to_process = []
    for cat_id in all_catalogues[start_idx:]:
        if progress.is_complete(cat_id):
            continue
        catalogues_to_process.append(cat_id)
        if args.max_catalogues and len(catalogues_to_process) >= args.max_catalogues:
            break

    print(f"To process: {len(catalogues_to_process)} catalogues\n")

    if not catalogues_to_process:
        print("✓ All done!")
        return 0

    print(f"{'='*80}\n")

    print("Initializing YOLO...", end=' ', flush=True)
    detector = YoloDetector(
        crop_dir=str(crop_dir),
        device=device,
        verbose=False,
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det
    )
    print("✓\n")

    run_stats = {'processed': 0, 'failed': 0, 'total_crops': 0, 'start_time': datetime.now()}

    try:
        for batch_idx in range(0, len(catalogues_to_process), args.catalogue_batch):
            batch_end = min(batch_idx + args.catalogue_batch, len(catalogues_to_process))
            batch_cat_ids = catalogues_to_process[batch_idx:batch_end]

            batch_start_time = datetime.now()

            print(f"\n{'#'*80}")
            print(f"BATCH {batch_idx//args.catalogue_batch + 1}: {batch_idx+1}-{batch_end}")
            print(f"{'#'*80}")

            batch_results = process_batch_of_catalogues(
                image_dir, batch_cat_ids, detector, progress,
                args.batch_size, args.yolo_batch_size, args.pin_memory
            )

            for catalogue_id, num_crops in batch_results.items():
                if num_crops > 0:
                    run_stats['processed'] += 1
                    run_stats['total_crops'] += num_crops
                else:
                    run_stats['failed'] += 1

            batch_time = (datetime.now() - batch_start_time).total_seconds()
            total_time = (datetime.now() - run_stats['start_time']).total_seconds()

            cat_per_hour = (run_stats['processed'] / total_time) * 3600 if total_time > 0 else 0
            crops_per_hour = (run_stats['total_crops'] / total_time) * 3600 if total_time > 0 else 0

            remaining = len(catalogues_to_process) - batch_end
            eta_hours = (remaining / cat_per_hour) if cat_per_hour > 0 else 0

            print(f"\n{'='*80}")
            print(f"PROGRESS: {batch_end}/{len(catalogues_to_process)} ({100*batch_end/len(catalogues_to_process):.1f}%)")
            print(f"Success: {run_stats['processed']} | Failed: {run_stats['failed']}")
            print(f"Total crops: {run_stats['total_crops']:,}")
            if run_stats['processed'] > 0:
                print(f"Avg: {run_stats['total_crops']/run_stats['processed']:.1f} crops/cat")
            print(f"⚡ {cat_per_hour:.1f} cat/hr | {crops_per_hour:.0f} crops/hr")
            print(f"⏱️  Batch: {batch_time:.1f}s | Total: {total_time/3600:.2f}h")
            if remaining > 0 and eta_hours > 0:
                print(f"📅 ETA: {eta_hours:.1f}h")

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(0) / 1e9
                util_pct = (mem / total_mem_gb) * 100 if total_mem_gb > 0 else 0
                print(f"🎮 GPU: {mem:.2f} / {total_mem_gb:.1f} GB ({util_pct:.0f}%)")

            print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n⚠ INTERRUPTED - Progress saved!")

    finally:
        total_time = (datetime.now() - run_stats['start_time']).total_seconds()

        print(f"\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Processed: {run_stats['processed']} | Failed: {run_stats['failed']}")
        print(f"Total Crops: {run_stats['total_crops']:,}")
        if run_stats['processed'] > 0:
            print(f"Avg: {run_stats['total_crops']/run_stats['processed']:.1f} crops/cat")
            print(f"Throughput: {(run_stats['processed']/total_time)*3600:.1f} cat/hr")
        print(f"Runtime: {total_time/3600:.2f} hours")
        print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
