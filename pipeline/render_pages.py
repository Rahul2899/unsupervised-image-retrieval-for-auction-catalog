#!/usr/bin/env python3
"""
Stage 1: PDF to Images Converter
Multi-process CPU-optimized extraction - one-time operation
ROBUST RESUME: Automatically detects and skips completed work
OUTPUT: Flat directory with {catalogue_id}_page{N}.jpg naming

Usage:
    python pipeline/render_pages.py --pdf-dir data/pdfs --image-dir data/pages --workers 8
"""

import os
import sys
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, Set
import traceback

import fitz  # PyMuPDF
import numpy as np
from PIL import Image


class ProgressTracker:
    """Thread-safe progress tracking with verification"""

    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.progress_file = image_dir / '.pdf_conversion_progress.json'

    def load(self) -> dict:
        """Load progress data"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {'completed': {}, 'last_catalogue': None, 'last_update': None}
        return {'completed': {}, 'last_catalogue': None, 'last_update': None}

    def save(self, data: dict):
        """Save progress atomically"""
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.progress_file.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump({
                    **data,
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
            tmp.replace(self.progress_file)
        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")

    def mark_complete(self, catalogue_id: str, num_pages: int):
        """Mark catalogue conversion complete"""
        data = self.load()
        data['completed'][catalogue_id] = num_pages
        data['last_catalogue'] = catalogue_id
        self.save(data)

    def is_complete(self, catalogue_id: str, verify: bool = True) -> bool:
        """
        Check if catalogue already converted
        If verify=True, also checks if image files actually exist
        """
        data = self.load()

        # Check progress file
        if catalogue_id not in data.get('completed', {}):
            return False

        # Optional verification: check if images exist in flat directory
        if verify:
            expected_pages = data['completed'][catalogue_id]

            # Count existing images with pattern: {catalogue_id}_page{N}.jpg
            actual_images = 0
            for page_num in range(1, expected_pages + 1):  # Pages start from 1 (skipping page 0)
                img_path = self.image_dir / f"{catalogue_id}_page{page_num}.jpg"
                if img_path.exists() and img_path.stat().st_size > 1000:
                    actual_images += 1

            # If mismatch, mark as incomplete
            if actual_images < expected_pages:
                print(f"  ⚠ {catalogue_id}: Expected {expected_pages} pages, found {actual_images} - will re-convert")
                return False

        return True

    def get_stats(self) -> dict:
        """Get conversion statistics"""
        data = self.load()
        completed = data.get('completed', {})
        return {
            'catalogues': len(completed),
            'total_pages': sum(completed.values()),
            'last_catalogue': data.get('last_catalogue'),
            'last_update': data.get('last_update')
        }

    def get_last_processed(self) -> Optional[str]:
        """Get the last successfully processed catalogue ID"""
        data = self.load()
        return data.get('last_catalogue')


def convert_pdf_to_images(
    pdf_path: Path,
    image_dir: Path,
    dpi: int = 200,
    quality: int = 75,
    resize_to_yolo: bool = True,
    yolo_size: int = 640,
    skip_first_page: bool = True
) -> Optional[int]:
    """
    Convert single PDF to images in flat directory
    Filename format: {catalogue_id}_page{N}.jpg (matches reference code)
    Returns: number of pages converted (excluding skipped pages), or None on failure
    """
    catalogue_id = pdf_path.stem

    doc = None
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        if num_pages == 0:
            return None

        # Determine page range (skip first page like reference code)
        start_page = 1 if skip_first_page else 0
        pages_converted = 0

        # Convert each page
        for page_num in range(start_page, num_pages):
            # Output filename: {catalogue_id}_page{page_num}.jpg
            # Note: page_num is the actual page index (1, 2, 3...) matching reference
            output_path = image_dir / f"{catalogue_id}_page{page_num}.jpg"

            # Skip if image already exists and is valid
            if output_path.exists() and output_path.stat().st_size > 1000:
                pages_converted += 1
                continue

            page = doc[page_num]

            if resize_to_yolo:
                scale = yolo_size / max(page.rect.width, page.rect.height)
            else:
                scale = dpi / 72

            # Render at output size so large pages never allocate a full 200-DPI raster.
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, pix.n)

            # Handle alpha channel
            if pix.n == 4:
                img_array = img_array[:, :, :3]

            # Save as JPEG
            img = Image.fromarray(img_array)
            img.save(output_path, 'JPEG', quality=quality, optimize=True)

            pages_converted += 1

            # Cleanup
            pix = None

        doc.close()
        return pages_converted

    except Exception:
        if doc:
            doc.close()
        raise


def worker_process(args_tuple):
    """Worker function for multiprocessing"""
    pdf_path, image_dir, dpi, quality, skip_first, worker_id = args_tuple

    catalogue_id = pdf_path.stem

    try:
        num_pages = convert_pdf_to_images(
            pdf_path, image_dir, dpi, quality,
            resize_to_yolo=True,
            skip_first_page=skip_first
        )

        if num_pages is not None:
            return {
                'catalogue_id': catalogue_id,
                'status': 'success',
                'pages': num_pages,
                'worker_id': worker_id
            }
        else:
            return {
                'catalogue_id': catalogue_id,
                'status': 'failed',
                'pages': 0,
                'error': 'PDF has no pages',
                'worker_id': worker_id
            }
    except Exception as e:
        return {
            'catalogue_id': catalogue_id,
            'status': 'error',
            'pages': 0,
            'error': str(e),
            'worker_id': worker_id
        }


def find_resume_point(all_pdfs: list, progress: ProgressTracker, start_from: Optional[str]) -> int:
    """
    Find where to resume processing
    Returns: index in all_pdfs to start from
    """
    if not start_from:
        # Auto-detect: use last processed catalogue
        last = progress.get_last_processed()
        if last:
            print(f"🔄 Auto-resume detected!")
            print(f"   Last processed: {last}")

            # Find index
            for idx, pdf in enumerate(all_pdfs):
                if pdf.stem == last:
                    # Start from NEXT catalogue
                    resume_idx = idx + 1
                    if resume_idx < len(all_pdfs):
                        print(f"   Resuming from: {all_pdfs[resume_idx].stem}")
                        return resume_idx
                    else:
                        print(f"   All catalogues after {last} already processed")
                        return len(all_pdfs)
        return 0

    # Manual start_from specified
    for idx, pdf in enumerate(all_pdfs):
        if pdf.stem == start_from:
            print(f"📍 Manual start point: {start_from}")
            return idx

    print(f"⚠ Warning: --start-from '{start_from}' not found, starting from beginning")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Convert PDFs to Images (Flat Directory, Multi-process with Auto-Resume)'
    )

    parser.add_argument('--pdf-dir', type=str, required=True,
                       help='Directory containing PDF catalogues')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Output directory for images (flat structure)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--dpi', type=int, default=200,
                       help='Image DPI (default: 200 to match reference)')
    parser.add_argument('--quality', type=int, default=75,
                       help='JPEG quality 1-100 (default: 75)')
    parser.add_argument('--no-skip-first', action='store_true',
                       help='Include first page (default: skip first page like reference)')
    parser.add_argument('--start-from', type=str, default=None,
                       help='Resume from specific catalogue ID (auto-detects if not specified)')
    parser.add_argument('--max-catalogues', type=int, default=None,
                       help='Maximum number of catalogues to process')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification of existing images (faster but less safe)')

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    image_dir = Path(args.image_dir)
    skip_first_page = not args.no_skip_first

    # Validate
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1

    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("📄 PDF TO IMAGES CONVERTER - STAGE 1 (FLAT DIRECTORY)")
    print(f"{'='*80}")
    print(f"PDF Directory:    {pdf_dir}")
    print(f"Image Directory:  {image_dir}")
    print(f"Output Format:    {'{catalogue_id}_page{N}.jpg'}")
    print(f"Workers:          {args.workers}")
    print(f"DPI:              {args.dpi}")
    print(f"JPEG Quality:     {args.quality}")
    print(f"Skip First Page:  {skip_first_page}")
    print(f"Verify existing:  {not args.no_verify}")
    if args.start_from:
        print(f"Start From:       {args.start_from}")
    if args.max_catalogues:
        print(f"Max Catalogues:   {args.max_catalogues}")
    print(f"{'='*80}\n")

    # Setup progress tracker
    progress = ProgressTracker(image_dir)

    # Show existing progress
    stats = progress.get_stats()
    if stats['catalogues'] > 0:
        print(f"📊 Existing Progress:")
        print(f"   Completed: {stats['catalogues']} catalogues ({stats['total_pages']:,} pages)")
        if stats['last_catalogue']:
            print(f"   Last processed: {stats['last_catalogue']}")
        if stats['last_update']:
            print(f"   Last update: {stats['last_update']}")
        print()

    # Find all PDFs
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(all_pdfs)} total PDFs")

    if not all_pdfs:
        print("No PDFs found!")
        return 1

    # Find resume point
    start_idx = find_resume_point(all_pdfs, progress, args.start_from)

    # Filter PDFs to process
    pdfs_to_process = []

    for pdf_path in all_pdfs[start_idx:]:
        catalogue_id = pdf_path.stem

        # Skip completed (with optional verification)
        if progress.is_complete(catalogue_id, verify=not args.no_verify):
            continue

        pdfs_to_process.append(pdf_path)

        # Handle max limit
        if args.max_catalogues and len(pdfs_to_process) >= args.max_catalogues:
            break

    # Show stats
    print(f"Already converted: {stats['catalogues']} catalogues")
    print(f"To process:        {len(pdfs_to_process)} catalogues")

    if not pdfs_to_process:
        print("\n✓ All PDFs already converted!")
        return 0

    print(f"\nWill process: {pdfs_to_process[0].stem} → {pdfs_to_process[-1].stem}")

    # Estimate
    avg_pages_per_catalogue = 50  # rough estimate
    total_estimated_pages = len(pdfs_to_process) * avg_pages_per_catalogue
    pages_per_hour = 2000  # rough estimate for multi-core
    estimated_hours = total_estimated_pages / pages_per_hour

    print(f"\nEstimated: ~{total_estimated_pages:,} pages")
    print(f"Est. time: ~{estimated_hours:.1f} hours @ {args.workers} workers")
    print(f"{'='*80}\n")

    # Prepare worker arguments
    worker_args = [
        (pdf_path, image_dir, args.dpi, args.quality, skip_first_page, i % args.workers)
        for i, pdf_path in enumerate(pdfs_to_process)
    ]

    # Process with multiprocessing
    start_time = datetime.now()
    completed = 0
    failed = 0
    total_pages = 0

    try:
        with mp.Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_process, worker_args), 1):
                catalogue_id = result['catalogue_id']
                status = result['status']
                pages = result['pages']

                if status == 'success':
                    completed += 1
                    total_pages += pages
                    progress.mark_complete(catalogue_id, pages)
                    print(f"[{i}/{len(pdfs_to_process)}] ✓ {catalogue_id} ({pages} pages)")
                else:
                    failed += 1
                    error = result.get('error', 'Unknown error')
                    print(f"[{i}/{len(pdfs_to_process)}] ✗ {catalogue_id} - {error}")

                # Progress update every 50 catalogues
                if i % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(pdfs_to_process) - i
                    eta = remaining / rate if rate > 0 else 0

                    print(f"\n{'='*80}")
                    print(f"Progress: {i}/{len(pdfs_to_process)} ({100*i/len(pdfs_to_process):.1f}%)")
                    print(f"Success: {completed} | Failed: {failed}")
                    print(f"Total pages: {total_pages:,}")
                    print(f"Rate: {rate*3600:.1f} catalogues/hour")
                    print(f"ETA: {eta/3600:.1f} hours")
                    print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted - Progress saved!")
        print(f"   Resume by running the same command again (auto-resume enabled)")

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("📊 CONVERSION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed:    {completed} catalogues")
    print(f"Failed:       {failed} catalogues")
    print(f"Total pages:  {total_pages:,}")
    if completed > 0:
        print(f"Avg pages:    {total_pages/completed:.1f} per catalogue")
        print(f"Rate:         {(completed/elapsed)*3600:.1f} catalogues/hour")
    print(f"Runtime:      {elapsed/3600:.2f} hours")

    # Show overall progress
    final_stats = progress.get_stats()
    print(f"\n📈 Overall Progress:")
    print(f"   Total completed: {final_stats['catalogues']} / {len(all_pdfs)} catalogues")
    print(f"   Completion: {100*final_stats['catalogues']/len(all_pdfs):.1f}%")

    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
