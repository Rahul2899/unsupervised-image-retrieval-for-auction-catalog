import os, re, io, gc, json, random, time, shutil
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, quote, urljoin
from typing import Optional, Tuple, List
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import numpy as np
import cv2

try:
    import fitz as pymupdf
except Exception:
    import pymupdf

from PIL import Image

# ==================== Failure Logger ====================

class FailureLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.download_log = self.log_dir / "failed_downloads.json"
        self.processing_log = self.log_dir / "failed_processing.json"
        self.summary_log = self.log_dir / "failure_summary.txt"
        self.lock = Lock()  # Thread safety

    def _append_to_log(self, log_file, entry):
        with self.lock:
            max_retries = 3
            for attempt in range(max_retries):
                tmp = log_file.with_suffix(f'.tmp.{os.getpid()}.{attempt}')
                try:
                    if log_file.exists():
                        with open(log_file, 'r', encoding='utf-8') as f:
                            try:
                                logs = json.load(f)
                            except json.JSONDecodeError:
                                logs = []
                    else:
                        logs = []

                    logs.append(entry)

                    with open(tmp, 'w', encoding='utf-8') as f:
                        json.dump(logs, f, indent=2, ensure_ascii=False)

                    tmp.replace(log_file)

                    # Clean stale temp files
                    for stale in log_file.parent.glob(f'{log_file.stem}.tmp.*'):
                        if stale != tmp:
                            try:
                                stale.unlink(missing_ok=True)
                            except:
                                pass

                    return

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                    else:
                        print(f"⚠️  Error appending to {log_file}: {e}")
                finally:
                    if tmp.exists():
                        try:
                            tmp.unlink(missing_ok=True)
                        except:
                            pass

    def _load_log(self, log_file):
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def log_failed_download(self, catalogue_entry, reason):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'uri': catalogue_entry.get('uri', 'Unknown'),
            'name': catalogue_entry.get('name', catalogue_entry.get('title', 'Unknown')),
            'filename': catalogue_entry.get('fn', 'Unknown'),
            'reason': reason,
            'type': 'download_failure'
        }
        self._append_to_log(self.download_log, entry)
        print(f"DOWNLOAD FAILED: {entry['name']} - {reason}")

    def log_failed_processing(self, catalogue_entry, stage, reason):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'uri': catalogue_entry.get('uri', 'Unknown'),
            'name': catalogue_entry.get('name', catalogue_entry.get('title', 'Unknown')),
            'stage': stage,
            'reason': reason,
            'type': 'processing_failure'
        }
        self._append_to_log(self.processing_log, entry)
        print(f"PROCESSING FAILED: {entry['name']} at {stage} - {reason}")

    def generate_summary(self):
        download_failures = self._load_log(self.download_log)
        processing_failures = self._load_log(self.processing_log)

        summary = []
        summary.append(f"FAILURE SUMMARY - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("=" * 60)
        summary.append(f"\nDOWNLOAD FAILURES: {len(download_failures)} total")
        summary.append("-" * 30)

        reasons = {}
        for f in download_failures:
            r = f.get('reason', 'Unknown')
            reasons[r] = reasons.get(r, 0) + 1
        for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  {r}: {c} failures")

        if download_failures:
            summary.append("\nFailed Downloads:")
            for f in download_failures[:20]:  # Limit output
                summary.append(f"  - {f.get('name', 'Unknown')}")
                summary.append(f"    URI: {f.get('uri', 'Unknown')}")
                summary.append(f"    Reason: {f.get('reason', 'Unknown')}\n")

        summary.append(f"\nPROCESSING FAILURES: {len(processing_failures)} total")
        summary.append("-" * 30)

        stages = {}
        preasons = {}
        for f in processing_failures:
            s = f.get('stage', 'Unknown')
            r = f.get('reason', 'Unknown')
            stages[s] = stages.get(s, 0) + 1
            preasons[r] = preasons.get(r, 0) + 1

        summary.append("By Stage:")
        for s, c in sorted(stages.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  {s}: {c} failures")

        summary.append("\nBy Reason:")
        for r, c in sorted(preasons.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  {r}: {c} failures")

        if processing_failures:
            summary.append("\nFailed Processing:")
            for f in processing_failures[:20]:  # Limit output
                summary.append(f"  - {f.get('name', 'Unknown')}")
                summary.append(f"    URI: {f.get('uri', 'Unknown')}")
                summary.append(f"    Stage: {f.get('stage', 'Unknown')}")
                summary.append(f"    Reason: {f.get('reason', 'Unknown')}\n")

        text = "\n".join(summary)
        with open(self.summary_log, 'w', encoding='utf-8') as fp:
            fp.write(text)
        print(f"\nSummary written to: {self.summary_log}")
        print(f"Total failures: {len(download_failures) + len(processing_failures)}")
        return text

    def get_failed_catalogues(self):
        dl = self._load_log(self.download_log)
        pl = self._load_log(self.processing_log)
        failed_uris = set()
        for f in dl + pl:
            u = f.get('uri')
            if u and u != 'Unknown':
                failed_uris.add(u)
        return list(failed_uris)

# ==================== Heidelberg Diglit Downloader ====================

class HeidelbergDiglitDownloader:
    """
    Robust downloader for Heidelberg University's diglit repository.
    Handles DOIs, IIIF, PDFs, and page-by-page scraping.
    """

    def __init__(self, download_dir, skip_existing=True, max_retries=3):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.max_retries = max_retries

        self.session = requests.Session()
        try:
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        except TypeError:
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    def _resolve_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve DOI or other redirects to actual diglit URL."""
        try:
            resp = self.session.head(url, headers=self.headers, allow_redirects=True, timeout=60)
            final_url = resp.url

            if 'digi.ub.uni-heidelberg.de/diglit/' in final_url:
                parts = final_url.rstrip('/').split('/')
                volume_id = parts[-1]
                if volume_id.isdigit():
                    volume_id = parts[-2]
                base_url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}"
                return base_url, volume_id
            else:
                return None, f"Not a valid diglit URL: {final_url}"
        except Exception as e:
            return None, f"URL resolution failed: {str(e)}"

    def _try_iiif_manifest(self, volume_id: str, output_path: Path) -> Tuple[Optional[Path], Optional[str]]:
        """Attempt to download via IIIF manifest."""
        manifest_url = f"https://digi.ub.uni-heidelberg.de/diglit/iiif/{volume_id}/manifest.json"

        try:

            print(f"  Trying IIIF: {manifest_url}")
            manifest = self.session.get(manifest_url, headers=self.headers, timeout=30).json()

            images = list(self._extract_iiif_images(manifest))
            if not images:
                return None, "No images found in IIIF manifest"

            print(f"  Found {len(images)} pages in IIIF manifest")
            return self._images_to_pdf(images, output_path), None

        except Exception as e:
            return None, f"IIIF failed: {str(e)}"

    def _extract_iiif_images(self, manifest: dict):
        """Extract image URLs from IIIF manifest (v2/v3 compatible)."""
        is_v3 = manifest.get("type") == "Manifest"

        canvases = []
        if is_v3:
            canvases = manifest.get("items", [])
        else:
            sequences = manifest.get("sequences", [])
            if sequences:
                canvases = sequences[0].get("canvases", [])

        for idx, canvas in enumerate(canvases):
            try:
                image_service_id = None

                # Try v2 structure
                if canvas.get("images"):
                    image_service_id = canvas["images"][0]["resource"]["service"]["@id"]

                # Try v3 structure
                elif canvas.get("items"):
                    anno_page = canvas["items"][0]
                    anno = anno_page["items"][0]
                    body = anno["body"]

                    services = body.get("service") or body.get("services") or []
                    if isinstance(services, dict):
                        services = [services]

                    for svc in services:
                        image_service_id = svc.get("@id") or svc.get("id")
                        if image_service_id:
                            break

                if not image_service_id:
                    continue

                image_service_id = image_service_id.rstrip("/")

                for pattern in ["/full/full/0/default.jpg", "/full/max/0/default.jpg"]:
                    img_url = image_service_id + pattern
                    yield img_url
                    break

            except Exception as e:
                print(f"  Warning: Could not extract image {idx}: {e}")
                continue

    def _try_direct_pdf(self, base_url: str, output_path: Path) -> Tuple[Optional[Path], Optional[str]]:
        """Try common PDF endpoint patterns."""
        pdf_endpoints = ["/download", "/pdf", "/0000.pdf", "/download.pdf"]

        for endpoint in pdf_endpoints:
            try:
                pdf_url = base_url + endpoint
                print(f"  Trying PDF: {pdf_url}")

                resp = self.session.get(pdf_url, headers=self.headers, timeout=60, stream=True)

                if resp.status_code == 200:
                    content = resp.content

                    if content.startswith(b'%PDF') and len(content) > 10000:
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        print(f"  ✓ Downloaded PDF ({len(content)} bytes)")
                        return output_path, None

            except Exception:
                continue

        return None, "No valid PDF endpoints found"

    def _scrape_page_images(self, base_url: str, output_path: Path) -> Tuple[Optional[Path], Optional[str]]:
        """Scrape individual page images - most reliable fallback."""
        try:
            print(f"  Scraping page images from: {base_url}")

            resp = self.session.get(base_url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            page_count = self._find_page_count(soup, base_url)

            if not page_count:
                return None, "Could not determine page count"

            print(f"  Found {page_count} pages to scrape")

            images = []
            volume_id = base_url.rstrip('/').split('/')[-1]

            for page_num in range(1, page_count + 1):
                img_url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}/{page_num:04d}/max"

                try:
                    img_resp = self.session.get(img_url, headers=self.headers, timeout=30)
                    if img_resp.status_code == 200:
                        images.append(img_resp.content)
                        if page_num % 10 == 0:
                            print(f"  Downloaded {page_num}/{page_count} pages")
                except Exception as e:
                    print(f"  Warning: Failed page {page_num}: {e}")

            if not images:
                return None, "No images could be downloaded"

            print(f"  Successfully downloaded {len(images)} pages")
            return self._images_to_pdf(images, output_path), None

        except Exception as e:
            return None, f"Page scraping failed: {str(e)}"

    def _find_page_count(self, soup, base_url: str) -> Optional[int]:
        """Try multiple methods to find the page count."""

        # Method 1: Navigation elements
        for selector in ['.navi-total', '.page-count', '#pageCount']:
            elem = soup.select_one(selector)
            if elem:
                match = re.search(r'\d+', elem.get_text())
                if match:
                    return int(match.group())

        # Method 2: Meta tags
        for meta in soup.find_all('meta'):
            if 'page' in str(meta.get('name', '')).lower():
                content = meta.get('content', '')
                match = re.search(r'\d+', content)
                if match:
                    return int(match.group())

        # Method 3: JavaScript config
        for script in soup.find_all('script'):
            script_text = script.string or ''
            if 'lastPage' in script_text or 'pageCount' in script_text:
                match = re.search(r'(?:lastPage|pageCount).*?(\d+)', script_text)
                if match:
                    return int(match.group(1))

        # Method 4: Binary search probe
        volume_id = base_url.rstrip('/').split('/')[-1]
        for test_page in [50, 100, 200, 500]:
            test_url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}/{test_page:04d}"
            try:
                resp = self.session.head(test_url, headers=self.headers, timeout=10)
                if resp.status_code == 404:
                    return self._binary_search_last_page(volume_id, test_page // 2, test_page)
            except:
                continue

        return None

    def _binary_search_last_page(self, volume_id: str, low: int, high: int) -> Optional[int]:
        """Binary search to find the last valid page."""
        while low < high:
            mid = (low + high + 1) // 2
            url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}/{mid:04d}"
            try:
                resp = self.session.head(url, headers=self.headers, timeout=10)
                if resp.status_code == 200:
                    low = mid
                else:
                    high = mid - 1
            except:
                high = mid - 1
        return low if low > 0 else None

    def _images_to_pdf(self, images: List, output_path: Path) -> Path:
        """Convert list of image URLs or bytes to PDF."""
        doc = pymupdf.open()

        for idx, img in enumerate(images):
            try:
                if isinstance(img, str):
                    resp = self.session.get(img, headers=self.headers, timeout=60)
                    img_bytes = resp.content
                else:
                    img_bytes = img

                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size

                page = doc.new_page(width=w, height=h)
                page.insert_image(pymupdf.Rect(0, 0, w, h), stream=img_bytes)

                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(images)} images")

            except Exception as e:
                print(f"  Warning: Failed to process image {idx}: {e}")

        doc.save(output_path)
        doc.close()

        return output_path

    def download_catalogue(self, catalogue_entry: dict) -> Tuple[Optional[Path], Optional[str]]:
        """
        Main download with fallback chain:
        1. Resolve URL (DOIs)
        2. IIIF manifest
        3. Direct PDF
        4. Page scraping
        """

        if 'uri' not in catalogue_entry:
            return None, "No URI in catalogue entry"

        original_url = catalogue_entry['uri']
        fn = catalogue_entry.get('fn', 'catalogue') + ".pdf"
        output_path = self.download_dir / fn

        if self.skip_existing and output_path.exists():
            if verify_pdf_exists(output_path):
                print(f'✓ {fn} already exists')
                return output_path, None

        print(f"\nDownloading: {catalogue_entry.get('name', original_url)[:60]}")

        # Step 1: Resolve URL
        base_url, volume_id = self._resolve_url(original_url)

        if not base_url:
            return None, volume_id

        catalogue_entry['resolved_catalogue_id'] = volume_id

        print(f"  Resolved: {volume_id}")

        # Step 2: IIIF
        result, error = self._try_iiif_manifest(volume_id, output_path)
        if result:
            print(f"  ✓ Downloaded via IIIF")
            return result, None

        # Step 3: Direct PDF
        result, error = self._try_direct_pdf(base_url, output_path)
        if result:
            print(f"  ✓ Downloaded via PDF")
            return result, None

        # Step 4: Page scraping
        result, error = self._scrape_page_images(base_url, output_path)
        if result:
            print(f"  ✓ Downloaded via scraping")
            return result, None

        return None, f"All methods failed. Last: {error}"

# Global instances
failure_logger = FailureLogger()
_downloader_cache = {}
_downloader_lock = Lock()

def get_downloader(download_dir):
    with _downloader_lock:
        if download_dir not in _downloader_cache:
            _downloader_cache[download_dir] = HeidelbergDiglitDownloader(download_dir)
        return _downloader_cache[download_dir]

def download_catalogue(catalogue_entry, download_dir, skip_existing=True):
    """Download catalogue with full fallback support."""
    downloader = get_downloader(download_dir)
    downloader.skip_existing = skip_existing

    pdf_path, error = downloader.download_catalogue(catalogue_entry)

    if error:
        failure_logger.log_failed_download(catalogue_entry, error)
        return None

    return pdf_path

# ==================== PDF/Image helpers ====================

def verify_pdf_exists(pdf_path):
    """Verify PDF exists and is valid"""
    if not pdf_path or not Path(pdf_path).exists():
        return False
    try:
        size = Path(pdf_path).stat().st_size
        if size < 1000:
            return False
        doc = pymupdf.open(pdf_path)
        n = len(doc)
        doc.close()
        return n > 0
    except Exception:
        return False

def verify_crop_exists(crop_path):
    """Verify crop image exists and is valid"""
    if not crop_path or not Path(crop_path).exists():
        return False
    try:
        size = Path(crop_path).stat().st_size
        if size < 100:
            return False
        img = cv2.imread(str(crop_path))
        return img is not None and img.size > 0
    except Exception:
        return False

def count_valid_crops(catalogue_id, crop_dir):
    """Count valid crop files for a catalogue"""
    if not crop_dir or not Path(crop_dir).exists():
        return 0
    crop_files = list(Path(crop_dir).glob(f"{catalogue_id}__pdf-*.jpg"))
    crop_files += list(Path(crop_dir).glob(f"{catalogue_id}_page*.jpg"))
    valid = 0
    for cf in crop_files:
        if verify_crop_exists(cf):
            valid += 1
    return valid

def count_valid_crops_for_page(catalogue_id, page_num, crop_dir):
    """Count valid crops for one page, supporting fresh and legacy names."""
    if not crop_dir or not Path(crop_dir).exists():
        return 0
    patterns = [
        f"{catalogue_id}__pdf-{int(page_num):04d}__crop-*.jpg",
        f"{catalogue_id}_page{page_num}_*.jpg",
    ]
    return sum(1 for pattern in patterns for path in Path(crop_dir).glob(pattern) if verify_crop_exists(path))

def _fetch_iiif_page_metadata(catalogue_id):
    """Return stable Heidelberg canvas/page metadata when a manifest exists."""
    if not catalogue_id:
        return []
    try:
        url = f"https://digi.ub.uni-heidelberg.de/diglit/iiif3/{catalogue_id}/manifest"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        manifest = response.json()
        pages = []
        for index, canvas in enumerate(manifest.get("items", [])):
            canvas_id = canvas.get("id", "")
            canvas_number = canvas_id.rsplit("/", 1)[-1] or f"{index + 1:04d}"
            label = canvas.get("label", {}).get("none", [None])[0]
            image_url = None
            try:
                image_url = canvas["items"][0]["items"][0]["body"]["id"]
            except (KeyError, IndexError, TypeError):
                pass
            pages.append({
                "canvas_number": canvas_number,
                "printed_page_label": label,
                "source_page_url": f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}/{canvas_number}",
                "source_image_url": image_url,
            })
        return pages
    except Exception as exc:
        print(f"[provenance] Manifest unavailable for {catalogue_id}: {exc}")
        return []


def extract_pages_as_images(input_pdf: str, skip_first_page=True, max_pages=None, catalogue_entry=None):
    """Extract PDF pages as arrays and retain source-page provenance."""
    def resize_yolosize(pix, yolosize=640):
        long_side = max(pix.w, pix.h)
        ratio = yolosize / long_side
        return pymupdf.Pixmap(pix, int(pix.w * ratio), int(pix.h * ratio), 1)

    try:
        doc = pymupdf.open(input_pdf)
        base = Path(input_pdf).stem
        catalogue_id = (catalogue_entry or {}).get('resolved_catalogue_id') or (catalogue_entry or {}).get('fn') or base
        manifest_pages = _fetch_iiif_page_metadata(catalogue_id)
        source_url = (catalogue_entry or {}).get('uri') or f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}"
        pages = []
        start = 1 if skip_first_page else 0
        end = len(doc)
        if max_pages:
            end = min(start + max_pages, end)

        print(f"Extracting pages {start+1} to {end} from {base}")
        for i in range(start, end):
            try:
                page = doc[i]
                pix = page.get_pixmap(dpi=200)
                resized = resize_yolosize(pix)
                pil = resized.pil_image()
                arr = np.array(pil)
                source_page = manifest_pages[i] if i < len(manifest_pages) else {
                    'canvas_number': f'{i + 1:04d}',
                    'printed_page_label': None,
                    'source_page_url': f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}/{i + 1:04d}",
                    'source_image_url': None,
                }
                pages.append({
                    'page_image': arr,
                    'source_catalogue': base,
                    'catalogue_page': i,
                    'catalogue_id': catalogue_id,
                    'pdf_page_index': i,
                    'source_catalogue_url': source_url,
                    **source_page,
                })
                del pix, resized, pil
            except Exception as e:
                print(f"Error processing page {i}: {e}")

        doc.close()
        return pages

    except Exception as e:
        print(f"Error opening PDF {input_pdf}: {e}")
        return []

def cleanup_gpu_memory():
    """Force GPU memory cleanup"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_catalogue_progress(catalogue_entry, feature_dir):
    """Get progress for a catalogue"""
    if 'uri' not in catalogue_entry:
        return 0
    cat_id = catalogue_entry['uri'].split('/')[-1]
    files = list(Path(feature_dir).glob(f"{cat_id}*"))
    return len(files)

def retry_failed_downloads(download_dir, max_retries=3):
    """Retry all failed downloads"""
    downloader = get_downloader(download_dir)
    failed_downloads = failure_logger._load_log(failure_logger.download_log)
    if not failed_downloads:
        print("No failed downloads to retry")
        return 0, []

    print(f"Retrying {len(failed_downloads)} failed downloads...")
    success, still_failed = 0, []

    for f in failed_downloads:
        entry = {
            'uri': f['uri'],
            'name': f.get('name', 'Unknown'),
            'fn': f.get('filename', 'unknown').replace('.pdf', '')
        }
        file_path, error = downloader.download_catalogue(entry)
        if file_path:
            print(f"✓ Retry successful: {f['name']}")
            success += 1
        else:
            print(f"✗ Retry failed: {f['name']} - {error}")
            f['retry_error'] = error
            f['retry_timestamp'] = datetime.now().isoformat()
            still_failed.append(f)

    if still_failed:
        with open(failure_logger.download_log, 'w', encoding='utf-8') as fp:
            json.dump(still_failed, fp, indent=2, ensure_ascii=False)
        print(f"✓ {success} downloads now successful")
        print(f"✗ {len(still_failed)} downloads still failing")
    else:
        if failure_logger.download_log.exists():
            failure_logger.download_log.unlink(missing_ok=True)
        print(f"✓ All {success} failed downloads now successful!")

    return success, still_failed
