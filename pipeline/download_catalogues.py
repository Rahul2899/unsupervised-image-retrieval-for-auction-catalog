#!/usr/bin/env python3
"""
Optimized download.py with comprehensive fixes for JSON format handling
"""

import os, sys, json, time, argparse, traceback, re, random
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from bs4 import BeautifulSoup

_LAST_HTTP_REQUEST = 0.0


def polite_get(session, url, **kwargs):
    """Rate-limit requests and back off when the host asks us to slow down."""
    global _LAST_HTTP_REQUEST
    minimum_gap = float(os.getenv('AUCTION_HTTP_MIN_GAP', '1.5'))
    wait = minimum_gap - (time.monotonic() - _LAST_HTTP_REQUEST)
    if wait > 0:
        time.sleep(wait)
    response = session.get(url, **kwargs)
    _LAST_HTTP_REQUEST = time.monotonic()
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After', '10')
        try:
            time.sleep(min(120.0, max(10.0, float(retry_after))))
        except ValueError:
            time.sleep(10.0)
    return response

# Import your existing modules
sys.path.insert(0, str(Path(__file__).parent))
from preparation.bibliography_parser import BibliographyParser
from preparation.utils import (
    HeidelbergDiglitDownloader,
    FailureLogger,
    verify_pdf_exists
)

class DownloadStats:
    """Track download statistics"""
    def __init__(self, pdf_dir):
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'skipped_existing': 0,
            'skipped_prev_failed': 0,
            'skipped_collection_pages': 0,
            'native_pdfs': 0,
            'scraped_pdfs': 0,
            'total_size_mb': 0.0
        }
        self.stats_file = pdf_dir / 'download_stats.json'

    def save(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def print_summary(self):
        elapsed = (datetime.now() - datetime.fromisoformat(self.stats['start_time'])).total_seconds()
        avg_size = self.stats['total_size_mb'] / max(1, self.stats['successful'])
        print(f"\n{'='*80}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        print(f"Runtime:              {elapsed/60:.1f} minutes")
        print(f"Attempted:            {self.stats['attempted']}")
        print(f"Successful:           {self.stats['successful']}")
        print(f"  Native PDFs:        {self.stats['native_pdfs']}")
        print(f"  Scraped PDFs:       {self.stats['scraped_pdfs']}")
        print(f"Failed:               {self.stats['failed']}")
        print(f"Skipped (existing):   {self.stats['skipped_existing']}")
        print(f"Skipped (prev failed):{self.stats['skipped_prev_failed']}")
        print(f"Skipped (collections):{self.stats['skipped_collection_pages']}")
        print(f"Total size:           {self.stats['total_size_mb']:.1f} MB")
        print(f"Average size:         {avg_size:.1f} MB")
        print(f"{'='*80}\n")


def normalize_catalogue_entry(entry):
    """
    Normalize catalogue entry to consistent dict format

    Handles:
    - String URIs -> convert to dict with 'uri' key
    - Dict entries -> ensure required keys exist
    - Generate reasonable defaults for missing fields
    - Fix URIs pointing to images instead of catalogue pages

    Returns:
        dict with keys: uri, fn, name, title or None if invalid
    """
    def recover_uri_from_text_lines(lines):
        lines = lines or []
        for index, line in enumerate(lines):
            if 'digitalausgabe:' not in line.lower():
                continue
            candidate = line.split(':', 1)[1].strip()
            for continuation in lines[index + 1:index + 3]:
                candidate += continuation.strip()
                if '/diglit/' in candidate:
                    break
            match = re.search(r'https?://digi\.ub\.uni-heidelberg\.de/diglit/[A-Za-z0-9_-]+', candidate)
            if match:
                return match.group(0)
        return None

    if isinstance(entry, str):
        # String entry - assume it's a URI
        uri = entry.replace('/auktionshus_', '/auktionshaus_')
        parsed = urlparse(uri)
        path_parts = parsed.path.strip('/').split('/')

        # Try to find volume ID for filename
        volume_id = None
        if 'diglit' in path_parts:
            idx = path_parts.index('diglit')
            if idx + 1 < len(path_parts):
                volume_id = path_parts[idx + 1].split('?')[0]

        if not volume_id and path_parts:
            volume_id = path_parts[-1].split('?')[0]

        if not volume_id:
            volume_id = f"catalogue_{hash(uri) % 100000:05d}"

        return {
            'uri': uri,
            'fn': volume_id,
            'name': volume_id.replace('_', ' ').title(),
            'title': volume_id.replace('_', ' ').title()
        }

    elif isinstance(entry, dict):
        # Dict entry - ensure required keys
        normalized = {
            'uri': entry.get('uri', ''),
            'fn': entry.get('fn', ''),
            'name': entry.get('name', ''),
            'title': entry.get('title', ''),
            'doi': entry.get('doi', '')
        }

        if not re.search(r'/diglit/[A-Za-z0-9_-]+$', normalized['uri'] or ''):
            recovered = recover_uri_from_text_lines(entry.get('text_lines'))
            if recovered:
                normalized['uri'] = recovered
                normalized['fn'] = recovered.rstrip('/').rsplit('/', 1)[-1]
        if normalized['uri']:
            normalized['uri'] = normalized['uri'].replace('/auktionshus_', '/auktionshaus_')
        if normalized['fn']:
            normalized['fn'] = normalized['fn'].replace('auktionshus_', 'auktionshaus_')

        # CRITICAL FIX: Check if URI points to an image instead of catalogue page
        if normalized['uri']:
            uri_lower = normalized['uri'].lower()
            # If URI points to an image file, we need to fix it
            if any(ext in uri_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.tif']):
                # Extract catalogue ID from the image path
                parsed = urlparse(normalized['uri'])
                path_parts = parsed.path.strip('/').split('/')

                # Get filename without extension
                if path_parts:
                    filename = path_parts[-1]
                    catalogue_id = filename.rsplit('.', 1)[0]  # Remove extension

                    # Construct proper diglit URL
                    fixed_uri = f"https://digi.ub.uni-heidelberg.de/diglit/{catalogue_id}"
                    normalized['uri'] = fixed_uri

                    # Use catalogue_id as fn if fn is missing or contains image extension
                    if not normalized['fn'] or any(ext in normalized['fn'].lower() for ext in ['.jpg', '.jpeg', '.png']):
                        normalized['fn'] = catalogue_id

        # Fill in missing fields
        if not normalized['uri']:
            # Try to construct from DOI if available
            if normalized.get('doi'):
                normalized['uri'] = f"https://doi.org/{normalized['doi']}"
            else:
                # No URI and no DOI - can't download
                return None

        if not normalized['fn']:
            # Generate filename from URI or other fields
            if normalized.get('name'):
                normalized['fn'] = re.sub(r'[^\w\-]', '_', normalized['name'])[:50]
            else:
                parsed = urlparse(normalized['uri'])
                path_parts = parsed.path.strip('/').split('/')
                if 'diglit' in path_parts:
                    idx = path_parts.index('diglit')
                    if idx + 1 < len(path_parts):
                        normalized['fn'] = path_parts[idx + 1].split('?')[0]
                else:
                    normalized['fn'] = f"catalogue_{hash(normalized['uri']) % 100000:05d}"

        # Clean up fn if it still has image extension
        if normalized['fn']:
            normalized['fn'] = normalized['fn'].rsplit('.', 1)[0] if '.' in normalized['fn'] else normalized['fn']

        if not normalized['name']:
            normalized['name'] = normalized['fn'].replace('_', ' ').title()

        if not normalized['title']:
            normalized['title'] = normalized['name']

        return normalized

    else:
        return None


def check_existing_pdfs(pdf_dir, base_filename):
    """
    Check for existing PDFs with base filename or numbered variants

    Args:
        pdf_dir: Path to PDF directory
        base_filename: Base filename without extension

    Returns:
        list of existing PDF paths
    """
    existing = []

    # Check base file
    base_path = pdf_dir / f"{base_filename}.pdf"
    if verify_pdf_exists(base_path):
        existing.append(base_path)

    # Check numbered variants (bd1, bd2, etc.)
    for i in range(1, 20):
        variant_path = pdf_dir / f"{base_filename}_bd{i}.pdf"
        if verify_pdf_exists(variant_path):
            existing.append(variant_path)
        else:
            break

    return existing


def deduplicate_catalogues(catalogues):
    """Keep one record per normalized source catalogue/filename."""
    unique = []
    seen = set()
    for entry in catalogues:
        uri = (entry.get('uri') or '').rstrip('/').lower()
        fn = (entry.get('fn') or '').lower()
        key = ('uri', uri) if uri else ('fn', fn)
        if key[1] and key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def resolve_doi_to_volume_id(uri, session, timeout=10):
    """
    Resolve DOI URL to actual diglit volume ID by following redirects

    Returns:
        (volume_id, error_msg) tuple
    """
    if not uri:
        return None, "Empty URI"

    SKIP_PAGES = {
        'leuchtpult', 'diglit', 'index', 'search',
    }

    try:
        parsed = urlparse(uri)

        # If it's a DOI URL, follow the redirect
        if 'doi.org' in parsed.netloc:
            print(f"  Resolving DOI: {uri}")
            resp = polite_get(session, uri, timeout=timeout, allow_redirects=True)

            if resp.status_code == 200:
                final_url = resp.url
                print(f"  Resolved to: {final_url}")

                parsed_final = urlparse(final_url)
                path = parsed_final.path.rstrip('/')
                parts = path.split('/')

                if parts and 'diglit' in parts:
                    diglit_idx = parts.index('diglit')
                    if diglit_idx + 1 < len(parts):
                        volume_id = parts[diglit_idx + 1]
                        volume_id = volume_id.split('?')[0].split('#')[0]

                        if volume_id in SKIP_PAGES:
                            return None, f"Skipped: Virtual collection/system page ({volume_id})"

                        return volume_id, None

            return None, "Could not resolve DOI"

        # Not a DOI - extract from direct diglit URL
        path = parsed.path.rstrip('/').split('?')[0]
        parts = path.split('/')

        if 'diglit' in parts:
            diglit_idx = parts.index('diglit')
            if diglit_idx + 1 < len(parts):
                volume_id = parts[diglit_idx + 1]

                if volume_id in SKIP_PAGES:
                    return None, f"Skipped: Virtual collection/system page ({volume_id})"

                return volume_id, None

        # Fallback: just take last part
        if parts:
            volume_id = parts[-1].split('?')[0].split('#')[0]

            if volume_id in SKIP_PAGES:
                return None, f"Skipped: Virtual collection/system page ({volume_id})"

            return volume_id, None

        return None, "Could not extract volume ID from path"

    except Exception as e:
        return None, f"URL resolution error: {e}"


def get_sub_volumes(volume_id, session, timeout):
    """
    Check if the page is a parent/collection page and extract sub-volume IDs

    Returns:
        list of sub_volume_ids (if any), or [volume_id] if not a collection
    """
    url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}"
    try:
        print(f"  Checking for sub-volumes: {url}")
        resp = polite_get(session, url, timeout=timeout)
        if resp.status_code != 200:
            return [volume_id]

        soup = BeautifulSoup(resp.text, 'html.parser')

        related_section = soup.find(string=re.compile("All related items", re.I))
        if not related_section:
            print("    No related items found - single volume")
            return [volume_id]

        related_container = related_section.find_parent('div') or soup.find('div', id='relatedItems')

        if not related_container:
            return [volume_id]

        sub_volumes = []
        for a in related_container.find_all('a', href=True):
            href = a['href']
            if '/diglit/' in href:
                parsed = urlparse(href)
                parts = parsed.path.strip('/').split('/')
                if 'diglit' in parts:
                    idx = parts.index('diglit')
                    if idx + 1 < len(parts):
                        sub_id = parts[idx + 1]
                        if sub_id != volume_id:
                            sub_volumes.append(sub_id)
                            print(f"    Found sub-volume: {sub_id}")

        if sub_volumes:
            return sub_volumes
        else:
            return [volume_id]

    except Exception as e:
        print(f"    Error checking sub-volumes: {e}")
        return [volume_id]


def download_all_catalogues(catalogues, download_dir, timeout=60,
                            retry_failed_only=False, skip_existing=True,
                            skip_prev_failed=True, compress_scraped=True,
                            max_width=1600, jpeg_quality=85):
    """
    Download catalogues with smart compression and duplicate checking
    """

    # Normalize all catalogue entries first
    print("\nNormalizing catalogue entries...")
    print(f"Input: {len(catalogues)} entries")

    normalized_catalogues = []
    skipped_count = 0

    for idx, entry in enumerate(catalogues):
        normalized = normalize_catalogue_entry(entry)
        if normalized:
            normalized_catalogues.append(normalized)
            if idx == 0:  # Show first normalized entry
                print(f"\n✓ Example normalized entry:")
                print(f"  URI: {normalized['uri']}")
                print(f"  FN:  {normalized['fn']}")
                print(f"  Name: {normalized['name'][:60]}...")
        else:
            skipped_count += 1
            if skipped_count <= 3:  # Show first few failures
                entry_type = type(entry).__name__
                print(f"✗ Skipped invalid entry #{idx} (type: {entry_type})")

    catalogues = deduplicate_catalogues(normalized_catalogues)
    print(f"\n✓ Normalized: {len(catalogues)} valid catalogues")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} invalid entries")

    if not catalogues:
        print("\n✗ No valid catalogues to download!")
        return

    print(f"\n{'='*80}")
    print("PDF DOWNLOAD MODE")
    print(f"{'='*80}")
    print(f"Total catalogues:     {len(catalogues)}")
    print(f"Download directory:   {download_dir}")
    print(f"Timeout:              {timeout}s")
    print(f"Skip existing:        {skip_existing}")
    print(f"Skip prev failed:     {skip_prev_failed}")
    print(f"Retry failed only:    {retry_failed_only}")
    print(f"Compress scraped imgs:{compress_scraped}")
    if compress_scraped:
        print(f"  Max width:          {max_width}px")
        print(f"  JPEG quality:       {jpeg_quality}")
    print(f"{'='*80}\n")

    # Initialize
    failure_logger = FailureLogger()
    stats = DownloadStats(download_dir)

    # Get previously failed URIs
    prev_failed = set()
    if skip_prev_failed:
        try:
            prev_failed = set(failure_logger.get_failed_catalogues())
            print(f"Found {len(prev_failed)} previously failed downloads\n")
        except:
            print("No previous failure log found\n")

    # Filter if retry mode
    if retry_failed_only:
        if not prev_failed:
            print("No failed downloads to retry!")
            return
        catalogues = [c for c in catalogues if c.get('uri') in prev_failed]
        print(f"Retrying {len(catalogues)} failed downloads\n")

    # Create custom downloader
    downloader = HeidelbergDiglitDownloader(
        download_dir,
        skip_existing=skip_existing,
        max_retries=5
    )

    # Setup session
    downloader.session = requests.Session()
    downloader.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        retry_strategy = Retry(
            total=10,
            status_forcelist=[429, 500, 502, 503, 504, 408],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=5
        )
    except TypeError:
        retry_strategy = Retry(
            total=10,
            status_forcelist=[429, 500, 502, 503, 504, 408],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=5
        )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    downloader.session.mount("http://", adapter)
    downloader.session.mount("https://", adapter)

    # Enhanced download function
    def enhanced_download_catalogue(entry):
        """Enhanced download with better URL handling"""
        uri = entry.get('uri', '')
        base_fn = entry.get('fn', 'catalogue')
        name = entry.get('name') or entry.get('title') or uri

        print(f"  Downloading: {uri}")

        def save_pdf_atomically(response, destination):
            partial = destination.with_suffix(destination.suffix + '.part')
            try:
                with open(partial, 'wb') as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                if not verify_pdf_exists(partial):
                    return False
                os.replace(partial, destination)
                return True
            finally:
                partial.unlink(missing_ok=True)

        # Check for existing files FIRST
        existing_files = check_existing_pdfs(download_dir, base_fn)
        if existing_files and skip_existing:
            print(f"  ✓ Already exists: {len(existing_files)} file(s)")
            for f in existing_files:
                print(f"    - {f.name}")
            stats.stats['skipped_existing'] += len(existing_files)
            return existing_files, None

        # Resolve DOI or extract volume ID
        volume_id, error_msg = resolve_doi_to_volume_id(uri, downloader.session, timeout)

        if not volume_id:
            if error_msg and "Skipped:" in error_msg:
                print(f"  ⊘ {error_msg}")
                return None, error_msg
            else:
                return None, error_msg or "Could not extract volume ID"

        print(f"  Volume ID: {volume_id}")

        # Get sub-volumes
        sub_volumes = get_sub_volumes(volume_id, downloader.session, timeout)

        downloaded = []
        errors = []

        for idx, sub_id in enumerate(sub_volumes, 1):
            sub_fn = f"{base_fn}_bd{idx}" if len(sub_volumes) > 1 else base_fn
            output_path = download_dir / (sub_fn + '.pdf')

            if verify_pdf_exists(output_path):
                print(f"    ✓ Exists: {sub_id} ({output_path.name})")
                stats.stats['skipped_existing'] += 1
                downloaded.append(output_path)
                continue

            print(f"    Processing: {sub_id}")

            # Try direct PDF downloads
            pdf_urls = [
                f"https://digi.ub.uni-heidelberg.de/diglitData/pdf/{sub_id}.pdf",
                f"https://digi.ub.uni-heidelberg.de/diglit/{sub_id}/download",
                f"https://digi.ub.uni-heidelberg.de/diglit/{sub_id}/pdf",
                f"https://digi.ub.uni-heidelberg.de/diglit/{sub_id}/0000.pdf",
                f"https://digi.ub.uni-heidelberg.de/diglit/{sub_id}/download-ocr.pdf",
                f"https://digi.ub.uni-heidelberg.de/diglit/{sub_id}/download-zoom4.pdf",

            ]

            success = False
            static_pdf_missing = False
            for pdf_url in pdf_urls:
                try:
                    print(f"      Trying: {pdf_url}")
                    resp = polite_get(downloader.session,
                        pdf_url,
                        headers=downloader.headers,
                        timeout=timeout,
                        stream=True
                    )

                    if resp.status_code == 200:
                        content_type = resp.headers.get('content-type', '').lower()
                        if 'pdf' in content_type:
                            if not save_pdf_atomically(resp, output_path):
                                continue
                            size_mb = output_path.stat().st_size / (1024 * 1024)
                            print(f"      ✓ Downloaded PDF ({size_mb:.1f} MB)")
                            stats.stats['native_pdfs'] += 1
                            stats.stats['total_size_mb'] += size_mb
                            downloaded.append(output_path)
                            success = True
                            break
                    elif pdf_url == pdf_urls[0] and resp.status_code == 404:
                        # The static route is authoritative; avoid challenge-page fallbacks.
                        static_pdf_missing = True
                        break

                except Exception as e:
                    print(f"        Failed: {e}")
                    continue

            # Try IIIF manifest as fallback
            if not success and not static_pdf_missing:
                try:
                    iiif_url = f"https://digi.ub.uni-heidelberg.de/diglit/iiif/{sub_id}/manifest.json"
                    print(f"      Trying IIIF: {iiif_url}")
                    resp = polite_get(downloader.session,
                        iiif_url,
                        headers=downloader.headers,
                        timeout=timeout
                    )

                    if resp.status_code == 200:
                        manifest = resp.json()
                        rendering = manifest.get('rendering', [])
                        if isinstance(rendering, list):
                            for r in rendering:
                                if isinstance(r, dict) and 'pdf' in r.get('format', '').lower():
                                    pdf_url = r.get('@id')
                                    if pdf_url:
                                        print(f"      Found PDF in IIIF: {pdf_url}")
                                        resp = polite_get(downloader.session,
                                            pdf_url,
                                            timeout=timeout,
                                            stream=True
                                        )
                                        if resp.status_code == 200:
                                            if not save_pdf_atomically(resp, output_path):
                                                continue
                                            size_mb = output_path.stat().st_size / (1024 * 1024)
                                            print(f"      ✓ Downloaded PDF from IIIF ({size_mb:.1f} MB)")
                                            stats.stats['native_pdfs'] += 1
                                            stats.stats['total_size_mb'] += size_mb
                                            downloaded.append(output_path)
                                            success = True
                                            break
                except Exception as e:
                    print(f"        IIIF failed: {e}")

            if not success:
                errors.append(f"{sub_id}: All download methods failed")

        if downloaded:
            stats.stats['successful'] += 1
            return downloaded, None
        else:
            error = "; ".join(set(errors)) if errors else "No downloadable files"
            return None, error

    downloader.download_catalogue = enhanced_download_catalogue

    # Process catalogues
    for idx, entry in enumerate(catalogues, 1):
        try:
            uri = entry.get('uri', 'unknown')
            name = entry.get('name') or entry.get('title') or uri

            # Skip previously failed
            if skip_prev_failed and not retry_failed_only and uri in prev_failed:
                print(f"[{idx}/{len(catalogues)}] ⊘ Skipped (prev failed): {name[:60]}")
                stats.stats['skipped_prev_failed'] += 1
                continue

            print(f"\n[{idx}/{len(catalogues)}] {name[:60]}")
            stats.stats['attempted'] += 1

            # Download
            pdf_paths, error = downloader.download_catalogue(entry)

            if pdf_paths:
                print(f"  ✓ Success: {len(pdf_paths)} file(s)")
            else:
                if error and ("Skipped:" in str(error) or "collection" in str(error).lower()):
                    print(f"  ⊘ Collection page: {error}")
                    stats.stats['skipped_collection_pages'] += 1
                else:
                    print(f"  ✗ Failed: {error}")
                    stats.stats['failed'] += 1
                    try:
                        failure_logger.log_failed_download(entry, error)
                    except Exception as log_err:
                        print(f"  Warning: Could not log failure: {log_err}")

            # Save stats periodically
            if idx % 10 == 0:
                stats.save()

            # Polite pacing: one shared session, bounded retries, and jitter
            # avoid hammering the catalogue host like a tight bot loop.
            delay_min = float(os.getenv('AUCTION_DOWNLOAD_DELAY_MIN', '2.0'))
            delay_max = float(os.getenv('AUCTION_DOWNLOAD_DELAY_MAX', '5.0'))
            time.sleep(random.uniform(delay_min, max(delay_min, delay_max)))

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
            stats.save()
            stats.print_summary()
            raise
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            traceback.print_exc()
            stats.stats['failed'] += 1
            try:
                failure_logger.log_failed_download(entry, str(e))
            except:
                pass

    # Final summary
    stats.save()
    stats.print_summary()
    try:
        failure_logger.generate_summary()
    except:
        print("Note: Could not generate failure summary")

    print("\n✓ Download complete!")


def main():
    parser = argparse.ArgumentParser(description='Download auction catalogue PDFs')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--no-skip-existing', action='store_true')
    parser.add_argument('--no-skip-failed', action='store_true')
    parser.add_argument('--no-compression', action='store_true')
    parser.add_argument('--max-width', type=int, default=1600)
    parser.add_argument('--quality', type=int, default=85)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--catalogue-file', type=str, default=None)
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    data_dir = Path(os.getenv("AUCTION_RETRIEVAL_DATA_DIR", os.getcwd()))

    if args.output_dir:
        pdf_dir = Path(args.output_dir)
    else:
        pdf_dir = data_dir / 'pdfs'
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"PDF directory:  {pdf_dir}")
    print(f"Logs directory: {logs_dir}\n")

    # Load bibliography
    try:
        print("Loading bibliography...")
        if args.catalogue_file:
            catalogue_path = Path(args.catalogue_file)
            if not catalogue_path.exists():
                raise FileNotFoundError(f"Catalogue file not found: {catalogue_path}")

            with open(catalogue_path) as f:
                catalogues_data = json.load(f)

            # Handle various JSON formats
            if isinstance(catalogues_data, dict):
                # Check if it has a 'metadata' and 'catalogues' structure
                if 'metadata' in catalogues_data and 'catalogues' in catalogues_data:
                    print(f"  Detected metadata+catalogues structure")
                    metadata = catalogues_data['metadata']
                    print(f"  Generated: {metadata.get('generated_at')}")
                    print(f"  Total in file: {metadata.get('total_catalogues')}")
                    catalogues_dict = catalogues_data['catalogues']
                # Check if it has a 'catalogues' key (without metadata)
                elif 'catalogues' in catalogues_data:
                    catalogues_dict = catalogues_data['catalogues']
                # Otherwise, assume all values are catalogues
                else:
                    catalogues_dict = list(catalogues_data.values())
            elif isinstance(catalogues_data, list):
                # Check for [metadata_dict, catalogue_list] structure
                if len(catalogues_data) >= 2:
                    if isinstance(catalogues_data[0], dict) and 'generated_at' in catalogues_data[0]:
                        print(f"  Detected list with metadata structure")
                        metadata = catalogues_data[0]
                        print(f"  Generated: {metadata.get('generated_at')}")
                        print(f"  Total in file: {metadata.get('total_catalogues')}")

                        # The actual catalogues should be in the second element
                        if isinstance(catalogues_data[1], list):
                            catalogues_dict = catalogues_data[1]
                        else:
                            # Fallback: find first list
                            for item in catalogues_data:
                                if isinstance(item, list):
                                    catalogues_dict = item
                                    break
                            else:
                                catalogues_dict = [c for c in catalogues_data
                                                 if isinstance(c, (dict, str)) and
                                                 not (isinstance(c, dict) and c.get('generated_at'))]
                    else:
                        catalogues_dict = catalogues_data
                else:
                    catalogues_dict = catalogues_data
            else:
                raise ValueError(f"Unexpected format: {type(catalogues_data)}")

            print(f"✓ Loaded {len(catalogues_dict)} catalogues from {catalogue_path}")
        else:
            parser_obj = BibliographyParser(cache_dir=data_dir, oai_delay=1.0)
            catalogues_dict = parser_obj.parse(
                use_cached=True,
                include_pdfs=True,
                include_oai=True,
                validate_uris=False,
                check_online=False
            )
            print(f"✓ Loaded {len(catalogues_dict)} catalogues")

    except Exception as e:
        print(f"✗ Failed to load bibliography: {e}")
        traceback.print_exc()
        return 1

    # Download
    try:
        download_all_catalogues(
            catalogues_dict,
            pdf_dir,
            timeout=args.timeout,
            retry_failed_only=args.retry_failed,
            skip_existing=not args.no_skip_existing,
            skip_prev_failed=not args.no_skip_failed,
            compress_scraped=not args.no_compression,
            max_width=args.max_width,
            jpeg_quality=args.quality
        )
    except KeyboardInterrupt:
        print("\n✓ Progress saved. Resume anytime.")
        return 0
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
