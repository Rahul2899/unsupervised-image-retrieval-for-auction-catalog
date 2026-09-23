# bibliography_parser.py - With collection page expansion

import pikepdf
import json
from pathlib import Path
from pdfminer.high_level import extract_text
import requests
import re
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class CollectionExpander:
    """Expands collection pages into individual catalogue URIs"""

    def __init__(self, session=None, delay=0.5):
        self.session = session or requests.Session()
        self.delay = delay
        self.cache = {}  # Cache collection page results

    def is_collection_page(self, volume_id):
        """
        Check if a volume is a collection page (≤3 pages)
        Returns: (is_collection, page_count)
        """
        try:
            mets_url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}/mets"
            time.sleep(self.delay)
            resp = self.session.get(mets_url, timeout=120)

            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns = {'mets': 'http://www.loc.gov/METS/'}
                files = root.findall('.//mets:file', ns)
                page_count = len(files)

                # Collection pages typically have ≤3 pages
                return (page_count <= 3, page_count)
        except Exception as e:
            print(f"  Warning: Could not check page count for {volume_id}: {e}")

        return (False, None)

    def extract_catalogue_links(self, volume_id):
        """
        Extract individual catalogue URIs from a collection page
        Returns: List of volume IDs (not full URIs)
        """
        if volume_id in self.cache:
            return self.cache[volume_id]

        try:
            page_url = f"https://digi.ub.uni-heidelberg.de/diglit/{volume_id}"
            time.sleep(self.delay)
            resp = self.session.get(page_url, timeout=120)

            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find all links to other diglit volumes
            catalogue_ids = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')

                # Match patterns like /diglit/achenbach1933_05_12
                if '/diglit/' in href and href != f'/diglit/{volume_id}':
                    parts = href.split('/diglit/')
                    if len(parts) > 1:
                        linked_id = parts[1].rstrip('/').split('/')[0]

                        # Filter out non-catalogue links
                        if linked_id and not any(x in linked_id.lower() for x in
                                                ['download', 'mets', 'iiif', 'manifest', 'image', 'thumb']):
                            catalogue_ids.append(linked_id)

            # Remove duplicates, preserve order
            seen = set()
            unique_ids = []
            for cid in catalogue_ids:
                if cid not in seen:
                    seen.add(cid)
                    unique_ids.append(cid)

            self.cache[volume_id] = unique_ids
            return unique_ids

        except Exception as e:
            print(f"  Warning: Could not extract links from {volume_id}: {e}")
            return []

    def expand_if_collection(self, volume_id, base_entry=None):
        """
        Check if volume is a collection page and expand it

        Args:
            volume_id: The diglit volume ID
            base_entry: Optional base catalogue entry to copy metadata from

        Returns:
            List of catalogue entries (original if not collection, expanded if collection)
        """
        is_collection, page_count = self.is_collection_page(volume_id)

        if not is_collection:
            # Not a collection, return original
            return [base_entry] if base_entry else []

        # It's a collection page - extract individual catalogues
        print(f"  ✓ Collection page detected: {volume_id} ({page_count} pages)")
        catalogue_ids = self.extract_catalogue_links(volume_id)

        if not catalogue_ids:
            print(f"  ⚠ No catalogues found in collection: {volume_id}")
            return []

        print(f"  ✓ Found {len(catalogue_ids)} catalogues: {', '.join(catalogue_ids[:3])}{'...' if len(catalogue_ids) > 3 else ''}")

        # Create entry for each catalogue
        entries = []
        for cat_id in catalogue_ids:
            # Copy base entry metadata if provided
            if base_entry:
                new_entry = base_entry.copy()
                # Update URI and fn
                new_entry['uri'] = f"https://digi.ub.uni-heidelberg.de/diglit/{cat_id}"
                new_entry['fn'] = cat_id
                # Add note about source
                if 'text_lines' in new_entry:
                    new_entry['text_lines'] = new_entry['text_lines'].copy()
                    new_entry['text_lines'].append(f"From collection: {volume_id}")
                entries.append(new_entry)
            else:
                # Create minimal entry
                entries.append({
                    'uri': f"https://digi.ub.uni-heidelberg.de/diglit/{cat_id}",
                    'fn': cat_id,
                    'title': f"Catalogue {cat_id}",
                    'text_lines': [f"From collection: {volume_id}"],
                    'location': '',
                    'year': -1,
                    'types': []
                })

        return entries


class URIValidator:
    """Validates and attempts to correct malformed Heidelberg digital library URIs"""

    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.base_pattern = r'^https?://digi\.ub\.uni-heidelberg\.de/diglit/[a-zA-Z0-9_\-]+$'
        self.doi_pattern = r'^https?://doi\.org/10\.11588/diglit\.[0-9]+$'

    def is_valid_format(self, uri):
        """Check if URI matches expected format"""
        if not uri:
            return False
        return bool(re.match(self.base_pattern, uri) or re.match(self.doi_pattern, uri))

    def is_complete(self, uri):
        """Check if URI is complete (not truncated)"""
        if not uri:
            return False
        parsed = urlparse(uri)
        return bool(parsed.scheme and parsed.netloc and parsed.path and parsed.path != '/diglit/')

    def fix_common_errors(self, uri):
        """Attempt to fix common URI errors"""
        if not uri:
            return None

        # Fix incomplete URIs
        if uri == 'http://digi.ub.uni-' or uri.endswith('digi.ub.uni-'):
            return None  # Cannot recover

        # Fix typos
        corrections = {
            'auktionshus': 'auktionshaus',
            'auctionshaus': 'auktionshaus',
        }

        for wrong, right in corrections.items():
            if wrong in uri:
                uri = uri.replace(wrong, right)

        # Ensure https
        if uri.startswith('http://'):
            uri = uri.replace('http://', 'https://', 1)

        # Remove trailing slashes
        uri = uri.rstrip('/')

        return uri

    def check_availability(self, uri, timeout=60):
        """Check if URI is accessible"""
        try:
            response = self.session.head(uri, timeout=timeout, allow_redirects=True)
            return response.status_code == 200
        except:
            return False

    def validate_and_fix(self, uri, check_online=False):
        """
        Validate URI and attempt corrections
        Returns: (is_valid, corrected_uri, error_message)
        """
        if not uri:
            return False, None, "Empty URI"

        original_uri = uri

        # Try to fix common errors
        uri = self.fix_common_errors(uri)
        if not uri:
            return False, None, f"Cannot fix incomplete URI: {original_uri}"

        # Check format
        if not self.is_valid_format(uri):
            return False, uri, f"Invalid URI format: {uri}"

        # Check completeness
        if not self.is_complete(uri):
            return False, uri, f"Incomplete URI: {uri}"

        # Optionally check online availability
        if check_online:
            if not self.check_availability(uri):
                return False, uri, f"URI not accessible: {uri}"

        return True, uri, None

    def validate_catalogue(self, catalogues, fix_errors=True, check_online=False):
        """
        Validate all URIs in catalogue list
        Returns: (valid_catalogues, invalid_catalogues, stats)
        """
        valid = []
        invalid = []
        stats = {
            'total': len(catalogues),
            'valid': 0,
            'fixed': 0,
            'unfixable': 0,
            'missing': 0
        }

        for cat in catalogues:
            uri = cat.get('uri')

            if not uri:
                stats['missing'] += 1
                invalid.append({
                    'catalogue': cat,
                    'error': 'Missing URI',
                    'original_uri': None
                })
                continue

            is_valid, corrected_uri, error = self.validate_and_fix(uri, check_online)

            if is_valid:
                if corrected_uri != uri and fix_errors:
                    cat['uri'] = corrected_uri
                    cat['fn'] = self._extract_fn(corrected_uri)
                    stats['fixed'] += 1
                stats['valid'] += 1
                valid.append(cat)
            else:
                stats['unfixable'] += 1
                invalid.append({
                    'catalogue': cat,
                    'error': error,
                    'original_uri': uri,
                    'corrected_uri': corrected_uri
                })

        return valid, invalid, stats

    def _extract_fn(self, uri):
        """Extract filename from URI"""
        fn = uri.split('/')[-1]
        return fn.replace('.', '_')

    def generate_report(self, invalid_catalogues):
        """Generate human-readable report of invalid URIs"""
        report = []
        report.append("=" * 80)
        report.append("INVALID URI REPORT")
        report.append("=" * 80)

        for item in invalid_catalogues:
            cat = item['catalogue']
            report.append(f"\nTitle: {cat.get('title', 'N/A')}")
            report.append(f"Location: {cat.get('location', 'N/A')}")
            report.append(f"Year: {cat.get('year', 'N/A')}")
            report.append(f"Original URI: {item['original_uri']}")
            if item.get('corrected_uri'):
                report.append(f"Attempted correction: {item['corrected_uri']}")
            report.append(f"Error: {item['error']}")
            report.append("-" * 80)

        return "\n".join(report)


class BibliographyParser:
    # Bibliographies are source data, not code. Pass them explicitly or use
    # --catalogue-file in download_catalogues.py.
    DEFAULT_BIBLIOGRAPHIES = []

    DEFAULT_OAI_SOURCES = [
        'https://digi.ub.uni-heidelberg.de/cgi-bin/digioai.cgi?verb=ListIdentifiers&metadataPrefix=mets&set=sammlung20',
        'https://digi.ub.uni-heidelberg.de/cgi-bin/digioai.cgi?verb=ListIdentifiers&metadataPrefix=mets&set=sammlung27',
        'https://digi.ub.uni-heidelberg.de/cgi-bin/digioai.cgi?verb=ListIdentifiers&metadataPrefix=mets&set=sammlung49',
        'https://digi.ub.uni-heidelberg.de/cgi-bin/digioai.cgi?verb=ListIdentifiers&metadataPrefix=mets&set=sammlung99',
        'https://digi.ub.uni-heidelberg.de/cgi-bin/digioai.cgi?verb=ListIdentifiers&metadataPrefix=mets&set=sammlung102',
        'https://digi.ub.uni-heidelberg.de/cgi-bin/digioai.cgi?verb=ListIdentifiers&metadataPrefix=mets&set=sammlung110'
    ]

    def _save_incremental(self):
        out_path = Path(self.cache_dir) / self.cache_file_name
        tmp_path = out_path.with_suffix(f'.tmp.{os.getpid()}')
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.catalogues, f, ensure_ascii=False, indent=4)
            tmp_path.replace(out_path)
            print(f"✓ Progress saved: {out_path} ({len(self.catalogues)} catalogues)")
        except Exception as e:
            print(f"⚠️  Error saving cache to {out_path}: {e}")
            # Keep the temp file for debugging
            if tmp_path.exists():
                backup_tmp = out_path.with_suffix('.tmp.failed')
                try:
                    tmp_path.rename(backup_tmp)
                    print(f"   Failed write saved to: {backup_tmp}")
                except:
                    pass
        finally:
            # Clean up only this process's temp files
            if tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except:
                    pass

    def _load_cached(self):
        """Load cached catalogues if available"""
        cache_path = Path(self.cache_dir) / self.cache_file_name
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def __init__(self, lit_pdfs=None, page_ranges=None, oai_sources=None, cache_dir='.', oai_delay=0.5, expand_collections=True):
        """
        Args:
            expand_collections: If True, automatically expand collection pages into individual catalogues
        """
        if lit_pdfs is None or page_ranges is None:
            if not self.DEFAULT_BIBLIOGRAPHIES:
                raise ValueError("Pass lit_pdfs/page_ranges or use a prepared --catalogue-file")
            self.lit_pdfs, self.page_ranges = zip(*self.DEFAULT_BIBLIOGRAPHIES)
        else:
            self.lit_pdfs = lit_pdfs
            self.page_ranges = page_ranges

        if oai_sources is None:
            self.oai_sources = self.DEFAULT_OAI_SOURCES
        else:
            self.oai_sources = oai_sources

        self.catalogues = []
        self.current_entry = {}
        self.cache_dir = cache_dir
        self.cache_file_name = 'catalogues_dict.json'
        self.progress_file_name = 'parse_progress.json'
        self.oai_delay = oai_delay
        self.expand_collections = expand_collections

        # URI tracking for deduplication
        self.seen_uris = set()
        self.duplicate_stats = {
            'duplicates_found': 0,
            'duplicates_by_source': {},
            'collections_expanded': 0,
            'catalogues_from_collections': 0
        }

        # Setup for OAI-PMH parsing
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.namespaces = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'mets': 'http://www.loc.gov/METS/',
            'mods': 'http://www.loc.gov/mods/v3',
            'xlink': 'http://www.w3.org/1999/xlink',
            'dv': 'http://dfg-viewer.de/'
        }

        # Initialize URI validator and collection expander
        self.uri_validator = URIValidator(session=self.session)
        self.collection_expander = CollectionExpander(session=self.session, delay=oai_delay)

    # ==================== Deduplication Methods ====================

    def _normalize_uri(self, uri):
        """Normalize URI for comparison (lowercase, no trailing slash, https)"""
        if not uri:
            return None
        uri = uri.strip().lower()
        uri = uri.rstrip('/')
        # Normalize to https
        if uri.startswith('http://'):
            uri = uri.replace('http://', 'https://', 1)
        return uri

    def _is_duplicate(self, uri):
        """Check if URI already exists in catalogues"""
        normalized = self._normalize_uri(uri)
        if not normalized:
            return False
        return normalized in self.seen_uris

    def _mark_uri_seen(self, uri):
        """Mark URI as seen for deduplication"""
        normalized = self._normalize_uri(uri)
        if normalized:
            self.seen_uris.add(normalized)

    def _add_catalogue_with_dedup(self, entry, source='unknown', expand=True):
        """
        Add catalogue entry with deduplication and optional collection expansion

        Args:
            entry: Catalogue entry dict
            source: Source identifier for tracking
            expand: If True, check if this is a collection page and expand it
        """
        uri = entry.get('uri')

        if not uri:
            # Entries without URI are always added
            self.catalogues.append(entry)
            return True

        if self._is_duplicate(uri):
            # Track duplicate statistics
            self.duplicate_stats['duplicates_found'] += 1
            if source not in self.duplicate_stats['duplicates_by_source']:
                self.duplicate_stats['duplicates_by_source'][source] = 0
            self.duplicate_stats['duplicates_by_source'][source] += 1

            print(f"[DUPLICATE] Skipping {uri} (already exists)")
            return False

        # Check if collection expansion is enabled
        if expand and self.expand_collections:
            volume_id = uri.rstrip('/').split('/')[-1]
            expanded_entries = self.collection_expander.expand_if_collection(volume_id, entry)

            if len(expanded_entries) > 1:
                # Collection was expanded
                self.duplicate_stats['collections_expanded'] += 1
                self.duplicate_stats['catalogues_from_collections'] += len(expanded_entries)

                # Add all expanded entries
                added_count = 0
                for exp_entry in expanded_entries:
                    exp_uri = exp_entry.get('uri')
                    if exp_uri and not self._is_duplicate(exp_uri):
                        self.catalogues.append(exp_entry)
                        self._mark_uri_seen(exp_uri)
                        added_count += 1

                print(f"  ✓ Added {added_count}/{len(expanded_entries)} catalogues from collection")
                return added_count > 0
            elif len(expanded_entries) == 1:
                # Not a collection, add normally
                self.catalogues.append(expanded_entries[0])
                self._mark_uri_seen(uri)
                return True
            else:
                # Expansion failed, skip
                return False
        else:
            # No expansion, add normally
            self.catalogues.append(entry)
            self._mark_uri_seen(uri)
            return True

    def _initialize_seen_uris(self):
        """Initialize seen URIs from existing catalogues (for resume functionality)"""
        self.seen_uris.clear()
        for entry in self.catalogues:
            uri = entry.get('uri')
            if uri:
                self._mark_uri_seen(uri)
        print(f"Initialized deduplication tracker with {len(self.seen_uris)} existing URIs")

    # ==================== Progress Tracking Methods ====================

    def _load_progress(self):
        """Load parsing progress to skip already-processed sources"""
        progress_path = Path(self.cache_dir, self.progress_file_name)
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                return json.load(f)
        return {'completed_oai_sets': [], 'completed_pdfs': []}

    def _save_progress(self, progress):
        """Save parsing progress"""
        progress_path = Path(self.cache_dir, self.progress_file_name)
        with open(progress_path, 'w') as f:
            json.dump(progress, f, indent=2)

    def _save_incremental(self):
        """Save catalogues incrementally after each source"""
        out_path = Path(self.cache_dir) / self.cache_file_name
        tmp_path = out_path.with_suffix(f'.tmp.{os.getpid()}')
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.catalogues, f, ensure_ascii=False, indent=4)
            tmp_path.replace(out_path)
            print(f"✓ Progress saved: {out_path} ({len(self.catalogues)} catalogues)")

        except Exception as e:
            print(f"⚠️  Error saving cache to {out_path}: {e}")
            # Keep the temp file for debugging
            if tmp_path.exists():
                backup_tmp = out_path.with_suffix('.tmp.failed')
                try:
                    tmp_path.rename(backup_tmp)
                    print(f"   Failed write saved to: {backup_tmp}")
                except:
                    pass
        finally:
            # Clean up only this process's temp files
            if tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except:
                    pass

    def _get_oai_set_name(self, oai_url):
        """Extract set name from OAI URL"""
        set_name = re.search(r'set=([^&]+)', oai_url)
        return set_name.group(1) if set_name else oai_url

    # ==================== PDF Bibliography Parsing Methods ====================
    # (Keep all existing PDF parsing methods unchanged)

    @staticmethod
    def _is_header(line):
        return bool(re.search(r'<[^>]+', line))

    @staticmethod
    def _has_link(line):
        return 'http://' in line or 'https://' in line

    @staticmethod
    def _has_type(line):
        return 'Lose' in line and '; ' in line

    @staticmethod
    def _sanitize_kw(kw):
        return kw.strip().replace('\xa0', '').replace(',', '')

    @classmethod
    def _get_types(cls, line):
        kws_raw = line.split('; ')[1].split(', ')
        return [cls._sanitize_kw(kw) for kw in kws_raw] if kws_raw else []

    @staticmethod
    def _get_link(line):
        if 'https://' in line:
            return 'https://' + line.split('https://')[-1]
        if 'http://' in line:
            return 'http://' + line.split('http://')[-1]
        return None

    @staticmethod
    def _get_fn(uri):
        fn = uri.split('/')[-1]
        return fn.replace('.','_')

    @staticmethod
    def _extract_location(line):
        location = line.split('<')[1].split('>')[0].strip()
        return (location.replace(' ', '')
                        .replace('imBreisgau', '')
                        .replace('a.M.', ' am Main')
                        .replace('\xa0a.\xa0M.', ' am Main')
                        .replace('amMain', ' am Main'))

    @staticmethod
    def _extract_date(line):
        dates = re.findall(r'[12]\d{3}', line)
        return int(dates[0]) if len(dates) == 1 else -1

    @classmethod
    def _has_year(cls, text):
        return cls._extract_date(text) != -1

    @classmethod
    def _start_entry(cls, title):
        return {
            'title': title,
            'text_lines': [],
            'location': cls._extract_location(title),
            'year': cls._extract_date(title),
            'types': []
        }

    @staticmethod
    def _is_valid_entry(entry):
        return 'title' in entry

    def _save_entry(self, entry, source='pdf'):
        """Save entry with deduplication"""
        if self._is_valid_entry(entry):
            self._add_catalogue_with_dedup(entry, source, expand=True)

    def _fill_entry(self, entry, text):
        if not self._is_valid_entry(entry):
            return entry
        if text == '':
            return entry

        entry['text_lines'].append(text)

        if self._has_link(text):
            link = self._get_link(text)
            if link:
                entry['uri'] = link
                entry['fn'] = self._get_fn(link)

        if self._has_type(text):
            entry['types'] = self._get_types(text)

        if entry['year'] == '' and self._has_year(text):
            entry['year'] = self._extract_date(text)

        return entry

    def parse_pdf_batchwise(self, pdf_path, start_page, end_page, progress=None):
        pdf_name = Path(pdf_path).name

        # Skip if already completed
        if progress and pdf_name in progress.get('completed_pdfs', []):
            print(f"Skipping {pdf_name} (already completed)")
            return

        # Check if file exists
        if not Path(pdf_path).exists():
            print(f"Skipping {pdf_name}: File not found at {pdf_path}")
            return

        text = extract_text(
            pdf_path,
            page_numbers=list(range(start_page, end_page)),
            laparams=None
        )

        lines = text.splitlines()
        for line in tqdm(lines, desc=f"Parsing {pdf_name}"):
            if self._is_header(line):
                self._save_entry(self.current_entry, source=f'pdf_{pdf_name}')
                self.current_entry = self._start_entry(line)
            else:
                self.current_entry = self._fill_entry(self.current_entry, line)

        self._save_entry(self.current_entry, source=f'pdf_{pdf_name}')

        # Mark as completed and save progress
        if progress is not None:
            progress['completed_pdfs'].append(pdf_name)
            self._save_progress(progress)
            self._save_incremental()
            print(f"Completed {pdf_name}, progress saved")

    # ==================== OAI-PMH Parsing Methods ====================
    # (Keep all existing OAI parsing methods unchanged, they call _add_catalogue_with_dedup which now handles expansion)

    def fetch_response(self, url, timeout=60):
        """Fetch and return response, handling errors"""
        try:
            time.sleep(self.oai_delay)
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_identifiers(self, oai_url):
        """Extract all record identifiers from ListIdentifiers response"""
        identifiers = []
        current_url = oai_url

        while current_url:
            response = self.fetch_response(current_url)
            if response is None:
                break

            try:
                root = ET.fromstring(response.content)
            except:
                break

            for identifier in root.findall('.//oai:identifier', self.namespaces):
                identifiers.append(identifier.text)

            resumption_token = root.find('.//oai:resumptionToken', self.namespaces)
            if resumption_token is not None and resumption_token.text:
                base_url = oai_url.split('?')[0]
                current_url = f"{base_url}?verb=ListIdentifiers&resumptionToken={resumption_token.text}"
            else:
                current_url = None

        return identifiers

    def get_record_metadata(self, base_url, identifier):
        """Fetch full record metadata using GetRecord"""
        url = f"{base_url}?verb=GetRecord&metadataPrefix=mets&identifier={identifier}"
        response = self.fetch_response(url)
        if response is None:
            return None
        try:
            return ET.fromstring(response.content)
        except:
            return None

    def extract_diglit_id_from_mets(self, root):
        """Extract the actual diglit ID from METS metadata"""
        ns = self.namespaces

        # Check for dv:links
        dv_links = root.find('.//dv:links', ns)
        if dv_links is not None:
            presentation = dv_links.find('dv:presentation', ns)
            if presentation is not None and presentation.text:
                match = re.search(r'/diglit/([^/\s]+)', presentation.text)
                if match:
                    return match.group(1)

        # Fallback: check file pointers
        file_sec = root.find('.//mets:fileSec', ns)
        if file_sec is not None:
            for file_grp in file_sec.findall('mets:fileGrp', ns):
                for file_elem in file_grp.findall('mets:file', ns):
                    flocat = file_elem.find('mets:FLocat', ns)
                    if flocat is not None:
                        href = flocat.get('{http://www.w3.org/1999/xlink}href')
                        if href and 'diglit' in href:
                            match = re.search(r'/diglit/([^/]+)/', href)
                            if match:
                                return match.group(1)

        return None

    def extract_oai_metadata(self, root):
        """Extract comprehensive metadata from MODS"""
        ns = self.namespaces
        metadata = {}

        title = root.find('.//mods:title', ns)
        metadata['title'] = title.text.strip() if title is not None and title.text else ''

        subtitle = root.find('.//mods:subTitle', ns)
        metadata['subtitle'] = subtitle.text.strip() if subtitle is not None and subtitle.text else ''

        date = root.find('.//mods:dateIssued', ns)
        metadata['date'] = date.text.strip() if date is not None and date.text else ''

        creators = []
        for name_elem in root.findall('.//mods:name', ns):
            name_part = name_elem.find('mods:namePart', ns)
            if name_part is not None and name_part.text:
                creators.append(name_part.text.strip())
        metadata['creators'] = creators

        publisher_elem = root.find('.//mods:publisher', ns)
        metadata['publisher'] = publisher_elem.text.strip() if publisher_elem is not None and publisher_elem.text else ''

        place_term = root.find('.//mods:placeTerm', ns)
        metadata['place'] = place_term.text.strip() if place_term is not None and place_term.text else ''

        extent = root.find('.//mods:extent', ns)
        metadata['extent'] = extent.text.strip() if extent is not None and extent.text else ''

        notes = []
        for note in root.findall('.//mods:note', ns):
            if note.text:
                notes.append(note.text.strip())
        metadata['notes'] = notes

        subjects = []
        for subject in root.findall('.//mods:subject/mods:topic', ns):
            if subject.text:
                subjects.append(subject.text.strip())
        metadata['subjects'] = subjects

        return metadata

    def count_pages_from_mets(self, root):
        """Count pages from METS structure"""
        ns = self.namespaces
        struct_map = root.find('.//mets:structMap[@TYPE="PHYSICAL"]', ns)
        if struct_map is None:
            struct_map = root.find('.//mets:structMap[@TYPE="LOGICAL"]', ns)
        if struct_map is None:
            struct_map = root.find('.//mets:structMap', ns)

        if struct_map is None:
            return 0

        page_count = 0
        for div in struct_map.findall('.//mets:div', ns):
            fptrs = div.findall('mets:fptr', ns)
            if fptrs:
                page_count += 1

        return page_count

    def build_oai_title(self, metadata):
        """Build title in format: Auctioneer <Location> / Date"""
        creators = metadata.get('creators', [])
        place = metadata.get('place', '')
        date = metadata.get('date', '')

        creator_parts = []
        for creator in creators:
            if place:
                creator_parts.append(f"{creator} <{place}>")
            else:
                creator_parts.append(creator)

        creator_str = " ; ".join(creator_parts) if creator_parts else "Unknown"

        if date:
            return f"{creator_str} / {date}"
        else:
            return creator_str

    def build_oai_text_lines(self, metadata, doc_id, page_count, uri):
        """Build detailed text_lines array"""
        text_lines = []

        title = metadata.get('title', '')
        subtitle = metadata.get('subtitle', '')
        if title:
            desc = title
            if subtitle:
                desc += f" : {subtitle}"

            creators = metadata.get('creators', [])
            if creators:
                desc += f" / {'; '.join(creators)}"

            place = metadata.get('place', '')
            date = metadata.get('date', '')
            if place or date:
                desc += ", "
                if place:
                    desc += place
                if date:
                    if place:
                        desc += " "
                    desc += date

            extent = metadata.get('extent', '')
            if extent:
                desc += f". - {extent}"

            text_lines.append(desc)

        if date:
            place = metadata.get('place', '')
            text_lines.append(f"Versteigerung: {place}, {date}" if place else f"Datum: {date}")

        subjects = metadata.get('subjects', [])
        if subjects:
            text_lines.append(f"{', '.join(subjects)}")

        notes = metadata.get('notes', [])
        for note in notes:
            text_lines.append(note)

        if page_count:
            text_lines.append(str(page_count))

        text_lines.append(f"Digitalausgabe: {uri}")

        return text_lines

    def extract_oai_location(self, metadata):
        """Extract location from metadata"""
        place = metadata.get('place', '')
        if place:
            place = place.strip()
            place = re.sub(r'\[.*?\]', '', place).strip()
            return place
        return ''

    def extract_oai_year(self, metadata):
        """Extract year from date field"""
        date_str = metadata.get('date', '')
        if not date_str:
            return -1

        years = re.findall(r'[12]\d{3}', date_str)
        if years:
            return int(years[0])

        return -1

    def extract_oai_types(self, metadata):
        """Extract specific object types from subjects and notes"""
        types = []

        subjects = metadata.get('subjects', [])
        for subject in subjects:
            if any(word in subject.lower() for word in ['gemälde', 'malerei', 'painting']):
                types.append('Gemälde')
            if any(word in subject.lower() for word in ['skulptur', 'plastik', 'sculpture']):
                types.append('Skulptur')
            if any(word in subject.lower() for word in ['möbel', 'furniture']):
                types.append('Möbel')
            if any(word in subject.lower() for word in ['textil', 'textile']):
                types.append('Textilien')
            if any(word in subject.lower() for word in ['graphik', 'druckgraphik', 'print']):
                types.append('Graphik')
            if any(word in subject.lower() for word in ['zeichnung', 'drawing']):
                types.append('Zeichnungen')
            if any(word in subject.lower() for word in ['buch', 'bücher', 'book']):
                types.append('Bücher')

        notes = metadata.get('notes', [])
        for note in notes:
            note_lower = note.lower()
            if 'lose' in note_lower:
                match = re.search(r'lose[;:]?\s*([^;]+)', note_lower, re.IGNORECASE)
                if match:
                    type_text = match.group(1)
                    for t in re.split(r'[,;]', type_text):
                        t = t.strip().capitalize()
                        if t and len(t) > 3:
                            types.append(t)

        seen = set()
        unique_types = []
        for t in types:
            if t not in seen:
                seen.add(t)
                unique_types.append(t)

        return unique_types

    def process_oai_record(self, identifier, record_root):
        """Process a single OAI record"""
        doc_id = self.extract_diglit_id_from_mets(record_root)

        if not doc_id:
            match = re.search(r'([^:]+)$', identifier)
            doc_id = match.group(1) if match else None

        if not doc_id:
            return None

        metadata = self.extract_oai_metadata(record_root)
        page_count = self.count_pages_from_mets(record_root)
        uri = f"https://digi.ub.uni-heidelberg.de/diglit/{doc_id}"

        catalogue_entry = {
            'title': self.build_oai_title(metadata),
            'text_lines': self.build_oai_text_lines(metadata, doc_id, page_count, uri),
            'location': self.extract_oai_location(metadata),
            'year': self.extract_oai_year(metadata),
            'types': self.extract_oai_types(metadata),
            'uri': uri,
            'fn': doc_id
        }

        return catalogue_entry

    def parse_oai_set(self, oai_url, limit=None, progress=None):
        """Process all records from an OAI set with deduplication and collection expansion"""
        set_name = re.search(r'set=([^&]+)', oai_url)
        set_name = set_name.group(1) if set_name else oai_url

        # Skip if already completed
        if progress and set_name in progress.get('completed_oai_sets', []):
            print(f"Skipping {set_name} (already completed)")
            return

        base_url = oai_url.split('?')[0]

        print(f"Fetching identifiers from {oai_url}...")
        identifiers = self.extract_identifiers(oai_url)

        if not identifiers:
            print("No records found!")
            return

        if limit:
            identifiers = identifiers[:limit]

        print(f"Processing {len(identifiers)} OAI records...")

        added_count = 0
        duplicate_count = 0

        for identifier in tqdm(identifiers, desc="Parsing OAI metadata"):
            record_root = self.get_record_metadata(base_url, identifier)
            if record_root is None:
                continue

            catalogue_entry = self.process_oai_record(identifier, record_root)
            if catalogue_entry:
                # This now automatically handles collection expansion
                was_added = self._add_catalogue_with_dedup(catalogue_entry, source=f'oai_{set_name}', expand=True)
                if was_added:
                    added_count += 1
                else:
                    duplicate_count += 1

        print(f"OAI set {set_name}: Added {added_count} new entries, skipped {duplicate_count} duplicates")

        # Mark as completed and save progress
        if progress is not None:
            progress['completed_oai_sets'].append(set_name)
            self._save_progress(progress)
            self._save_incremental()
            print(f"Completed {set_name}, progress saved")

    # ==================== URI Validation Methods ====================

    def validate_and_clean_catalogues(self, check_online=False):
        """Validate and clean catalogue URIs"""
        if not self.catalogues:
            print("No catalogues to validate")
            return [], [], {}

        print(f"\nValidating {len(self.catalogues)} catalogue URIs...")
        valid, invalid, stats = self.uri_validator.validate_catalogue(
            self.catalogues,
            fix_errors=True,
            check_online=check_online
        )

        print(f"\n{'='*80}")
        print("URI Validation Results:")
        print(f"{'='*80}")
        print(f"Total catalogues:    {stats['total']}")
        print(f"Valid URIs:          {stats['valid']}")
        print(f"Auto-fixed:          {stats['fixed']}")
        print(f"Unfixable errors:    {stats['unfixable']}")
        print(f"Missing URIs:        {stats['missing']}")
        print(f"{'='*80}")

        if invalid:
            report = self.uri_validator.generate_report(invalid)
            report_path = Path(self.cache_dir, 'invalid_uris_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n⚠ Invalid URI report saved to: {report_path}")

            # Save invalid catalogues as separate JSON
            invalid_json_path = Path(self.cache_dir, 'invalid_catalogues.json')
            with open(invalid_json_path, 'w', encoding='utf-8') as f:
                json.dump(invalid, f, ensure_ascii=False, indent=2)
            print(f"⚠ Invalid catalogues saved to: {invalid_json_path}")

        # Update catalogues with validated ones
        self.catalogues = valid

        # Rebuild seen_uris after validation (URIs may have changed)
        self._initialize_seen_uris()

        # Save validated catalogues
        if valid:
            validated_path = Path(self.cache_dir, 'catalogues_dict_validated.json')
            with open(validated_path, 'w', encoding='utf-8') as f:
                json.dump(valid, f, ensure_ascii=False, indent=4)
            print(f"✓ Validated catalogues saved to: {validated_path}")

        return valid, invalid, stats

    # ==================== Main Parsing Methods ====================

    def write(self, out_dir):
        """Write with atomic save to prevent corruption"""
        out_path = Path(out_dir) / self.cache_file_name
        tmp_path = out_path.with_suffix(f'.tmp.{os.getpid()}')

        try:
            # Ensure directory exists
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.catalogues, f, ensure_ascii=False, indent=4)

            # Atomic replace
            tmp_path.replace(out_path)
            print(f'✓ Catalogue dict written to {out_path}')
            print(f'  Total catalogues: {len(self.catalogues)}')

        except Exception as e:
            print(f'⚠️  Error writing catalogue dict to {out_path}: {e}')
            if tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except:
                    pass

    def parse(self, use_cached=True, include_pdfs=True, include_oai=True, oai_limit=None, validate_uris=True, check_online=False):
        """
        Parse bibliographies from both PDF sources and OAI-PMH sources with deduplication and collection expansion
        OAI sources are parsed first, then PDF bibliographies
        Supports incremental parsing with progress tracking

        Args:
            use_cached: Load from cache if available
            include_pdfs: Parse PDF bibliographies
            include_oai: Parse OAI-PMH sources
            oai_limit: Limit number of OAI records per set (for testing)
            validate_uris: Run URI validation after parsing
            check_online: Check if URIs are accessible online (slower)
        """
        # Load progress first, regardless of cache usage
        progress = self._load_progress()

        cached = self._load_cached()
        if cached and use_cached:
            print(f"Loaded {len(cached)} entries from cache")
            self.catalogues = cached

            # Initialize deduplication tracker with existing URIs
            self._initialize_seen_uris()

            # Check if there's new work to do
            has_new_oai = include_oai and any(
                self._get_oai_set_name(url) not in progress.get('completed_oai_sets', [])
                for url in self.oai_sources
            )
            has_new_pdfs = include_pdfs and any(
                Path(pdf_path).name not in progress.get('completed_pdfs', [])
                for pdf_path in self.lit_pdfs
            )

            if not has_new_oai and not has_new_pdfs:
                print("All sources already processed according to progress tracking")

                # Print deduplication stats
                self._print_dedup_summary()

                if validate_uris:
                    print("\nValidating cached entries...")
                    self.validate_and_clean_catalogues(check_online=check_online)
                return self.catalogues
            else:
                print(f"Found new sources to process (OAI: {has_new_oai}, PDFs: {has_new_pdfs})")
        else:
            # Starting fresh - initialize seen URIs from any existing catalogues
            if self.catalogues:
                self._initialize_seen_uris()

        # Load existing catalogues if resuming
        if not cached and Path(self.cache_dir, self.cache_file_name).exists():
            with open(Path(self.cache_dir, self.cache_file_name), 'r') as f:
                self.catalogues = json.load(f)
            self._initialize_seen_uris()
            print(f"Resuming from {len(self.catalogues)} existing entries")

        # Parse OAI-PMH sources FIRST
        if include_oai:
            print('\n' + '='*80)
            print('PHASE 1: Parsing OAI-PMH sources')
            print('='*80)
            for oai_url in self.oai_sources:
                set_name = re.search(r'set=([^&]+)', oai_url)
                set_name = set_name.group(1) if set_name else 'unknown'
                print(f"\nProcessing OAI set: {set_name}")
                try:
                    self.parse_oai_set(oai_url, limit=oai_limit, progress=progress)
                except Exception as e:
                    print(f"Error processing {set_name}: {e}")
                    print("Continuing with next set...")
                    continue

        # Parse PDF bibliographies SECOND
        if include_pdfs:
            print('\n' + '='*80)
            print('PHASE 2: Parsing PDF bibliography files')
            print('='*80)
            for pdf_path, page_range in zip(self.lit_pdfs, self.page_ranges):
                start_page, end_page = page_range
                try:
                    # Check file exists before opening with pikepdf
                    if not Path(pdf_path).exists():
                        print(f"Skipping {Path(pdf_path).name}: File not found")
                        continue

                    with pikepdf.open(pdf_path) as pdf:
                        num_pages = len(pdf.pages)
                        assert end_page <= num_pages, f"Requested end_page {end_page} exceeds total pages {num_pages}"

                    self.parse_pdf_batchwise(pdf_path, start_page, end_page, progress=progress)
                except Exception as e:
                    print(f"Failed to parse {Path(pdf_path).name}: {e}")
                    continue

        print(f"\n{'='*80}")
        print(f"Parsing complete!")
        print(f"{'='*80}")
        print(f"Total catalogues: {len(self.catalogues)}")
        print(f"Unique URIs: {len(self.seen_uris)}")
        print(f"{'='*80}")

        # Print deduplication and collection expansion statistics
        self._print_dedup_summary()

        # Validate URIs
        if validate_uris and self.catalogues:
            print('\n' + '='*80)
            print('PHASE 3: URI Validation')
            print('='*80)
            self.validate_and_clean_catalogues(check_online=check_online)

        print(f"\n{'='*80}")
        print("SAVING FINAL BIBLIOGRAPHY")
        print(f"{'='*80}")
        self.write(self.cache_dir)  # Save to data directory
        print(f"✓ Saved to: {Path(self.cache_dir) / self.cache_file_name}")

        return self.catalogues

    def _print_dedup_summary(self):
        """Print deduplication and collection expansion statistics"""
        print(f"\n{'='*80}")
        print(f"DEDUPLICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total unique catalogues: {len(self.catalogues)}")
        print(f"Total unique URIs tracked: {len(self.seen_uris)}")

        if self.duplicate_stats['duplicates_found'] > 0:
            print(f"Total duplicates skipped: {self.duplicate_stats['duplicates_found']}")
            print(f"\nDuplicates by source:")
            for source, count in sorted(self.duplicate_stats['duplicates_by_source'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count} duplicates")

        if self.expand_collections and self.duplicate_stats['collections_expanded'] > 0:
            print(f"\nCollection pages expanded: {self.duplicate_stats['collections_expanded']}")
            print(f"Catalogues from collections: {self.duplicate_stats['catalogues_from_collections']}")

        print(f"{'='*80}")

    def get_statistics(self):
        """Generate statistics about parsed catalogues"""
        if not self.catalogues:
            return "No catalogues loaded"

        stats = {
            'total': len(self.catalogues),
            'unique_uris': len(self.seen_uris),
            'with_uri': sum(1 for c in self.catalogues if c.get('uri')),
            'with_year': sum(1 for c in self.catalogues if c.get('year') and c.get('year') != -1),
            'with_location': sum(1 for c in self.catalogues if c.get('location')),
            'with_types': sum(1 for c in self.catalogues if c.get('types')),
            'duplicates_removed': self.duplicate_stats['duplicates_found'],
            'collections_expanded': self.duplicate_stats['collections_expanded'],
            'from_collections': self.duplicate_stats['catalogues_from_collections']
        }

        # Year distribution
        years = [c.get('year') for c in self.catalogues if c.get('year') and c.get('year') != -1]
        if years:
            stats['year_range'] = f"{min(years)} - {max(years)}"

        # Location distribution
        locations = {}
        for c in self.catalogues:
            loc = c.get('location')
            if loc:
                locations[loc] = locations.get(loc, 0) + 1
        stats['top_locations'] = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]

        # Types distribution
        types_count = {}
        for c in self.catalogues:
            for t in c.get('types', []):
                types_count[t] = types_count.get(t, 0) + 1
        stats['top_types'] = sorted(types_count.items(), key=lambda x: x[1], reverse=True)[:10]

        return stats

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()

        if isinstance(stats, str):
            print(stats)
            return

        print(f"\n{'='*80}")
        print("CATALOGUE STATISTICS")
        print(f"{'='*80}")
        print(f"\nTotal catalogues:        {stats['total']}")
        print(f"Unique URIs:             {stats['unique_uris']}")
        print(f"Duplicates removed:      {stats['duplicates_removed']}")
        if self.expand_collections:
            print(f"Collections expanded:    {stats['collections_expanded']}")
            print(f"From collections:        {stats['from_collections']}")
        print(f"With URI:                {stats['with_uri']} ({stats['with_uri']/stats['total']*100:.1f}%)")
        print(f"With year:               {stats['with_year']} ({stats['with_year']/stats['total']*100:.1f}%)")
        print(f"With location:           {stats['with_location']} ({stats['with_location']/stats['total']*100:.1f}%)")
        print(f"With types:              {stats['with_types']} ({stats['with_types']/stats['total']*100:.1f}%)")

        if 'year_range' in stats:
            print(f"\nYear range:              {stats['year_range']}")

        if stats['top_locations']:
            print(f"\nTop 10 locations:")
            for loc, count in stats['top_locations']:
                print(f"  {loc:30s} {count:5d} catalogues")

        if stats['top_types']:
            print(f"\nTop 10 object types:")
            for typ, count in stats['top_types']:
                print(f"  {typ:30s} {count:5d} catalogues")

        print(f"{'='*80}\n")


# ==================== Usage Example ====================

def main():
    """Example usage of the BibliographyParser with collection expansion"""

    # Initialize parser with collection expansion enabled
    parser = BibliographyParser(
        cache_dir='./bibliography_cache',
        oai_delay=1.0,
        expand_collections=True  # Enable automatic collection expansion
    )

    # Parse all sources with validation and deduplication
    catalogues = parser.parse(
        use_cached=False,          # Set to True to use existing cache
        include_pdfs=True,          # Parse PDF bibliographies
        include_oai=True,           # Parse OAI-PMH sources
        oai_limit=None,             # Set to number for testing (e.g., 10)
        validate_uris=True,         # Validate and fix URIs
        check_online=False          # Set to True to check URI accessibility (slow)
    )

    # Print statistics
    parser.print_statistics()

    # Save final output
    parser.write('./bibliography_cache')

    print(f"\n✓ Processing complete!")
    print(f"✓ Valid catalogues: {len(catalogues)}")
    print(f"✓ Output saved to: ./bibliography_cache/catalogues_dict.json")


if __name__ == "__main__":
    main()
