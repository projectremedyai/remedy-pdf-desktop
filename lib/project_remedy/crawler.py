"""Stage 1: Web Crawling & Document Discovery.

Uses Firecrawl's map API or crawl4ai (local browser) to discover all URLs
on a site, then filters for document links (PDF, Word, PowerPoint, Excel).
Discovered document links are deduplicated and persisted as DocumentJob
entries in the pipeline database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote, urljoin

import httpx

from project_remedy.config import CampusConfig, PipelineConfig
from project_remedy.database import DatabaseManager
from project_remedy.models import DocumentJob, FileType, JobStatus

logger = logging.getLogger(__name__)

# File extensions we want to discover, mapped to FileType enum values.
_DOCUMENT_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".doc": FileType.DOC,
    ".docx": FileType.DOCX,
    ".ppt": FileType.PPT,
    ".pptx": FileType.PPTX,
    ".xls": FileType.XLS,
    ".xlsx": FileType.XLSX,
}

# Regex that matches any of the target extensions at the end of a URL path
# (before any query string).
_EXT_PATTERN = re.compile(
    r"\.(?:pdf|docx?|pptx?|xlsx?)(?:\?.*)?$",
    re.IGNORECASE,
)

# Common CMS patterns for embedded document viewers / file fields.
_CMS_DOC_PATTERNS = [
    # Drupal file field: /sites/default/files/...
    re.compile(r"/sites/default/files/.*\.(?:pdf|docx?|pptx?|xlsx?)", re.IGNORECASE),
    # Generic /files/ or /documents/ directories
    re.compile(r"/(?:files|documents|uploads|media|assets)/.*\.(?:pdf|docx?|pptx?|xlsx?)", re.IGNORECASE),
    # Embedded Google Docs viewer links that reference a source document
    re.compile(r"docs\.google\.com/(?:viewer|gview)\?.*url=", re.IGNORECASE),
    # SharePoint / OneDrive embedded links
    re.compile(r"sharepoint\.com/.*\.(?:pdf|docx?|pptx?|xlsx?)", re.IGNORECASE),
    # OneDrive sharing links (1drv.ms/b/ = binary/PDF, /w/ = Word, /p/ = PowerPoint, /x/ = Excel)
    re.compile(r"1drv\.ms/[bwpx]/", re.IGNORECASE),
    # OneDrive full sharing URLs (onedrive.live.com with redeem parameter)
    re.compile(r"onedrive\.live\.com/.*(?:redeem|cid)=", re.IGNORECASE),
    # Adobe Acrobat shared PDF links
    re.compile(r"acrobat\.adobe\.com/link/review\?uri=", re.IGNORECASE),
]

# Map OneDrive short path segments to FileType.
_ONEDRIVE_TYPE_MAP: dict[str, FileType] = {
    "/b/": FileType.PDF,
    "/w/": FileType.DOCX,
    "/p/": FileType.PPTX,
    "/x/": FileType.XLSX,
}


def _file_type_from_url(url: str) -> FileType | None:
    """Determine the FileType from a URL's path extension or sharing pattern."""
    parsed = urlparse(url)
    path = unquote(parsed.path).lower()
    for ext, ftype in _DOCUMENT_EXTENSIONS.items():
        if path.endswith(ext):
            return ftype

    # OneDrive sharing links: 1drv.ms/{type_char}/...
    if "1drv.ms" in parsed.netloc:
        for segment, ftype in _ONEDRIVE_TYPE_MAP.items():
            if segment in parsed.path:
                return ftype

    # OneDrive full URLs with redeem parameter
    if "onedrive.live.com" in parsed.netloc:
        return FileType.PDF  # assume PDF for generic OneDrive shares

    # Adobe Acrobat shared links are always PDFs
    if "acrobat.adobe.com" in parsed.netloc:
        return FileType.PDF

    return None


def _normalise_url(url: str) -> str:
    """Strip fragments and trailing whitespace for deduplication."""
    parsed = urlparse(url)
    # Reconstruct without fragment
    return parsed._replace(fragment="").geturl().strip()


class DocumentCrawler:
    """Stage 1 -- discover document URLs on a website using Firecrawl.

    Parameters
    ----------
    config:
        Full pipeline configuration (uses ``config.crawl`` for crawl settings
        and ``config.output`` for state persistence paths).
    db:
        An already-connected :class:`DatabaseManager` instance.
    """

    def __init__(
        self,
        config: PipelineConfig,
        db: DatabaseManager,
        campus: CampusConfig | None = None,
    ) -> None:
        self._config = config
        self._crawl_cfg = config.crawl
        self._firecrawl_api_key = config.crawl.firecrawl_api_key
        self._db = db
        self._campus = campus
        self._campus_id = campus.code if campus else ""

        # Per-campus start_url takes precedence over config.crawl.start_url.
        self._start_url = (
            campus.start_url if campus and campus.start_url
            else config.crawl.start_url
        )

        # Track discovered document URLs to avoid duplicates within a run.
        self._seen_urls: set[str] = set()

        # Mapping of document URL → source page URL (for hybrid crawl).
        self._doc_source_pages: dict[str, str] = {}

        # Counters for progress logging.
        self._pages_crawled: int = 0
        self._docs_found: int = 0

        # Where to persist crawl state snapshots for resumability.
        self._state_path = config.output.output_dir / "crawl_state.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(self) -> list[DocumentJob]:
        """Discover document URLs and return newly created DocumentJobs.

        Supports two crawl engines:
        - ``firecrawl`` (default): Uses the Firecrawl cloud map API.
        - ``crawl4ai``: Uses a local headless browser via crawl4ai.

        The crawl:
        1. Loads any previously-seen URLs from the database so they are
           skipped (resumability).
        2. Calls the selected crawl engine to discover all URLs on the site.
        3. Filters returned URLs for document extensions.
        4. Deduplicates and persists new DocumentJob entries.
        5. Saves crawl state for future resumption.

        Returns
        -------
        list[DocumentJob]
            All document jobs created during this crawl session.
        """
        engine = self._crawl_cfg.crawl_engine.lower()
        logger.info(
            "Starting %s crawl from %s (max_pages=%d, campus=%s)",
            engine,
            self._start_url,
            self._crawl_cfg.max_pages,
            self._campus_id or "all",
        )

        # Pre-load already-known document URLs so we skip them.
        await self._load_existing_urls()

        # Run the selected crawl engine.
        if engine == "hybrid":
            all_urls = await self._run_hybrid()
        elif engine == "crawl4ai":
            all_urls = await self._run_crawl4ai()
        elif engine == "firecrawl_scrape":
            all_urls = await self._run_firecrawl_scrape()
        else:
            all_urls = await self._run_firecrawl_map()
        self._pages_crawled = len(all_urls)
        logger.info("%s returned %d URLs.", engine, len(all_urls))

        # Filter for document URLs and register jobs.
        new_jobs: list[DocumentJob] = []
        for url in all_urls:
            normalised = _normalise_url(url)
            # Use tracked source page if available (hybrid mode),
            # otherwise fall back to start URL.
            source_page = self._doc_source_pages.get(
                normalised, self._start_url
            )

            # Check by extension pattern first.
            if _EXT_PATTERN.search(normalised):
                file_type = _file_type_from_url(normalised)
                if file_type is not None:
                    job = await self._register_document(
                        url=normalised,
                        source_page_url=source_page,
                        link_text="",
                        link_context="",
                        file_type=file_type,
                    )
                    if job is not None:
                        new_jobs.append(job)
                    continue

            # Check CMS-specific URL patterns.
            for pattern in _CMS_DOC_PATTERNS:
                if pattern.search(normalised):
                    file_type = _file_type_from_url(normalised)
                    if file_type is not None:
                        job = await self._register_document(
                            url=normalised,
                            source_page_url=source_page,
                            link_text="",
                            link_context="",
                            file_type=file_type,
                        )
                        if job is not None:
                            new_jobs.append(job)
                    break  # Only match first CMS pattern per URL.

        # Save crawl state for resumability.
        await self._save_state()

        logger.info(
            "Crawl complete: %d URLs scanned, %d documents discovered.",
            self._pages_crawled,
            self._docs_found,
        )
        return new_jobs

    # ------------------------------------------------------------------
    # Firecrawl integration
    # ------------------------------------------------------------------

    async def _run_firecrawl_map(self) -> list[str]:
        """Discover all URLs on the site using Firecrawl map API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.firecrawl.dev/v1/map",
                headers={"Authorization": f"Bearer {self._firecrawl_api_key}"},
                json={
                    "url": self._start_url,
                    "limit": self._crawl_cfg.max_pages,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("links", [])

    async def _run_firecrawl_scrape(self) -> list[str]:
        """Scrape a single page with Firecrawl and extract all links."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {self._firecrawl_api_key}"},
                json={
                    "url": self._start_url,
                    "formats": ["links"],
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            links = data.get("data", {}).get("links", [])
            logger.info(
                "Firecrawl scrape of %s returned %d links.",
                self._start_url,
                len(links),
            )
            return links

    # ------------------------------------------------------------------
    # Hybrid: Firecrawl map + crawl4ai page rendering
    # ------------------------------------------------------------------

    async def _run_hybrid(self) -> list[str]:
        """Discover documents using Firecrawl map for page discovery
        and crawl4ai for JS-rendered link extraction.

        This is the recommended mode for LACCD Drupal sites which render
        document links via JavaScript.

        Steps:
            1. Call Firecrawl map to get all page URLs (fast).
            2. Filter for internal HTML pages.
            3. Render each page with crawl4ai to extract document links.
            4. Return all discovered URLs.
        """
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        except ImportError:
            raise RuntimeError(
                "crawl4ai is not installed. Install with: pip install crawl4ai"
            )

        # Step 1: Get page URLs from Firecrawl map.
        logger.info("Hybrid crawl: getting page list from Firecrawl map...")
        map_urls = await self._run_firecrawl_map()
        logger.info("Firecrawl map returned %d URLs.", len(map_urls))

        start_parsed = urlparse(self._start_url)
        base_domain = start_parsed.netloc
        # Accept both www and non-www variants
        alt_domain = (
            base_domain.replace("www.", "") if base_domain.startswith("www.")
            else f"www.{base_domain}"
        )
        allowed_domains = {base_domain, alt_domain}

        # Separate document URLs (already discovered) from page URLs.
        all_discovered: set[str] = set()
        page_urls: list[str] = []

        for url in map_urls:
            all_discovered.add(url)
            parsed = urlparse(url)
            # Skip non-internal or asset URLs.
            if parsed.netloc not in allowed_domains:
                continue
            if _EXT_PATTERN.search(url):
                continue  # Already a document URL, no need to render.
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in (
                ".css", ".js", ".svg", ".ico", ".png", ".jpg", ".jpeg",
                ".gif", ".webp", ".woff", ".woff2", ".ttf",
            )):
                continue
            page_urls.append(url)

        logger.info(
            "Hybrid crawl: %d document URLs from map, %d pages to render.",
            len(all_discovered) - len(page_urls),
            len(page_urls),
        )

        # Step 2: Render each page with crawl4ai and extract links.
        browser_config = BrowserConfig(headless=True)
        crawler_config = CrawlerRunConfig()
        semaphore = asyncio.Semaphore(20)  # Concurrent browser tabs.
        rendered = 0
        doc_from_pages = 0

        async def _render_page(
            crawler: AsyncWebCrawler, url: str
        ) -> list[str]:
            nonlocal rendered, doc_from_pages
            async with semaphore:
                try:
                    result = await crawler.arun(url=url, config=crawler_config)
                    html = result.html if hasattr(result, "html") else ""

                    # Extract all href values from the rendered HTML.
                    hrefs = re.findall(r'href="([^"]+)"', html, re.IGNORECASE)
                    links: list[str] = []
                    for href in hrefs:
                        abs_url = urljoin(url, href)
                        links.append(abs_url)
                        # Track source page for document URLs.
                        if _EXT_PATTERN.search(abs_url):
                            normalised = _normalise_url(abs_url)
                            if normalised not in self._doc_source_pages:
                                self._doc_source_pages[normalised] = url

                    rendered += 1
                    page_docs = [l for l in links if _EXT_PATTERN.search(l)]
                    doc_from_pages += len(page_docs)

                    if rendered % 100 == 0:
                        logger.info(
                            "Hybrid crawl: rendered %d/%d pages, "
                            "%d document links found so far.",
                            rendered,
                            len(page_urls),
                            doc_from_pages,
                        )

                    return links
                except Exception as exc:
                    logger.debug(
                        "Hybrid crawl: failed to render %s: %s", url, exc
                    )
                    return []

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process in batches to avoid memory issues.
            batch_size = 50
            for i in range(0, len(page_urls), batch_size):
                batch = page_urls[i : i + batch_size]
                results = await asyncio.gather(
                    *[_render_page(crawler, url) for url in batch]
                )
                for links in results:
                    for link in links:
                        all_discovered.add(link)

        doc_urls = [u for u in all_discovered if _EXT_PATTERN.search(u)]
        logger.info(
            "Hybrid crawl complete: rendered %d pages, "
            "discovered %d total URLs (%d documents).",
            rendered,
            len(all_discovered),
            len(doc_urls),
        )
        return list(all_discovered)

    # ------------------------------------------------------------------
    # crawl4ai integration
    # ------------------------------------------------------------------

    async def _run_crawl4ai(self) -> list[str]:
        """Discover URLs using crawl4ai's local headless browser.

        Performs a BFS crawl starting from ``self._start_url``, visiting
        pages up to ``max_depth`` and collecting all discovered URLs.
        """
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        except ImportError:
            raise RuntimeError(
                "crawl4ai is not installed. Install with: pip install crawl4ai"
            )

        max_depth = self._crawl_cfg.max_depth
        max_pages = self._crawl_cfg.max_pages
        delay = self._crawl_cfg.rate_limit

        start_parsed = urlparse(self._start_url)
        base_domain = start_parsed.netloc

        # BFS state
        visited: set[str] = set()
        all_discovered: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self._start_url, 0)])

        browser_config = BrowserConfig(headless=True)
        crawler_config = CrawlerRunConfig()

        async with AsyncWebCrawler(config=browser_config) as crawler:
            while queue and len(visited) < max_pages:
                current_url, depth = queue.popleft()

                normalised = _normalise_url(current_url)
                if normalised in visited:
                    continue
                visited.add(normalised)

                logger.debug("crawl4ai visiting [depth=%d]: %s", depth, current_url)

                try:
                    result = await crawler.arun(
                        url=current_url,
                        config=crawler_config,
                    )
                except Exception as exc:
                    logger.warning("crawl4ai failed on %s: %s", current_url, exc)
                    continue

                # Collect all links from the page
                page_links: list[str] = []
                if hasattr(result, "links"):
                    links_data = result.links or {}
                    for link_type in ("internal", "external"):
                        for link_obj in links_data.get(link_type, []):
                            href = link_obj.get("href", "") if isinstance(link_obj, dict) else str(link_obj)
                            if href:
                                absolute = urljoin(current_url, href)
                                page_links.append(absolute)

                for link in page_links:
                    all_discovered.add(link)

                    # Enqueue internal pages for further crawling
                    link_parsed = urlparse(link)
                    if (
                        depth < max_depth
                        and link_parsed.netloc == base_domain
                        and _normalise_url(link) not in visited
                        and not _EXT_PATTERN.search(link)
                    ):
                        queue.append((link, depth + 1))

                if delay > 0 and queue:
                    await asyncio.sleep(delay)

        logger.info(
            "crawl4ai complete: visited %d pages, discovered %d total URLs.",
            len(visited),
            len(all_discovered),
        )
        return list(all_discovered)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _register_document(
        self,
        *,
        url: str,
        source_page_url: str,
        link_text: str,
        link_context: str,
        file_type: FileType,
    ) -> DocumentJob | None:
        """Create a DocumentJob if the URL has not been seen before.

        Returns the new job, or ``None`` if it was a duplicate.
        """
        if url in self._seen_urls:
            return None
        self._seen_urls.add(url)

        # Also check the database in case of a resumed crawl.
        if await self._db.job_exists_for_url(url):
            logger.debug("Skipping already-known document: %s", url)
            return None

        job = DocumentJob(
            url=url,
            campus_id=self._campus_id,
            source_page_url=source_page_url,
            link_text=link_text[:500] if link_text else "",
            link_context=link_context[:1000] if link_context else "",
            file_type=file_type,
            status=JobStatus.DISCOVERED,
        )

        await self._db.create_job(job)
        self._docs_found += 1

        if self._docs_found % 50 == 0:
            logger.info(
                "Progress: %d documents discovered across %d URLs.",
                self._docs_found,
                self._pages_crawled,
            )

        return job

    async def _load_existing_urls(self) -> None:
        """Pre-populate ``_seen_urls`` from previously stored jobs."""
        existing_jobs = await self._db.get_all_jobs()
        for job in existing_jobs:
            if job.url:
                self._seen_urls.add(job.url)
        if self._seen_urls:
            logger.info(
                "Loaded %d previously discovered URLs for deduplication.",
                len(self._seen_urls),
            )

    async def _save_state(self) -> None:
        """Persist a snapshot of crawl progress for resumability."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pages_crawled": self._pages_crawled,
            "documents_found": self._docs_found,
            "seen_urls_count": len(self._seen_urls),
        }
        self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info("Crawl state saved to %s", self._state_path)
