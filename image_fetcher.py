"""Image fetching with fallback chain: DuckDuckGo → Bing scraping.

icrawler/Google removed — Google parser is broken (returns no results).
DDG is fast and reliable; Bing HTML scraping as backup.
"""

import logging
import re
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Callable

import warnings

import requests

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ── Source 1: DuckDuckGo ─────────────────────────────────────────────

def _fetch_ddg(query: str) -> Path | None:
    """Fetch image via DuckDuckGo image search."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))
        if not results:
            logger.debug("[ddg] No results for '%s'", query)
            return None

        for result in results:
            url = result.get("image", "")
            if not url:
                continue
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=8, stream=True)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "png" in ct:
                    suffix = ".png"
                elif "webp" in ct:
                    suffix = ".webp"
                else:
                    suffix = ".jpg"
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="ce_img_"
                )
                for chunk in resp.iter_content(8192):
                    tmp.write(chunk)
                tmp.close()
                logger.debug("[ddg] OK for '%s': %s", query, tmp.name)
                return Path(tmp.name)
            except (requests.RequestException, OSError):
                continue
    except Exception as e:
        logger.debug("[ddg] Exception for '%s': %s", query, e)
    return None


# ── Source 2: Bing HTML scraping ─────────────────────────────────────

def _fetch_bing(query: str) -> Path | None:
    """Scrape Bing Images for the first usable image URL."""
    try:
        bing_url = (
            f"https://www.bing.com/images/search"
            f"?q={urllib.parse.quote(query)}&first=1"
        )
        resp = requests.get(bing_url, headers=_HEADERS, timeout=8)
        resp.raise_for_status()

        # Bing encodes original image URLs as mediaurl= in query strings
        raw_urls = re.findall(r"mediaurl=(https?[^&\"]+)", resp.text, re.IGNORECASE)
        # Decode and filter to actual image URLs
        urls = []
        for raw in raw_urls:
            decoded = urllib.parse.unquote(raw)
            if any(decoded.lower().endswith(ext) or f".{ext}?" in decoded.lower()
                   for ext in ("jpg", "jpeg", "png", "webp")):
                urls.append(decoded)
            elif "bing.net" not in decoded:
                urls.append(decoded)  # accept non-bing URLs even without extension
        if not urls:
            logger.debug("[bing] No mediaurl matches for '%s'", query)
            return None

        for url in urls[:5]:
            try:
                img_resp = requests.get(url, headers=_HEADERS, timeout=8, stream=True)
                img_resp.raise_for_status()
                ct = img_resp.headers.get("content-type", "")
                if "image" not in ct and "octet" not in ct:
                    continue
                suffix = ".png" if "png" in ct else ".webp" if "webp" in ct else ".jpg"
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="ce_img_"
                )
                for chunk in img_resp.iter_content(8192):
                    tmp.write(chunk)
                tmp.close()
                logger.debug("[bing] OK for '%s': %s", query, tmp.name)
                return Path(tmp.name)
            except (requests.RequestException, OSError):
                continue
    except Exception as e:
        logger.debug("[bing] Exception for '%s': %s", query, e)
    return None


# ── Fallback chain ───────────────────────────────────────────────────

def fetch_image(query: str) -> Path | None:
    """Fetch a single image: DDG first, then Bing scraping."""
    if not query:
        return None

    result = _fetch_ddg(query)
    if result:
        return result

    logger.info("[images] DDG failed for '%s', trying Bing", query)
    result = _fetch_bing(query)
    if result:
        return result

    logger.warning("[images] All sources failed for '%s'", query)
    return None


# ── Parallel fetcher ───────────────────────────────────────────────────

_TOTAL_TIMEOUT = 30   # wall-clock seconds for all images
_PER_IMAGE_TIMEOUT = 15  # max seconds per individual image


def fetch_images_parallel(
    queries: list[str],
    on_progress: Callable[[int, int], None] | None = None,
    max_workers: int = 5,
) -> list[Path | None]:
    """Fetch images for multiple queries in parallel with timeouts.

    All images fetched simultaneously (one thread per query).
    30 second total wall-clock limit — any image not done is skipped (None).
    """
    results: list[Path | None] = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(fetch_image, q): i for i, q in enumerate(queries)
        }
        done_count = 0
        try:
            for future in as_completed(future_to_idx, timeout=_TOTAL_TIMEOUT):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=_PER_IMAGE_TIMEOUT)
                except Exception:
                    results[idx] = None
                done_count += 1
                if on_progress:
                    on_progress(done_count, len(queries))
        except TimeoutError:
            pass

    success = sum(1 for r in results if r is not None)
    total = len(queries)
    failed = total - success
    logger.info("Images fetched: %d/%d (%d fallback to white)", success, total, failed)

    return results
