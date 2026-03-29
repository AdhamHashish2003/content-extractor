"""Multi-platform video transcript extraction.

Tier 1 — Native captions (YouTube only): youtube-transcript-api, fast + free.
Tier 2 — Groq Whisper API (Instagram, X, YouTube fallback): yt-dlp audio → Groq cloud transcription.
Tier 3 — Local Whisper fallback: if Groq returns 429 (rate limit), fall back to local openai-whisper.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from groq import Groq, RateLimitError
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi

from config import Platform, get_groq_keys

logger = logging.getLogger(__name__)

# ── YouTube Cookies (for bypassing bot detection on cloud servers) ────

_YT_COOKIE_FILE = Path("/tmp/youtube_cookies.txt")
_yt_cookies_ready = False


def _init_youtube_cookies() -> bool:
    """Write YOUTUBE_COOKIES env var to a Netscape cookie file.

    The env var should contain cookies in Netscape/Mozilla format, e.g.:
    .youtube.com\tTRUE\t/\tTRUE\t0\tSID\tvalue...
    One cookie per line, tab-separated.
    """
    global _yt_cookies_ready
    raw = os.getenv("YOUTUBE_COOKIES", "").strip()
    if not raw:
        return False
    try:
        # Write with the required Netscape header
        content = raw if raw.startswith("# Netscape") else "# Netscape HTTP Cookie File\n" + raw
        _YT_COOKIE_FILE.write_text(content)
        _yt_cookies_ready = True
        logger.info("[yt-cookies] Cookie file written: %d bytes, %d lines",
                    len(content), content.count("\n"))
        return True
    except Exception as e:
        logger.warning("[yt-cookies] Failed to write cookie file: %s", e)
        return False


def _load_cookies_into_session(session) -> None:
    """Load YouTube cookies from the cookie file into a requests.Session."""
    if not _yt_cookies_ready or not _YT_COOKIE_FILE.exists():
        return
    try:
        from http.cookiejar import MozillaCookieJar
        jar = MozillaCookieJar(str(_YT_COOKIE_FILE))
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies.update(jar)
        logger.info("[yt-cookies] Loaded %d cookies into session", len(jar))
    except Exception as e:
        # MozillaCookieJar is strict — fall back to manual parsing
        logger.warning("[yt-cookies] MozillaCookieJar failed (%s), trying manual parse", e)
        try:
            for line in _YT_COOKIE_FILE.read_text().strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    domain, _, path, secure, _, name, value = parts[:7]
                    session.cookies.set(name, value, domain=domain, path=path)
            logger.info("[yt-cookies] Manual parse loaded cookies into session")
        except Exception as e2:
            logger.warning("[yt-cookies] Manual parse also failed: %s", e2)


def _generate_sapisidhash(sapisid: str, origin: str = "https://www.youtube.com") -> str | None:
    """Generate SAPISIDHASH authorization header from SAPISID cookie."""
    import hashlib
    import time as _time
    if not sapisid:
        return None
    timestamp = int(_time.time())
    hash_input = f"{timestamp} {sapisid} {origin}"
    sha1 = hashlib.sha1(hash_input.encode()).hexdigest()
    return f"SAPISIDHASH {timestamp}_{sha1}"


def _get_yt_cookie_session():
    """Create a requests.Session pre-loaded with YouTube cookies and SAPISIDHASH auth."""
    import requests
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    session.cookies.set("CONSENT", "YES+cb.20210328-17-p0.en+FX+999", domain=".youtube.com")
    _load_cookies_into_session(session)

    # Generate SAPISIDHASH auth header if SAPISID cookie is present
    sapisid = None
    for cookie in session.cookies:
        if cookie.name == "SAPISID":
            sapisid = cookie.value
            break
        if cookie.name == "__Secure-3PAPISID":
            sapisid = cookie.value
            break
    if sapisid:
        auth = _generate_sapisidhash(sapisid)
        if auth:
            session.headers["Authorization"] = auth
            session.headers["X-Origin"] = "https://www.youtube.com"
            logger.info("[yt-cookies] SAPISIDHASH auth header set")

    return session


# Initialize cookies on module load
_init_youtube_cookies()
if _yt_cookies_ready and _YT_COOKIE_FILE.exists():
    _lines = _YT_COOKIE_FILE.read_text().strip().split("\n")
    _real = [l for l in _lines if l.strip() and not l.startswith("#")]
    print(f"YouTube cookies: LOADED ({len(_real)} cookies, file={_YT_COOKIE_FILE})")
    if _real:
        print(f"  first: {_real[0][:60]}...")
        print(f"  last:  {_real[-1][:60]}...")
else:
    print(f"YouTube cookies: NOT SET (env var empty: {not os.getenv('YOUTUBE_COOKIES', '')})")


# ── Transcript cache ──────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_FILE = _CACHE_DIR / "transcripts.json"
_CACHE_TTL = 7 * 24 * 3600  # 7 days


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _get_cached_transcript(url: str) -> tuple[str, str] | None:
    """Return (transcript, title) from cache if fresh, else None."""
    cache = _load_cache()
    key = _cache_key(url)
    entry = cache.get(key)
    if entry and (time.time() - entry.get("timestamp", 0)) < _CACHE_TTL:
        logger.info("Transcript cache hit for %s", key)
        return entry["transcript"], entry.get("title", "")
    return None


def _set_cached_transcript(url: str, transcript: str, title: str) -> None:
    cache = _load_cache()
    key = _cache_key(url)
    cache[key] = {
        "transcript": transcript,
        "title": title,
        "timestamp": time.time(),
    }
    _save_cache(cache)

# 24 MB in bytes — Groq's limit is 25 MB, leave margin
_CHUNK_SIZE_BYTES = 24 * 1024 * 1024


@dataclass
class VideoMeta:
    video_id: str
    title: str
    duration_str: str
    author: str
    platform: Platform


@dataclass
class VideoInfo:
    meta: VideoMeta
    transcript: str

    @property
    def video_id(self) -> str:
        return self.meta.video_id

    @property
    def title(self) -> str:
        return self.meta.title

    @property
    def duration_str(self) -> str:
        return self.meta.duration_str

    @property
    def platform(self) -> Platform:
        return self.meta.platform


# ── Platform Detection ─────────────────────────────────────────────────

def detect_platform(url: str) -> Platform:
    """Detect the platform from a URL."""
    url_lower = url.lower()
    if any(d in url_lower for d in ("youtube.com", "youtu.be")):
        return Platform.YOUTUBE
    if "instagram.com" in url_lower:
        return Platform.INSTAGRAM
    if any(d in url_lower for d in ("x.com", "twitter.com")):
        return Platform.TWITTER
    if "tiktok.com" in url_lower:
        return Platform.TIKTOK
    return Platform.UNKNOWN


PLATFORM_LABELS: dict[Platform, str] = {
    Platform.YOUTUBE: "YouTube",
    Platform.INSTAGRAM: "Instagram",
    Platform.TWITTER: "X / Twitter",
    Platform.TIKTOK: "TikTok",
    Platform.UNKNOWN: "Direct URL",
}


# ── Video ID Extraction ────────────────────────────────────────────────

def _extract_youtube_id(url: str) -> str:
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract YouTube video ID from: {url}")


def _extract_generic_id(url: str) -> str:
    """Extract a usable ID from Instagram/X/TikTok/other URLs via URL parsing."""
    match = re.search(r"instagram\.com/(?:reel|p)/([a-zA-Z0-9_-]+)", url)
    if match:
        return f"ig_{match.group(1)}"
    match = re.search(r"(?:x\.com|twitter\.com)/.+/status/(\d+)", url)
    if match:
        return f"x_{match.group(1)}"
    match = re.search(r"tiktok\.com/.+/video/(\d+)", url)
    if match:
        return f"tt_{match.group(1)}"
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _extract_video_id(url: str, platform: Platform) -> str:
    if platform == Platform.YOUTUBE:
        return _extract_youtube_id(url)
    return _extract_generic_id(url)


# ── Metadata ─────────────────────────────────────────────────────────

def _format_duration(seconds: int) -> str:
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fetch_youtube_metadata(video_id: str) -> dict:
    """Get YouTube title via simple HTTP request — no yt-dlp, no bot detection."""
    try:
        page_url = f"https://www.youtube.com/watch?v={video_id}"
        req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
        title_match = re.search(r"<title>(.*?)</title>", html)
        title = title_match.group(1).replace(" - YouTube", "").strip() if title_match else ""
        author_match = re.search(r'"author":"(.*?)"', html)
        author = author_match.group(1) if author_match else ""
        return {"title": title, "author": author}
    except Exception:
        return {"title": "", "author": ""}


def _fetch_ytdlp_metadata(url: str) -> dict:
    """Get metadata via yt-dlp Python API — used for non-YouTube platforms only."""
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)
        return {
            "title": data.get("title") or data.get("description", "")[:80] or "",
            "duration": int(data.get("duration", 0) or 0),
            "author": data.get("uploader") or data.get("channel") or "",
        }
    except Exception:
        return {"title": "", "duration": 0, "author": ""}


def fetch_metadata(url: str, platform: Platform) -> VideoMeta:
    """Fetch video metadata. YouTube uses HTTP scrape; others use yt-dlp."""
    video_id = _extract_video_id(url, platform)

    if platform == Platform.YOUTUBE:
        data = _fetch_youtube_metadata(video_id)
        return VideoMeta(
            video_id=video_id,
            title=data.get("title", ""),
            duration_str="",
            author=data.get("author", "YouTube"),
            platform=platform,
        )

    # Non-YouTube: use yt-dlp
    data = _fetch_ytdlp_metadata(url)
    return VideoMeta(
        video_id=video_id,
        title=data.get("title", ""),
        duration_str=_format_duration(data.get("duration", 0)),
        author=data.get("author", ""),
        platform=platform,
    )


# ── Caption XML parsing ──────────────────────────────────────────────

def _parse_caption_xml(xml_text: str) -> str:
    """Parse YouTube caption XML.

    Handles:
    - Format 1: ``<text start="..." dur="...">words</text>``
    - Format 3 with segments: ``<p t="..."><s>word</s><s>word</s></p>``
    - Format 3 bare: ``<p t="...">[Applause]</p>`` (no ``<s>`` children)
    """
    import html as htmlmod

    # Format 1: <text> tags
    texts = re.findall(r"<text[^>]*>(.*?)</text>", xml_text, re.DOTALL)
    if texts:
        clean = [htmlmod.unescape(t).strip() for t in texts if t.strip()]
        return " ".join(clean)

    # Format 3 with <s> segments (most auto-generated captions)
    segments = re.findall(r"<s[^>]*>(.*?)</s>", xml_text, re.DOTALL)
    if segments:
        clean = [htmlmod.unescape(s).strip() for s in segments if s.strip()]
        return " ".join(clean)

    # Format 3 bare <p> tags (e.g. music-only videos with [Applause])
    paras = re.findall(r"<p[^>]*>(.*?)</p>", xml_text, re.DOTALL)
    if paras:
        # Strip any nested tags first
        clean = []
        for p in paras:
            stripped = re.sub(r"<[^>]+>", "", p)
            stripped = htmlmod.unescape(stripped).strip()
            if stripped:
                clean.append(stripped)
        return " ".join(clean)

    return ""


# ── Language priority helper ──────────────────────────────────────────

def _track_sort_key(t) -> tuple:
    """Sort caption tracks: en > ar > other, manual > auto-generated."""
    lang = t.get("languageCode", "") if isinstance(t, dict) else getattr(t, "language_code", "")
    kind = t.get("kind", "") if isinstance(t, dict) else ("asr" if getattr(t, "is_generated", False) else "")
    is_auto = 1 if kind == "asr" else 0
    if lang == "en" or lang.startswith("en-"):
        return (0, is_auto)
    if lang == "ar" or lang.startswith("ar-"):
        return (1, is_auto)
    return (2, is_auto)


# ── YouTube Transcript (multi-source fallback) ──────────────────────

def _try_youtube_captions(video_id: str) -> str | None:
    """Try every caption source. Returns transcript text or None."""
    import requests

    MIN_WORDS = 20

    # ── Method 1: youtube-transcript-api (fast, free) ──
    logger.info("[transcript] Method 1 (youtube-transcript-api) for %s (cookies=%s)", video_id, _yt_cookies_ready)
    try:
        yt_session = _get_yt_cookie_session() if _yt_cookies_ready else None
        api = YouTubeTranscriptApi(http_client=yt_session) if yt_session else YouTubeTranscriptApi()
        available = list(api.list(video_id))
        if available:
            langs = [(t.language_code, "auto" if t.is_generated else "manual") for t in available]
            logger.info("[transcript] Method 1: found %d transcripts: %s", len(available), langs)
            available.sort(key=lambda t: _track_sort_key(t))
            chosen = available[0]
            text = " ".join(s.text for s in chosen.fetch())
            if len(text.split()) > MIN_WORDS:
                logger.info("[transcript] Method 1 OK: %s, %d words", chosen.language_code, len(text.split()))
                return text
            logger.info("[transcript] Method 1: fetched but too short (%d words)", len(text.split()))
        else:
            logger.info("[transcript] Method 1: no transcripts listed")
    except Exception as e:
        logger.info("[transcript] Method 1 ERROR: %s: %s", type(e).__name__, str(e)[:300])

    # ── Method 2: Innertube player API — try WEB (cookies), ANDROID, TV embedded ──
    _clients = [
        ("WEB", "2.20240101.00.00"),
        ("ANDROID", "20.10.38"),
        ("TVHTML5_SIMPLY_EMBEDDED_PLAYER", "2.0"),
    ]
    for client_name, client_ver in _clients:
        logger.info("[transcript] Method 2 (Innertube %s) for %s", client_name, video_id)
        try:
            session = _get_yt_cookie_session()
            page = session.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
            key_m = re.search(r'"INNERTUBE_API_KEY":\s*"([a-zA-Z0-9_-]+)"', page.text)
            api_key = key_m.group(1) if key_m else "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"

            resp = session.post(
                f"https://www.youtube.com/youtubei/v1/player?key={api_key}",
                json={"context": {"client": {"clientName": client_name, "clientVersion": client_ver}}, "videoId": video_id},
                timeout=15,
            )
            if not resp.ok:
                logger.info("[transcript] Method 2 (%s): player API returned %d", client_name, resp.status_code)
                continue

            player_data = resp.json()
            caps = player_data.get("captions", {})
            tracks = (caps.get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
                      or caps.get("playerCaptionsRenderer", {}).get("captionTracks", []))

            if not tracks:
                logger.info("[transcript] Method 2 (%s): no caption tracks", client_name)
                continue

            logger.info("[transcript] Method 2 (%s): %d tracks found", client_name, len(tracks))
            tracks.sort(key=_track_sort_key)

            for t in tracks:
                track_url = t.get("baseUrl", "")
                lang = t.get("languageCode", "?")
                if not track_url:
                    continue
                cr = session.get(track_url, timeout=15)
                if cr.ok and len(cr.text) > 100:
                    text = _parse_caption_xml(cr.text)
                    if len(text.split()) > MIN_WORDS:
                        logger.info("[transcript] Method 2 (%s) OK: %s, %d words", client_name, lang, len(text.split()))
                        return text
                    logger.info("[transcript] Method 2 (%s): track %s parsed but too short (%d)", client_name, lang, len(text.split()))
                else:
                    logger.info("[transcript] Method 2 (%s): track %s response empty (%d bytes)", client_name, lang, len(cr.text))
        except Exception as e:
            logger.info("[transcript] Method 2 (%s) ERROR: %s: %s", client_name, type(e).__name__, str(e)[:200])

    # ── Method 3: External transcript APIs (work from any IP) ──
    logger.info("[transcript] Method 3 (external APIs) for %s", video_id)
    external_urls = [
        f"https://deserving-harmony-production.up.railway.app/transcript?url=https://youtube.com/watch?v={video_id}",
    ]
    for svc_url in external_urls:
        try:
            r = requests.get(svc_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if not r.ok:
                continue
            data = r.json()
            # Handle different response formats
            if isinstance(data, list):
                text = " ".join(item.get("text", "") for item in data if item.get("text"))
            elif isinstance(data, dict):
                text = data.get("transcript", "") or data.get("text", "") or data.get("content", "")
                # Some APIs nest transcript in a list
                if not text and "transcription" in data:
                    parts = data["transcription"]
                    if isinstance(parts, list):
                        text = " ".join(p.get("text", "") for p in parts if p.get("text"))
            else:
                continue
            if text and len(text.split()) > MIN_WORDS:
                logger.info("[transcript] Method 3 OK via %s: %d words", svc_url[:50], len(text.split()))
                return text
        except Exception as e:
            logger.info("[transcript] Method 3 service %s failed: %s", svc_url[:40], e)

    logger.warning("[transcript] All caption methods failed for %s", video_id)
    return None


def get_transcript(video_id: str, url: str) -> tuple[str, str]:
    """Get transcript using all methods. Returns (text, language_code).

    Tries captions first (Methods 1-3), then Groq Whisper as last resort.
    Raises RuntimeError if everything fails.
    """
    # Try all caption methods first
    text = _try_youtube_captions(video_id)
    if text and len(text.split()) > 20:
        from analyzer import detect_language
        return text, detect_language(text)

    # Method 4: Groq Whisper (downloads audio via yt-dlp with cookies — last resort)
    logger.info("[transcript] Method 4 (Groq Whisper) for %s (cookies=%s)", url[:60], _yt_cookies_ready)
    try:
        text = _groq_pipeline(url)
        if text and len(text.split()) > 20:
            from analyzer import detect_language
            return text, detect_language(text)
    except Exception as e:
        logger.warning("[transcript] Method 4 (Whisper) failed: %s", e)

    raise RuntimeError(
        "Could not get transcript. All methods failed — captions unavailable "
        "and audio download blocked. Try a different video or one with subtitles."
    )


# ── Tier 2: yt-dlp Download + Groq Whisper API ─────────────────────────

class TranscriptionProgress:
    """Callback hooks for the CLI to show progress."""
    def on_download_start(self) -> None: ...
    def on_download_done(self) -> None: ...
    def on_transcribe_start(self) -> None: ...
    def on_transcribe_done(self, word_count: int) -> None: ...
    def on_rate_limit_fallback(self) -> None: ...


def _download_audio_innertube(video_id: str, output_path: Path) -> Path | None:
    """Download audio via Innertube ANDROID API — bypasses bot detection on cloud servers."""
    logger.info("[audio-download] Trying Innertube ANDROID for %s (cookies=%s)", video_id, "YES" if _yt_cookies_ready else "NO")
    try:
        session = _get_yt_cookie_session()

        # Get API key
        page = session.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        key_match = re.search(r'"INNERTUBE_API_KEY":\s*"([a-zA-Z0-9_-]+)"', page.text)
        api_key = key_match.group(1) if key_match else "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"

        # Get audio streams via ANDROID client
        player = session.post(
            f"https://www.youtube.com/youtubei/v1/player?key={api_key}",
            json={
                "context": {"client": {"clientName": "ANDROID", "clientVersion": "20.10.38"}},
                "videoId": video_id,
            },
            timeout=15,
        )
        if not player.ok:
            logger.info("[audio-download] Innertube player API returned %d", player.status_code)
            return None

        data = player.json()
        formats = data.get("streamingData", {}).get("adaptiveFormats", [])
        # Find best audio-only stream
        audio_fmts = [f for f in formats if f.get("mimeType", "").startswith("audio/")]
        if not audio_fmts:
            logger.info("[audio-download] No audio formats in Innertube response")
            return None

        # Prefer m4a/mp4a, lowest bitrate that's still reasonable (>48kbps)
        audio_fmts.sort(key=lambda f: f.get("bitrate", 0))
        chosen = None
        for fmt in audio_fmts:
            if fmt.get("bitrate", 0) >= 48000:
                chosen = fmt
                break
        if not chosen:
            chosen = audio_fmts[-1]  # highest bitrate if all are low

        audio_url = chosen.get("url", "")
        if not audio_url:
            logger.info("[audio-download] Audio format has no URL (possibly encrypted)")
            return None

        # Download the audio stream
        ext = "m4a" if "mp4a" in chosen.get("mimeType", "") or "m4a" in chosen.get("mimeType", "") else "webm"
        audio_path = output_path / f"audio.{ext}"
        logger.info("[audio-download] Downloading %s audio (%d kbps)...",
                     ext, chosen.get("bitrate", 0) // 1000)

        dl_headers = {
            "Referer": "https://www.youtube.com/",
            "Origin": "https://www.youtube.com",
        }
        with session.get(audio_url, stream=True, timeout=120, headers=dl_headers) as r:
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

        size_kb = audio_path.stat().st_size // 1024
        logger.info("[audio-download] Innertube download OK: %s (%d KB)", audio_path.name, size_kb)
        if size_kb < 10:
            logger.info("[audio-download] File too small, likely empty")
            return None
        return audio_path

    except Exception as e:
        logger.info("[audio-download] Innertube audio download failed: %s", e)
        return None


def _download_audio(url: str, output_path: Path) -> Path:
    """Download audio — tries Innertube first (cloud-friendly), then yt-dlp."""

    # For YouTube URLs, try Innertube ANDROID API first (bypasses bot detection)
    yt_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if yt_match:
        video_id = yt_match.group(1)
        innertube_path = _download_audio_innertube(video_id, output_path)
        if innertube_path:
            return innertube_path
        logger.info("[audio-download] Innertube failed, falling back to yt-dlp")

    # Fallback: yt-dlp (works locally, may fail on cloud without cookies)
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "audioquality": "5",
        "outtmpl": str(output_path / "audio_ytdlp.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "64",
        }],
    }
    # Pass cookie file if available (bypasses bot detection)
    if _yt_cookies_ready and _YT_COOKIE_FILE.exists():
        cookie_path = str(_YT_COOKIE_FILE)
        opts["cookiefile"] = cookie_path
        logger.info("[audio-download] yt-dlp cookiefile=%s (%d bytes)", cookie_path, _YT_COOKIE_FILE.stat().st_size)
    else:
        logger.info("[audio-download] yt-dlp NO cookies (ready=%s, exists=%s)", _yt_cookies_ready, _YT_COOKIE_FILE.exists())
    logger.info("[audio-download] yt-dlp opts: format=%s", opts.get("format"))
    with yt_dlp.YoutubeDL(opts) as ydl:
        rc = ydl.download([url])
        if rc != 0:
            raise RuntimeError("yt-dlp failed to download audio")

    audio_files = list(output_path.glob("audio_ytdlp.*"))
    if not audio_files:
        raise RuntimeError("yt-dlp completed but no audio file was produced.")
    logger.info("[audio-download] yt-dlp OK: %s (%d KB)", audio_files[0].name, audio_files[0].stat().st_size // 1024)
    return audio_files[0]


def _convert_to_mp3(audio_path: Path, tmp_dir: Path) -> Path:
    """Convert any audio file to mp3 at 64k bitrate for smaller size."""
    logger.info("[groq-pipeline] Converting to mp3 (64k)...")
    audio = AudioSegment.from_file(str(audio_path))
    mp3_path = tmp_dir / "converted.mp3"
    audio.export(str(mp3_path), format="mp3", bitrate="64k")
    logger.info("[groq-pipeline] Converted: %d KB, %d sec", mp3_path.stat().st_size // 1024, len(audio) // 1000)
    return mp3_path


def _prepare_chunks(mp3_path: Path, tmp_dir: Path) -> list[Path]:
    """Return a list of ≤24 MB mp3 files ready for Groq. Splits by duration if needed."""
    file_size = mp3_path.stat().st_size
    if file_size <= _CHUNK_SIZE_BYTES:
        return [mp3_path]

    audio = AudioSegment.from_file(str(mp3_path))
    total_ms = len(audio)

    # Calculate chunk duration proportional to the 24 MB target
    chunk_duration_ms = int(total_ms * (_CHUNK_SIZE_BYTES / file_size))
    chunks: list[Path] = []
    start = 0

    while start < total_ms:
        end = min(start + chunk_duration_ms, total_ms)
        chunk_path = tmp_dir / f"chunk_{len(chunks):03d}.mp3"
        audio[start:end].export(str(chunk_path), format="mp3", bitrate="64k")
        chunks.append(chunk_path)
        start = end

    return chunks


def _transcribe_chunk_with_rotation(audio_path: Path, keys: list[str]) -> str:
    """Transcribe a single audio chunk, rotating through keys on 429."""
    import time
    chunk_size_kb = audio_path.stat().st_size // 1024
    logger.info("[groq-pipeline] Transcribing chunk %s (%d KB)...", audio_path.name, chunk_size_kb)

    last_err: Exception | None = None
    for key in keys:
        try:
            client = Groq(api_key=key, timeout=120.0)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    response_format="text",
                )
            words = len(result.strip().split())
            logger.info("[groq-pipeline] Chunk transcribed: %d words", words)
            return result.strip()
        except RateLimitError as e:
            logger.info("[groq-pipeline] Key %s...%s rate limited, trying next", key[:8], key[-4:])
            last_err = e
            time.sleep(2)  # Brief pause before trying next key
            continue
        except Exception as e:
            logger.warning("[groq-pipeline] Transcription error with key %s: %s", key[:8], e)
            last_err = e
            continue
    if isinstance(last_err, RateLimitError):
        raise last_err
    raise RuntimeError(f"All Groq keys failed: {last_err}")


def _transcribe_with_local_whisper(audio_path: Path) -> str:
    """Tier 3: Transcribe using local openai-whisper model."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "Local Whisper fallback unavailable — openai-whisper not installed. "
            "Install with: pip install openai-whisper"
        )

    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def _groq_pipeline(url: str, progress: TranscriptionProgress | None = None) -> str:
    """Full pipeline: download audio → convert to 64k mp3 → Groq API (with local Whisper fallback)."""
    import time as _t
    _start = _t.time()
    logger.info("[groq-pipeline] Starting for %s", url[:80])
    with tempfile.TemporaryDirectory(prefix="content_extractor_") as tmp:
        tmp_path = Path(tmp)

        # Download
        if progress:
            progress.on_download_start()
        audio_path = _download_audio(url, tmp_path)
        if progress:
            progress.on_download_done()

        # Convert to small mp3, split if still >24 MB
        if progress:
            progress.on_transcribe_start()
        mp3_path = _convert_to_mp3(audio_path, tmp_path)
        chunks = _prepare_chunks(mp3_path, tmp_path)

        # Tier 2: try Groq with key rotation; Tier 3: local Whisper if ALL keys exhausted
        import random
        keys = get_groq_keys()
        random.shuffle(keys)

        try:
            transcripts = [_transcribe_chunk_with_rotation(chunk, keys) for chunk in chunks]
        except RateLimitError:
            logger.warning("All Groq keys rate limited, using local Whisper (slower)...")
            if progress:
                progress.on_rate_limit_fallback()
            transcripts = [_transcribe_with_local_whisper(chunk) for chunk in chunks]

        transcript = " ".join(transcripts)

        word_count = len(transcript.split())
        elapsed = _t.time() - _start
        logger.info("[groq-pipeline] Complete: %d words in %.1fs", word_count, elapsed)
        if progress:
            progress.on_transcribe_done(word_count)

        return transcript


# ── Video Frame Extraction (Podcast Mode) ─────────────────────────────

def extract_video_frames(url: str, num_frames: int = 5) -> list[Path]:
    """Download video and extract evenly-spaced frames for podcast mode.

    Returns a list of frame file paths (JPEG). Filters out dark/blank frames.
    """
    import subprocess
    import yt_dlp
    from PIL import Image

    tmp_dir = Path(tempfile.mkdtemp(prefix="podcast_frames_"))
    video_path = tmp_dir / "video.mp4"

    # Download video at low quality (360p max)
    ydl_opts = {
        "format": "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]/best",
        "outtmpl": str(video_path),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if _yt_cookies_ready and _YT_COOKIE_FILE.exists():
        ydl_opts["cookiefile"] = str(_YT_COOKIE_FILE)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error("Video download for frames failed: %s", e)
        return []

    # Find the actual downloaded file (yt-dlp may change extension)
    video_files = list(tmp_dir.glob("video.*"))
    if not video_files:
        return []
    video_path = video_files[0]

    # Extract more frames than needed so we can filter dark ones
    raw_count = num_frames * 3
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    try:
        # Get video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 60.0

        # Calculate interval to get evenly-spaced frames
        interval = max(1, duration / (raw_count + 1))

        # Extract frames using ffmpeg select filter
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vf",
             f"fps=1/{interval:.1f}", "-frames:v", str(raw_count),
             "-q:v", "3", str(frames_dir / "frame_%03d.jpg")],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        logger.error("Frame extraction failed: %s", e)
        return []

    # Filter out dark/blank frames
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    good_frames: list[Path] = []

    for fp in frame_paths:
        try:
            img = Image.open(fp).convert("L")  # grayscale
            avg_brightness = sum(img.getdata()) / (img.width * img.height)
            if avg_brightness > 30:  # skip nearly black frames
                good_frames.append(fp)
        except Exception:
            continue

    # If we filtered too aggressively, fall back to all frames
    if len(good_frames) < num_frames and frame_paths:
        good_frames = list(frame_paths)

    # Evenly pick num_frames from good_frames
    if len(good_frames) <= num_frames:
        return good_frames

    step = len(good_frames) / num_frames
    selected = [good_frames[int(i * step)] for i in range(num_frames)]
    return selected


# ── Public API ─────────────────────────────────────────────────────────

def fetch_video_info(
    url: str,
    progress: TranscriptionProgress | None = None,
) -> VideoInfo:
    """Extract metadata and transcript for any supported video URL."""
    platform = detect_platform(url)
    meta = fetch_metadata(url, platform)

    # Check cache first — instant if hit
    cached = _get_cached_transcript(url)
    if cached:
        transcript, cached_title = cached
        if not meta.title and cached_title:
            meta = VideoMeta(
                video_id=meta.video_id,
                title=cached_title,
                duration_str=meta.duration_str,
                author=meta.author,
                platform=meta.platform,
            )
        return VideoInfo(meta=meta, transcript=transcript)

    transcript: str | None = None

    # Tier 1: try native captions for YouTube
    if platform == Platform.YOUTUBE:
        transcript = _try_youtube_captions(meta.video_id)

    # Tier 2: Groq Whisper API for everything else (or YT without captions)
    if transcript is None:
        transcript = _groq_pipeline(url, progress)

    if len(transcript.split()) < 50:
        raise RuntimeError(
            "Transcript is too short (fewer than 50 words). "
            "Cannot extract meaningful content."
        )

    # Save to cache for next time
    _set_cached_transcript(url, transcript, meta.title)

    return VideoInfo(meta=meta, transcript=transcript)
