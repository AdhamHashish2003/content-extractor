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


# ── YouTube Transcript (multi-source fallback) ──────────────────────

def _try_youtube_captions(video_id: str) -> str | None:
    """Fetch YouTube transcript with multiple fallbacks for cloud servers.

    1. youtube-transcript-api (uses Innertube internally — best option)
    2. Direct page scrape → extract captionTrack URLs → fetch XML
    """
    import html as htmlmod
    import requests

    # Source 1: youtube-transcript-api — list available transcripts, then fetch best match
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        available = list(transcript_list)
        if available:
            codes = [t.language_code for t in available]
            logger.info("[yt-captions] available languages for %s: %s", video_id, codes)

            # Priority order: manual first, then auto-generated
            manual = [t for t in available if not t.is_generated]
            generated = [t for t in available if t.is_generated]

            # Try preferred languages in order across both pools
            preferred = ["en", "ar"]
            chosen = None
            for pool in [manual, generated]:
                for pref in preferred:
                    for t in pool:
                        if t.language_code == pref or t.language_code.startswith(pref + "-"):
                            chosen = t
                            break
                    if chosen:
                        break
                if chosen:
                    break

            # If no preferred match, take the first available transcript
            if not chosen:
                chosen = manual[0] if manual else available[0]

            result = chosen.fetch()
            text = " ".join(snippet.text for snippet in result)
            if len(text.split()) >= 50:
                logger.info("[yt-captions] youtube-transcript-api OK for %s (lang=%s)", video_id, chosen.language_code)
                return text
    except Exception as e:
        logger.info("[yt-captions] youtube-transcript-api failed: %s", e)

    # Source 2: Scrape caption track URLs from page HTML and fetch XML directly
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        # Set CONSENT cookie to bypass EU consent page
        session.cookies.set("CONSENT", "YES+1", domain=".youtube.com")

        page = session.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        match = re.search(r'"captionTracks":\s*(\[.*?\])', page.text)
        if match:
            import json as _json
            tracks = _json.loads(match.group(1))
            for track in tracks:
                base_url = track.get("baseUrl", "")
                if not base_url:
                    continue
                cc_resp = session.get(base_url, timeout=15)
                if cc_resp.ok and len(cc_resp.text) > 100:
                    texts = re.findall(r"<text[^>]*>(.*?)</text>", cc_resp.text, re.DOTALL)
                    clean = [htmlmod.unescape(t).strip() for t in texts if t.strip()]
                    text = " ".join(clean)
                    if len(text.split()) >= 50:
                        logger.info("[yt-captions] direct page scrape OK for %s", video_id)
                        return text
    except Exception as e:
        logger.info("[yt-captions] direct page scrape failed: %s", e)

    return None


# ── Tier 2: yt-dlp Download + Groq Whisper API ─────────────────────────

class TranscriptionProgress:
    """Callback hooks for the CLI to show progress."""
    def on_download_start(self) -> None: ...
    def on_download_done(self) -> None: ...
    def on_transcribe_start(self) -> None: ...
    def on_transcribe_done(self, word_count: int) -> None: ...
    def on_rate_limit_fallback(self) -> None: ...


def _download_audio(url: str, output_path: Path) -> Path:
    """Download audio via yt-dlp Python API, return path to the audio file."""
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "audioquality": "5",
        "outtmpl": str(output_path / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "5",
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        rc = ydl.download([url])
        if rc != 0:
            raise RuntimeError("yt-dlp failed to download audio")

    audio_files = list(output_path.glob("audio.*"))
    if not audio_files:
        raise RuntimeError("yt-dlp completed but no audio file was produced.")
    return audio_files[0]


def _convert_to_mp3(audio_path: Path, tmp_dir: Path) -> Path:
    """Convert any audio file to mp3 at 64k bitrate for smaller size."""
    audio = AudioSegment.from_file(str(audio_path))
    mp3_path = tmp_dir / "converted.mp3"
    audio.export(str(mp3_path), format="mp3", bitrate="64k")
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
    last_err: RateLimitError | None = None
    for key in keys:
        try:
            client = Groq(api_key=key)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    response_format="text",
                )
            return result.strip()
        except RateLimitError as e:
            logger.info("Key %s...%s rate limited, trying next", key[:8], key[-4:])
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


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
        if progress:
            progress.on_transcribe_done(word_count)

        return transcript


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
