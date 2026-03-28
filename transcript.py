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


# ── YouTube Transcript (multi-source fallback with debug logging) ────

def _try_youtube_captions(video_id: str) -> str | None:
    """Fetch YouTube transcript with multiple fallbacks for cloud servers.

    Source 1: youtube-transcript-api (Innertube internally)
    Source 2: Innertube ANDROID player API (bypasses exp=xpe, works on cloud)
    Returns None if all sources fail — caller can fall back to Groq Whisper.
    """
    import requests

    debug: list[str] = []  # Accumulate debug info for error messages

    # ── Source 1: youtube-transcript-api ──────────────────────────────
    debug.append(f"=== CAPTION DEBUG for {video_id} ===")
    debug.append("Source 1 (youtube-transcript-api): trying...")
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        available = list(transcript_list)
        if available:
            codes = [(t.language_code, "auto" if t.is_generated else "manual") for t in available]
            debug.append(f"Source 1: languages found: {codes}")
            logger.info("[yt-captions] Source 1 languages for %s: %s", video_id, codes)

            manual = [t for t in available if not t.is_generated]
            generated = [t for t in available if t.is_generated]

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
            if not chosen:
                chosen = manual[0] if manual else available[0]

            debug.append(f"Source 1: chose {chosen.language_code}, fetching...")
            result = chosen.fetch()
            text = " ".join(snippet.text for snippet in result)
            word_count = len(text.split())
            debug.append(f"Source 1: fetched {word_count} words")

            if word_count >= 50:
                logger.info("[yt-captions] Source 1 OK for %s (lang=%s, words=%d)",
                            video_id, chosen.language_code, word_count)
                return text
            else:
                debug.append(f"Source 1: too few words ({word_count}), skipping")
        else:
            debug.append("Source 1: no transcripts available")
    except Exception as e:
        err_type = type(e).__name__
        debug.append(f"Source 1 FAILED: {err_type}: {e}")
        logger.info("[yt-captions] Source 1 failed for %s: %s: %s", video_id, err_type, e)

    # ── Source 2: Innertube ANDROID player API ───────────────────────
    debug.append("Source 2 (Innertube ANDROID): trying...")
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        # Robust consent cookie to bypass EU consent page
        session.cookies.set("CONSENT", "YES+cb.20210328-17-p0.en+FX+999", domain=".youtube.com")

        # Fetch page to get API key (with fallback to hardcoded public key)
        page = session.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        debug.append(f"Source 2: page fetch status={page.status_code} len={len(page.text)}")

        api_key_match = re.search(r'"INNERTUBE_API_KEY":\s*"([a-zA-Z0-9_-]+)"', page.text)
        api_key = api_key_match.group(1) if api_key_match else "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
        debug.append(f"Source 2: API key {'from page' if api_key_match else 'HARDCODED FALLBACK'}: {api_key[:20]}...")

        # Call player API with ANDROID client
        player_resp = session.post(
            f"https://www.youtube.com/youtubei/v1/player?key={api_key}",
            json={
                "context": {"client": {"clientName": "ANDROID", "clientVersion": "20.10.38"}},
                "videoId": video_id,
            },
            timeout=15,
        )
        debug.append(f"Source 2: player response status={player_resp.status_code}")

        if player_resp.ok:
            player_data = player_resp.json()

            # Check both possible caption paths
            captions_obj = player_data.get("captions", {})
            captions_data = (
                captions_obj.get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
                or captions_obj.get("playerCaptionsRenderer", {}).get("captionTracks", [])
            )

            if captions_data:
                langs = [(t.get("languageCode", "?"), t.get("kind", "")) for t in captions_data]
                debug.append(f"Source 2: {len(captions_data)} tracks found: {langs}")
                logger.info("[yt-captions] Source 2 found %d tracks for %s: %s",
                            len(captions_data), video_id, langs)

                captions_data.sort(key=_track_sort_key)

                for track in captions_data:
                    base_url = track.get("baseUrl", "")
                    lang = track.get("languageCode", "?")
                    if not base_url:
                        debug.append(f"Source 2: track {lang} has no baseUrl, skipping")
                        continue

                    has_xpe = "exp=xpe" in base_url
                    debug.append(f"Source 2: fetching track {lang} (xpe={has_xpe})")

                    cc_resp = session.get(base_url, timeout=15)
                    debug.append(f"Source 2: caption response status={cc_resp.status_code} len={len(cc_resp.text)}")

                    if cc_resp.ok and len(cc_resp.text) > 100:
                        text = _parse_caption_xml(cc_resp.text)
                        word_count = len(text.split())
                        debug.append(f"Source 2: parsed {word_count} words from {lang}")

                        if word_count >= 50:
                            logger.info("[yt-captions] Source 2 OK for %s (lang=%s, words=%d)",
                                        video_id, lang, word_count)
                            return text
                        else:
                            debug.append(f"Source 2: too few words from {lang}, trying next track")
                    else:
                        debug.append(f"Source 2: empty/short response for {lang}")
            else:
                debug.append("Source 2: no captionTracks in player response")
                # Log what keys ARE present for debugging
                cap_keys = list(captions_obj.keys()) if captions_obj else []
                debug.append(f"Source 2: captions object keys: {cap_keys}")
        else:
            debug.append(f"Source 2: player API returned {player_resp.status_code}")
    except Exception as e:
        err_type = type(e).__name__
        debug.append(f"Source 2 FAILED: {err_type}: {e}")
        logger.info("[yt-captions] Source 2 failed for %s: %s: %s", video_id, err_type, e)

    # ── All sources failed — log full debug trace ────────────────────
    debug.append(f"=== FINAL RESULT: FAILED (all sources exhausted) ===")
    debug_text = "\n".join(debug)
    logger.warning("[yt-captions] All caption sources failed for %s:\n%s", video_id, debug_text)

    # Store debug info so the API layer can include it in the error message
    _try_youtube_captions._last_debug = debug_text  # type: ignore[attr-defined]
    return None


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
    import requests

    logger.info("[audio-download] Trying Innertube ANDROID for %s", video_id)
    try:
        session = requests.Session()
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        session.cookies.set("CONSENT", "YES+cb.20210328-17-p0.en+FX+999", domain=".youtube.com")

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

    # Fallback: yt-dlp (works locally, may fail on cloud)
    import yt_dlp

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
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
    logger.info("[audio-download] Trying yt-dlp for %s", url[:80])
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
        "format": "worst[height>=360][ext=mp4]/worst[ext=mp4]/worst",
        "outtmpl": str(video_path),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
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
