"""FastAPI web server for ContentExtractor."""
print("Starting ContentExtractor server...")

import json
import os
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="ContentExtractor")


# ── Health check (defined early so Railway sees it immediately) ───────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Heavy imports (wrapped so server starts even if a dep has issues) ─

try:
    import asyncio
    import base64
    import shutil
    import zipfile
    from datetime import datetime, timezone

    from config import BRANDS, CONTENT_TYPES, OUTPUT_DIR, LOGOS_DIR, DATA_DIR, Palette
    from transcript import detect_platform, fetch_metadata, _try_youtube_captions, _groq_pipeline, PLATFORM_LABELS, Platform
    from analyzer import extract_content  # also registers extra content types
    from image_fetcher import fetch_images_parallel
    from designer import generate_carousel
    from output_formatter import format_all

    _deps_loaded = True
except Exception as _import_err:
    print(f"WARNING: Some imports failed: {_import_err}")
    _deps_loaded = False

# ── Startup key check (warn but don't exit) ──────────────────────────
print(f"NVIDIA_API_KEY loaded: {bool(os.getenv('NVIDIA_API_KEY'))}")
print(f"GROQ_API_KEY loaded: {bool(os.getenv('GROQ_API_KEY'))}")

_missing = []
if not os.getenv("NVIDIA_API_KEY"):
    _missing.append("NVIDIA_API_KEY")
if not os.getenv("GROQ_API_KEY") and not os.getenv("GROQ_API_KEYS"):
    _missing.append("GROQ_API_KEY (or GROQ_API_KEYS)")
if _missing:
    print(f"WARNING: Missing env vars: {', '.join(_missing)}")
    print(f"Check your .env file at: {_env_path}")

# ── Static files ───────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
DOWNLOADS_DIR = Path(__file__).parent / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Request / Response models ──────────────────────────────────────────

class BrandSettings(BaseModel):
    handle: str = "@undercurrenthq"
    name: str = "UndercurrentHQ"
    tagline: str = "The force beneath the surface."
    bg_color: str = "#0A0A0A"
    text_color: str = "#FFFFFF"
    accent_color: str = "#8B5CF6"
    logo: str = ""  # filename of uploaded logo

class ExtractRequest(BaseModel):
    url: str
    mode: str = "facts"
    format: str = "carousel"
    brand: BrandSettings | None = None


class BulkExtractRequest(BaseModel):
    urls: list[str]
    mode: str = "facts"
    format: str = "all"
    brand: BrandSettings | None = None


# ── History store ─────────────────────────────────────────────────────

HISTORY_FILE = DATA_DIR / "history.json"
_MAX_HISTORY = 100


def _load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(entries: list[dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(entries, indent=2))


def _add_history_entry(entry: dict) -> None:
    history = _load_history()
    history.insert(0, entry)
    if len(history) > _MAX_HISTORY:
        history = history[:_MAX_HISTORY]
    _save_history(history)


# ── Main page ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# ── Pipeline helpers (run sync code in thread) ─────────────────────────

async def _run_pipeline(url: str, mode: str, output_format: str, brand: BrandSettings | None = None) -> dict:
    """Run the full extraction pipeline in a background thread."""
    if not _deps_loaded:
        raise HTTPException(503, "Server still starting or missing dependencies. Check logs.")

    if mode not in CONTENT_TYPES:
        valid = ", ".join(CONTENT_TYPES.keys())
        raise HTTPException(400, f"Invalid mode: {mode}. Use: {valid}")

    content_type = CONTENT_TYPES[mode]

    if brand:
        logo_path = ""
        if brand.logo:
            logo_file = LOGOS_DIR / brand.logo
            if logo_file.exists():
                logo_path = str(logo_file)
        palette = Palette(
            bg=brand.bg_color,
            accent=brand.accent_color,
            text=brand.text_color,
            name=brand.name,
            handle=brand.handle,
            tagline=brand.tagline,
            logo_path=logo_path,
        )
    else:
        palette = BRANDS["undercurrent"]

    # 1. Detect platform + fetch metadata
    platform = detect_platform(url)
    meta = await asyncio.to_thread(fetch_metadata, url, platform)

    title_display = meta.title or "Untitled Video"

    # 2. Fetch transcript (two-tier)
    transcript = None
    if platform == Platform.YOUTUBE:
        transcript = await asyncio.to_thread(_try_youtube_captions, meta.video_id)

    if transcript is None:
        transcript = await asyncio.to_thread(_groq_pipeline, url)

    if len(transcript.split()) < 50:
        raise HTTPException(400, "Transcript too short (< 50 words)")

    # 3. Extract content via Groq
    video_title_for_prompt = meta.title or "this video"
    result = await asyncio.to_thread(extract_content, content_type, transcript, video_title_for_prompt)
    items = result.items

    # 4. Fetch background images
    queries = [result.title_image_query] + [item.image_query for item in items]
    image_paths = await asyncio.to_thread(fetch_images_parallel, queries)
    title_image = image_paths[0]
    content_images = image_paths[1:]

    # 5. Generate slides
    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / job_id / content_type.key
    slide_paths = await asyncio.to_thread(
        generate_carousel,
        items=items,
        video_title=title_display,
        content_type=content_type,
        palette=palette,
        output_dir=output_path,
        source=meta.author,
        title_image=title_image,
        content_images=content_images,
    )

    # 6. Create ZIP
    zip_name = f"{job_id}.zip"
    zip_path = DOWNLOADS_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sp in slide_paths:
            zf.write(sp, sp.name)

    # 7. Build slide previews as base64
    slides_data = []
    for sp in slide_paths:
        with open(sp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        slides_data.append({
            "filename": sp.name,
            "data_url": f"data:image/png;base64,{b64}",
        })

    # 8. Generate text formats
    text_formats = {}
    if output_format != "carousel":
        source_channel = meta.author or ""
        handle = palette.handle or f"@{palette.name}"
        all_formats = format_all(result, handle=handle, source_channel=source_channel)
        if output_format == "all":
            text_formats = all_formats
        else:
            key_map = {
                "twitter": "twitter_thread",
                "linkedin": "linkedin_post",
                "newsletter": "newsletter",
                "tiktok": "tiktok_script",
                "caption": "caption",
                "ad_copy": "ad_copy",
            }
            fk = key_map.get(output_format)
            if fk and fk in all_formats:
                text_formats = {fk: all_formats[fk]}

    # 9. Schedule cleanup after 1 hour
    async def _cleanup():
        await asyncio.sleep(3600)
        shutil.rmtree(output_path.parent, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

    asyncio.create_task(_cleanup())

    platform_label = PLATFORM_LABELS.get(platform, "Unknown")

    # 10. Save to history
    thumb_b64 = slides_data[0]["data_url"] if slides_data else ""
    _add_history_entry({
        "job_id": job_id,
        "url": url,
        "title": title_display,
        "source": meta.author or "",
        "platform": platform_label,
        "mode": mode,
        "item_count": len(items),
        "slide_count": len(slide_paths),
        "zip_url": f"/downloads/{zip_name}",
        "thumbnail": thumb_b64,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    response = {
        "title": title_display,
        "source": meta.author,
        "platform": platform_label,
        "item_count": len(items),
        "slides": slides_data,
        "zip_url": f"/downloads/{zip_name}",
    }
    response.update(text_formats)
    return response


# ── API endpoints ──────────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(req: ExtractRequest):
    try:
        result = await _run_pipeline(req.url, req.mode, req.format, req.brand)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR in /api/generate: {e}\n{tb}")
        return JSONResponse(
            {"detail": f"{type(e).__name__}: {str(e)}", "traceback": tb},
            status_code=500,
        )


@app.post("/api/generate-bulk")
async def api_generate_bulk(req: BulkExtractRequest):
    """Process multiple URLs sequentially, return combined results."""
    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:
        raise HTTPException(400, "No URLs provided")
    if len(urls) > 10:
        raise HTTPException(400, "Maximum 10 URLs per batch")

    results: list[dict] = []
    errors: list[dict] = []

    for url in urls:
        try:
            result = await _run_pipeline(url, req.mode, req.format, req.brand)
            result["url"] = url
            result["status"] = "success"
            results.append(result)
        except HTTPException as e:
            errors.append({"url": url, "status": "error", "detail": e.detail})
        except Exception as e:
            errors.append({"url": url, "status": "error", "detail": str(e)[:200]})

    # Build a combined ZIP if multiple successes
    combined_zip_url = ""
    if len(results) > 1:
        bulk_id = str(uuid.uuid4())[:8]
        bulk_zip_name = f"bulk_{bulk_id}.zip"
        bulk_zip_path = DOWNLOADS_DIR / bulk_zip_name
        with zipfile.ZipFile(bulk_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                # Copy each individual zip's contents into a subfolder
                individual_zip = DOWNLOADS_DIR / Path(r["zip_url"]).name
                if individual_zip.exists():
                    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in r["title"])[:40]
                    with zipfile.ZipFile(individual_zip, "r") as izf:
                        for entry in izf.namelist():
                            data = izf.read(entry)
                            zf.writestr(f"{safe_title}/{entry}", data)
        combined_zip_url = f"/downloads/{bulk_zip_name}"

        async def _cleanup_bulk():
            await asyncio.sleep(3600)
            bulk_zip_path.unlink(missing_ok=True)
        asyncio.create_task(_cleanup_bulk())

    return JSONResponse({
        "results": results,
        "errors": errors,
        "total": len(urls),
        "succeeded": len(results),
        "failed": len(errors),
        "combined_zip_url": combined_zip_url,
    })


@app.get("/downloads/{filename}")
async def download_file(filename: str):
    path = DOWNLOADS_DIR / filename
    if not path.exists() or not path.name.endswith(".zip"):
        raise HTTPException(404, "File not found or expired")
    return FileResponse(path, media_type="application/zip", filename=filename)


# ── History API ───────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history():
    return JSONResponse(_load_history())


@app.delete("/api/history")
async def clear_history():
    _save_history([])
    return {"ok": True}


@app.delete("/api/history/{job_id}")
async def delete_history_entry(job_id: str):
    history = _load_history()
    history = [e for e in history if e.get("job_id") != job_id]
    _save_history(history)
    return {"ok": True}


# ── Logo upload ────────────────────────────────────────────────────────

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/svg+xml"}
MAX_LOGO_SIZE = 2 * 1024 * 1024  # 2 MB


@app.post("/api/upload-logo")
async def upload_logo(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Invalid file type: {file.content_type}. Use PNG, JPEG, or WebP.")

    data = await file.read()
    if len(data) > MAX_LOGO_SIZE:
        raise HTTPException(400, "Logo must be under 2 MB")

    ext = Path(file.filename).suffix or ".png"
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    logo_path = LOGOS_DIR / filename
    logo_path.write_bytes(data)

    return {"filename": filename}


@app.get("/api/logos/{filename}")
async def get_logo(filename: str):
    path = LOGOS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Logo not found")
    return FileResponse(path)

