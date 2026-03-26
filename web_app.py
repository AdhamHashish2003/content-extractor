"""FastAPI web server for ContentExtractor."""
print("Starting ContentExtractor server...")

import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

import re

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("contentextractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

    from config import BRANDS, DEFAULT_BRAND, CONTENT_TYPES, OUTPUT_DIR, LOGOS_DIR, DATA_DIR, Palette
    from transcript import detect_platform, fetch_metadata, _try_youtube_captions, _groq_pipeline, PLATFORM_LABELS, Platform
    from analyzer import extract_content  # also registers extra content types
    from image_fetcher import fetch_images_parallel, cleanup_temp_images
    from designer import generate_carousel
    from output_formatter import format_all

    _generation_sem = asyncio.Semaphore(5)
    _deps_loaded = True
except Exception as _import_err:
    print(f"WARNING: Some imports failed: {_import_err}")
    _deps_loaded = False

# ── Auth / Stripe imports ────────────────────────────────────────────
try:
    import stripe as _stripe_mod
    from jose import jwt, JWTError

    import database as db

    _STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
    _STRIPE_PUB_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    _JWT_SECRET = os.getenv("JWT_SECRET", "")

    _PLANS = {
        "pro": {"name": "ContentExtractor AI Pro", "amount": 1900, "interval": "month"},
        "agency": {"name": "ContentExtractor AI Agency", "amount": 4900, "interval": "month"},
    }
    _JWT_ALGORITHM = "HS256"
    _JWT_EXPIRE_DAYS = 30

    if _STRIPE_SECRET:
        _stripe_mod.api_key = _STRIPE_SECRET
    _auth_loaded = True
except Exception as _auth_err:
    print(f"WARNING: Auth/Stripe imports failed: {_auth_err}")
    _auth_loaded = False

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
    handle: str = ""
    name: str = ""
    tagline: str = ""
    bg_color: str = "#0A0A0A"
    text_color: str = "#FFFFFF"
    accent_color: str = "#8B5CF6"
    logo: str = ""  # filename of uploaded logo

class ExtractRequest(BaseModel):
    url: str
    mode: str = "facts"
    format: str = "carousel"
    brand: BrandSettings | None = None
    num_items: int = 5


# ── Rate limiting & concurrency ─────────────────────────────────────
_rate_limits: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 3600  # 1 hour

TEXT_ONLY_MODES = {"summary", "hooks"}


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed."""
    now = time.time()
    _rate_limits[ip] = [t for t in _rate_limits[ip] if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_limits[ip]) >= _RATE_LIMIT_MAX:
        return False
    _rate_limits[ip].append(now)
    return True


def _friendly_error(detail: str) -> str:
    """Map technical errors to user-friendly messages."""
    d = detail.lower()
    if "no captions" in d or "could not get transcript" in d:
        return "This video doesn't have captions available. Try a different video."
    if "ratelimit" in d:
        return "We're experiencing high demand. Please try again in a minute."
    if "timeout" in d or "timed out" in d:
        return "Processing took too long. Try a shorter video."
    if "transcript too short" in d:
        return "This video doesn't have enough spoken content. Try a different video."
    return "Something went wrong. Please try again."


class BulkExtractRequest(BaseModel):
    urls: list[str]
    mode: str = "facts"
    format: str = "all"
    brand: BrandSettings | None = None


# ── History (now in database) ─────────────────────────────────────────


# ── Main page ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# ── Auth helpers ──────────────────────────────────────────────────────

def _create_jwt(user_id: str, email: str) -> str:
    from datetime import timedelta
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=_JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict | None:
    """Decode a JWT token. Returns payload dict or None on failure."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except JWTError:
        return None


def _get_current_user(authorization: str | None) -> dict | None:
    """Extract user from Authorization header. Returns user dict or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    payload = _decode_jwt(token)
    if not payload:
        return None
    return db.get_user_by_id(payload["sub"])


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ── Auth endpoints ───────────────────────────────────────────────────

@app.post("/api/auth/signup")
async def auth_signup(request: Request):
    if not _auth_loaded:
        raise HTTPException(503, "Auth system unavailable")
    body = await request.json()
    email = (body.get("email") or "").strip()
    password = body.get("password") or ""

    if not email or not _EMAIL_RE.match(email):
        raise HTTPException(400, "Invalid email address")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    try:
        user = db.create_user(email, password)
    except ValueError:
        raise HTTPException(409, "Email already registered")

    token = _create_jwt(user["id"], user["email"])
    return JSONResponse({
        "token": token,
        "user": {"email": user["email"], "plan": user["plan"]},
    })


@app.post("/api/auth/login")
async def auth_login(request: Request):
    if not _auth_loaded:
        raise HTTPException(503, "Auth system unavailable")
    body = await request.json()
    email = (body.get("email") or "").strip()
    password = body.get("password") or ""

    user = db.get_user_by_email(email)
    if not user or not db.verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    token = _create_jwt(user["id"], user["email"])
    return JSONResponse({
        "token": token,
        "user": {"email": user["email"], "plan": user["plan"]},
    })


@app.get("/api/auth/me")
async def auth_me(request: Request):
    if not _auth_loaded:
        raise HTTPException(503, "Auth system unavailable")
    user = _get_current_user(request.headers.get("authorization"))
    if not user:
        raise HTTPException(401, "Not authenticated")
    usage = db.get_daily_usage(user["id"])
    can_extract = db.check_can_extract(user["id"])
    return JSONResponse({
        "email": user["email"],
        "plan": user["plan"],
        "daily_usage": usage,
        "can_extract": can_extract,
    })


# ── Stripe endpoints ─────────────────────────────────────────────────

@app.get("/api/config/stripe")
async def stripe_config():
    """Return the publishable key for the frontend."""
    return JSONResponse({"publishable_key": _STRIPE_PUB_KEY if _auth_loaded else ""})


@app.post("/api/checkout")
async def create_checkout(request: Request):
    if not _auth_loaded or not _STRIPE_SECRET:
        raise HTTPException(503, "Payments unavailable")
    user = _get_current_user(request.headers.get("authorization"))
    if not user:
        raise HTTPException(401, "Not authenticated")

    body = await request.json()
    plan = body.get("plan", "pro")
    plan_info = _PLANS.get(plan)
    if not plan_info:
        raise HTTPException(400, f"Invalid plan: {plan}")

    base_url = str(request.base_url).rstrip("/")
    session = _stripe_mod.checkout.Session.create(
        mode="subscription",
        customer_email=user["email"],
        line_items=[{
            "price_data": {
                "product_data": {"name": plan_info["name"]},
                "currency": "usd",
                "unit_amount": plan_info["amount"],
                "recurring": {"interval": plan_info["interval"]},
            },
            "quantity": 1,
        }],
        success_url=f"{base_url}?payment=success&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base_url}?payment=cancelled",
        metadata={"user_id": user["id"], "plan": plan},
    )
    return JSONResponse({"checkout_url": session.url})


@app.get("/api/checkout/verify")
async def verify_checkout(request: Request):
    """Verify a completed checkout session and activate the plan."""
    if not _auth_loaded or not _STRIPE_SECRET:
        raise HTTPException(503, "Payments unavailable")
    user = _get_current_user(request.headers.get("authorization"))
    if not user:
        raise HTTPException(401, "Not authenticated")

    session_id = request.query_params.get("session_id", "")
    if not session_id:
        raise HTTPException(400, "Missing session_id")

    try:
        session = _stripe_mod.checkout.Session.retrieve(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session")

    if session.payment_status == "paid":
        plan = session.metadata.get("plan", "pro")
        customer_id = session.customer or ""
        sub_id = session.subscription or ""
        db.update_plan(user["id"], plan, customer_id, sub_id)
        return JSONResponse({"status": "success", "plan": plan})

    return JSONResponse({"status": "pending"})


@app.get("/api/billing/portal")
async def billing_portal(request: Request):
    if not _auth_loaded or not _STRIPE_SECRET:
        raise HTTPException(503, "Payments unavailable")
    user = _get_current_user(request.headers.get("authorization"))
    if not user:
        raise HTTPException(401, "Not authenticated")
    if not user.get("stripe_customer_id"):
        raise HTTPException(400, "No active subscription")

    base_url = str(request.base_url).rstrip("/")
    session = _stripe_mod.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=base_url,
    )
    return JSONResponse({"portal_url": session.url})


# ── Pipeline helpers (run sync code in thread) ─────────────────────────

_PIPELINE_TIMEOUT = 120  # seconds
_MAX_TRANSCRIPT_WORDS = 15000


async def _run_pipeline(url: str, mode: str, output_format: str, brand: BrandSettings | None = None, num_items: int = 5, watermark: bool = False, user_id: str | None = None) -> dict:
    """Run the full extraction pipeline with per-step error handling."""
    if not _deps_loaded:
        raise HTTPException(503, "Server still starting or missing dependencies. Check logs.")

    if mode not in CONTENT_TYPES:
        valid = ", ".join(CONTENT_TYPES.keys())
        raise HTTPException(400, f"Invalid mode: {mode}. Use: {valid}")

    content_type = CONTENT_TYPES[mode]
    num_items = max(3, min(7, num_items))
    is_text_only = mode in TEXT_ONLY_MODES

    if brand:
        logo_path = ""
        if brand.logo:
            logo_file = LOGOS_DIR / brand.logo
            print(f"[PIPELINE] Logo filename from request: {brand.logo!r}, full path: {logo_file}, exists: {logo_file.exists()}")
            if logo_file.exists():
                logo_path = str(logo_file)
        palette = Palette(
            bg=brand.bg_color, accent=brand.accent_color, text=brand.text_color,
            name=brand.name, handle=brand.handle, tagline=brand.tagline, logo_path=logo_path,
        )
    else:
        palette = BRANDS.get(DEFAULT_BRAND, list(BRANDS.values())[0])

    # Track temp files for cleanup
    temp_image_files: list[Path | None] = []

    try:
        # 1. Detect platform + fetch metadata
        try:
            platform = detect_platform(url)
            meta = await asyncio.wait_for(
                asyncio.to_thread(fetch_metadata, url, platform), timeout=30
            )
        except asyncio.TimeoutError:
            raise HTTPException(400, "Metadata fetch timed out. Check the URL.")
        except Exception as e:
            raise HTTPException(400, f"Could not fetch video metadata. {e}")

        title_display = meta.title or "Untitled Video"

        # 2. Fetch transcript
        transcript = None
        if platform == Platform.YOUTUBE:
            try:
                transcript = await asyncio.wait_for(
                    asyncio.to_thread(_try_youtube_captions, meta.video_id), timeout=15
                )
            except Exception:
                transcript = None
            if not transcript:
                raise HTTPException(400,
                    "This YouTube video has no captions available. "
                    "Try a different video with subtitles, or use an Instagram/TikTok/X link instead.")
        else:
            try:
                transcript = await asyncio.wait_for(
                    asyncio.to_thread(_groq_pipeline, url), timeout=90
                )
            except asyncio.TimeoutError:
                raise HTTPException(400, "Transcript extraction timed out.")
            except Exception as e:
                raise HTTPException(400, f"Could not get transcript. ({e})")

        if not transcript or len(transcript.split()) < 50:
            raise HTTPException(400, "Transcript too short (< 50 words). The video may not have spoken content.")

        words = transcript.split()
        if len(words) > _MAX_TRANSCRIPT_WORDS:
            transcript = " ".join(words[:_MAX_TRANSCRIPT_WORDS])

        # 3. Extract content via LLM
        try:
            video_title_for_prompt = meta.title or "this video"
            result = await asyncio.wait_for(
                asyncio.to_thread(extract_content, content_type, transcript, video_title_for_prompt, num_items),
                timeout=120,
            )
            items = result.items
        except asyncio.TimeoutError:
            raise HTTPException(400, "AI analysis timed out. Try a shorter video.")
        except Exception as e:
            raise HTTPException(400, f"AI content extraction error. {e}")

        job_id = str(uuid.uuid4())[:8]
        slides_data = []
        slide_paths = []
        zip_name = ""
        zip_path = None
        output_path = None

        if not is_text_only:
            # 4. Fetch background images (non-fatal)
            title_image = None
            content_images = [None] * len(items)
            try:
                queries = [result.title_image_query] + [item.image_query for item in items]
                image_paths = await asyncio.wait_for(
                    asyncio.to_thread(fetch_images_parallel, queries), timeout=25
                )
                temp_image_files = list(image_paths)
                title_image = image_paths[0]
                content_images = image_paths[1:]
            except Exception as e:
                logger.warning("Image fetching failed, using white fallback: %s", e)

            # 5. Generate slides
            try:
                output_path = OUTPUT_DIR / job_id / content_type.key
                slide_paths = await asyncio.to_thread(
                    generate_carousel, items=items, video_title=title_display,
                    content_type=content_type, palette=palette, output_dir=output_path,
                    source=meta.author, title_image=title_image, content_images=content_images,
                    watermark=watermark,
                )
            except Exception as e:
                raise HTTPException(500, f"Slide generation error. {e}")

            # 6. Create ZIP
            zip_name = f"{job_id}.zip"
            zip_path = DOWNLOADS_DIR / zip_name
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for sp in slide_paths:
                    zf.write(sp, sp.name)

            # 7. Build slide previews as base64
            for sp in slide_paths:
                with open(sp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                slides_data.append({"filename": sp.name, "data_url": f"data:image/png;base64,{b64}"})

        # 8. Generate text formats
        text_formats = {}
        if output_format != "carousel" or is_text_only:
            try:
                source_channel = meta.author or ""
                handle = palette.handle or (f"@{palette.name}" if palette.name else "")
                all_formats = format_all(result, handle=handle, source_channel=source_channel)
                if output_format == "all" or is_text_only:
                    text_formats = all_formats
                else:
                    key_map = {"twitter": "twitter_thread", "linkedin": "linkedin_post",
                               "newsletter": "newsletter", "tiktok": "tiktok_script",
                               "caption": "caption", "ad_copy": "ad_copy"}
                    fk = key_map.get(output_format)
                    if fk and fk in all_formats:
                        text_formats = {fk: all_formats[fk]}
            except Exception as e:
                logger.warning("Text format generation failed: %s", e)

        # 9. Schedule cleanup after 1 hour
        if output_path or zip_path:
            _output_path = output_path
            _zip_path = zip_path
            async def _cleanup():
                await asyncio.sleep(3600)
                if _output_path:
                    shutil.rmtree(_output_path.parent, ignore_errors=True)
                if _zip_path:
                    _zip_path.unlink(missing_ok=True)
            asyncio.create_task(_cleanup())

        platform_label = PLATFORM_LABELS.get(platform, "Unknown")

        # 10. Save to history (non-fatal)
        try:
            thumb_b64 = slides_data[0]["data_url"] if slides_data else ""
            if _auth_loaded:
                db.add_extraction({
                    "job_id": job_id, "user_id": user_id, "url": url, "title": title_display,
                    "source": meta.author or "", "platform": platform_label, "mode": mode,
                    "item_count": len(items), "slide_count": len(slide_paths),
                    "zip_url": f"/downloads/{zip_name}" if zip_name else "", "thumbnail": thumb_b64,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            pass

        response = {
            "title": title_display, "source": meta.author, "platform": platform_label,
            "item_count": len(items), "slides": slides_data,
            "zip_url": f"/downloads/{zip_name}" if zip_name else "",
            "has_slides": not is_text_only,
        }
        response.update(text_formats)
        return response

    finally:
        # Clean up temp image files from fetcher
        cleanup_temp_images(temp_image_files)


# ── API endpoints ──────────────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(req: ExtractRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        return JSONResponse(
            {"detail": "We're experiencing high demand. Please try again in a minute."},
            status_code=429,
        )

    # ── Plan enforcement ──
    watermark = True  # default for anonymous / free
    user = None
    if _auth_loaded:
        user = _get_current_user(request.headers.get("authorization"))
    if user:
        plan = user.get("plan", "free")
        if plan in ("pro", "agency"):
            watermark = False
        else:
            # free plan — check daily limit
            if not db.check_can_extract(user["id"]):
                return JSONResponse(
                    {"detail": "Daily limit reached. Upgrade to Pro for unlimited extractions.",
                     "limit_reached": True},
                    status_code=429,
                )
    # anonymous users get watermark but no limit check (IP rate limit still applies)

    start_time = time.time()
    # Concurrency gate — return 503 immediately if all slots are busy
    try:
        await asyncio.wait_for(_generation_sem.acquire(), timeout=0.5)
    except (asyncio.TimeoutError, Exception):
        return JSONResponse(
            {"detail": "Server busy, please try again in a moment."},
            status_code=503,
        )
    try:
        _uid = user["id"] if user else None
        result = await _run_pipeline(req.url, req.mode, req.format, req.brand, req.num_items, watermark=watermark, user_id=_uid)
        # Increment usage for logged-in users
        if user and _auth_loaded:
            db.increment_daily_usage(user["id"])
        result["watermark"] = watermark
        elapsed = time.time() - start_time
        logger.info("OK ip=%s url=%s mode=%s items=%d time=%.1fs", ip, req.url[:80], req.mode, req.num_items, elapsed)
        return JSONResponse(result)
    except HTTPException as e:
        elapsed = time.time() - start_time
        logger.warning("FAIL ip=%s url=%s mode=%s time=%.1fs err=%s", ip, req.url[:80], req.mode, elapsed, e.detail[:120])
        return JSONResponse({"detail": _friendly_error(e.detail)}, status_code=e.status_code)
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("ERROR ip=%s url=%s mode=%s time=%.1fs", ip, req.url[:80], req.mode, elapsed, exc_info=True)
        return JSONResponse({"detail": "Something went wrong. Please try again."}, status_code=500)
    finally:
        _generation_sem.release()


@app.post("/api/generate-bulk")
async def api_generate_bulk(req: BulkExtractRequest, request: Request):
    """Process multiple URLs sequentially, return combined results."""
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        return JSONResponse(
            {"detail": "We're experiencing high demand. Please try again in a minute."},
            status_code=429,
        )

    # ── Require pro/agency for bulk ──
    user = None
    if _auth_loaded:
        user = _get_current_user(request.headers.get("authorization"))
    if not user or user.get("plan", "free") not in ("pro", "agency"):
        return JSONResponse(
            {"detail": "Upgrade to Pro for bulk processing."},
            status_code=403,
        )

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
    if _auth_loaded:
        return JSONResponse(db.get_history())
    return JSONResponse([])


@app.delete("/api/history")
async def clear_history():
    if _auth_loaded:
        db.clear_history()
    return {"ok": True}


@app.delete("/api/history/{job_id}")
async def delete_history_entry(job_id: str):
    if _auth_loaded:
        db.delete_extraction(job_id)
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

