"""Database layer: PostgreSQL in production (Railway), SQLite for local dev."""

import json
import os
import uuid
from datetime import date, datetime, timezone

from passlib.hash import bcrypt

DATABASE_URL = os.getenv("DATABASE_URL")

# ── Plan limits ──────────────────────────────────────────────────────
FREE_DAILY_LIMIT = 3
PLAN_LIMITS = {
    "free": FREE_DAILY_LIMIT,
    "pro": None,
    "agency": None,
}

_MAX_HISTORY = 100

# ── Connection helpers ───────────────────────────────────────────────

_use_pg = bool(DATABASE_URL)


def _connect():
    """Return (conn, placeholder) — placeholder is %s for PG, ? for SQLite."""
    if _use_pg:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn, "%s"
    else:
        import sqlite3
        from config import DATA_DIR
        path = DATA_DIR / "users.db"
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn, "?"


def _row_to_dict(row) -> dict | None:
    if row is None:
        return None
    return dict(row)


def _fetchone(cur) -> dict | None:
    row = cur.fetchone()
    return _row_to_dict(row)


def _fetchall(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


# ── Schema ───────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist."""
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            plan TEXT DEFAULT 'free',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            daily_extractions INTEGER DEFAULT 0,
            last_extraction_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS extractions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            url TEXT,
            platform TEXT,
            video_title TEXT,
            mode TEXT,
            num_slides INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result_json TEXT,
            zip_path TEXT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


# ── User CRUD ─────────────────────────────────────────────────────────

def create_user(email: str, password: str) -> dict:
    """Create a new user. Raises ValueError if email exists."""
    user_id = str(uuid.uuid4())
    pw_hash = bcrypt.hash(password)
    conn, ph = _connect()
    cur = conn.cursor()
    try:
        cur.execute(
            f"INSERT INTO users (id, email, password_hash) VALUES ({ph}, {ph}, {ph})",
            (user_id, email.lower().strip(), pw_hash),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        if "unique" in str(e).lower() or "duplicate" in str(e).lower() or "integrity" in str(e).lower():
            raise ValueError("Email already registered")
        raise
    cur.execute(f"SELECT * FROM users WHERE id = {ph}", (user_id,))
    user = _fetchone(cur)
    cur.close()
    conn.close()
    return user


def get_user_by_email(email: str) -> dict | None:
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE email = {ph}", (email.lower().strip(),))
    user = _fetchone(cur)
    cur.close()
    conn.close()
    return user


def get_user_by_id(user_id: str) -> dict | None:
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE id = {ph}", (user_id,))
    user = _fetchone(cur)
    cur.close()
    conn.close()
    return user


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.verify(plain, hashed)


# ── Plan management ───────────────────────────────────────────────────

def update_plan(user_id: str, plan: str, stripe_customer_id: str, stripe_sub_id: str) -> None:
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE users SET plan = {ph}, stripe_customer_id = {ph}, stripe_subscription_id = {ph} WHERE id = {ph}",
        (plan, stripe_customer_id, stripe_sub_id, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def cancel_plan(user_id: str) -> None:
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE users SET plan = 'free', stripe_subscription_id = NULL WHERE id = {ph}",
        (user_id,),
    )
    conn.commit()
    cur.close()
    conn.close()


def find_user_by_stripe_customer(customer_id: str) -> dict | None:
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE stripe_customer_id = {ph}", (customer_id,))
    user = _fetchone(cur)
    cur.close()
    conn.close()
    return user


# ── Usage tracking ────────────────────────────────────────────────────

def increment_daily_usage(user_id: str) -> int:
    """Increment daily counter, resetting if new day. Returns new count."""
    today = date.today().isoformat()
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT daily_extractions, last_extraction_date FROM users WHERE id = {ph}", (user_id,))
    row = _fetchone(cur)
    if row is None:
        cur.close()
        conn.close()
        return 0
    if row["last_extraction_date"] != today:
        count = 1
        cur.execute(
            f"UPDATE users SET daily_extractions = 1, last_extraction_date = {ph} WHERE id = {ph}",
            (today, user_id),
        )
    else:
        count = row["daily_extractions"] + 1
        cur.execute(
            f"UPDATE users SET daily_extractions = {ph} WHERE id = {ph}",
            (count, user_id),
        )
    conn.commit()
    cur.close()
    conn.close()
    return count


def get_daily_usage(user_id: str) -> int:
    today = date.today().isoformat()
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT daily_extractions, last_extraction_date FROM users WHERE id = {ph}", (user_id,))
    row = _fetchone(cur)
    cur.close()
    conn.close()
    if row is None:
        return 0
    if row["last_extraction_date"] != today:
        return 0
    return row["daily_extractions"]


def check_can_extract(user_id: str) -> bool:
    """Return True if user hasn't exceeded their plan limit."""
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"SELECT plan, daily_extractions, last_extraction_date FROM users WHERE id = {ph}", (user_id,))
    row = _fetchone(cur)
    cur.close()
    conn.close()
    if row is None:
        return False
    limit = PLAN_LIMITS.get(row["plan"])
    if limit is None:
        return True
    today = date.today().isoformat()
    used = row["daily_extractions"] if row["last_extraction_date"] == today else 0
    return used < limit


# ── Extraction history ────────────────────────────────────────────────

def add_extraction(entry: dict) -> None:
    """Save an extraction record to the database."""
    ext_id = entry.get("job_id") or str(uuid.uuid4())[:8]
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(
        f"""INSERT INTO extractions (id, user_id, url, platform, video_title, mode, num_slides, created_at, result_json, zip_path)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
        (
            ext_id,
            entry.get("user_id"),
            entry.get("url", ""),
            entry.get("platform", ""),
            entry.get("title", ""),
            entry.get("mode", ""),
            entry.get("slide_count", 0),
            entry.get("created_at", datetime.now(timezone.utc).isoformat()),
            json.dumps({
                "source": entry.get("source", ""),
                "item_count": entry.get("item_count", 0),
                "thumbnail": entry.get("thumbnail", ""),
            }),
            entry.get("zip_url", ""),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_history(user_id: str | None = None, limit: int = _MAX_HISTORY) -> list[dict]:
    """Return extraction history, newest first. If user_id given, filter to that user."""
    conn, ph = _connect()
    cur = conn.cursor()
    if user_id:
        cur.execute(
            f"SELECT * FROM extractions WHERE user_id = {ph} ORDER BY created_at DESC LIMIT {ph}",
            (user_id, limit),
        )
    else:
        cur.execute(
            f"SELECT * FROM extractions ORDER BY created_at DESC LIMIT {ph}",
            (limit,),
        )
    rows = _fetchall(cur)
    cur.close()
    conn.close()
    # Re-map to the format the frontend expects
    result = []
    for r in rows:
        extra = {}
        try:
            extra = json.loads(r.get("result_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        result.append({
            "job_id": r["id"],
            "url": r.get("url", ""),
            "title": r.get("video_title", ""),
            "source": extra.get("source", ""),
            "platform": r.get("platform", ""),
            "mode": r.get("mode", ""),
            "item_count": extra.get("item_count", 0),
            "slide_count": r.get("num_slides", 0),
            "zip_url": r.get("zip_path", ""),
            "thumbnail": extra.get("thumbnail", ""),
            "created_at": r.get("created_at", ""),
        })
    return result


def clear_history(user_id: str | None = None) -> None:
    conn, ph = _connect()
    cur = conn.cursor()
    if user_id:
        cur.execute(f"DELETE FROM extractions WHERE user_id = {ph}", (user_id,))
    else:
        cur.execute("DELETE FROM extractions")
    conn.commit()
    cur.close()
    conn.close()


def delete_extraction(job_id: str) -> None:
    conn, ph = _connect()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM extractions WHERE id = {ph}", (job_id,))
    conn.commit()
    cur.close()
    conn.close()


# ── Initialize on import ──────────────────────────────────────────────
init_db()
