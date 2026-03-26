"""User database: SQLite-backed auth and plan management."""

import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from passlib.hash import bcrypt
from config import DATA_DIR

DB_PATH = DATA_DIR / "users.db"

# ── Plan limits ──────────────────────────────────────────────────────
FREE_DAILY_LIMIT = 3

PLAN_LIMITS = {
    "free": FREE_DAILY_LIMIT,
    "pro": None,      # unlimited
    "agency": None,    # unlimited
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create users table if it doesn't exist."""
    conn = _connect()
    conn.execute("""
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
    conn.commit()
    conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


# ── User CRUD ─────────────────────────────────────────────────────────

def create_user(email: str, password: str) -> dict:
    """Create a new user. Returns user dict. Raises ValueError if email exists."""
    user_id = str(uuid.uuid4())
    pw_hash = bcrypt.hash(password)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
            (user_id, email.lower().strip(), pw_hash),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Email already registered")
    user = _row_to_dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
    conn.close()
    return user


def get_user_by_email(email: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    return _row_to_dict(row)


def get_user_by_id(user_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.verify(plain, hashed)


# ── Plan management ───────────────────────────────────────────────────

def update_plan(user_id: str, plan: str, stripe_customer_id: str, stripe_sub_id: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE users SET plan = ?, stripe_customer_id = ?, stripe_subscription_id = ? WHERE id = ?",
        (plan, stripe_customer_id, stripe_sub_id, user_id),
    )
    conn.commit()
    conn.close()


def cancel_plan(user_id: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE users SET plan = 'free', stripe_subscription_id = NULL WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def find_user_by_stripe_customer(customer_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


# ── Usage tracking ────────────────────────────────────────────────────

def increment_daily_usage(user_id: str) -> int:
    """Increment daily counter, resetting if new day. Returns new count."""
    today = date.today().isoformat()
    conn = _connect()
    row = conn.execute("SELECT daily_extractions, last_extraction_date FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        return 0
    if row["last_extraction_date"] != today:
        count = 1
        conn.execute(
            "UPDATE users SET daily_extractions = 1, last_extraction_date = ? WHERE id = ?",
            (today, user_id),
        )
    else:
        count = row["daily_extractions"] + 1
        conn.execute(
            "UPDATE users SET daily_extractions = ? WHERE id = ?",
            (count, user_id),
        )
    conn.commit()
    conn.close()
    return count


def get_daily_usage(user_id: str) -> int:
    today = date.today().isoformat()
    conn = _connect()
    row = conn.execute("SELECT daily_extractions, last_extraction_date FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        return 0
    if row["last_extraction_date"] != today:
        return 0
    return row["daily_extractions"]


def check_can_extract(user_id: str) -> bool:
    """Return True if user hasn't exceeded their plan limit."""
    conn = _connect()
    row = conn.execute("SELECT plan, daily_extractions, last_extraction_date FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        return False
    limit = PLAN_LIMITS.get(row["plan"])
    if limit is None:
        return True  # unlimited
    today = date.today().isoformat()
    used = row["daily_extractions"] if row["last_extraction_date"] == today else 0
    return used < limit


# ── Initialize on import ──────────────────────────────────────────────
init_db()
