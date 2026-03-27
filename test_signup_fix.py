"""Targeted tests for the signup validation fix."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
DB = Path(__file__).parent / "data" / "users.db"

def api(method, path, data=None, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    r = requests.request(method, BASE + path, json=data, headers=headers, timeout=10)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}

def db_query(sql, params=()):
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows

def db_exec(sql, params=()):
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    cur.close()
    conn.close()

# Wait for server
for _ in range(15):
    try:
        if requests.get(BASE + "/health", timeout=2).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)

print("=" * 65)
print("SIGNUP VALIDATION FIX — TARGETED TESTS")
print("=" * 65)

EMAIL = "fixtest@example.com"
GOOD_PW = "CorrectPass123"
BAD_PW = "abc"  # too short
MISMATCH_PW = "DifferentPass999"

all_pass = True

# ── Scenario 1: Mismatched passwords should be blocked client-side.
# But to test the BACKEND safety net, simulate what happens if the
# client-side check FAILS and the request reaches the server with
# a short/empty password.
print("\n--- Scenario 1: Backend rejects short password ---")
st, body = api("POST", "/api/auth/signup", {"email": EMAIL, "password": BAD_PW})
print(f"  POST /api/auth/signup (pw='{BAD_PW}')  →  status={st}")
print(f"  Response: {body}")
rows = db_query("SELECT * FROM users WHERE email = ?", (EMAIL,))
user_created = len(rows) > 0
print(f"  User in DB? {user_created}")
if st == 400 and not user_created:
    print("  ✓ PASS: Backend blocked short password, no user created")
else:
    print("  ✗ FAIL: Expected 400 + no user record")
    all_pass = False

# ── Scenario 2: Backend rejects empty password
print("\n--- Scenario 2: Backend rejects empty password ---")
st, body = api("POST", "/api/auth/signup", {"email": EMAIL, "password": ""})
print(f"  POST /api/auth/signup (pw='')  →  status={st}")
rows = db_query("SELECT * FROM users WHERE email = ?", (EMAIL,))
user_created = len(rows) > 0
print(f"  User in DB? {user_created}")
if st == 400 and not user_created:
    print("  ✓ PASS: Backend blocked empty password")
else:
    print("  ✗ FAIL")
    all_pass = False

# ── Scenario 3: Valid signup with matching passwords → should succeed
print("\n--- Scenario 3: Valid signup (matching passwords) ---")
st, body = api("POST", "/api/auth/signup", {"email": EMAIL, "password": GOOD_PW})
print(f"  POST /api/auth/signup (pw='{GOOD_PW}')  →  status={st}")
token = body.get("token", "")
rows = db_query("SELECT id, email, plan, daily_extractions, last_extraction_date FROM users WHERE email = ?", (EMAIL,))
print(f"  User in DB? {len(rows) > 0}")
if rows:
    print(f"  User record: {rows[0]}")
if st == 200 and token and len(rows) == 1:
    print("  ✓ PASS: User created with valid password")
else:
    print("  ✗ FAIL")
    all_pass = False

# ── Scenario 4: Same email again → should say "already exists"
print("\n--- Scenario 4: Duplicate email → 409 ---")
st, body = api("POST", "/api/auth/signup", {"email": EMAIL, "password": GOOD_PW})
print(f"  POST /api/auth/signup (same email)  →  status={st}")
print(f"  Detail: {body.get('detail', '')}")
if st == 409 and "already exists" in body.get("detail", ""):
    print("  ✓ PASS: Correctly rejected duplicate")
else:
    print("  ✗ FAIL")
    all_pass = False

# ── Scenario 5: Log in with the account → should work
print("\n--- Scenario 5: Login with created account ---")
st, body = api("POST", "/api/auth/login", {"email": EMAIL, "password": GOOD_PW})
print(f"  POST /api/auth/login  →  status={st}")
login_token = body.get("token", "")
if st == 200 and login_token:
    # Verify the token works
    st2, me = api("GET", "/api/auth/me", token=login_token)
    print(f"  GET /api/auth/me  →  status={st2}  email={me.get('email')}")
    if st2 == 200 and me.get("email") == EMAIL:
        print("  ✓ PASS: Login works, token valid")
    else:
        print("  ✗ FAIL: Token invalid")
        all_pass = False
else:
    print("  ✗ FAIL: Login returned no token")
    all_pass = False

# ── Scenario 6: Ghost user recovery
# Simulate the old bug: insert a ghost user directly into the DB with
# last_extraction_date=NULL (as if the old broken frontend created it).
print("\n--- Scenario 6: Ghost user recovery ---")
GHOST_EMAIL = "ghost@example.com"
GHOST_PW2 = "NewGhostPass2"

import uuid as _uuid
from passlib.hash import bcrypt as _bc
_ghost_id = str(_uuid.uuid4())
_ghost_hash = _bc.hash("BuggyPassword1")
db_exec(
    "INSERT INTO users (id, email, password_hash, plan, daily_extractions) VALUES (?, ?, ?, 'free', 0)",
    (_ghost_id, GHOST_EMAIL, _ghost_hash),
)
rows = db_query("SELECT daily_extractions, last_extraction_date FROM users WHERE email = ?", (GHOST_EMAIL,))
print(f"  Inserted ghost: extractions={rows[0]['daily_extractions']}, last_date={rows[0]['last_extraction_date']}")

# Now re-register with new password → should succeed (ghost recovery)
st, body = api("POST", "/api/auth/signup", {"email": GHOST_EMAIL, "password": GHOST_PW2})
print(f"  Re-register ghost  →  status={st}")
if st == 200 and body.get("token"):
    # Verify new password works
    st3, _ = api("POST", "/api/auth/login", {"email": GHOST_EMAIL, "password": GHOST_PW2})
    # Verify old (buggy) password does NOT work
    st4, _ = api("POST", "/api/auth/login", {"email": GHOST_EMAIL, "password": "BuggyPassword1"})
    print(f"  Login with new pw  →  {st3}")
    print(f"  Login with old pw  →  {st4}")
    if st3 == 200 and st4 == 401:
        print("  ✓ PASS: Ghost user recovered with new password")
    else:
        print("  ✗ FAIL: Password not properly updated")
        all_pass = False
else:
    print(f"  ✗ FAIL: Expected 200, got {st}")
    all_pass = False

# ── Scenario 7: Active user should NOT be overwritable
print("\n--- Scenario 7: Active user NOT recoverable as ghost ---")
# The user from Scenario 3 logged in (has a token) but daily_extractions=0.
# To make it "active", set daily_extractions > 0
db_exec("UPDATE users SET daily_extractions = 1, last_extraction_date = '2026-03-26' WHERE email = ?", (EMAIL,))
st, body = api("POST", "/api/auth/signup", {"email": EMAIL, "password": "HackerPass999"})
print(f"  Re-register active user  →  status={st}")
print(f"  Detail: {body.get('detail','')}")
# Verify original password still works
st5, _ = api("POST", "/api/auth/login", {"email": EMAIL, "password": GOOD_PW})
st6, _ = api("POST", "/api/auth/login", {"email": EMAIL, "password": "HackerPass999"})
print(f"  Login with original pw  →  {st5}")
print(f"  Login with attacker pw  →  {st6}")
if st == 409 and st5 == 200 and st6 == 401:
    print("  ✓ PASS: Active user protected from ghost recovery")
else:
    print("  ✗ FAIL")
    all_pass = False

# ── Scenario 8: Verify frontend JS has correct validation structure
print("\n--- Scenario 8: Frontend JS validation structure ---")
html = (Path(__file__).parent / "static" / "index.html").read_text()

# The handler must be a non-async function that calls e.preventDefault()
has_prevent = "e.preventDefault()" in html
has_stop = "e.stopImmediatePropagation()" in html
# Validation must occur BEFORE the async IIFE with fetch()
signup_block = html[html.find("function handleSignup"):html.find("/* --- Login ---")]
fetch_pos = signup_block.find("fetch('/api/auth/signup'")
email_check_pos = signup_block.find("Please enter a valid email")
pw_len_pos = signup_block.find("Password must be at least 6 characters")
pw_match_pos = signup_block.find("Passwords don\\'t match") if "Passwords don\\'t match" in signup_block else signup_block.find("Passwords don't match")

checks_before_fetch = (
    0 < email_check_pos < fetch_pos and
    0 < pw_len_pos < fetch_pos and
    0 < pw_match_pos < fetch_pos
)
# No onsubmit="return false" on the form
no_inline = 'id="auth-form-signup" onsubmit' not in html

print(f"  e.preventDefault(): {has_prevent}")
print(f"  e.stopImmediatePropagation(): {has_stop}")
print(f"  All checks before fetch(): {checks_before_fetch}")
print(f"  No inline onsubmit: {no_inline}")
if has_prevent and has_stop and checks_before_fetch and no_inline:
    print("  ✓ PASS: Frontend validation is structurally correct")
else:
    print("  ✗ FAIL")
    all_pass = False


print("\n" + "=" * 65)
if all_pass:
    print("ALL 8 SCENARIOS PASSED.")
else:
    print("SOME SCENARIOS FAILED — see above.")
print("=" * 65)

sys.exit(0 if all_pass else 1)
