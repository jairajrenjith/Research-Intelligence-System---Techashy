"""
core/cache.py
-------------
Two-level caching system:

Level 1 — Domain report cache (no login needed)
  - Saves final report JSON to data/cache/<domain_slug>.json
  - Any user searching the same domain gets instant results
  - TTL: 24 hours (configurable)

Level 2 — Per-user report history (requires login)
  - Each user has their own report history in data/users.json
  - Survives server restarts and browser refreshes
  - Reports never expire (user can clear manually)
"""

import json
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from passlib.context import CryptContext
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(os.getenv("CACHE_DIR",  "./data/cache"))
USERS_FILE  = Path(os.getenv("USERS_FILE", "./data/users.json"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 24))

SECRET_KEY  = os.getenv("SECRET_KEY", "ris-secret-key-change-in-production")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 24

CACHE_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Domain report cache (shared, no login)
# ══════════════════════════════════════════════════════════════════════════════

def _domain_cache_path(domain: str) -> Path:
    slug = hashlib.md5(domain.strip().lower().encode()).hexdigest()[:16]
    return CACHE_DIR / f"{slug}.json"


def get_cached_report(domain: str) -> dict | None:
    """
    Return cached report for a domain if it exists and hasn't expired.
    Returns None if not found or expired.
    """
    path = _domain_cache_path(domain)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        # Make both timezone-aware for comparison
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        expiry = cached_at + timedelta(hours=CACHE_TTL_HOURS)
        if datetime.now(timezone.utc) > expiry:
            path.unlink(missing_ok=True)   # expired — delete
            logger.info(f"Cache expired for domain: {domain}")
            return None
        logger.info(f"Cache HIT for domain: {domain}")
        return data["report"]
    except Exception as e:
        logger.warning(f"Cache read error for {domain}: {e}")
        return None


def save_cached_report(domain: str, report: dict) -> None:
    """Save report to domain cache with timestamp."""
    path = _domain_cache_path(domain)
    try:
        path.write_text(json.dumps({
            "domain":    domain,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "report":    report,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Cached report for domain: {domain}")
    except Exception as e:
        logger.warning(f"Cache write error for {domain}: {e}")


def list_cached_domains() -> list[dict]:
    """Return all cached domains with their timestamps."""
    results = []
    for path in CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            expiry = cached_at + timedelta(hours=CACHE_TTL_HOURS)
            if datetime.now(timezone.utc) <= expiry:
                results.append({
                    "domain":    data.get("domain", ""),
                    "cached_at": data.get("cached_at", ""),
                })
        except Exception:
            continue
    return sorted(results, key=lambda x: x["cached_at"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — Per-user auth + report history
# ══════════════════════════════════════════════════════════════════════════════

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(
        json.dumps(users, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ── Auth ──────────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> bool:
    """Register a new user. Returns False if username taken."""
    if not username or not password:
        return False
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "hashed_password": pwd_context.hash(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reports": {},
    }
    _save_users(users)
    logger.info(f"Registered user: {username}")
    return True


def verify_login(username: str, password: str) -> bool:
    """Verify username + password. Returns True if valid."""
    users = _load_users()
    if username not in users:
        return False
    return pwd_context.verify(password, users[username]["hashed_password"])


def create_token(username: str) -> str:
    """Create a JWT token for a user."""
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": username, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def verify_token(token: str) -> str | None:
    """Verify JWT token. Returns username or None if invalid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ── Per-user report history ───────────────────────────────────────────────────

def save_user_report(username: str, domain: str, report: dict) -> None:
    """Save a report to a user's personal history."""
    users = _load_users()
    if username not in users:
        return
    users[username]["reports"][domain] = {
        "report":    report,
        "saved_at":  datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    logger.info(f"Saved report for user '{username}', domain: {domain}")


def get_user_report(username: str, domain: str) -> dict | None:
    """Get a specific report from a user's history."""
    users = _load_users()
    return users.get(username, {}).get("reports", {}).get(domain, {}).get("report")


def get_user_history(username: str) -> list[dict]:
    """Return all report summaries for a user (domain + timestamp, no full text)."""
    users = _load_users()
    reports = users.get(username, {}).get("reports", {})
    return [
        {"domain": domain, "saved_at": data["saved_at"]}
        for domain, data in sorted(
            reports.items(),
            key=lambda x: x[1]["saved_at"],
            reverse=True
        )
    ]


def delete_user_report(username: str, domain: str) -> bool:
    """Delete a report from user's history."""
    users = _load_users()
    if username not in users:
        return False
    if domain not in users[username].get("reports", {}):
        return False
    del users[username]["reports"][domain]
    _save_users(users)
    return True