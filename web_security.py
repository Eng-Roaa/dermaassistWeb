"""
web_security.py  -  security layer for the DermaAssist web proof of concept

* Passwords: salted scrypt hashes (never stored in clear text)
* Roles: "doctor" (read + write + audit + FHIR send) and "viewer" (read-only + export)
* Brute-force protection: temporary lock-out after repeated failed logins
* Audit log: who did what and when
* Clinician reviews: the human decision recorded next to any AI result
* Vault: every stored image is encrypted at rest (Fernet = AES-128-CBC + HMAC-SHA256)
* Upload hygiene: size limit, format check, EXIF/GPS metadata removed by re-encoding

This is a proof of concept for synthetic/demo data. It is NOT a certified or audited system.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import secrets
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from PIL import Image, ImageOps

ROLE_PERMISSIONS = {
    "doctor": {"read", "write", "audit", "fhir_send", "export"},
    "viewer": {"read", "export"},
}
MAX_FAILURES = 5
LOCK_MINUTES = 5
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SIDE = 2000                       # larger uploads are scaled down (speed + memory)
AI_AGREEMENT = ("Agree with the AI result", "Disagree with the AI result",
                "Image not suitable for assessment")
CLINICIAN_ACTIONS = ("No action", "Routine follow-up", "Refer to a dermatologist",
                     "Start or change treatment (record it under Treatments)")


def now_iso() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def can(role: str | None, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role or "", set())


# ------------------------------------------------------------------ passwords
def hash_password(password: str, n: int = 2 ** 14, r: int = 8, p: int = 1) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                                n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


@dataclass
class AuthResult:
    ok: bool
    user: dict | None = None
    message: str = ""


class SecurityStore:
    """Users, failed-login counters, audit log and clinician reviews (same SQLite file)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('doctor','viewer')),
            display_name TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS login_attempts (
            username TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0, locked_until TEXT);
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, username TEXT,
            role TEXT, action TEXT NOT NULL, target TEXT, detail TEXT);
        CREATE TABLE IF NOT EXISTS clinician_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE RESTRICT,
            capture_id INTEGER REFERENCES captures(id) ON DELETE RESTRICT,
            username TEXT NOT NULL, ai_agreement TEXT NOT NULL, action TEXT NOT NULL,
            note TEXT, created_at TEXT NOT NULL);
        """)
        self.conn.commit()

    # ------------------------------------------------------------ users
    def upsert_user(self, username: str, password: str, role: str, display_name: str | None = None):
        if role not in ROLE_PERMISSIONS:
            raise ValueError("Unknown role.")
        if len(password) < 10:
            raise ValueError("Password must have at least 10 characters.")
        username = username.strip().lower()
        with self.conn:
            self.conn.execute(
                "INSERT INTO users (username, password_hash, role, display_name, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(username) DO UPDATE SET "
                "password_hash = excluded.password_hash, role = excluded.role, "
                "display_name = excluded.display_name",
                (username, hash_password(password), role, display_name or username, now_iso()))

    def authenticate(self, username: str, password: str) -> AuthResult:
        username = (username or "").strip().lower()
        row = self.conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        att = self.conn.execute("SELECT * FROM login_attempts WHERE username = ?", (username,)).fetchone()
        now = datetime.now()
        if att and att["locked_until"] and datetime.fromisoformat(att["locked_until"]) > now:
            minutes = int((datetime.fromisoformat(att["locked_until"]) - now).total_seconds() // 60) + 1
            self.log(username, None, "login_blocked", detail="account temporarily locked")
            return AuthResult(False, None, f"Too many failed attempts. Try again in about {minutes} minute(s).")

        # always do the same amount of hashing work, so timing does not reveal valid usernames
        stored = row["password_hash"] if row else hash_password("dummy-password-for-timing")
        ok = verify_password(password or "", stored) and row is not None
        if ok:
            with self.conn:
                self.conn.execute("DELETE FROM login_attempts WHERE username = ?", (username,))
            user = {"username": row["username"], "role": row["role"], "display_name": row["display_name"]}
            self.log(username, row["role"], "login_ok")
            return AuthResult(True, user, "")

        failures = (att["failures"] if att else 0) + 1
        locked = (now + timedelta(minutes=LOCK_MINUTES)).isoformat() if failures >= MAX_FAILURES else None
        with self.conn:
            self.conn.execute(
                "INSERT INTO login_attempts (username, failures, locked_until) VALUES (?, ?, ?) "
                "ON CONFLICT(username) DO UPDATE SET failures = excluded.failures, "
                "locked_until = excluded.locked_until", (username, 0 if locked else failures, locked))
        self.log(username, None, "login_failed")
        return AuthResult(False, None, "Invalid username or password.")

    # ------------------------------------------------------------ audit
    def log(self, username, role, action, target="", detail=""):
        with self.conn:
            self.conn.execute(
                "INSERT INTO audit_log (ts, username, role, action, target, detail) VALUES (?, ?, ?, ?, ?, ?)",
                (now_iso(), username, role, action, str(target), str(detail)))

    def audit_events(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, username, role, action, target, detail FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ clinician reviews
    def add_review(self, user: dict, patient_id: int, capture_id: int | None,
                   ai_agreement: str, action: str, note: str = "") -> int:
        if not can(user["role"], "write"):
            raise PermissionError("Only a clinician (doctor role) can record a review.")
        if ai_agreement not in AI_AGREEMENT or action not in CLINICIAN_ACTIONS:
            raise ValueError("Please choose valid options.")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO clinician_reviews (patient_id, capture_id, username, ai_agreement, "
                "action, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (patient_id, capture_id, user["username"], ai_agreement, action,
                 (note or "").strip() or None, now_iso()))
        self.log(user["username"], user["role"], "clinician_review", f"patient {patient_id}", action)
        return cur.lastrowid

    def reviews(self, patient_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM clinician_reviews WHERE patient_id = ? ORDER BY id DESC", (patient_id,)).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------------ vault (encryption at rest)
class Vault:
    """Encrypted file store. Files are named with random tokens, so names reveal nothing."""

    def __init__(self, root: Path, key: str | bytes):
        self.root = Path(root)
        (self.root / "vault").mkdir(parents=True, exist_ok=True)
        self.fernet = Fernet(key)

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()

    def put(self, data: bytes) -> str:
        rel = f"vault/{secrets.token_urlsafe(16)}.enc"
        (self.root / rel).write_bytes(self.fernet.encrypt(data))
        return rel

    def get(self, rel: str) -> bytes:
        path = (self.root / rel).resolve()
        if (self.root / "vault").resolve() not in path.parents:
            raise PermissionError("Path outside the vault.")
        try:
            return self.fernet.decrypt(path.read_bytes())
        except InvalidToken:
            raise ValueError("Cannot decrypt this file: wrong or missing IMAGE_KEY.")


def sanitize_image(data: bytes) -> tuple[Image.Image, bytes]:
    """Validate an uploaded image and return (RGB image, PNG bytes without metadata)."""
    if not data:
        raise ValueError("Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than 8 MB.")
    Image.MAX_IMAGE_PIXELS = 40_000_000             # refuse "decompression bombs"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            if probe.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPEG or PNG images are accepted.")
            probe.verify()
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)       # respect phone orientation, then drop EXIF
            img = img.convert("RGB")
    except ValueError:
        raise
    except Exception:
        raise ValueError("This file is not a valid image.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "PNG")                              # re-encoding removes EXIF and GPS data
    return img, out.getvalue()
