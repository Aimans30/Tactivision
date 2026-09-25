"""Local browser-upload helpers for the portfolio dashboard.

Not production infrastructure: stores uploads under ``data/raw/uploads/``
and invokes the same ``run_match_pipeline`` used by the CLI.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from tactivision.pipeline.paths import validate_run_id

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_ROOT = PROJECT_ROOT / "data" / "raw" / "uploads"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

ALLOWED_EXTENSIONS = frozenset({".mp4", ".avi", ".mkv", ".mov"})
MAX_UPLOAD_BYTES = int(os.environ.get("TACTIVISION_MAX_UPLOAD_MB", "500")) * 1024 * 1024
DEFAULT_MAX_SECONDS = 30
MAX_MAX_SECONDS = 300

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_upload_filename(filename: str | None) -> str:
    """Return a basename-only safe filename (no path traversal)."""
    if not filename or not str(filename).strip():
        raise ValueError("Missing upload filename")
    name = Path(str(filename)).name
    if not name or name in {".", ".."}:
        raise ValueError("Invalid upload filename")
    name = _SAFE_NAME.sub("_", name).strip("._")
    if not name:
        raise ValueError("Invalid upload filename")
    return name[:180]


def extension_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def new_upload_run_id(*, processed_root: Path | None = None) -> str:
    """Filesystem-safe unique run id that does not collide with existing runs."""
    root = processed_root or PROCESSED_ROOT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for _ in range(32):
        candidate = f"upload_{stamp}_{secrets.token_hex(3)}"
        validate_run_id(candidate)
        if not (root / candidate).exists():
            return candidate
    raise RuntimeError("Could not allocate a unique run id")


def upload_destination(run_id: str, safe_name: str, *, upload_root: Path | None = None) -> Path:
    rid = validate_run_id(run_id)
    if not extension_allowed(safe_name):
        raise ValueError(
            "Unsupported video type. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    dest_dir = (upload_root or UPLOAD_ROOT) / rid
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Store under a fixed stem so the original name cannot escape the folder.
    ext = Path(safe_name).suffix.lower()
    return dest_dir / f"source{ext}"


def parse_max_seconds(raw: str | None) -> float | None:
    """``None`` / empty / 'full' → None (process until EOF, still capped by caller policy).

    Numeric values are clamped to ``(0, MAX_MAX_SECONDS]``.
    """
    if raw is None:
        return float(DEFAULT_MAX_SECONDS)
    text = str(raw).strip().lower()
    if text in {"", "full", "none", "all"}:
        # Portfolio default: still enforce a hard upper bound for "full".
        return float(MAX_MAX_SECONDS)
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("max_seconds must be a number, 'full', or empty") from exc
    if value <= 0:
        raise ValueError("max_seconds must be positive")
    return float(min(value, MAX_MAX_SECONDS))


def public_error_message(exc: BaseException) -> str:
    """Strip absolute paths / traceback noise from user-facing errors."""
    text = str(exc).strip() or exc.__class__.__name__
    text = text.replace("\\", "/")
    text = re.sub(r"[A-Za-z]:/\S+", "<path>", text)
    text = re.sub(r"/Users/\S+", "<path>", text)
    text = re.sub(r"/home/\S+", "<path>", text)
    if len(text) > 280:
        text = text[:277] + "..."
    return text


def ensure_run_slot_free(run_id: str, *, processed_root: Path | None = None) -> None:
    root = (processed_root or PROCESSED_ROOT) / validate_run_id(run_id)
    if root.exists():
        raise ValueError(f"Run id already exists: {run_id}")
