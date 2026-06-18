"""Load `.env` into process environment before OpenAI/other clients read vars."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_early() -> None:
    """Load `.env` from repo root (parent of `aide/`) and from cwd. Idempotent-ish."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    # Repo root = parent of package `aide/`
    here = Path(__file__).resolve()
    pkg_aide_dir = here.parent  # aide/
    repo_root = pkg_aide_dir.parent
    cwd = Path.cwd()

    load_dotenv(repo_root / ".env")
    load_dotenv(cwd / ".env")
