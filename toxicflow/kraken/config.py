"""Credential and path configuration, loaded from the repo-root `.env` file."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = Path(os.getenv("TOXICFLOW_DATA_DIR") or REPO_ROOT / "data")


class MissingCredentials(RuntimeError):
    pass


def get_credentials() -> tuple[str, str]:
    key = os.getenv("KRAKEN_API_KEY", "").strip()
    secret = os.getenv("KRAKEN_API_SECRET", "").strip()
    if not key or not secret:
        raise MissingCredentials(
            "KRAKEN_API_KEY / KRAKEN_API_SECRET not set. "
            f"Copy .env.example to {REPO_ROOT / '.env'} and fill them in."
        )
    return key, secret
