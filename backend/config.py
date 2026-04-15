"""Configuration for the standalone Chinese Prompt Optimizer."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()

HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = int(os.getenv("PORT", "8008"))
ALLOWED_CORS_ORIGINS = tuple(
    item.strip()
    for item in os.getenv("ALLOWED_CORS_ORIGINS", "http://127.0.0.1:8008,http://localhost:8008").split(",")
    if item.strip()
)
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "local-user").strip() or "local-user"

DATA_DIR = PROJECT_ROOT / "data"
USERS_DIR = DATA_DIR / "users"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)
