"""Simple JSON-backed storage for local prompt copilot sessions."""

from __future__ import annotations

from typing import Any

from backend.config import USERS_DIR
from backend.core.utils import load_json, save_json


class DatabaseService:
    def read_user_file(self, user_id: str, filename: str, default: Any) -> Any:
        path = USERS_DIR / user_id / filename
        return load_json(path, default)

    def write_user_file(self, user_id: str, filename: str, payload: Any) -> None:
        path = USERS_DIR / user_id / filename
        save_json(path, payload)


database_service = DatabaseService()
