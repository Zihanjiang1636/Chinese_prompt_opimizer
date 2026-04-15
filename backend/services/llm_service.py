"""Minimal LLM wrapper with deterministic fallback mode."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER


class LLMService:
    DEFAULT_TIMEOUT_SECONDS = 45
    OPENAI_COMPATIBLE_PROVIDERS = {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "dashscope",
        "qwen",
        "deepseek",
    }

    def __init__(self) -> None:
        self.provider = LLM_PROVIDER
        self.model = LLM_MODEL
        self.base_url = LLM_BASE_URL
        self.api_key = LLM_API_KEY

    @property
    def stub_mode(self) -> bool:
        return not bool(self.api_key)

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        *,
        timeout_seconds: int | float | None = None,
    ) -> dict[str, Any]:
        timeout_value = float(timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS)
        started_at = perf_counter()

        if self.stub_mode:
            return self._build_payload("stub", fallback_text, timeout_value, started_at, True)

        try:
            if self.provider in self.OPENAI_COMPATIBLE_PROVIDERS:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_value)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                text = response.choices[0].message.content or fallback_text
                return self._build_payload("live", text, timeout_value, started_at, text == fallback_text)
        except Exception as exc:  # pragma: no cover
            return self._build_payload("fallback", fallback_text, timeout_value, started_at, True, error=str(exc))

        return self._build_payload(
            "fallback",
            fallback_text,
            timeout_value,
            started_at,
            True,
            error=f"Unsupported provider: {self.provider}",
        )

    def _build_payload(
        self,
        mode: str,
        text: str,
        timeout_seconds: float,
        started_at: float,
        fallback_used: bool,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        return {
            "mode": mode,
            "provider": self.provider,
            "model": self.model,
            "text": text,
            "timeout_ms": int(timeout_seconds * 1000),
            "elapsed_ms": elapsed_ms,
            "fallback_used": fallback_used,
            "error": error,
        }


llm_service = LLMService()
