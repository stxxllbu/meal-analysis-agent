"""Runtime settings from environment variables only (no TOML, no pydantic)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    vlm_model: str
    temperature: float
    api_key: str | None


def load_config() -> AppConfig:
    raw_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MEAL_OPENAI_API_KEY")
    key = (raw_key or "").strip() or None

    return AppConfig(
        vlm_model=os.environ.get("MEAL_VLM_MODEL", "gpt-4o-mini").strip(),
        temperature=float(os.environ.get("MEAL_TEMPERATURE", "0.2")),
        api_key=key,
    )
