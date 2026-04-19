"""Business step: prompt + call vision model + parse JSON → RecognitionResult."""

from __future__ import annotations

import json
from typing import Any

from config import AppConfig
from schema import RecognitionResult
from vision_model import complete_vision


def _build_instructions() -> str:
    return """You analyze a food photo.

List each distinct food or drink you can identify. Use common English labels (e.g. white rice, grilled salmon).

Respond with ONLY valid JSON and no other text, in exactly this shape:
{"items": [{"name": "<string>", "confidence": <number from 0 to 1>}]}

Rules:
- confidence is your subjective certainty for that item (0–1).
- Omit items you are not reasonably sure about.
- If unsure about everything, return {"items": []}.
"""


def _extract_json_object(raw: str) -> str:
    """Strip optional ``` / ```json fences, then take the first balanced {...} block."""
    text = raw.strip()
    fence = "```"
    if fence in text:
        first = text.find(fence)
        after_open = text.find("\n", first)
        if after_open == -1:
            after_open = first + len(fence)
        else:
            after_open += 1
        rest = text[after_open:]
        close = rest.find(fence)
        if close != -1:
            text = rest[:close].strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

    start = text.find("{")
    if start < 0:
        raise ValueError("Model output did not contain a JSON object.")

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces in model output.")


def detect_foods_from_image(image_bytes: bytes, cfg: AppConfig) -> RecognitionResult:
    instructions = _build_instructions()
    raw_text = complete_vision(image_bytes=image_bytes, instructions=instructions, cfg=cfg)

    try:
        json_str = _extract_json_object(raw_text)
        payload: Any = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Could not parse model JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("Model JSON root must be an object.")

    items = payload.get("items")
    if items is None:
        raise ValueError('Model JSON must contain an "items" array.')

    return RecognitionResult.model_validate(
        {
            "items": items,
            "model": cfg.vlm_model,
            "raw_text": raw_text,
        }
    )
