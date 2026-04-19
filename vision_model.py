"""Thin call to the vision-capable chat model: image + instructions → raw assistant text."""

from __future__ import annotations

import base64

from openai import OpenAI

from config import AppConfig


def _image_mime(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 3 and image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/jpeg"


def complete_vision(*, image_bytes: bytes, instructions: str, cfg: AppConfig) -> str:
    if not cfg.api_key:
        raise ValueError("Missing API key (OPENAI_API_KEY or MEAL_OPENAI_API_KEY).")

    mime = _image_mime(image_bytes)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"

    client = OpenAI(api_key=cfg.api_key)
    response = client.chat.completions.create(
        model=cfg.vlm_model,
        temperature=cfg.temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )
    choice = response.choices[0].message
    text = (choice.content or "").strip()
    return text
