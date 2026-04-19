"""Structured types for: image → food list."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FoodItem(BaseModel):
    """One food the model believes is in the image."""

    name: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("name cannot be empty")
        return s


class RecognitionResult(BaseModel):
    """What the recognition step returns to callers / JSON."""

    items: list[FoodItem] = Field(default_factory=list)
    model: str | None = Field(default=None, description="Model id used for this run, if known.")
    raw_text: str | None = Field(
        default=None,
        description="Optional raw assistant text for debugging.",
    )
