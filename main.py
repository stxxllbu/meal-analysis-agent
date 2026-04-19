"""CLI entry: args → image bytes → food detection → JSON on stdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import load_config
from food_detection import detect_foods_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect foods in a meal photo.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an image file (e.g. JPEG or PNG).",
    )
    args = parser.parse_args()

    path: Path = args.image
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    if not cfg.api_key:
        print("Set OPENAI_API_KEY or MEAL_OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    image_bytes = path.read_bytes()
    result = detect_foods_from_image(image_bytes, cfg)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
