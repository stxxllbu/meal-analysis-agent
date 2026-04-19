# meal-analysis-agent

MVP: given a **meal photo**, call a **vision-capable model** and print a **JSON** list of foods with confidences (`name`, `confidence`).

## Requirements

- Python **3.11+**
- An **OpenAI API key** with access to a vision-capable model (default: `gpt-4o-mini`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` or `MEAL_OPENAI_API_KEY` | **Yes** | — | API key for OpenAI |
| `MEAL_VLM_MODEL` | No | `gpt-4o-mini` | Chat vision model id |
| `MEAL_TEMPERATURE` | No | `0.2` | Sampling temperature |

## Input images

Put local meal photos under **`input/`** (ignored by git except `.gitkeep`). Example:

```bash
python main.py --image input/lunch.jpg
```

## Run

```bash
export OPENAI_API_KEY=sk-...   # or MEAL_OPENAI_API_KEY
python main.py --image path/to/your/photo.jpg
```

JSON is written to **stdout** (pretty-printed). Errors go to **stderr** with a non-zero exit code.

## Layout

| File | Role |
|------|------|
| `main.py` | CLI: args, read image bytes, call detection, print JSON |
| `vision_model.py` | Thin OpenAI call: image + instructions → raw text |
| `food_detection.py` | Prompt, parse JSON (incl. code fences), validate → `RecognitionResult` |
| `schema.py` | Pydantic types for output |
| `config.py` | Small env-based config (`AppConfig`) |

## Output shape

The printed JSON matches `RecognitionResult` in `schema.py`: an `items` array of `{ "name", "confidence" }`, plus optional `model` and `raw_text` fields.

## License

See `LICENSE`.
