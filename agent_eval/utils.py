from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def json_loads(value: str | None, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def token_estimate(parts: Iterable[str | None]) -> int:
    total_chars = sum(len(part or "") for part in parts)
    return max(1, round(total_chars / 4))


def cost_estimate(tokens: int, multiplier: float = 0.0000025) -> float:
    return round(tokens * multiplier, 6)


def fts_query(text: str) -> str:
    terms = [term for term in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(term) > 1]
    if not terms:
        return text.strip() or "*"
    return " OR ".join(f'"{term}"' for term in terms[:8])
