from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, Iterable, Mapping


def is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def coerce_primitive(value: Any) -> str | bool | int | float | None:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _fallback_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, PurePath):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, default=_fallback_default, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def flatten_str_dict(mapping: Mapping[str, Any], *, prefix: str = "") -> dict[str, str | bool | int | float]:
    result: dict[str, str | bool | int | float] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            result.update(flatten_str_dict(value, prefix=full_key))
            continue
        coerced = coerce_primitive(value)
        if coerced is not None:
            result[full_key] = coerced
    return result


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def compact_dict(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if v is not None}


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def find_nested_keys(payload: Any, names: Iterable[str]) -> list[Any]:
    wanted = {name.lower() for name in names}
    results: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key).lower() in wanted:
                    results.append(value)
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return results
