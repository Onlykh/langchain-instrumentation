from __future__ import annotations

from typing import Any, Mapping

from .config import LangChainConfig
from .semconv import MimeTypes
from .utils import compact_dict, ensure_list, safe_json_dumps


def redact_payload(value: Any, config: LangChainConfig) -> Any:
    if isinstance(value, Mapping):
        output = {}
        for key, child in value.items():
            if config.should_redact(str(key)):
                output[key] = "***"
            else:
                output[key] = redact_payload(child, config)
        return output
    if isinstance(value, list):
        return [redact_payload(item, config) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item, config) for item in value]
    return value


def _message_like_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "dict"):
        return message.dict()
    data = {
        "type": getattr(message, "type", None),
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", None),
        "content": getattr(message, "content", None),
        "additional_kwargs": getattr(message, "additional_kwargs", None),
        "response_metadata": getattr(message, "response_metadata", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "usage_metadata": getattr(message, "usage_metadata", None),
        "artifact": getattr(message, "artifact", None),
        "status": getattr(message, "status", None),
    }
    return compact_dict(data)


def extract_tool_calls_from_message(message_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    candidates = [
        message_dict.get("tool_calls"),
        (message_dict.get("additional_kwargs") or {}).get("tool_calls") if isinstance(message_dict.get("additional_kwargs"), Mapping) else None,
        (message_dict.get("kwargs") or {}).get("tool_calls") if isinstance(message_dict.get("kwargs"), Mapping) else None,
    ]
    for candidate in candidates:
        for item in ensure_list(candidate):
            if not isinstance(item, Mapping):
                output.append({"raw": str(item)})
                continue
            function = item.get("function") if isinstance(item.get("function"), Mapping) else {}
            output.append(
                compact_dict(
                    {
                        "id": item.get("id"),
                        "type": item.get("type") or "function",
                        "name": item.get("name") or function.get("name"),
                        "arguments": item.get("arguments") or item.get("args") or function.get("arguments"),
                    }
                )
            )
    return output


def normalize_messages(messages: Any, config: LangChainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for message in ensure_list(messages):
        data = _message_like_to_dict(message)
        if config.record_content:
            data = redact_payload(data, config)
        normalized.append(data)
        tool_calls.extend(extract_tool_calls_from_message(data))
    return normalized, tool_calls


def serialize_io(value: Any, config: LangChainConfig) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, str):
        return value, MimeTypes.TEXT

    if isinstance(value, Mapping) and "messages" in value:
        payload = redact_payload(value, config)
        return safe_json_dumps(payload), MimeTypes.JSON

    if isinstance(value, list):
        payload = redact_payload(value, config)
        return safe_json_dumps(payload), MimeTypes.JSON

    payload = redact_payload(value, config)
    try:
        return safe_json_dumps(payload), MimeTypes.JSON
    except Exception:
        return str(payload), MimeTypes.TEXT


def extract_prompt_template(serialized: Mapping[str, Any] | None) -> tuple[str | None, list[str]]:
    if not serialized:
        return None, []
    kwargs = serialized.get("kwargs") if isinstance(serialized.get("kwargs"), Mapping) else {}
    template = kwargs.get("template") or kwargs.get("prompt")
    input_variables = kwargs.get("input_variables")
    return (str(template) if template is not None else None, [str(v) for v in ensure_list(input_variables)])
