from __future__ import annotations

from typing import Any, Iterable, Mapping

from .models import UsagePayload
from .utils import ensure_list, find_nested_keys, first_non_empty


class UsageExtractor:
    name = "base"

    def extract(self, payload: Any) -> UsagePayload | None:
        raise NotImplementedError


class LangChainUsageExtractor(UsageExtractor):
    name = "langchain_usage_metadata"

    def extract(self, payload: Any) -> UsagePayload | None:
        normalized_payload = _to_mapping(payload)
        usage = first_non_empty(
            getattr(payload, "usage_metadata", None),
            normalized_payload.get("usage_metadata") if normalized_payload else None,
            normalized_payload.get("usageMetadata") if normalized_payload else None,
        )
        if usage is None:
            return None
        usage = _to_mapping(usage)
        if not isinstance(usage, Mapping):
            return None
        return UsagePayload(
            input_tokens=_maybe_int(first_non_empty(usage.get("input_tokens"), usage.get("prompt_tokens"))),
            output_tokens=_maybe_int(first_non_empty(usage.get("output_tokens"), usage.get("completion_tokens"))),
            total_tokens=_maybe_int(usage.get("total_tokens")),
            reasoning_tokens=_maybe_int(usage.get("reasoning_tokens")),
            raw=dict(usage),
            source=self.name,
        )


class StandardUsageDictExtractor(UsageExtractor):
    name = "standard_usage_dict"

    def extract(self, payload: Any) -> UsagePayload | None:
        candidates: list[Any] = []
        normalized_payload = _to_mapping(payload)
        if normalized_payload:
            candidates.extend(
                find_nested_keys(
                    normalized_payload,
                    ["usage", "token_usage", "usage_metadata", "usageMetadata", "response_metadata", "llm_output"],
                )
            )
        for attr in ("usage", "token_usage", "usage_metadata", "usageMetadata", "response_metadata", "llm_output"):
            attr_value = getattr(payload, attr, None)
            if attr_value is not None:
                candidates.append(attr_value)
        for candidate in candidates:
            candidate = _to_mapping(candidate)
            if not isinstance(candidate, Mapping):
                continue
            input_tokens = _maybe_int(
                first_non_empty(
                    candidate.get("input_tokens"),
                    candidate.get("prompt_tokens"),
                    candidate.get("inputTokenCount"),
                    candidate.get("promptTokenCount"),
                )
            )
            output_tokens = _maybe_int(
                first_non_empty(
                    candidate.get("output_tokens"),
                    candidate.get("completion_tokens"),
                    candidate.get("candidates_token_count"),
                    candidate.get("outputTokenCount"),
                    candidate.get("candidatesTokenCount"),
                )
            )
            total_tokens = _maybe_int(
                first_non_empty(
                    candidate.get("total_tokens"),
                    candidate.get("totalTokenCount"),
                )
            )
            if input_tokens is None and output_tokens is None and total_tokens is None:
                continue
            payload_out = UsagePayload(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=_maybe_int(first_non_empty(candidate.get("reasoning_tokens"), candidate.get("reasoningTokenCount"))),
                audio_input_tokens=_maybe_int(first_non_empty(candidate.get("audio_input_tokens"), candidate.get("audioInputTokenCount"))),
                audio_output_tokens=_maybe_int(first_non_empty(candidate.get("audio_output_tokens"), candidate.get("audioOutputTokenCount"))),
                cache_read_tokens=_maybe_int(first_non_empty(candidate.get("cache_read_input_tokens"), candidate.get("cacheReadInputTokens"))),
                cache_write_tokens=_maybe_int(first_non_empty(candidate.get("cache_creation_input_tokens"), candidate.get("cacheCreationInputTokens"))),
                raw=dict(candidate),
                source=self.name,
            )
            if payload_out.total_tokens is None and payload_out.input_tokens is not None and payload_out.output_tokens is not None:
                payload_out.total_tokens = payload_out.input_tokens + payload_out.output_tokens
            return payload_out
        return None


class AnthropicUsageExtractor(UsageExtractor):
    name = "anthropic_usage"

    def extract(self, payload: Any) -> UsagePayload | None:
        payload = _to_mapping(payload)
        if not isinstance(payload, Mapping):
            return None
        usage = first_non_empty(payload.get("usage"), payload.get("response_metadata", {}).get("usage") if isinstance(payload.get("response_metadata"), Mapping) else None)
        usage = _to_mapping(usage)
        if not isinstance(usage, Mapping):
            return None
        input_tokens = _maybe_int(usage.get("input_tokens"))
        output_tokens = _maybe_int(usage.get("output_tokens"))
        if input_tokens is None and output_tokens is None:
            return None
        return UsagePayload(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(input_tokens or 0) + (output_tokens or 0),
            cache_read_tokens=_maybe_int(usage.get("cache_read_input_tokens")),
            cache_write_tokens=_maybe_int(usage.get("cache_creation_input_tokens")),
            raw=dict(usage),
            source=self.name,
        )


class OpenAIUsageExtractor(UsageExtractor):
    name = "openai_usage"

    def extract(self, payload: Any) -> UsagePayload | None:
        payload = _to_mapping(payload)
        if not isinstance(payload, Mapping):
            return None
        usage = first_non_empty(payload.get("usage"), payload.get("token_usage"))
        usage = _to_mapping(usage)
        if not isinstance(usage, Mapping):
            return None
        prompt = _maybe_int(first_non_empty(usage.get("prompt_tokens"), usage.get("input_tokens")))
        completion = _maybe_int(first_non_empty(usage.get("completion_tokens"), usage.get("output_tokens")))
        total = _maybe_int(usage.get("total_tokens"))
        if prompt is None and completion is None and total is None:
            return None
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), Mapping) else {}
        prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), Mapping) else {}
        return UsagePayload(
            input_tokens=prompt,
            output_tokens=completion,
            total_tokens=total if total is not None else ((prompt or 0) + (completion or 0)),
            reasoning_tokens=_maybe_int(first_non_empty(details.get("reasoning_tokens"), prompt_details.get("reasoning_tokens"))),
            audio_input_tokens=_maybe_int(prompt_details.get("audio_tokens")),
            audio_output_tokens=_maybe_int(details.get("audio_tokens")),
            cache_read_tokens=_maybe_int(prompt_details.get("cached_tokens")),
            raw=dict(usage),
            source=self.name,
        )


class GeminiUsageExtractor(UsageExtractor):
    name = "gemini_usage"

    def extract(self, payload: Any) -> UsagePayload | None:
        payload = _to_mapping(payload)
        if not isinstance(payload, Mapping):
            return None
        usage = first_non_empty(payload.get("usageMetadata"), payload.get("usage_metadata"))
        usage = _to_mapping(usage)
        if not isinstance(usage, Mapping):
            return None
        input_tokens = _maybe_int(first_non_empty(usage.get("promptTokenCount"), usage.get("inputTokenCount")))
        output_tokens = _maybe_int(first_non_empty(usage.get("candidatesTokenCount"), usage.get("outputTokenCount")))
        total_tokens = _maybe_int(usage.get("totalTokenCount"))
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return UsagePayload(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens if total_tokens is not None else ((input_tokens or 0) + (output_tokens or 0)),
            raw=dict(usage),
            source=self.name,
        )


class OllamaUsageExtractor(UsageExtractor):
    name = "ollama_usage"

    def extract(self, payload: Any) -> UsagePayload | None:
        payload = _to_mapping(payload)
        if not isinstance(payload, Mapping):
            return None
        response_metadata = payload.get("response_metadata") if isinstance(payload.get("response_metadata"), Mapping) else {}
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        llm_output = payload.get("llm_output") if isinstance(payload.get("llm_output"), Mapping) else {}
        eval_count = _maybe_int(first_non_empty(payload.get("eval_count"), response_metadata.get("eval_count"), llm_output.get("eval_count")))
        prompt_eval_count = _maybe_int(
            first_non_empty(
                payload.get("prompt_eval_count"),
                response_metadata.get("prompt_eval_count"),
                llm_output.get("prompt_eval_count"),
            )
        )
        output_tokens = _maybe_int(first_non_empty(usage.get("output_tokens"), eval_count))
        input_tokens = _maybe_int(first_non_empty(usage.get("input_tokens"), prompt_eval_count))
        total_tokens = _maybe_int(first_non_empty(usage.get("total_tokens")))
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return UsagePayload(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens if total_tokens is not None else ((input_tokens or 0) + (output_tokens or 0)),
            raw={
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "response_metadata": response_metadata,
                "usage": usage,
            },
            source=self.name,
        )


class UsageRegistry:
    def __init__(self) -> None:
        self.extractors: list[UsageExtractor] = [
            LangChainUsageExtractor(),
            OpenAIUsageExtractor(),
            AnthropicUsageExtractor(),
            GeminiUsageExtractor(),
            OllamaUsageExtractor(),
            StandardUsageDictExtractor(),
        ]

    def extract(self, payloads: list[Any], provider_hint: str | None = None) -> UsagePayload | None:
        for payload in payloads:
            provider = provider_hint or _detect_provider(payload)
            for extractor in _ordered_extractors(self.extractors, provider):
                extracted = extractor.extract(payload)
                if extracted is not None:
                    return extracted
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return None


def _ordered_extractors(extractors: list[UsageExtractor], provider: str | None) -> Iterable[UsageExtractor]:
    if not provider:
        return extractors
    lowered = provider.lower()
    preferred: list[str] = []
    if "openai" in lowered:
        preferred = ["openai_usage"]
    elif "anthropic" in lowered:
        preferred = ["anthropic_usage"]
    elif "gemini" in lowered or "vertex" in lowered or "google" in lowered:
        preferred = ["gemini_usage"]
    elif "ollama" in lowered:
        preferred = ["ollama_usage"]
    if not preferred:
        return extractors
    ordered: list[UsageExtractor] = []
    used = set()
    for name in preferred:
        for extractor in extractors:
            if extractor.name == name and extractor.name not in used:
                ordered.append(extractor)
                used.add(extractor.name)
    for extractor in extractors:
        if extractor.name not in used:
            ordered.append(extractor)
    return ordered


def _detect_provider(payload: Any) -> str | None:
    normalized = _to_mapping(payload)
    if not normalized:
        return None
    candidates = [
        normalized.get("provider"),
        normalized.get("ls_provider"),
        normalized.get("model_provider"),
        normalized.get("model"),
        normalized.get("model_name"),
    ]
    response_metadata = normalized.get("response_metadata")
    if isinstance(response_metadata, Mapping):
        candidates.extend(
            [
                response_metadata.get("model_provider"),
                response_metadata.get("model"),
                response_metadata.get("model_name"),
            ]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        if text:
            return text
    return None
