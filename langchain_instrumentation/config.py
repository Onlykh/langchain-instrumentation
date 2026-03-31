from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(slots=True)
class LangChainConfig:
    service_name: str = "langchain-instrumentation"
    instrumentation_scope_name: str = "otel.instrumentation.langchain"
    emit_metrics: bool = True
    emit_logs: bool = True
    attach_current_span_for_children: bool = True
    patch_callback_manager: bool = True
    patch_langgraph: bool = True
    patch_langchain_factories: bool = True
    record_content: bool = True
    record_invocation_params: bool = True
    record_raw_payloads: bool = False
    redact_keys: tuple[str, ...] = (
        "api_key",
        "authorization",
        "token",
        "password",
        "secret",
        "access_token",
        "refresh_token",
    )
    extra_metadata_keys: tuple[str, ...] = ()
    suppressed_exception_prefixes: tuple[str, ...] = (
        "Command(",
        "ParentCommand(",
    )
    emit_json_input_output: bool = True
    enable_root_stream_workflow_spans: bool = False
    debug_lifecycle_logging: bool = False
    log_callback_payloads: bool = True
    extra_span_attributes: Mapping[str, str | int |
                                   float | bool] = field(default_factory=dict)

    def should_redact(self, key: str) -> bool:
        lowered = key.lower()
        return any(fragment in lowered for fragment in self.redact_keys)

    def merge_metadata_keys(self, keys: Iterable[str]) -> tuple[str, ...]:
        merged = set(self.extra_metadata_keys)
        merged.update(keys)
        return tuple(sorted(merged))
