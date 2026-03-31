from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .context import SafeContextToken


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(slots=True)
class UsagePayload:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    raw: dict[str, Any] | None = None
    source: str | None = None


@dataclass(slots=True)
class RunPayload:
    input_value: str | None = None
    output_value: str | None = None
    input_mime_type: str | None = None
    output_mime_type: str | None = None
    message_payloads: list[dict[str, Any]] = field(default_factory=list)
    completion_payloads: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    invocation_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    usage: UsagePayload | None = None
    raw_input: Any = None
    raw_output: Any = None


@dataclass(slots=True)
class RunState:
    run_id: UUID
    parent_run_id: UUID | None
    root_run_id: UUID
    run_type: str
    name: str
    start_time: datetime = field(default_factory=utc_now)
    end_time: datetime | None = None
    first_token_time: datetime | None = None
    span: Any = None
    context_token: SafeContextToken | None = None
    suppression_token: SafeContextToken | None = None
    children: set[UUID] = field(default_factory=set)
    workflow_name: str | None = None
    langgraph_node: str | None = None
    command_source: str | None = None
    serialized: dict[str, Any] | None = None
    payload: RunPayload = field(default_factory=RunPayload)
    error: BaseException | None = None
    stream_chunk_count: int = 0
