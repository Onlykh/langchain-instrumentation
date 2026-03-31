from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import context as context_api
from opentelemetry.trace import Span, set_span_in_context

CURRENT_WORKFLOW_NAME: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "otel_langchain_current_workflow_name",
    default=None,
)
CURRENT_LANGGRAPH_NODE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "otel_langgraph_current_node",
    default=None,
)
CURRENT_COMMAND_SOURCE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "otel_langgraph_current_command_source",
    default=None,
)


class SafeContextToken:
    __slots__ = ("token",)

    def __init__(self, token: object | None) -> None:
        self.token = token

    @classmethod
    def attach_span(cls, span: Span) -> "SafeContextToken":
        try:
            token = context_api.attach(set_span_in_context(span))
            return cls(token)
        except Exception:
            return cls(None)

    @classmethod
    def attach_value(cls, key: str, value: object) -> "SafeContextToken":
        try:
            token = context_api.attach(context_api.set_value(key, value))
            return cls(token)
        except Exception:
            return cls(None)

    def detach(self) -> None:
        if self.token is None:
            return
        try:
            from opentelemetry.context import _RUNTIME_CONTEXT

            _RUNTIME_CONTEXT.detach(self.token)
        except Exception:
            pass


@contextmanager
def set_context_var(var: contextvars.ContextVar[str | None], value: str | None) -> Iterator[None]:
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)
