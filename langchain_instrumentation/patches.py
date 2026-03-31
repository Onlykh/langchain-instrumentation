from __future__ import annotations

import importlib
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from opentelemetry import trace as trace_api
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GenAiOperationNameValues,
)
from opentelemetry.trace import Status, StatusCode
from wrapt import wrap_function_wrapper

from .config import LangChainConfig
from .context import CURRENT_LANGGRAPH_NODE, CURRENT_WORKFLOW_NAME, SafeContextToken, set_context_var
from .semconv import LangChain

logger = logging.getLogger(__name__)

try:
    from opentelemetry.instrumentation.utils import unwrap as _otel_unwrap
except ImportError:  # pragma: no cover
    _otel_unwrap = None


def _unwrap_wrapped_function(module_name: str, qualified_name: str) -> None:
    """Reverse ``wrap_function_wrapper(module=..., name=...)`` (one wrapt layer)."""
    parts = qualified_name.split(".")
    if _otel_unwrap is not None:
        try:
            if len(parts) == 1:
                mod = importlib.import_module(module_name)
                _otel_unwrap(mod, parts[0])
            else:
                owner_path = f"{module_name}.{parts[0]}"
                attr = ".".join(parts[1:])
                _otel_unwrap(owner_path, attr)
        except (ImportError, AttributeError, ValueError):
            logger.debug(
                "Unwrap skipped for %s.%s", module_name, qualified_name, exc_info=True
            )
        return
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return
    if len(parts) == 1:
        owner, attr = mod, parts[0]
    else:
        owner = mod
        for segment in parts[:-1]:
            owner = getattr(owner, segment)
        attr = parts[-1]
    fn = getattr(owner, attr, None)
    if fn is None:
        return
    orig = fn
    while hasattr(orig, "__wrapped__"):
        orig = orig.__wrapped__
    if orig is not fn:
        setattr(owner, attr, orig)


def _begin_langgraph_root_workflow_span(
    tracer: trace_api.Tracer, instance: Any
) -> tuple[Any, Any, str]:
    workflow_name = getattr(instance, "name", None) or instance.__class__.__name__
    span_name = str(workflow_name).strip() or GenAiOperationNameValues.INVOKE_AGENT.value
    span = tracer.start_span(span_name)
    span.set_attribute(GEN_AI_OPERATION_NAME, GenAiOperationNameValues.INVOKE_AGENT.value)
    span.set_attribute(GEN_AI_AGENT_NAME, str(workflow_name))
    span_context_token = SafeContextToken.attach_span(span)
    return span, span_context_token, str(workflow_name)


def _invoke_pregel_wrapped(
    instance: Any,
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    workflow_name: str,
) -> Any:
    with set_context_var(CURRENT_WORKFLOW_NAME, str(workflow_name)):
        with set_context_var(CURRENT_LANGGRAPH_NODE, getattr(instance, "node_name", None)):
            return wrapped(*args, **kwargs)


def _tool_definitions_json(tools: Any, log: logging.Logger) -> str | None:
    tool_definitions: list[dict[str, Any]] = []
    for tool in tools:
        definition: dict[str, Any] = {}
        if hasattr(tool, "name"):
            definition["name"] = str(tool.name)
        elif isinstance(tool, dict) and "name" in tool:
            definition["name"] = str(tool["name"])
        if hasattr(tool, "description"):
            definition["description"] = str(tool.description)
        elif isinstance(tool, dict) and "description" in tool:
            definition["description"] = str(tool["description"])
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                if hasattr(args_schema, "model_json_schema"):
                    definition["parameters"] = args_schema.model_json_schema()
                elif hasattr(args_schema, "schema"):
                    definition["parameters"] = args_schema.schema()
            except Exception:
                log.debug(
                    "Failed to serialize args_schema for tool %s",
                    definition.get("name"),
                    exc_info=True,
                )
        tool_args = getattr(tool, "args", None)
        if "parameters" not in definition and isinstance(tool_args, Mapping):
            definition["parameters"] = dict(tool_args)
        if "parameters" not in definition and isinstance(tool, dict):
            if isinstance(tool.get("parameters"), Mapping):
                definition["parameters"] = dict(tool["parameters"])
            elif isinstance(tool.get("args"), Mapping):
                definition["parameters"] = dict(tool["args"])
        if definition:
            tool_definitions.append(definition)
    if not tool_definitions:
        return None
    return json.dumps(tool_definitions)


@dataclass(slots=True)
class PatchHandle:
    module: str
    name: str
    enabled: bool = False


class PatchManager:
    def __init__(self, config: LangChainConfig, handler: Any | None = None) -> None:
        self._config = config
        self._handler = handler
        self._handles: list[PatchHandle] = []

    def apply_all(self) -> None:
        if self._config.patch_callback_manager:
            self._patch_callback_manager_init()
        if self._config.patch_langgraph:
            self._patch_langgraph()
        if self._config.patch_langchain_factories:
            self._patch_factories()

    def remove_all(self) -> None:
        for handle in reversed(self._handles):
            if not handle.enabled:
                continue
            try:
                _unwrap_wrapped_function(handle.module, handle.name)
            except Exception:
                logger.debug(
                    "Failed to unwrap %s.%s", handle.module, handle.name, exc_info=True
                )
        self._handles.clear()

    def _safe_patch(self, module: str, name: str, wrapper: Callable[..., Any]) -> None:
        handle = PatchHandle(module=module, name=name, enabled=False)
        try:
            wrap_function_wrapper(module=module, name=name, wrapper=wrapper)
            handle.enabled = True
        except Exception:
            logger.debug("Failed to patch %s.%s", module, name, exc_info=True)
        self._handles.append(handle)

    def _safe_patch_many(self, targets: list[tuple[str, str]], wrapper: Callable[..., Any]) -> None:
        for module, name in targets:
            self._safe_patch(module, name, wrapper)

    def _patch_callback_manager_init(self) -> None:
        def wrapper(wrapped: Callable[..., Any], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            result = wrapped(*args, **kwargs)
            handler = kwargs.get("langchain_otel_handler")
            if handler is None:
                handler = getattr(instance, "_langchain_otel_handler", None)
            if handler is None:
                handler = self._handler
            if handler is None:
                return result
            handlers = getattr(instance, "inheritable_handlers", None)
            if handlers is None:
                handlers = getattr(instance, "handlers", None)
            if handlers is not None and not any(existing is handler for existing in handlers):
                try:
                    instance.add_handler(handler, inherit=True)
                except Exception:
                    try:
                        handlers.append(handler)
                    except Exception:
                        logger.debug("Failed to inject LangChain OTel handler", exc_info=True)
            return result

        self._safe_patch_many(
            [
                ("langchain_core.callbacks.base", "BaseCallbackManager.__init__"),
                ("langchain_core.callbacks.manager", "BaseCallbackManager.__init__"),
                ("langchain_core.callbacks", "BaseCallbackManager.__init__"),
            ],
            wrapper,
        )

    def _patch_langgraph(self) -> None:
        def command_wrapper(wrapped: Callable[..., Any], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            result = wrapped(*args, **kwargs)
            goto = getattr(instance, "goto", None)
            source = getattr(instance, "source", None)
            if source is not None:
                try:
                    setattr(instance, "__langgraph_command_source__", str(source))
                except Exception:
                    pass
            if goto is not None:
                try:
                    setattr(instance, "__langgraph_command_target__", str(goto))
                except Exception:
                    pass
            goto_targets: list[str] = []
            if isinstance(goto, str):
                goto_targets = [goto]
            elif isinstance(goto, (list, tuple)):
                goto_targets = [str(target) for target in goto]
            elif goto is not None:
                goto_targets = [str(goto)]
            if source is not None and goto_targets:
                target_str = ", ".join(goto_targets)
                tracer = trace_api.get_tracer(self._config.instrumentation_scope_name)
                with tracer.start_as_current_span(f"goto {target_str}") as span:
                    span.set_attribute(GEN_AI_OPERATION_NAME, "goto")
                    span.set_attribute(LangChain.LANGGRAPH_COMMAND_SOURCE, str(source))
                    span.set_attribute(LangChain.LANGGRAPH_COMMAND_TARGET, target_str)
            return result

        tracer = trace_api.get_tracer(self._config.instrumentation_scope_name)

        def stream_wrapper(wrapped: Callable[..., Any], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            span, span_context_token, workflow_name = _begin_langgraph_root_workflow_span(
                tracer, instance
            )
            iterator = _invoke_pregel_wrapped(
                instance, wrapped, args, kwargs, workflow_name
            )
            if not inspect.isgenerator(iterator):
                span_context_token.detach()
                span.end()
                return iterator

            def traced_generator() -> Any:
                try:
                    for item in iterator:
                        yield item
                except BaseException as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
                finally:
                    span_context_token.detach()
                    span.end()

            return traced_generator()

        def astream_wrapper(wrapped: Callable[..., Any], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            span, span_context_token, workflow_name = _begin_langgraph_root_workflow_span(
                tracer, instance
            )
            async_iterator = _invoke_pregel_wrapped(
                instance, wrapped, args, kwargs, workflow_name
            )

            async def traced_async_generator() -> Any:
                try:
                    async for item in async_iterator:
                        yield item
                except BaseException as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
                finally:
                    span_context_token.detach()
                    span.end()

            return traced_async_generator()

        self._safe_patch("langgraph.types", "Command.__init__", command_wrapper)
        self._safe_patch("langgraph.pregel", "Pregel.stream", stream_wrapper)
        self._safe_patch("langgraph.pregel", "Pregel.astream", astream_wrapper)

    def _patch_factories(self) -> None:
        def create_agent_wrapper(wrapped: Callable[..., Any], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            name = kwargs.get("name") or getattr(wrapped, "__name__", "agent")
            tracer = trace_api.get_tracer(self._config.instrumentation_scope_name)
            with tracer.start_as_current_span(f"create_agent {name}") as span:
                span.set_attribute(GEN_AI_OPERATION_NAME, GenAiOperationNameValues.CREATE_AGENT.value)
                span.set_attribute(GEN_AI_AGENT_NAME, str(name))
                tools = kwargs.get("tools")
                if tools is None and len(args) > 1:
                    tools = args[1]
                if tools:
                    dumped = _tool_definitions_json(tools, logger)
                    if dumped is not None:
                        span.set_attribute("gen_ai.tool.definitions", dumped)
                result = wrapped(*args, **kwargs)
            try:
                setattr(result, "__langchain_workflow_name__", str(name))
            except Exception:
                pass
            return result

        self._safe_patch_many(
            [
                ("langchain.agents.factory", "create_agent"),
                ("langgraph.prebuilt.chat_agent_executor", "create_react_agent"),
                ("langgraph.prebuilt", "create_react_agent"),
            ],
            create_agent_wrapper,
        )
