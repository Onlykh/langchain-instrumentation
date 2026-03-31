from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Mapping
from uuid import UUID

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - optional dependency
    class BaseCallbackHandler:  # type: ignore
        pass

from ..config import LangChainConfig
from ..emitters import LogEmitter, MetricsEmitter, SpanEmitter
from ..registry import RunRegistry
from ..semconv import LangChain, SpanKinds
from ..serialization import normalize_messages, redact_payload
from ..usage import UsageRegistry
from ..utils import compact_dict, ensure_list, first_non_empty, safe_json_dumps

from .lifecycle import CallbackRunLifecycle
from .naming import resolve_run_name

logger = logging.getLogger("otel.instrumentation.langchain.lifecycle")


def _swallow_callback_exceptions(func: Any) -> Any:
    @wraps(func)
    def wrapper(self: "LangChainCallbackHandler", *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:
            try:
                self._lifecycle.debug_lifecycle(
                    "callback.internal_error",
                    callback=func.__name__,
                    error_class=exc.__class__.__name__,
                    error_message=str(exc),
                    callback_args=args,
                    callback_kwargs=kwargs,
                )
            except Exception:
                logger.exception(
                    "Callback guard failed for callback=%s", func.__name__)
            return None

    return wrapper


class LangChainCallbackHandler(BaseCallbackHandler):
    run_inline = True

    def __init__(
        self,
        *,
        config: LangChainConfig,
        registry: RunRegistry,
        span_emitter: SpanEmitter,
        metrics_emitter: MetricsEmitter,
        log_emitter: LogEmitter,
        usage_registry: UsageRegistry,
    ) -> None:
        super().__init__()
        self._config = config
        self._registry = registry
        self._span_emitter = span_emitter
        self._metrics_emitter = metrics_emitter
        self._log_emitter = log_emitter
        self._usage_registry = usage_registry
        self._lifecycle = CallbackRunLifecycle(
            config=config,
            registry=registry,
            span_emitter=span_emitter,
            metrics_emitter=metrics_emitter,
            log_emitter=log_emitter,
            usage_registry=usage_registry,
        )
    @_swallow_callback_exceptions
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = resolve_run_name(
            run_type=SpanKinds.CHAT_MODEL,
            serialized=serialized,
            metadata=metadata,
            callback_kwargs=kwargs,
            fallback="chat_model",
            is_root=parent_run_id is None,
        )
        state = self._lifecycle.start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            run_type=SpanKinds.CHAT_MODEL,
            name=name,
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            inputs={"messages": messages},
            callback_kwargs=kwargs,
        )
        flattened_messages = [message for group in ensure_list(
            messages) for message in ensure_list(group)]
        state.payload.message_payloads, state.payload.tool_calls = normalize_messages(
            flattened_messages, self._config)
        self._lifecycle.debug_lifecycle(
            "chat_model.start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            messages=messages,
            message_groups=len(ensure_list(messages)),
            flattened_messages=len(flattened_messages),
            normalized_messages=len(state.payload.message_payloads),
            extracted_tool_calls=len(state.payload.tool_calls),
            sample_message=(
                state.payload.message_payloads[0] if state.payload.message_payloads else None),
            sample_tool_call=(
                state.payload.tool_calls[0] if state.payload.tool_calls else None),
        )
        return run_id

    @_swallow_callback_exceptions
    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            run_type=SpanKinds.LLM,
            name=resolve_run_name(
                run_type=SpanKinds.LLM,
                serialized=serialized,
                metadata=metadata,
                callback_kwargs=kwargs,
                fallback="llm",
                is_root=parent_run_id is None,
            ),
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            inputs={"prompts": prompts},
            callback_kwargs=kwargs,
        )
        self._lifecycle.debug_lifecycle(
            "llm.start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            prompts=prompts,
            prompt_count=len(ensure_list(prompts)),
            first_prompt=(ensure_list(prompts)[
                          0] if ensure_list(prompts) else None),
            kwarg_keys=sorted(kwargs.keys()) if kwargs else None,
        )
        return run_id

    @_swallow_callback_exceptions
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            run_type=SpanKinds.CHAIN,
            name=resolve_run_name(
                run_type=SpanKinds.CHAIN,
                serialized=serialized,
                metadata=metadata,
                callback_kwargs=kwargs,
                fallback="chain",
                is_root=parent_run_id is None,
            ),
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            inputs=inputs,
            callback_kwargs=kwargs,
        )
        self._lifecycle.debug_lifecycle(
            "chain.start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            inputs=inputs,
            input_keys=sorted(inputs.keys()) if isinstance(
                inputs, Mapping) else None,
            kwarg_keys=sorted(kwargs.keys()) if kwargs else None,
        )
        return run_id

    @_swallow_callback_exceptions
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            run_type=SpanKinds.TOOL,
            name=resolve_run_name(
                run_type=SpanKinds.TOOL,
                serialized=serialized,
                metadata=metadata,
                callback_kwargs=kwargs,
                fallback="tool",
                is_root=parent_run_id is None,
            ),
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            inputs=inputs if inputs is not None else {"input": input_str},
            callback_kwargs=kwargs,
        )
        self._lifecycle.debug_lifecycle(
            "tool.start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            input_str=input_str,
            inputs=inputs,
            metadata=metadata,
            kwarg_keys=sorted(kwargs.keys()) if kwargs else None,
        )
        return run_id

    @_swallow_callback_exceptions
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            run_type=SpanKinds.RETRIEVER,
            name=resolve_run_name(
                run_type=SpanKinds.RETRIEVER,
                serialized=serialized,
                metadata=metadata,
                callback_kwargs=kwargs,
                fallback="retriever",
                is_root=parent_run_id is None,
            ),
            serialized=serialized,
            metadata=metadata,
            tags=tags,
            inputs={"query": query},
            callback_kwargs=kwargs,
        )
        self._lifecycle.debug_lifecycle(
            "retriever.start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            query=query,
            metadata=metadata,
            kwarg_keys=sorted(kwargs.keys()) if kwargs else None,
        )
        return run_id

    @_swallow_callback_exceptions
    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._registry.get(run_id)
        if state is not None:
            state.stream_chunk_count += 1
            if state.first_token_time is None:
                state.first_token_time = datetime.now(tz=timezone.utc)
            self._lifecycle.debug_lifecycle(
                "llm.new_token",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                stream_chunk_count=state.stream_chunk_count,
                token_preview=token,
                chunk=chunk,
                kwargs=kwargs,
            )

    @_swallow_callback_exceptions
    def on_chat_model_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)

    @_swallow_callback_exceptions
    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        state = self._registry.get(run_id)
        outputs = response
        messages: list[Any] = []
        if state is not None:
            generations = getattr(response, "generations", None)
            if generations:
                flattened = [generation for group in ensure_list(
                    generations) for generation in ensure_list(group)]
                completions: list[dict[str, Any]] = []
                for generation in flattened:
                    text = getattr(generation, "text", None)
                    if text is not None:
                        completions.append({"text": text, "generation_info": getattr(
                            generation, "generation_info", None)})
                    message = getattr(generation, "message", None)
                    if message is not None:
                        messages.append(message)
                state.payload.completion_payloads = completions
                if messages:
                    normalized_messages, tool_calls = normalize_messages(
                        messages, self._config)
                    state.payload.message_payloads.extend(normalized_messages)
                    state.payload.tool_calls.extend(tool_calls)
            # Prefer usage on the LLM/chat span itself if available.
            if state.payload.usage is None:
                llm_output = getattr(response, "llm_output", None)
                provider_hint = first_non_empty(
                    state.payload.metadata.get("ls_provider") if isinstance(
                        state.payload.metadata, Mapping) else None,
                    state.payload.metadata.get("provider") if isinstance(
                        state.payload.metadata, Mapping) else None,
                )
                usage_candidates: list[Any] = [
                    response,
                    llm_output,
                    getattr(response, "response_metadata", None),
                    getattr(response, "usage_metadata", None),
                    *messages,
                ]
                usage_candidates.extend(
                    getattr(message, "response_metadata", None) for message in messages)
                usage_candidates.extend(
                    getattr(message, "usage_metadata", None) for message in messages)
                usage_candidates.append(state.payload.metadata)
                state.payload.usage = self._usage_registry.extract(
                    usage_candidates,
                    provider_hint=str(
                        provider_hint) if provider_hint is not None else None,
                )
            self._lifecycle.debug_lifecycle(
                "llm.end.before_finish",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                response=response,
                generation_groups=len(ensure_list(
                    generations)) if generations is not None else 0,
                completion_count=len(state.payload.completion_payloads),
                normalized_message_count=len(state.payload.message_payloads),
                extracted_tool_call_count=len(state.payload.tool_calls),
                usage=(asdict(state.payload.usage)
                       if state.payload.usage is not None else None),
                sample_completion=(
                    state.payload.completion_payloads[0] if state.payload.completion_payloads else None),
                sample_message=(
                    state.payload.message_payloads[0] if state.payload.message_payloads else None),
                sample_tool_call=(
                    state.payload.tool_calls[0] if state.payload.tool_calls else None),
            )
        self._lifecycle.finish_run(run_id, outputs=outputs)
        return response

    @_swallow_callback_exceptions
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "chain.end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            outputs=outputs,
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=outputs)
        return outputs

    @_swallow_callback_exceptions
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "tool.end",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            output=output,
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=output)
        return output

    @_swallow_callback_exceptions
    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        state = self._registry.get(run_id)
        if state is not None:
            state.payload.documents = [redact_payload(getattr(document, "metadata", {"content": getattr(
                document, "page_content", str(document))}), self._config) for document in documents]
            self._lifecycle.debug_lifecycle(
                "retriever.end.before_finish",
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                documents=documents,
                document_count=len(state.payload.documents),
                sample_document=(
                    state.payload.documents[0] if state.payload.documents else None),
            )
        self._lifecycle.finish_run(run_id, outputs=documents)
        return documents

    @_swallow_callback_exceptions
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "llm.error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            error_class=error.__class__.__name__,
            error_message=str(error),
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=None, error=error)
        return error

    @_swallow_callback_exceptions
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "chain.error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            error_class=error.__class__.__name__,
            error_message=str(error),
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=None, error=error)
        return error

    @_swallow_callback_exceptions
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "tool.error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            error_class=error.__class__.__name__,
            error_message=str(error),
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=None, error=error)
        return error

    @_swallow_callback_exceptions
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "retriever.error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            error_class=error.__class__.__name__,
            error_message=str(error),
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=None, error=error)
        return error

    @_swallow_callback_exceptions
    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._lifecycle.debug_lifecycle(
            "agent.error",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            error_class=error.__class__.__name__,
            error_message=str(error),
            kwargs=kwargs,
        )
        self._lifecycle.finish_run(run_id, outputs=None, error=error)
        return error

    @_swallow_callback_exceptions
    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        state = self._registry.get(run_id)
        if state is not None and state.span is not None:
            state.span.add_event("callback.on_text",
                                 attributes=compact_dict({"text": text}))
        self._lifecycle.debug_lifecycle(
            "text",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            text=text,
            kwargs=kwargs,
        )
        return text

    @_swallow_callback_exceptions
    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        attempt_number = getattr(retry_state, "attempt_number", None)
        state = self._registry.get(run_id)
        if state is not None and state.span is not None:
            state.span.add_event(
                "callback.on_retry",
                attributes=compact_dict(
                    {
                        "attempt_number": attempt_number,
                        "retry_state": safe_json_dumps(str(retry_state)),
                    }
                ),
            )
        self._lifecycle.debug_lifecycle(
            "retry",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            attempt_number=attempt_number,
            retry_state=str(retry_state),
            kwargs=kwargs,
        )
        return retry_state

    @_swallow_callback_exceptions
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_name = getattr(action, "tool", None)
        tool_input = getattr(action, "tool_input", None)
        state = self._registry.get(run_id)
        if state is not None and state.span is not None:
            if tool_name is not None:
                state.span.set_attribute(GEN_AI_TOOL_NAME, str(tool_name))
            if tool_input is not None:
                state.span.set_attribute(
                    GEN_AI_TOOL_CALL_ARGUMENTS, safe_json_dumps(tool_input))
            state.span.add_event(
                "callback.on_agent_action",
                attributes=compact_dict(
                    {
                        "agent.tool": str(tool_name) if tool_name is not None else None,
                        "agent.tool_input": safe_json_dumps(tool_input),
                    }
                ),
            )
        self._lifecycle.debug_lifecycle(
            "agent.action",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            tool=tool_name,
            tool_input=tool_input,
            kwargs=kwargs,
        )
        return action

    @_swallow_callback_exceptions
    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        return_values = getattr(finish, "return_values", None)
        state = self._registry.get(run_id)
        if state is not None and state.span is not None:
            if return_values is not None:
                state.span.set_attribute(
                    GEN_AI_TOOL_CALL_RESULT, safe_json_dumps(return_values))
            state.span.set_attribute(GEN_AI_AGENT_NAME, state.name)
            state.span.add_event(
                "callback.on_agent_finish",
                attributes=compact_dict(
                    {"agent.return_values": safe_json_dumps(return_values)}),
            )
        self._lifecycle.debug_lifecycle(
            "agent.finish",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            return_values=return_values,
            kwargs=kwargs,
        )
        return finish

    @_swallow_callback_exceptions
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        state = self._registry.get(run_id)
        if state is not None and state.span is not None:
            state.span.add_event(
                f"callback.custom.{name}",
                attributes=compact_dict(
                    {
                        LangChain.METADATA: safe_json_dumps(metadata) if metadata else None,
                        LangChain.TAGS: safe_json_dumps(tags) if tags else None,
                        "custom.data": safe_json_dumps(data),
                    }
                ),
            )
        self._lifecycle.debug_lifecycle(
            "custom_event",
            run_id=str(run_id),
            name=name,
            tags=tags,
            metadata=metadata,
            data=data,
            kwargs=kwargs,
        )
        return data
