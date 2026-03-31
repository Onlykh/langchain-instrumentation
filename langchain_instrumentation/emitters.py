from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, Mapping

from opentelemetry import trace as trace_api, semconv
from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_CHOICE_COUNT,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_TOKEN_TYPE,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_DESCRIPTION,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAiOperationNameValues,
    GenAiOutputTypeValues,
)
from opentelemetry.semconv._incubating.metrics.gen_ai_metrics import (
    create_gen_ai_client_operation_duration,
    create_gen_ai_client_token_usage,
    create_gen_ai_server_request_duration,
    create_gen_ai_server_time_per_output_token,
    create_gen_ai_server_time_to_first_token,
)
from opentelemetry.trace import SpanKind, Status, StatusCode

from .config import LangChainConfig
from .models import RunState
from .semconv import LangChain, MimeTypes, SpanKinds
from .utils import compact_dict, flatten_str_dict, safe_json_dumps


class MetricsEmitter:
    def __init__(self, meter: Meter | None) -> None:
        self._meter = meter
        self._duration_histogram: Histogram | None = None
        self._token_histogram: Histogram | None = None
        self._genai_operation_duration_histogram: Histogram | None = None
        self._genai_token_usage_histogram: Histogram | None = None
        self._genai_server_request_duration_histogram: Histogram | None = None
        self._genai_server_time_to_first_token_histogram: Histogram | None = None
        self._genai_server_time_per_output_token_histogram: Histogram | None = None
        if meter is not None:
            self._duration_histogram = meter.create_histogram(
                name="langchain.instrumentation.run.duration_ms",
                unit="ms",
                description="Runtime for LangChain/LangGraph run nodes",
            )
            self._token_histogram = meter.create_histogram(
                name="langchain.instrumentation.tokens",
                unit="1",
                description="Observed token usage for LangChain/LangGraph spans",
            )
            self._genai_operation_duration_histogram = create_gen_ai_client_operation_duration(
                meter)
            self._genai_token_usage_histogram = create_gen_ai_client_token_usage(
                meter)
            self._genai_server_request_duration_histogram = create_gen_ai_server_request_duration(
                meter)
            self._genai_server_time_to_first_token_histogram = create_gen_ai_server_time_to_first_token(
                meter)
            self._genai_server_time_per_output_token_histogram = create_gen_ai_server_time_per_output_token(
                meter)

    def record_run_end(self, state: RunState) -> None:
        if self._duration_histogram is None and self._genai_operation_duration_histogram is None:
            return
        if state.end_time is None:
            return
        duration_ms = max(
            (state.end_time - state.start_time).total_seconds() * 1000.0, 0.0)
        if self._duration_histogram is not None:
            self._duration_histogram.record(
                duration_ms,
                {
                    LangChain.SPAN_KIND: state.run_type,
                    LangChain.WORKFLOW_NAME: state.workflow_name or "",
                },
            )
        operation_name = _operation_name_for_run_type(state)
        provider_name = _provider_name_for_state(state)
        response_model = _response_model_for_state(state)
        metric_attrs = compact_dict(
            {
                GEN_AI_OPERATION_NAME: operation_name,
                GEN_AI_PROVIDER_NAME: provider_name,
                GEN_AI_RESPONSE_MODEL: response_model,
            }
        )
        if self._genai_operation_duration_histogram is not None and metric_attrs:
            self._genai_operation_duration_histogram.record(
                duration_ms / 1000.0,
                metric_attrs,
            )
        if self._genai_server_request_duration_histogram is not None and metric_attrs:
            self._genai_server_request_duration_histogram.record(
                duration_ms / 1000.0,
                metric_attrs,
            )
            if (
                state.error is None
                and state.first_token_time is not None
                and state.run_type in {SpanKinds.LLM, SpanKinds.CHAT_MODEL}
            ):
                ttft_s = max(
                    (state.first_token_time - state.start_time).total_seconds(), 0.0
                )
                self._genai_server_time_to_first_token_histogram.record(
                    ttft_s, metric_attrs)
                if state.stream_chunk_count > 1:
                    post_first_duration = max(
                        (state.end_time - state.first_token_time).total_seconds(), 0.0
                    )
                    self._genai_server_time_per_output_token_histogram.record(
                        post_first_duration / (state.stream_chunk_count - 1),
                        metric_attrs,
                    )
        if self._token_histogram is None and self._genai_token_usage_histogram is None:
            return
        if state.payload.usage is None:
            return
        usage = state.payload.usage
        token_map = {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.total_tokens,
            "reasoning": usage.reasoning_tokens,
        }
        for token_type, token_value in token_map.items():
            if token_value is None:
                continue
            if self._token_histogram is not None:
                self._token_histogram.record(
                    token_value, {"langchain.token.type": token_type, LangChain.SPAN_KIND: state.run_type})
            if self._genai_token_usage_histogram is not None and token_type in {"input", "output"}:
                self._genai_token_usage_histogram.record(
                    token_value,
                    compact_dict(
                        {
                            GEN_AI_TOKEN_TYPE: token_type,
                            GEN_AI_OPERATION_NAME: operation_name,
                            GEN_AI_PROVIDER_NAME: provider_name,
                            GEN_AI_RESPONSE_MODEL: response_model,
                        }
                    ),
                )


class LogEmitter:
    def __init__(self, config: LangChainConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("otel.instrumentation.langchain.events")
        self._otel_logger = None
        if config.emit_logs:
            try:
                from opentelemetry._logs import get_logger
                self._otel_logger = get_logger(
                    config.instrumentation_scope_name)
            except Exception:
                self._otel_logger = None

    def emit(self, event_name: str, attributes: Mapping[str, Any]) -> None:
        payload = safe_json_dumps(
            {"event": event_name, "attributes": dict(attributes)})
        if self._otel_logger is not None:
            try:
                self._otel_logger.emit(body=payload)
            except Exception:
                pass
        self._logger.info(payload)


class SpanEmitter:
    def __init__(self, tracer: trace_api.Tracer, config: LangChainConfig) -> None:
        self._tracer = tracer
        self._config = config

    def start_span(self, state: RunState, parent_span: Any | None) -> Any:
        parent_context = trace_api.set_span_in_context(
            parent_span) if parent_span is not None else None
        span = self._tracer.start_span(
            name=_span_name_for_state(state),
            context=parent_context,
            kind=_span_kind_for_run_type(state.run_type),
            start_time=_datetime_to_unix_nanos(state.start_time),
        )
        return span

    def populate_common_attributes(self, state: RunState) -> None:
        if state.span is None:
            return
        attrs = {
            LangChain.SPAN_KIND: state.run_type,
            LangChain.RUN_ID: str(state.run_id),
            LangChain.ROOT_RUN_ID: str(state.root_run_id),
            LangChain.RUN_TYPE: state.run_type,
            LangChain.WORKFLOW_NAME: state.workflow_name,
            LangChain.LANGGRAPH_NODE: state.langgraph_node,
            LangChain.LANGGRAPH_COMMAND_SOURCE: state.command_source,
            LangChain.RESPONSE_STREAM_CHUNK_COUNT: state.stream_chunk_count or None,
        }
        if state.parent_run_id is not None:
            attrs[LangChain.PARENT_RUN_ID] = str(state.parent_run_id)
        serialized = state.serialized or {}
        if isinstance(serialized, Mapping):
            attrs[LangChain.SERIALIZED_CLASS] = _extract_serialized_class(
                serialized)
            attrs[LangChain.SERIALIZED_NAME] = serialized.get("name")
            attrs[LangChain.SERIALIZED_ID] = safe_json_dumps(
                serialized.get("id")) if serialized.get("id") is not None else None

            attrs[LangChain.ENTITY_NAME] = state.name
        for key, value in compact_dict(attrs).items():
            state.span.set_attribute(key, value)
        for key, value in self._config.extra_span_attributes.items():
            state.span.set_attribute(key, value)

    def populate_payload_attributes(self, state: RunState) -> None:
        if state.span is None:
            return
        payload = state.payload
        provider_name = _provider_name_for_state(state)
        if provider_name is not None:
            state.span.set_attribute(GEN_AI_PROVIDER_NAME, provider_name)
        operation_name = _operation_name_for_run_type(state)
        if operation_name is not None:
            state.span.set_attribute(GEN_AI_OPERATION_NAME, operation_name)
        request_model = _request_model_for_state(state)
        if request_model is not None:
            state.span.set_attribute(GEN_AI_REQUEST_MODEL, request_model)
        response_model = _response_model_for_state(state)
        if response_model is not None:
            state.span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
        response_id = _response_id_for_state(state)
        if response_id is not None:
            state.span.set_attribute(GEN_AI_RESPONSE_ID, response_id)
        finish_reasons = _response_finish_reasons_for_state(state)
        if finish_reasons:
            state.span.set_attribute(
                GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
        if state.run_type == SpanKinds.RETRIEVER:
            retrieval_query = _retrieval_query_text_for_state(state)
            if retrieval_query is not None:
                state.span.set_attribute(
                    GEN_AI_RETRIEVAL_QUERY_TEXT, retrieval_query)
        if state.run_type == SpanKinds.TOOL:
            state.span.set_attribute(GEN_AI_TOOL_NAME, state.name)
            state.span.set_attribute(GEN_AI_TOOL_TYPE, "function")
            if isinstance(state.serialized, Mapping):
                description = state.serialized.get("description")
                if description is not None:
                    state.span.set_attribute(
                        GEN_AI_TOOL_DESCRIPTION, str(description))
        request_params = _normalized_request_params(payload.invocation_params)
        for key, value in request_params.items():
            state.span.set_attribute(key, value)
        tool_definitions = _extract_tool_definitions(payload.invocation_params)
        if tool_definitions:
            state.span.set_attribute(
                GEN_AI_TOOL_DEFINITIONS, safe_json_dumps(tool_definitions))
        if payload.input_value is not None:
            state.span.set_attribute(LangChain.INPUT_VALUE, payload.input_value)
            if payload.input_mime_type is not None:
                state.span.set_attribute(
                    LangChain.INPUT_MIME_TYPE, payload.input_mime_type)
        if payload.output_value is not None:
            state.span.set_attribute(
                LangChain.OUTPUT_VALUE, payload.output_value)
            if payload.output_mime_type is not None:
                state.span.set_attribute(
                    LangChain.OUTPUT_MIME_TYPE, payload.output_mime_type)
        if payload.metadata:
            state.span.set_attribute(
                LangChain.METADATA, safe_json_dumps(payload.metadata))
        if payload.tags:
            state.span.set_attribute(
                LangChain.TAGS, safe_json_dumps(payload.tags))
        if payload.message_payloads:
            state.span.set_attribute(
                LangChain.INPUT_MESSAGES_COUNT, len(payload.message_payloads))
            state.span.set_attribute(
                GEN_AI_INPUT_MESSAGES, safe_json_dumps(payload.message_payloads))
        if payload.completion_payloads:
            state.span.set_attribute(
                LangChain.OUTPUT_MESSAGES_COUNT, len(payload.completion_payloads))
            state.span.set_attribute(
                GEN_AI_OUTPUT_MESSAGES, safe_json_dumps(payload.completion_payloads))
            state.span.set_attribute(
                GEN_AI_REQUEST_CHOICE_COUNT, len(payload.completion_payloads))
        output_type = _output_type_for_payload(payload)
        if output_type is not None:
            state.span.set_attribute(GEN_AI_OUTPUT_TYPE, output_type)
        if payload.tool_calls:
            state.span.set_attribute(
                LangChain.TOOL_CALLS, safe_json_dumps(payload.tool_calls))
            first_tool_call = payload.tool_calls[0] if payload.tool_calls else None
            if isinstance(first_tool_call, Mapping):
                tool_call_id = first_tool_call.get("id")
                if tool_call_id is not None:
                    state.span.set_attribute(
                        GEN_AI_TOOL_CALL_ID, str(tool_call_id))
                tool_call_args = first_tool_call.get("arguments")
                if tool_call_args is not None:
                    state.span.set_attribute(
                        GEN_AI_TOOL_CALL_ARGUMENTS, safe_json_dumps(tool_call_args))
                tool_call_name = first_tool_call.get("name")
                if tool_call_name is not None:
                    state.span.set_attribute(
                        GEN_AI_TOOL_NAME, str(tool_call_name))
                tool_call_type = first_tool_call.get("type")
                if tool_call_type is not None:
                    state.span.set_attribute(
                        GEN_AI_TOOL_TYPE, str(tool_call_type))
            for index, tool_call in enumerate(payload.tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                event_attrs = compact_dict(
                    {
                        GEN_AI_TOOL_CALL_ID: str(tool_call.get("id")) if tool_call.get("id") is not None else None,
                        GEN_AI_TOOL_NAME: str(tool_call.get("name")) if tool_call.get("name") is not None else None,
                        GEN_AI_TOOL_TYPE: str(tool_call.get("type")) if tool_call.get("type") is not None else None,
                        GEN_AI_TOOL_CALL_ARGUMENTS: safe_json_dumps(tool_call.get("arguments")) if tool_call.get("arguments") is not None else None,
                    }
                )
                if event_attrs:
                    state.span.add_event(
                        f"gen_ai.tool.call.{index}", attributes=event_attrs)
        if payload.invocation_params:
            flattened = flatten_str_dict(
                payload.invocation_params, prefix="langchain.request")
            for key, value in flattened.items():
                state.span.set_attribute(key, value)
        usage = payload.usage
        if usage is not None:
            if usage.input_tokens is not None:
                state.span.set_attribute(
                    GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)

            if usage.output_tokens is not None:
                state.span.set_attribute(
                    GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)

            if usage.reasoning_tokens is not None:
                state.span.set_attribute(
                    LangChain.USAGE_REASONING_TOKENS, usage.reasoning_tokens)
            if usage.audio_input_tokens is not None:
                state.span.set_attribute(
                    LangChain.USAGE_AUDIO_INPUT_TOKENS, usage.audio_input_tokens)
            if usage.audio_output_tokens is not None:
                state.span.set_attribute(
                    LangChain.USAGE_AUDIO_OUTPUT_TOKENS, usage.audio_output_tokens)
            if usage.cache_read_tokens is not None:
                state.span.set_attribute(
                    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, usage.cache_read_tokens)
                state.span.set_attribute(
                    LangChain.USAGE_CACHE_READ_TOKENS, usage.cache_read_tokens)
            if usage.cache_write_tokens is not None:
                state.span.set_attribute(
                    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, usage.cache_write_tokens)
                state.span.set_attribute(
                    LangChain.USAGE_CACHE_WRITE_TOKENS, usage.cache_write_tokens)
            if usage.source is not None:
                state.span.set_attribute(
                    LangChain.PROVIDER_USAGE_SOURCE, usage.source)
        if state.run_type == SpanKinds.TOOL and payload.output_value is not None:
            state.span.set_attribute(
                GEN_AI_TOOL_CALL_RESULT, payload.output_value)
        if payload.documents:
            state.span.set_attribute(
                LangChain.RETRIEVER_DOCUMENT_COUNT, len(payload.documents))
            state.span.set_attribute(
                GEN_AI_RETRIEVAL_DOCUMENTS, safe_json_dumps(payload.documents))
        if self._config.record_raw_payloads:
            if payload.raw_input is not None:
                state.span.set_attribute(
                    LangChain.RAW_INPUT, safe_json_dumps(payload.raw_input))
            if payload.raw_output is not None:
                state.span.set_attribute(
                    LangChain.RAW_OUTPUT, safe_json_dumps(payload.raw_output))

    def record_error(self, state: RunState, error: BaseException) -> None:
        if state.span is None:
            return
        state.span.set_status(Status(StatusCode.ERROR, str(error)))
        state.span.set_attribute(LangChain.ERROR_CLASS, error.__class__.__name__)
        state.span.set_attribute(LangChain.ERROR_MESSAGE, str(error))
        try:
            state.span.record_exception(error)
        except Exception:
            pass

    def end_span(self, state: RunState) -> None:
        if state.span is None:
            return
        end_time = _datetime_to_unix_nanos(
            state.end_time) if state.end_time is not None else None
        state.span.end(end_time=end_time)


def _extract_serialized_class(serialized: Mapping[str, Any]) -> str | None:
    class_id = serialized.get("id")
    if isinstance(class_id, list) and class_id:
        return str(class_id[-1])
    if class_id is not None:
        return str(class_id)
    return None


def _span_kind_for_run_type(run_type: str) -> SpanKind:
    if run_type in {SpanKinds.LLM, SpanKinds.CHAT_MODEL, SpanKinds.EMBEDDING}:
        return SpanKind.CLIENT
    return SpanKind.INTERNAL


def _datetime_to_unix_nanos(value: Any) -> int:
    # OpenTelemetry SDKs expect Unix nanoseconds for explicit timestamps.
    return int(value.timestamp() * 1_000_000_000)


def _operation_name_for_run_type(state: RunState) -> str | None:
    if state.run_type == SpanKinds.CHAT_MODEL:
        return GenAiOperationNameValues.CHAT.value
    if state.run_type == SpanKinds.LLM:
        return GenAiOperationNameValues.TEXT_COMPLETION.value
    if state.run_type == SpanKinds.EMBEDDING:
        return GenAiOperationNameValues.EMBEDDINGS.value
    if state.run_type == SpanKinds.TOOL:
        return GenAiOperationNameValues.EXECUTE_TOOL.value
    if state.run_type == SpanKinds.RETRIEVER:
        return GenAiOperationNameValues.RETRIEVAL.value
    if state.run_type == SpanKinds.CHAIN and state.parent_run_id is None:
        return GenAiOperationNameValues.INVOKE_AGENT.value
    return None


def _span_name_for_state(state: RunState) -> str:
    name = str(state.name).strip() if state.name is not None else ""
    if name:
        return name
    if state.run_type == SpanKinds.CHAIN and state.parent_run_id is None:
        return GenAiOperationNameValues.INVOKE_AGENT.value
    if state.run_type == SpanKinds.TOOL:
        return GenAiOperationNameValues.EXECUTE_TOOL.value
    if state.run_type == SpanKinds.RETRIEVER:
        return GenAiOperationNameValues.RETRIEVAL.value
    if state.run_type == SpanKinds.CHAT_MODEL:
        return GenAiOperationNameValues.CHAT.value
    if state.run_type == SpanKinds.LLM:
        return GenAiOperationNameValues.TEXT_COMPLETION.value
    return state.run_type.lower()


def _provider_name_for_state(state: RunState) -> str | None:
    metadata = state.payload.metadata if isinstance(
        state.payload.metadata, Mapping) else {}
    provider = metadata.get("ls_provider") or metadata.get("provider")
    return str(provider) if provider is not None else None


def _request_model_for_state(state: RunState) -> str | None:
    params = state.payload.invocation_params if isinstance(
        state.payload.invocation_params, Mapping) else {}
    for key in ("model", "model_name", "model_id"):
        value = params.get(key)
        if value is not None:
            return str(value)
    metadata = state.payload.metadata if isinstance(
        state.payload.metadata, Mapping) else {}
    for key in ("ls_model_name", "model_name"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return _response_model_for_state(state)


def _llm_output_for_state(state: RunState) -> Mapping[str, Any]:
    raw_output = state.payload.raw_output
    llm_output = getattr(raw_output, "llm_output", None)
    return llm_output if isinstance(llm_output, Mapping) else {}


def _response_model_for_state(state: RunState) -> str | None:
    llm_output = _llm_output_for_state(state)
    for key in ("model_name", "model_id"):
        value = llm_output.get(key)
        if value is not None:
            return str(value)
    return _request_model_for_state_fallback(state)


def _request_model_for_state_fallback(state: RunState) -> str | None:
    metadata = state.payload.metadata if isinstance(
        state.payload.metadata, Mapping) else {}
    value = metadata.get("ls_model_name") or metadata.get("model_name")
    return str(value) if value is not None else None


def _response_id_for_state(state: RunState) -> str | None:
    llm_output = _llm_output_for_state(state)
    value = llm_output.get("id")
    return str(value) if value is not None else None


def _response_finish_reasons_for_state(state: RunState) -> list[str]:
    output: list[str] = []
    for item in state.payload.completion_payloads:
        if not isinstance(item, Mapping):
            continue
        generation_info = item.get("generation_info")
        if not isinstance(generation_info, Mapping):
            continue
        finish_reason = generation_info.get("finish_reason")
        if finish_reason is not None:
            output.append(str(finish_reason))
    return output


def _retrieval_query_text_for_state(state: RunState) -> str | None:
    raw_input = state.payload.raw_input
    if isinstance(raw_input, Mapping):
        value = raw_input.get("query")
        if value is not None:
            return str(value)
    if isinstance(raw_input, str):
        return raw_input
    return None


def _extract_tool_definitions(invocation_params: Mapping[str, Any] | None) -> list[Any]:
    if not isinstance(invocation_params, Mapping):
        return []
    for key in ("tools", "functions"):
        value = invocation_params.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalized_request_params(invocation_params: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(invocation_params, Mapping):
        return {}
    out: dict[str, Any] = {}
    mapping = {
        GEN_AI_REQUEST_MAX_TOKENS: ("max_tokens", "max_new_tokens"),
        GEN_AI_REQUEST_TEMPERATURE: ("temperature",),
        GEN_AI_REQUEST_TOP_P: ("top_p",),
        GEN_AI_REQUEST_TOP_K: ("top_k",),
        GEN_AI_REQUEST_PRESENCE_PENALTY: ("presence_penalty",),
        GEN_AI_REQUEST_FREQUENCY_PENALTY: ("frequency_penalty",),
        GEN_AI_REQUEST_STOP_SEQUENCES: ("stop_sequences", "stop"),
        GEN_AI_REQUEST_SEED: ("seed",),
        GEN_AI_REQUEST_CHOICE_COUNT: ("n",),
    }
    for target_key, source_keys in mapping.items():
        for source in source_keys:
            if invocation_params.get(source) is not None:
                value = invocation_params.get(source)
                if isinstance(value, (dict, list)):
                    out[target_key] = safe_json_dumps(value)
                else:
                    out[target_key] = value
                break
    return out


def _output_type_for_payload(payload: Any) -> str | None:
    mime_type = payload.output_mime_type
    if mime_type == MimeTypes.JSON:
        return GenAiOutputTypeValues.JSON.value
    if mime_type == MimeTypes.TEXT:
        return GenAiOutputTypeValues.TEXT.value
    return None
