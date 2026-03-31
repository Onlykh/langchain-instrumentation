from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from ..config import LangChainConfig
from ..context import CURRENT_COMMAND_SOURCE, CURRENT_LANGGRAPH_NODE, CURRENT_WORKFLOW_NAME, SafeContextToken
from ..emitters import LogEmitter, MetricsEmitter, SpanEmitter
from ..models import RunState
from ..registry import RunRegistry
from ..semconv import LangChain, SpanKinds
from ..serialization import extract_prompt_template, normalize_messages, redact_payload, serialize_io
from ..usage import UsageRegistry
from ..utils import compact_dict, ensure_list, safe_json_dumps

logger = logging.getLogger("otel.instrumentation.langchain.lifecycle")

SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY = "suppress_language_model_instrumentation"


class CallbackRunLifecycle:
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
        self._config = config
        self._registry = registry
        self._span_emitter = span_emitter
        self._metrics_emitter = metrics_emitter
        self._log_emitter = log_emitter
        self._usage_registry = usage_registry

    def debug_lifecycle(self, event: str, **payload: Any) -> None:
        if not self._config.debug_lifecycle_logging and not self._config.log_callback_payloads:
            return
        try:
            compact_payload = compact_dict(payload)
            if self._config.debug_lifecycle_logging:
                logger.info("%s %s", event, safe_json_dumps(compact_payload))
            if self._config.log_callback_payloads:
                self._log_emitter.emit(f"callback.{event}", compact_payload)
        except Exception:
            logger.exception(
                "Failed to emit lifecycle debug log for event=%s", event)

    def start_run(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        run_type: str,
        name: str,
        serialized: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        inputs: Any = None,
        callback_kwargs: Mapping[str, Any] | None = None,
    ) -> RunState:
        parent_state = self._registry.get(parent_run_id)
        root_run_id = parent_state.root_run_id if parent_state else run_id
        state = RunState(
            run_id=run_id,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            run_type=run_type,
            name=name,
            workflow_name=CURRENT_WORKFLOW_NAME.get(),
            langgraph_node=CURRENT_LANGGRAPH_NODE.get(),
            command_source=CURRENT_COMMAND_SOURCE.get(),
            serialized=dict(serialized) if serialized else None,
        )
        state.payload.metadata = redact_payload(
            dict(metadata or {}), self._config)
        state.payload.tags = [str(tag) for tag in ensure_list(tags)]
        state.payload.raw_input = inputs
        state.payload.input_value, state.payload.input_mime_type = serialize_io(
            inputs, self._config)
        if self._config.record_invocation_params:
            state.payload.invocation_params = self.extract_invocation_params(
                serialized=serialized,
                callback_kwargs=callback_kwargs,
            )
        prompt_template, prompt_variables = extract_prompt_template(serialized)
        if prompt_template is not None:
            state.payload.metadata.setdefault(
                LangChain.PROMPT_TEMPLATE, prompt_template)
        if prompt_variables:
            state.payload.metadata.setdefault(
                LangChain.PROMPT_TEMPLATE_VARIABLES, prompt_variables)
        parent_span = parent_state.span if parent_state else None
        state.span = self._span_emitter.start_span(state, parent_span)
        self._span_emitter.populate_common_attributes(state)
        if self._config.attach_current_span_for_children:
            state.context_token = SafeContextToken.attach_span(state.span)
        if run_type in {SpanKinds.LLM, SpanKinds.CHAT_MODEL}:
            state.suppression_token = SafeContextToken.attach_value(
                SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True
            )
        self._registry.add(state)
        self.debug_lifecycle(
            "run.start",
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            root_run_id=str(root_run_id),
            run_type=run_type,
            name=name,
            workflow_name=state.workflow_name,
            langgraph_node=state.langgraph_node,
            command_source=state.command_source,
            serialized_keys=sorted(
                (serialized or {}).keys()) if serialized else None,
            metadata_keys=sorted((metadata or {}).keys()
                                 ) if metadata else None,
            tags=state.payload.tags,
            input_mime_type=state.payload.input_mime_type,
            input_preview=state.payload.input_value,
        )
        self._log_emitter.emit("run.start", {LangChain.RUN_ID: str(
            run_id), LangChain.RUN_TYPE: run_type, "name": name})
        return state

    def extract_invocation_params(
        self,
        *,
        serialized: Mapping[str, Any] | None,
        callback_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if callback_kwargs and isinstance(callback_kwargs.get("invocation_params"), Mapping):
            params.update(dict(callback_kwargs["invocation_params"]))
        serialized_kwargs = (
            serialized.get("kwargs")
            if isinstance(serialized, Mapping) and isinstance(serialized.get("kwargs"), Mapping)
            else {}
        )
        if isinstance(serialized_kwargs, Mapping):
            for key in (
                "model",
                "model_name",
                "model_id",
                "max_tokens",
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "frequency_penalty",
                "stop",
                "stop_sequences",
                "seed",
                "n",
                "tools",
                "functions",
                "response_format",
            ):
                value = serialized_kwargs.get(key)
                if value is not None and key not in params:
                    params[key] = value
        if callback_kwargs and isinstance(callback_kwargs.get("config"), Mapping):
            params.setdefault("config", callback_kwargs.get("config"))
        return redact_payload(params, self._config) if params else {}

    def finish_run(self, run_id: UUID, *, outputs: Any = None, error: BaseException | None = None) -> None:
        state = self._registry.pop(run_id)
        if state is None:
            return
        try:
            state.end_time = datetime.now(tz=timezone.utc)
            state.payload.raw_output = outputs
            state.payload.output_value, state.payload.output_mime_type = serialize_io(
                outputs, self._config)
            extracted_usage = self._usage_registry.extract(
                [outputs, state.payload.raw_output, state.payload.metadata]
            )
            if extracted_usage is not None:
                state.payload.usage = extracted_usage
            self.debug_lifecycle(
                "run.finish.pre_emit",
                run_id=str(run_id),
                run_type=state.run_type,
                name=state.name,
                output_mime_type=state.payload.output_mime_type,
                output_preview=state.payload.output_value,
                usage=(asdict(state.payload.usage)
                       if state.payload.usage is not None else None),
                message_count=len(state.payload.message_payloads),
                completion_count=len(state.payload.completion_payloads),
                tool_call_count=len(state.payload.tool_calls),
                document_count=len(state.payload.documents),
                invocation_param_keys=sorted(state.payload.invocation_params.keys(
                )) if state.payload.invocation_params else None,
            )
            if error is not None:
                state.error = error
                if not self.is_suppressed_error(error):
                    self._span_emitter.record_error(state, error)
            self._span_emitter.populate_payload_attributes(state)
            self._metrics_emitter.record_run_end(state)
        finally:
            if state.suppression_token is not None:
                state.suppression_token.detach()
            if state.context_token is not None:
                state.context_token.detach()
            self._span_emitter.end_span(state)
            self._log_emitter.emit(
                "run.end",
                {
                    LangChain.RUN_ID: str(run_id),
                    LangChain.RUN_TYPE: state.run_type,
                    "name": state.name,
                    "error": str(error) if error is not None else None,
                },
            )

    def is_suppressed_error(self, error: BaseException) -> bool:
        text = str(error)
        return any(text.startswith(prefix) for prefix in self._config.suppressed_exception_prefixes)
