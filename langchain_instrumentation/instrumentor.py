from __future__ import annotations

from typing import Any

from opentelemetry import metrics, trace

try:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
except ImportError:  # pragma: no cover
    class BaseInstrumentor:  # type: ignore
        def instrument(self, **kwargs: Any) -> Any:
            return self._instrument(**kwargs)

        def uninstrument(self, **kwargs: Any) -> Any:
            return self._uninstrument(**kwargs)

from .callbacks import LangChainCallbackHandler
from .config import LangChainConfig
from .emitters import LogEmitter, MetricsEmitter, SpanEmitter
from .patches import PatchManager
from .registry import RunRegistry
from .usage import UsageRegistry


class LangChainInstrumentor(BaseInstrumentor):
    def __init__(self, config: LangChainConfig | None = None) -> None:
        super().__init__()
        self._config = config or LangChainConfig()
        self._registry: RunRegistry | None = None
        self._handler: LangChainCallbackHandler | None = None
        self._patch_manager: PatchManager | None = None

    def instrumentation_dependencies(self) -> list[str]:
        return [
            "langchain-core",
            "opentelemetry-instrumentation",
            "wrapt",
        ]

    @property
    def handler(self) -> LangChainCallbackHandler | None:
        return self._handler

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        tracer = trace.get_tracer(
            self._config.instrumentation_scope_name, tracer_provider=tracer_provider)
        meter = metrics.get_meter(
            self._config.instrumentation_scope_name, meter_provider=meter_provider)
        self._registry = RunRegistry()
        span_emitter = SpanEmitter(tracer, self._config)
        metrics_emitter = MetricsEmitter(
            meter if self._config.emit_metrics else None)
        log_emitter = LogEmitter(self._config)
        usage_registry = UsageRegistry()
        self._handler = LangChainCallbackHandler(
            config=self._config,
            registry=self._registry,
            span_emitter=span_emitter,
            metrics_emitter=metrics_emitter,
            log_emitter=log_emitter,
            usage_registry=usage_registry,
        )
        self._patch_manager = PatchManager(self._config, handler=self._handler)
        self._patch_manager.apply_all()

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._patch_manager is not None:
            self._patch_manager.remove_all()
        self._patch_manager = None
        if self._registry is not None:
            self._registry.clear()
        self._registry = None
        self._handler = None
