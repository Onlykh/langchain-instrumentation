# langchain-instrumentation

OpenTelemetry instrumentation SDK for LangChain and LangGraph.

## Install

```bash
pip install langchain-instrumentation
```

## Quickstart (OTEL collector)

```python
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from langchain_instrumentation import LangChainConfig, LangChainInstrumentor

# Example collector endpoint (OTLP gRPC)
endpoint = "http://localhost:4317"
resource = Resource.create({SERVICE_NAME: "langchain-app"})

# Trace pipeline
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
)

# Metrics pipeline
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=10000,
        )
    ],
)

# Instrument LangChain and LangGraph integrations
instrumentor = LangChainInstrumentor(
    LangChainConfig(
        emit_logs=True,
        emit_metrics=True,
        record_content=True,
        record_raw_payloads=True,
    )
)
instrumentor.instrument(
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)

# Optional: pass callback handler explicitly in custom callback-manager flows
handler = instrumentor.handler
```

## What it instruments

- LangChain callback lifecycle spans
- Optional LangGraph tracing hooks
- Optional agent factory tracing hooks
- Metrics and structured log emission

Use `instrumentor.uninstrument()` during shutdown or test teardown.

## Configuration

`LangChainConfig` controls behavior. Most-used flags:

- `emit_metrics`, `emit_logs`
- `patch_callback_manager`, `patch_langgraph`, `patch_langchain_factories`
- `record_content`, `record_raw_payloads`, `redact_keys`
