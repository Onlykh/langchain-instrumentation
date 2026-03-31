# langchain-instrumentation

OpenTelemetry instrumentation SDK for LangChain and LangGraph.

## Install

```bash
pip install langchain-instrumentation
```

## Quickstart

```python
from langchain_instrumentation import LangChainConfig, LangChainInstrumentor

instrumentor = LangChainInstrumentor(LangChainConfig())
instrumentor.instrument()

# Optional manual handler injection in custom callback manager flows
handler = instrumentor.handler

# Shutdown or test cleanup
instrumentor.uninstrument()
```

## What it instruments

- LangChain callback lifecycle spans
- Optional LangGraph tracing hooks
- Optional agent factory tracing hooks
- Metrics and structured log emission

## Configuration

`LangChainConfig` controls behavior. Most-used flags:

- `emit_metrics`, `emit_logs`
- `patch_callback_manager`, `patch_langgraph`, `patch_langchain_factories`
- `record_content`, `record_raw_payloads`, `redact_keys`
