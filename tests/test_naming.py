from __future__ import annotations

from langchain_instrumentation.callbacks.naming import resolve_run_name
from langchain_instrumentation.context import CURRENT_WORKFLOW_NAME
from langchain_instrumentation.semconv import SpanKinds


def test_resolve_run_name_prefers_workflow_for_root_chain() -> None:
    token = CURRENT_WORKFLOW_NAME.set("my_workflow")
    try:
        name = resolve_run_name(
            run_type=SpanKinds.CHAIN,
            serialized=None,
            metadata={},
            callback_kwargs={},
            fallback="chain",
            is_root=True,
        )
        assert name == "my_workflow"
    finally:
        CURRENT_WORKFLOW_NAME.reset(token)


def test_resolve_run_name_chain_non_root() -> None:
    token = CURRENT_WORKFLOW_NAME.set("wf")
    try:
        name = resolve_run_name(
            run_type=SpanKinds.CHAIN,
            serialized=None,
            metadata={"langgraph_node": "node_a"},
            callback_kwargs={},
            fallback="chain",
            is_root=False,
        )
        assert name == "node_a"
    finally:
        CURRENT_WORKFLOW_NAME.reset(token)
