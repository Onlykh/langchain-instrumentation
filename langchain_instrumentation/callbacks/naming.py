from __future__ import annotations

from typing import Any, Mapping

from ..context import CURRENT_WORKFLOW_NAME
from ..semconv import SpanKinds
from ..utils import first_non_empty


def name_from_serialized(serialized: Mapping[str, Any] | None, fallback: str = "unknown") -> str:
    if not serialized:
        return fallback
    kwargs = serialized.get("kwargs") if isinstance(
        serialized.get("kwargs"), Mapping) else {}
    return str(
        first_non_empty(
            kwargs.get("name"),
            serialized.get("name"),
            serialized.get("id")[-1] if isinstance(serialized.get("id"),
                                                   list) and serialized.get("id") else None,
            fallback,
        )
    )


def resolve_run_name(
    *,
    run_type: str,
    serialized: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    callback_kwargs: Mapping[str, Any] | None,
    fallback: str,
    is_root: bool,
) -> str:
    callback_kwargs = callback_kwargs or {}
    metadata = metadata or {}
    serialized_kwargs = (
        serialized.get("kwargs")
        if isinstance(serialized, Mapping) and isinstance(serialized.get("kwargs"), Mapping)
        else {}
    )
    serialized_id_leaf = (
        serialized.get("id")[-1]
        if isinstance(serialized, Mapping)
        and isinstance(serialized.get("id"), list)
        and serialized.get("id")
        else None
    )
    langgraph_node = metadata.get("langgraph_node")
    langgraph_path = metadata.get("langgraph_path")
    langgraph_path_leaf = (
        str(langgraph_path).split(":")[-1]
        if langgraph_path is not None and str(langgraph_path)
        else None
    )
    workflow_name = CURRENT_WORKFLOW_NAME.get()
    common_candidates = [
        callback_kwargs.get("name"),
        callback_kwargs.get("run_name"),
        serialized_kwargs.get("name"),
        serialized.get("name") if isinstance(
            serialized, Mapping) else None,
        serialized_id_leaf,
        langgraph_path_leaf,
        workflow_name if is_root else None,
        fallback,
    ]
    if run_type == SpanKinds.CHAIN:
        candidates = [
            callback_kwargs.get("name"),
            callback_kwargs.get("run_name"),
            langgraph_node,
            serialized_kwargs.get("name"),
            serialized.get("name") if isinstance(
                serialized, Mapping) else None,
            serialized_id_leaf,
            langgraph_path_leaf,
            workflow_name if is_root else None,
            fallback,
        ]
    elif run_type == SpanKinds.TOOL:
        candidates = [
            callback_kwargs.get("name"),
            callback_kwargs.get("run_name"),
            serialized_kwargs.get("name"),
            serialized.get("name") if isinstance(
                serialized, Mapping) else None,
            serialized_id_leaf,
            langgraph_node,
            langgraph_path_leaf,
            fallback,
        ]
    else:
        candidates = common_candidates
    name = str(first_non_empty(*candidates))
    generic_names = {"chain", "chat_model",
                     "llm", "tool", "retriever", "unknown"}
    if name in generic_names:
        name = str(first_non_empty(
            langgraph_node, langgraph_path_leaf, workflow_name if is_root else None, name))
    if run_type == SpanKinds.CHAIN and name == "chain" and is_root and workflow_name:
        return str(workflow_name)
    return name
