from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_instrumentation.config import LangChainConfig
from langchain_instrumentation.patches import PatchManager, _unwrap_wrapped_function


def test_patch_manager_remove_all_calls_unwrap_for_enabled_handles() -> None:
    config = LangChainConfig(
        patch_callback_manager=False,
        patch_langgraph=False,
        patch_langchain_factories=False,
    )
    pm = PatchManager(config)
    pm._handles.clear()
    from langchain_instrumentation.patches import PatchHandle

    pm._handles.append(PatchHandle(module="fake.mod", name="Fn", enabled=True))
    pm._handles.append(PatchHandle(module="other", name="Other.fn", enabled=False))

    with patch(
        "langchain_instrumentation.patches._unwrap_wrapped_function"
    ) as uw:
        pm.remove_all()
        uw.assert_called_once_with("fake.mod", "Fn")
    assert pm._handles == []


def test_unwrap_fallback_strips_wrapt_layer() -> None:
    import sys
    import types

    def original() -> str:
        return "ok"

    wrapped = MagicMock()
    wrapped.__wrapped__ = original

    mod = types.ModuleType("fake_unwrap_mod")
    mod.target = wrapped
    sys.modules["fake_unwrap_mod"] = mod
    try:
        with patch(
            "langchain_instrumentation.patches._otel_unwrap", None
        ):
            _unwrap_wrapped_function("fake_unwrap_mod", "target")
        assert mod.target is original
    finally:
        del sys.modules["fake_unwrap_mod"]
