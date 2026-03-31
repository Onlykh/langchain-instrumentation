from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_instrumentation import LangChainConfig, LangChainInstrumentor


def test_uninstrument_invokes_patch_manager_remove_all() -> None:
    config = LangChainConfig(
        patch_callback_manager=False,
        patch_langgraph=False,
        patch_langchain_factories=False,
    )
    inst = LangChainInstrumentor(config)
    mock_pm = MagicMock()
    with patch(
        "langchain_instrumentation.instrumentor.PatchManager",
        return_value=mock_pm,
    ):
        inst._instrument()
    assert inst._patch_manager is mock_pm
    mock_pm.apply_all.assert_called_once()

    inst._uninstrument()
    mock_pm.remove_all.assert_called_once()
    assert inst._patch_manager is None
    assert inst._handler is None
