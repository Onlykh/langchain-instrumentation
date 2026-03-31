from __future__ import annotations

from langchain_instrumentation import __version__


def test_public_version_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_new_import_path_exports_main_symbols() -> None:
    from langchain_instrumentation import LangChainConfig, LangChainInstrumentor

    assert LangChainConfig is not None
    assert LangChainInstrumentor is not None
