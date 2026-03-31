from __future__ import annotations

from threading import RLock
from typing import Iterable
from uuid import UUID

from .models import RunState


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[UUID, RunState] = {}
        self._lock = RLock()

    def add(self, state: RunState) -> None:
        with self._lock:
            self._runs[state.run_id] = state
            if state.parent_run_id and state.parent_run_id in self._runs:
                self._runs[state.parent_run_id].children.add(state.run_id)

    def get(self, run_id: UUID | None) -> RunState | None:
        if run_id is None:
            return None
        with self._lock:
            return self._runs.get(run_id)

    def pop(self, run_id: UUID) -> RunState | None:
        with self._lock:
            state = self._runs.pop(run_id, None)
            if state and state.parent_run_id and state.parent_run_id in self._runs:
                self._runs[state.parent_run_id].children.discard(run_id)
            return state

    def values(self) -> Iterable[RunState]:
        with self._lock:
            return tuple(self._runs.values())

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
