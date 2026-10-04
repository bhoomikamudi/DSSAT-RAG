"""Per-session conversation context for follow-up analysis questions.

Only the structured AnalysisRequest (variables, filters, grouping) is kept,
never the retrieved data: follow-ups always re-query PostgreSQL so results
are computed from the underlying records, not from rendered chart points.

The store is in-process memory. It is lost on restart and not shared across
worker processes; a shared store (e.g. Redis) would be needed for that.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Optional

from app.agent.models import AnalysisRequest


def describe_analysis_context(request: AnalysisRequest) -> str:
    """Planner prompt section describing the previous analysis request.

    Lets the LLM plan follow-ups ("now fit a quadratic instead"). The area is
    left out: locations are resolved separately before planning.
    """
    previous = request.model_dump(
        mode="json",
        include={"operation", "x_variable", "y_variable", "filters", "group_by"},
    )
    return (
        "PREVIOUS ANALYSIS IN THIS CONVERSATION:\n"
        f"{json.dumps(previous)}\n"
        "If the user query is a follow-up to this analysis (for example "
        "\"now fit a quadratic instead\" or \"what about LNG?\"), return an "
        "analysis operation that keeps its variables and filters except for "
        "what the user changes."
    )


class AnalysisContextStore:
    """Bounded LRU map of session_id -> last AnalysisRequest."""

    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self._items: "OrderedDict[str, AnalysisRequest]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str]) -> Optional[AnalysisRequest]:
        if not session_id:
            return None
        with self._lock:
            request = self._items.get(session_id)
            if request is not None:
                self._items.move_to_end(session_id)
            return request.model_copy(deep=True) if request else None

    def set(self, session_id: Optional[str], request: AnalysisRequest) -> None:
        if not session_id:
            return
        with self._lock:
            self._items[session_id] = request.model_copy(deep=True)
            self._items.move_to_end(session_id)
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


analysis_context_store = AnalysisContextStore()
