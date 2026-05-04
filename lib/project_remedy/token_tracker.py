"""Centralized token usage tracker for all LLM API calls.

Provides a thread-safe global counter that aggregates token usage
across Ollama and standalone vision providers.

Usage::

    from project_remedy.token_tracker import tracker

    # Record usage (called automatically by clients/providers):
    tracker.record("ollama", input_tokens=100, output_tokens=50)

    # Read totals:
    print(tracker.summary())

    # Reset between runs:
    tracker.reset()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    calls: int = 0


class TokenTracker:
    """Thread-safe global token usage tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._providers: dict[str, _ProviderUsage] = {}
        self._start_time: float = time.monotonic()

    def record(
        self,
        provider: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thought_tokens: int = 0,
    ) -> None:
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = _ProviderUsage()
            p = self._providers[provider]
            p.input_tokens += input_tokens
            p.output_tokens += output_tokens
            p.thought_tokens += thought_tokens
            p.calls += 1

    @property
    def total_input(self) -> int:
        with self._lock:
            return sum(p.input_tokens for p in self._providers.values())

    @property
    def total_output(self) -> int:
        with self._lock:
            return sum(p.output_tokens for p in self._providers.values())

    @property
    def total_thoughts(self) -> int:
        with self._lock:
            return sum(p.thought_tokens for p in self._providers.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    @property
    def billed_total_tokens(self) -> int:
        return self.total_input + self.total_output + self.total_thoughts

    @property
    def total_calls(self) -> int:
        with self._lock:
            return sum(p.calls for p in self._providers.values())

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def summary(self) -> dict:
        with self._lock:
            by_provider = {
                name: {
                    "input_tokens": p.input_tokens,
                    "output_tokens": p.output_tokens,
                    "thought_tokens": p.thought_tokens,
                    "total_tokens": p.input_tokens + p.output_tokens,
                    "billed_total_tokens": (
                        p.input_tokens + p.output_tokens + p.thought_tokens
                    ),
                    "calls": p.calls,
                }
                for name, p in self._providers.items()
            }
        return {
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "thought_tokens": self.total_thoughts,
            "total_tokens": self.total_tokens,
            "billed_total_tokens": self.billed_total_tokens,
            "api_calls": self.total_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "by_provider": by_provider,
        }

    def reset(self) -> None:
        with self._lock:
            self._providers.clear()
            self._start_time = time.monotonic()


# Module-level singleton.
tracker = TokenTracker()
