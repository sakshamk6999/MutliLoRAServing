"""
Serving metrics: throughput, end-to-end latency, first-token latency (TTFT).

Adapter loading (vs naive get_peft_model + LoraConfig only)
------------------------------------------------------------
``get_peft_model(base, LoraConfig(r=rank))`` **creates a new LoRA structure** with
**random (untrained) weights** unless you load checkpoints yourself. Parsing
``rank`` from a path only fixes the **architecture**, not task-specific weights.

``PeftModel.from_pretrained(base, adapter_path, ...)`` / ``load_adapter(path)``
loads **trained** tensors from disk; behavior matches the adapter that was saved.

Those are **not** interchangeable: one is a random init / shape check, the other is
real fine-tuned weights. For serving, always use **from_pretrained / load_adapter**
with paths produced by training (as in this backend).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

logger = logging.getLogger(__name__)

# Rolling window for "recent" throughput (seconds).
THROUGHPUT_WINDOW_SEC = 60.0


@dataclass
class ServingMetrics:
    """Cumulative + rolling stats; thread assumes single-threaded gRPC handler."""

    _t0: float = field(default_factory=time.monotonic)
    _completed: int = 0
    _latencies: List[float] = field(default_factory=list)
    _first_token_latencies: List[float] = field(default_factory=list)
    _completion_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=50000)
    )

    def record_completion(
        self,
        *,
        latency_sec: float,
        first_token_latency_sec: Optional[float],
    ) -> None:
        """Call once per finished request."""
        now = time.monotonic()
        self._completed += 1
        self._latencies.append(latency_sec)
        if first_token_latency_sec is not None:
            self._first_token_latencies.append(first_token_latency_sec)
        else:
            self._first_token_latencies.append(latency_sec)
        self._completion_times.append(now)
        self._print_snapshot(latency_sec, first_token_latency_sec)

    def _window_throughput(self) -> float:
        cutoff = time.monotonic() - THROUGHPUT_WINDOW_SEC
        times = self._completion_times
        while times and times[0] < cutoff:
            times.popleft()
        if not times:
            return 0.0
        return len(times) / THROUGHPUT_WINDOW_SEC

    def _print_snapshot(
        self,
        last_lat: float,
        last_ttft: Optional[float],
    ) -> None:
        elapsed = time.monotonic() - self._t0
        thr_global = self._completed / elapsed if elapsed > 0 else 0.0
        thr_win = self._window_throughput()
        mean_lat = sum(self._latencies) / len(self._latencies)
        if self._first_token_latencies:
            mean_ttft = sum(self._first_token_latencies) / len(
                self._first_token_latencies
            )
        else:
            mean_ttft = 0.0
        ttft_s = f"{last_ttft:.4f}s" if last_ttft is not None else "n/a"
        msg = (
            f"[metrics] completed={self._completed} "
            f"throughput_global={thr_global:.4f} req/s "
            f"throughput_{int(THROUGHPUT_WINDOW_SEC)}s_window={thr_win:.4f} req/s "
            f"latency_mean={mean_lat:.4f}s "
            f"first_token_latency_mean={mean_ttft:.4f}s "
            f"| last: latency={last_lat:.4f}s first_token={ttft_s}"
        )
        logger.info(msg)
        print(msg, flush=True)
