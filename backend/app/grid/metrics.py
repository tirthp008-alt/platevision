"""Lightweight in-process metrics/instrumentation.

Tracks the acceptance-target instrumentation from the spec: API latency,
dashboard latency, processing FPS, candidate rejection rate, abstention rate,
OCR accuracy and accepted-link precision (the latter two against optional
labelled ground truth). Values are in-memory for the prototype but the
``snapshot`` shape is designed to be exported to Prometheus/OTel later.
"""

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional


class MetricsRegistry:
    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self._latencies: Dict[str, Deque[float]] = {}
        self._counters: Dict[str, float] = {}
        # Labelled evaluation accumulators
        self._ocr_correct = 0
        self._ocr_total = 0
        self._links_accepted = 0
        self._links_accepted_correct = 0
        self._candidates_total = 0
        self._candidates_rejected = 0
        self._candidates_abstained = 0
        self._window = window

    def observe_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            series = self._latencies.setdefault(name, deque(maxlen=self._window))
            series.append(seconds)

    def incr(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def record_ocr(self, correct: bool) -> None:
        with self._lock:
            self._ocr_total += 1
            if correct:
                self._ocr_correct += 1

    def record_link_decision(self, decision: str, correct: Optional[bool] = None) -> None:
        with self._lock:
            self._candidates_total += 1
            if decision == "ACCEPTED":
                self._links_accepted += 1
                if correct:
                    self._links_accepted_correct += 1
            elif decision == "REJECTED":
                self._candidates_rejected += 1
            elif decision == "ABSTAINED":
                self._candidates_abstained += 1

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
        return ordered[idx]

    def snapshot(self) -> dict:
        with self._lock:
            latencies = {}
            for name, series in self._latencies.items():
                vals = list(series)
                if not vals:
                    continue
                latencies[name] = {
                    "count": len(vals),
                    "p50_ms": round(self._percentile(vals, 50) * 1000, 2),
                    "p95_ms": round(self._percentile(vals, 95) * 1000, 2),
                    "max_ms": round(max(vals) * 1000, 2),
                }
            ocr_accuracy = (
                self._ocr_correct / self._ocr_total if self._ocr_total else None
            )
            link_precision = (
                self._links_accepted_correct / self._links_accepted
                if self._links_accepted
                else None
            )
            rejection_rate = (
                self._candidates_rejected / self._candidates_total
                if self._candidates_total
                else None
            )
            abstention_rate = (
                self._candidates_abstained / self._candidates_total
                if self._candidates_total
                else None
            )
            counters = dict(self._counters)
        return {
            "latency": latencies,
            "counters": counters,
            "evaluation": {
                "ocr_accuracy": ocr_accuracy,
                "ocr_samples": self._ocr_total,
                "accepted_link_precision": link_precision,
                "accepted_links": self._links_accepted,
                "candidate_rejection_rate": rejection_rate,
                "abstention_rate": abstention_rate,
                "candidates_total": self._candidates_total,
            },
            "targets": {
                "ocr_accuracy": 0.90,
                "accepted_link_precision": 0.95,
                "dashboard_latency_p95_ms": 5000,
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._ocr_correct = self._ocr_total = 0
            self._links_accepted = self._links_accepted_correct = 0
            self._candidates_total = self._candidates_rejected = self._candidates_abstained = 0


metrics = MetricsRegistry()


class LatencyTimer:
    """Context manager measuring latency into a named metric."""

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        metrics.observe_latency(self.name, time.time() - self.start)
        return False
