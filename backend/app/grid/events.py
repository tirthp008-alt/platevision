"""Event streaming abstraction (EventPublisher / EventConsumer).

The full architecture is Camera -> Edge AI -> Event Stream -> Central Fusion.
Kafka is appropriate for that, but it is not mandatory for the prototype: the
default backend is an in-process queue, and a Kafka backend is used when
``EVENT_BACKEND=kafka`` and a client library is available.

Events carry lightweight descriptors (plate text, embedding, bbox) rather than
raw video, which keeps the stream small and privacy-preserving.
"""

import json
import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger


class BaseEventPublisher(ABC):
    @abstractmethod
    def publish(self, topic: str, event: Dict) -> None: ...


class BaseEventConsumer(ABC):
    @abstractmethod
    def subscribe(self, handler: Callable[[Dict], None]) -> None: ...


class InProcessEventBus(BaseEventPublisher, BaseEventConsumer):
    """Thread-safe in-process pub/sub used for local development and tests."""

    def __init__(self, maxsize: int = 5000):
        self._queue: "queue.Queue[Dict]" = queue.Queue(maxsize=maxsize)
        self._subscribers: List[Callable[[Dict], None]] = []
        self._lock = threading.Lock()
        self._pump: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def publish(self, topic: str, event: Dict) -> None:
        payload = {"topic": topic, **event}
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Event bus full; dropping event to protect ingestion threads.")
            return
        self._ensure_pump()

    def _ensure_pump(self) -> None:
        with self._lock:
            if self._pump is None or not self._pump.is_alive():
                self._stop.clear()
                self._pump = threading.Thread(
                    target=self._run, name="drishti-event-bus", daemon=True
                )
                self._pump.start()

    def subscribe(self, handler: Callable[[Dict], None]) -> None:
        with self._lock:
            self._subscribers.append(handler)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                subs = list(self._subscribers)
            for handler in subs:
                try:
                    handler(event)
                except Exception as e:  # isolate subscriber failures
                    logger.error(f"Event subscriber error: {e}")

    def drain(self, max_items: int = 1000) -> None:
        """Process pending events synchronously (useful for deterministic tests)."""
        count = 0
        while count < max_items:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            for handler in list(self._subscribers):
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event subscriber error (drain): {e}")
            count += 1

    def stop(self) -> None:
        self._stop.set()


class KafkaEventPublisher(BaseEventPublisher):
    """Kafka publisher used when EVENT_BACKEND=kafka and kafka-python is installed."""

    def __init__(self, bootstrap_servers: str, topic: str):
        from kafka import KafkaProducer  # type: ignore  # pragma: no cover

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=20,
        )
        self._topic = topic
        logger.info(f"Kafka publisher connected to {bootstrap_servers} topic={topic}")

    def publish(self, topic: str, event: Dict) -> None:
        self._producer.send(topic or self._topic, event)


def build_event_bus() -> InProcessEventBus:
    """Return the process-wide in-process event bus. Kafka is additive/optional."""
    return InProcessEventBus()


# Process-wide singleton bus
event_bus = build_event_bus()
