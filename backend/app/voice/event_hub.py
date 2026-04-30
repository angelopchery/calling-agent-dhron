"""
Async pub/sub broadcaster for pipeline events.

Decouples the pipeline (producer) from websocket clients (consumers):
- Producer calls emit_nowait() — non-blocking, microseconds.
- Each consumer has a bounded queue with drop-oldest on overflow.
- A slow or disconnected client never blocks the pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

CLIENT_QUEUE_SIZE = 32


class EventHub:
    def __init__(self) -> None:
        self._clients: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
        self._clients.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._clients.remove(q)
        except ValueError:
            pass

    def emit_nowait(self, event: dict[str, Any]) -> None:
        """Broadcast to all subscribers without blocking the caller.

        Slow client → its queue fills → drop oldest, push newest.
        Pipeline never waits on network I/O.
        """
        for q in self._clients:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
