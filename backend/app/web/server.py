"""
FastAPI app exposing a localhost chat UI for the voice pipeline.

Read-only view: shows transcripts (user) and responses (agent) emitted by
the pipeline via EventHub. The only thing the UI sends back is mic toggle
state.

Latency contract: this server NEVER blocks the pipeline.
- Pipeline calls hub.emit_nowait() (microseconds, in-memory).
- Each websocket has its own send-loop task that drains its own queue.
- Slow / disconnected clients drop messages on their own queue, never
  affecting the pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..voice.event_hub import EventHub

logger = logging.getLogger(__name__)

_INDEX_PATH = Path(__file__).parent / "index.html"


def make_app(hub: EventHub, pipeline) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse(_INDEX_PATH.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        queue = hub.subscribe()

        try:
            await ws.send_json({"type": "mic_state", "enabled": pipeline.mic_enabled})
        except Exception:
            hub.unsubscribe(queue)
            return

        async def send_loop() -> None:
            try:
                while True:
                    msg = await queue.get()
                    await ws.send_json(msg)
            except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                pass

        async def recv_loop() -> None:
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "mic":
                        pipeline.set_mic_enabled(bool(data.get("enabled")))
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass

        send_task = asyncio.create_task(send_loop())
        recv_task = asyncio.create_task(recv_loop())
        try:
            done, pending = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            hub.unsubscribe(queue)

    return app
