"""
Cartesia streaming TTS via WebSocket.

Uses a persistent WebSocket connection to stream audio chunks as they're
generated, eliminating the wait-for-full-audio latency of the REST API.
Requires CARTESIA_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid

import websockets
from websockets.asyncio.client import ClientConnection

from .tts_stream import StreamingTTS, TTSChunk

logger = logging.getLogger(__name__)

CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2024-06-10"
DEFAULT_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"
DEFAULT_MODEL_ID = "sonic-2"
CONNECT_TIMEOUT = 3.0
RECV_TIMEOUT = 5.0
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_BASE = 0.5


class CartesiaTTS(StreamingTTS):
    """
    Cartesia WebSocket TTS — streams audio chunks with minimal latency.

    Maintains a persistent WebSocket connection. On each speak() call,
    sends the text and yields PCM audio chunks as they arrive from the
    server, achieving time-to-first-byte typically under 300ms.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        output_sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("CARTESIA_API_KEY", "")
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate = output_sample_rate
        self._ws: ClientConnection | None = None
        self._connect_lock = asyncio.Lock()

        if not self._api_key:
            logger.warning("[TTS:Cartesia] CARTESIA_API_KEY not set — calls will fail")

    async def _ensure_connected(self) -> ClientConnection:
        async with self._connect_lock:
            if self._ws is not None:
                try:
                    await self._ws.ping()
                    return self._ws
                except Exception:
                    logger.info("[TTS:Cartesia] Stale connection — reconnecting")
                    self._ws = None

            for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                try:
                    url = (
                        f"{CARTESIA_WS_URL}"
                        f"?api_key={self._api_key}"
                        f"&cartesia_version={CARTESIA_VERSION}"
                    )
                    self._ws = await asyncio.wait_for(
                        websockets.connect(url),
                        timeout=CONNECT_TIMEOUT,
                    )
                    logger.info("[TTS:Cartesia] WebSocket connected")
                    return self._ws
                except Exception as exc:
                    delay = RECONNECT_DELAY_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "[TTS:Cartesia] Connect attempt %d/%d failed: %s — retry in %.1fs",
                        attempt, MAX_RECONNECT_ATTEMPTS, exc, delay,
                    )
                    if attempt < MAX_RECONNECT_ATTEMPTS:
                        await asyncio.sleep(delay)

            raise ConnectionError("Failed to connect to Cartesia WebSocket")

    async def speak(self, text: str) -> None:
        """Stream-synthesize and simulate playback with real timing."""
        self._reset()
        self._playing = True

        if not self._api_key:
            logger.warning("[TTS:Cartesia] No API key — skipping")
            self._playing = False
            return

        try:
            async for chunk in self._stream_audio(text):
                if self._cancelled:
                    logger.info("[TTS:Cartesia] Cancelled mid-stream")
                    break
                audio_duration_s = len(chunk.audio) / (self._sample_rate * 2)
                await asyncio.sleep(audio_duration_s)
        except Exception:
            logger.exception("[TTS:Cartesia] speak() failed")
        finally:
            self._playing = False
            if not self._cancelled:
                logger.info("[TTS:Cartesia] Playback complete")
            self._reset()

    async def _stream_audio(self, text: str):
        """
        Send text over WebSocket and yield TTSChunk as audio arrives.

        Each server message is a JSON object with either:
        - type="chunk": contains base64-encoded PCM audio in "data"
        - type="done": signals end of audio for this context
        """
        context_id = str(uuid.uuid4())
        ws = await self._ensure_connected()

        request = {
            "model_id": self._model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": self._voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self._sample_rate,
            },
            "context_id": context_id,
            "continue": False,
        }

        try:
            await ws.send(json.dumps(request))
            logger.info("[TTS:Cartesia] Sent request ctx=%s: %r", context_id[:8], text)

            while True:
                if self._cancelled:
                    break

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("[TTS:Cartesia] Receive timeout — ending stream")
                    break

                msg = json.loads(raw)

                if msg.get("type") == "chunk" and "data" in msg:
                    audio_bytes = base64.b64decode(msg["data"])
                    yield TTSChunk(
                        audio=audio_bytes,
                        text_segment=text,
                        is_last=False,
                    )

                if msg.get("done", False) or msg.get("type") == "done":
                    logger.info("[TTS:Cartesia] Stream done ctx=%s", context_id[:8])
                    break

                status = msg.get("status_code", 0)
                if status >= 400:
                    logger.error("[TTS:Cartesia] Error status %d: %s", status, msg)
                    break

        except websockets.exceptions.ConnectionClosed:
            logger.warning("[TTS:Cartesia] Connection closed mid-stream — will reconnect next call")
            self._ws = None

    async def stop(self) -> None:
        await super().stop()

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("[TTS:Cartesia] WebSocket closed")
