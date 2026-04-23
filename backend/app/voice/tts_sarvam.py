"""
Sarvam AI Bulbul streaming TTS via HTTP stream endpoint.

Uses the /text-to-speech/stream endpoint for low-latency chunked audio
delivery. Plays audio through the system speaker using sounddevice with
true streaming playback — chunks are written to the output as they arrive,
minimizing time-to-first-audio.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue as _queue
import threading

import httpx
import numpy as np
import sounddevice as sd

from .tts_stream import StreamingTTS, TTSChunk

logger = logging.getLogger(__name__)

SARVAM_TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
DEFAULT_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "shubh"
DEFAULT_SAMPLE_RATE = 16_000
STREAM_TIMEOUT = 10.0
CONNECT_TIMEOUT = 5.0
CHUNK_READ_SIZE = 4096

_LANG_CODE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
}


class SarvamTTS(StreamingTTS):
    """
    Sarvam AI Bulbul TTS — streams audio via HTTP and plays through speakers.

    Uses true streaming playback: a background thread runs an sd.OutputStream
    and writes PCM chunks as they arrive from the HTTP stream. This eliminates
    the buffering delay of the old collect-then-play approach.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        speaker: str = DEFAULT_SPEAKER,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        pace: float = 1.0,
        output_device: int | str | None = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("SARVAMAI_API_KEY", "")
        self._model = model
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._pace = pace
        self._client: httpx.AsyncClient | None = None
        self._language = "en"
        self._output_device = output_device
        self._playback_finished = threading.Event()
        self._audio_q: _queue.Queue | None = None

        if not self._api_key:
            logger.warning("[TTS:Sarvam] SARVAMAI_API_KEY not set — calls will fail")

    def set_language(self, lang: str) -> None:
        self._language = lang

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(STREAM_TIMEOUT, connect=CONNECT_TIMEOUT),
            )
        return self._client

    async def speak(self, text: str) -> None:
        self._reset()
        self._playing = True

        if not self._api_key:
            logger.warning("[TTS:Sarvam] No API key — skipping")
            self._playing = False
            return

        self._audio_q = _queue.Queue(maxsize=200)
        self._playback_finished.clear()
        loop = asyncio.get_running_loop()
        chunk_count = 0

        def _play_streaming():
            try:
                with sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self._output_device,
                ) as stream:
                    while True:
                        try:
                            data = self._audio_q.get(timeout=0.5)
                        except _queue.Empty:
                            continue
                        if data is None:
                            break
                        try:
                            stream.write(data.reshape(-1, 1))
                        except sd.PortAudioError:
                            break
            except Exception as exc:
                logger.error("[TTS:Sarvam] Playback error: %s", exc)
            finally:
                loop.call_soon_threadsafe(self._playback_finished.set)

        thread = threading.Thread(target=_play_streaming, daemon=True)
        thread.start()

        try:
            async for chunk in self._stream_audio(text):
                if self._cancelled:
                    logger.info("[TTS:Sarvam] Cancelled mid-stream")
                    break
                samples = np.frombuffer(chunk.audio, dtype=np.int16)
                audio_float = samples.astype(np.float32) / 32768.0
                await loop.run_in_executor(None, self._audio_q.put, audio_float)
                chunk_count += 1
                if chunk_count == 1:
                    logger.info("[TTS:Sarvam] First chunk sent to playback")

        except Exception:
            logger.exception("[TTS:Sarvam] speak() failed")
        finally:
            if self._audio_q:
                self._audio_q.put(None)

            while not self._playback_finished.is_set():
                if self._cancelled:
                    sd.stop()
                    logger.info("[TTS:Sarvam] Playback stopped by cancellation")
                    break
                await asyncio.sleep(0.05)

            self._playing = False
            if not self._cancelled:
                logger.info("[TTS:Sarvam] Playback complete (%d chunks)", chunk_count)
            self._reset()

    async def _stream_audio(self, text: str):
        lang_code = _LANG_CODE_MAP.get(self._language, "en-IN")
        client = self._get_client()

        payload = {
            "text": text,
            "target_language_code": lang_code,
            "speaker": self._speaker,
            "model": self._model,
            "pace": self._pace,
            "speech_sample_rate": self._sample_rate,
            "output_audio_codec": "linear16",
        }

        try:
            async with client.stream(
                "POST",
                SARVAM_TTS_STREAM_URL,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(
                        "[TTS:Sarvam] HTTP %d — %s",
                        response.status_code, body[:500],
                    )
                    return

                logger.info("[TTS:Sarvam] Streaming started for %r (%s, %s)",
                            text[:50], lang_code, self._speaker)

                async for raw_chunk in response.aiter_bytes(CHUNK_READ_SIZE):
                    if self._cancelled:
                        break
                    yield TTSChunk(
                        audio=raw_chunk,
                        text_segment=text,
                        is_last=False,
                    )

                logger.info("[TTS:Sarvam] Stream complete")

        except httpx.TimeoutException:
            logger.warning("[TTS:Sarvam] Stream timeout for %r", text[:50])
        except Exception:
            logger.exception("[TTS:Sarvam] Stream error")

    async def stop(self) -> None:
        await super().stop()
        if self._audio_q:
            try:
                self._audio_q.put_nowait(None)
            except _queue.Full:
                pass
        try:
            sd.stop()
        except Exception:
            pass

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("[TTS:Sarvam] Client closed")
