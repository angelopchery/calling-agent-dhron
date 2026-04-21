"""
Audio input stream abstraction.

Sources:
  - MicrophoneStream  — real-time mic capture via sounddevice (default)
  - (future) FreeSWITCH RTP, WebSocket, etc.

All sources yield fixed-size PCM16 mono frames through the same
async iterator interface so downstream components are source-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import AsyncIterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # 16-bit mono PCM


class AudioInputStream:
    """Base async audio source yielding fixed-size PCM16 frames."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION_MS,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = sample_rate * frame_duration_ms // 1000
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def _read_frame(self) -> bytes | None:
        raise NotImplementedError

    async def frames(self) -> AsyncIterator[bytes]:
        await self.start()
        try:
            while self._running:
                frame = await self._read_frame()
                if frame is None:
                    break
                yield frame
        finally:
            await self.stop()


class MicrophoneStream(AudioInputStream):
    """
    Real-time microphone capture using sounddevice.

    Architecture:
      sounddevice.InputStream runs a C-level callback on a PortAudio
      thread. The callback normalises the audio to int16 PCM bytes
      (handling both int16 and float32 input from different backends)
      and pushes into an asyncio.Queue via loop.call_soon_threadsafe.
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION_MS,
        queue_max_size: int = 200,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_duration_ms=frame_duration_ms)
        self._device = device
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_max_size)
        self._sd_stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._overflow_count = 0
        self._frame_count = 0
        self._debug_interval = 250  # log raw values every 250 frames (5s)

    async def start(self) -> None:
        import sounddevice as sd

        logger.info("[MIC] Available audio devices:\n%s", sd.query_devices())

        self._loop = asyncio.get_running_loop()
        await super().start()

        blocksize = self.samples_per_frame

        try:
            self._sd_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                channels=1,
                dtype="int16",
                device=self._device,
                callback=self._audio_callback,
            )
            self._sd_stream.start()
        except sd.PortAudioError as exc:
            self._running = False
            raise RuntimeError(f"[MIC] Failed to open audio device: {exc}") from exc

        logger.info(
            "[MIC] Stream opened — device=%s, rate=%d, blocksize=%d (%dms)",
            self._device if self._device is not None else "system default",
            self.sample_rate,
            blocksize,
            self.frame_duration_ms,
        )

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """
        Called from the PortAudio thread. Must be fast and non-blocking.

        Handles both int16 and float32 input — some PortAudio backends
        deliver float32 regardless of the requested dtype. We normalise
        to int16 PCM bytes before queueing.
        """
        if status:
            logger.warning("[MIC] PortAudio status: %s", status)

        if indata.dtype == np.float32:
            pcm16 = (indata * 32767).astype(np.int16)
        else:
            pcm16 = indata

        pcm_bytes = pcm16.tobytes()

        self._frame_count += 1
        if self._frame_count % self._debug_interval == 1:
            audio_flat = pcm16.flatten()
            logger.info(
                "[MIC] chunk #%d — dtype=%s, samples=%d, "
                "min=%d, max=%d, energy=%.1f",
                self._frame_count,
                indata.dtype,
                len(audio_flat),
                int(audio_flat.min()),
                int(audio_flat.max()),
                float(np.abs(audio_flat.astype(np.float32)).mean()),
            )

        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, pcm_bytes)
        except asyncio.QueueFull:
            self._overflow_count += 1
            if self._overflow_count % 50 == 1:
                logger.warning(
                    "[MIC] Queue overflow (dropped %d frames) — event loop may be blocked",
                    self._overflow_count,
                )

    async def stop(self) -> None:
        await super().stop()
        if self._sd_stream is not None:
            self._sd_stream.stop()
            self._sd_stream.close()
            self._sd_stream = None
            logger.info("[MIC] Stream closed after %d frames", self._frame_count)
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def _read_frame(self) -> bytes | None:
        if not self._running and self._queue.empty():
            return None
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            return frame
        except asyncio.TimeoutError:
            if not self._running:
                return None
            return b"\x00" * (self.samples_per_frame * 2)
