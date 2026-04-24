"""
Audio input stream abstraction.

Sources:
  - MicrophoneStream  — real-time mic capture via sounddevice (default)
  - (future) FreeSWITCH RTP, WebSocket, etc.

All sources yield fixed-size PCM16 mono frames through the same
async iterator interface so downstream components are source-agnostic.

Includes a built-in noise suppression filter (NoiseGate) that runs
inline on every frame before it reaches VAD/STT.
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


# ---------------------------------------------------------------------------
# Noise suppression — spectral noise gate with adaptive floor
# ---------------------------------------------------------------------------

class NoiseGate:
    """
    Adaptive spectral noise gate for reducing background noise.

    Learns the noise floor from the first `calibration_frames` frames,
    then applies per-frequency-bin attenuation: energy below the noise
    floor is suppressed, energy above it passes through with smooth
    gain transitions to avoid artefacts.
    """

    def __init__(
        self,
        calibration_frames: int = 30,
        suppression_factor: float = 0.08,
        smoothing_alpha: float = 0.15,
        gate_threshold_multiplier: float = 2.5,
    ) -> None:
        self._calibration_frames = calibration_frames
        self._suppression_factor = suppression_factor
        self._smoothing_alpha = smoothing_alpha
        self._gate_threshold_mult = gate_threshold_multiplier

        self._noise_spectrum: np.ndarray | None = None
        self._calibration_spectra: list[np.ndarray] = []
        self._calibrated = False
        self._prev_gain: np.ndarray | None = None
        self._frame_count = 0

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def process(self, pcm16: np.ndarray) -> np.ndarray:
        """Process a PCM16 frame, returning noise-suppressed PCM16."""
        self._frame_count += 1
        audio_f = pcm16.astype(np.float32)
        spectrum = np.abs(np.fft.rfft(audio_f))

        if not self._calibrated:
            self._calibration_spectra.append(spectrum)
            if len(self._calibration_spectra) >= self._calibration_frames:
                self._finish_calibration()
            return pcm16

        threshold = self._noise_spectrum * self._gate_threshold_mult
        gain = np.ones_like(spectrum)
        below = spectrum < threshold
        gain[below] = self._suppression_factor

        if self._prev_gain is not None:
            gain = self._smoothing_alpha * gain + (1 - self._smoothing_alpha) * self._prev_gain
        self._prev_gain = gain.copy()

        fft_data = np.fft.rfft(audio_f)
        fft_data *= gain
        result = np.fft.irfft(fft_data, n=len(audio_f))
        return np.clip(result, -32768, 32767).astype(np.int16)

    def _finish_calibration(self) -> None:
        stacked = np.stack(self._calibration_spectra)
        self._noise_spectrum = np.percentile(stacked, 75, axis=0)
        self._calibrated = True
        self._calibration_spectra.clear()
        logger.info(
            "[NOISE_GATE] Calibrated from %d frames — noise floor mean=%.1f",
            self._calibration_frames,
            float(self._noise_spectrum.mean()),
        )


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
        noise_suppression: bool = True,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_duration_ms=frame_duration_ms)
        self._device = device
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_max_size)
        self._sd_stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._overflow_count = 0
        self._frame_count = 0
        self._debug_interval = 250  # log raw values every 250 frames (5s)
        self._noise_gate = NoiseGate() if noise_suppression else None

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
            pcm16 = indata.copy()

        if self._noise_gate is not None:
            pcm16 = self._noise_gate.process(pcm16.flatten()).reshape(pcm16.shape)

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
