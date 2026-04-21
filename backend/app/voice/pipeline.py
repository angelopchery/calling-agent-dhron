"""
Pipeline orchestrator — connects all layers via async queues.

Architecture:
    [AudioLayer] → vad_queue → [VADLayer] → stt_queue → [STTWorker]
        → transcript_queue → [ConversationEngine] → tts_queue → [TTSPlayer]

Each layer is an independent async task. Queues are bounded.
Failure in one layer does not crash others.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass

from .audio_input import AudioInputStream, MicrophoneStream, SAMPLE_RATE, SAMPLES_PER_FRAME
from .vad import VoiceActivityDetector
from .stt.router import STTRouter
from .tts_stream import StreamingTTS, MockStreamingTTS
from .turn_manager import TurnManager, TurnManagerConfig
from .conversation import ConversationRouter
from .memory import ConversationMemory

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """A complete speech segment ready for STT."""
    audio: bytes
    duration_ms: float
    timestamp: float


@dataclass
class Transcript:
    """STT result ready for conversation engine."""
    text: str
    duration_ms: float
    timestamp: float


@dataclass
class Response:
    """Engine response ready for TTS."""
    text: str
    is_shortcut: bool
    timestamp: float


QUEUE_SIZE_AUDIO = 300
QUEUE_SIZE_STT = 5
QUEUE_SIZE_TRANSCRIPT = 5
QUEUE_SIZE_TTS = 5

MAX_SPEECH_SECONDS = 7
MAX_BUFFER_BYTES = SAMPLE_RATE * 2 * MAX_SPEECH_SECONDS
MIN_SPEECH_MS = 300.0
BARGE_IN_COOLDOWN_MS = 300.0
BARGE_IN_MIN_FRAMES = 12  # 240ms of sustained speech to interrupt


class VoicePipeline:
    """
    Production voice pipeline with independent async layers.

    Each layer runs as a separate task communicating via bounded queues.
    The audio loop never blocks on downstream processing.
    """

    def __init__(
        self,
        audio_source: AudioInputStream | None = None,
        vad: VoiceActivityDetector | None = None,
        stt: STTRouter | None = None,
        tts: StreamingTTS | None = None,
        conversation: ConversationRouter | None = None,
    ) -> None:
        self.audio = audio_source or MicrophoneStream()
        self.vad = vad or VoiceActivityDetector()
        self.stt = stt or STTRouter()
        self.tts = tts or MockStreamingTTS()
        self.conversation = conversation or ConversationRouter()

        # Inter-layer queues
        self._stt_queue: asyncio.Queue[AudioSegment] = asyncio.Queue(maxsize=QUEUE_SIZE_STT)
        self._transcript_queue: asyncio.Queue[Transcript] = asyncio.Queue(maxsize=QUEUE_SIZE_TRANSCRIPT)
        self._tts_queue: asyncio.Queue[Response] = asyncio.Queue(maxsize=QUEUE_SIZE_TTS)

        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._tts_playing = False
        self._tts_start_time = 0.0

        # VAD + buffering state
        self._audio_buffer = bytearray()
        self._speech_duration_ms = 0.0
        self._prev_speech = False
        self._speech_frame_streak = 0
        self._frame_count = 0

        # Turn manager
        self._turn = TurnManager(TurnManagerConfig(
            silence_threshold_ms=600.0,
            grace_window_ms=150.0,
            min_words_short=2,
            min_words_long=4,
        ))

    async def start(self) -> None:
        self._running = True
        logger.info("=" * 60)
        logger.info("[PIPELINE] Starting voice pipeline")
        logger.info("=" * 60)

        self._tasks = [
            asyncio.create_task(self._audio_vad_layer(), name="audio_vad"),
            asyncio.create_task(self._stt_worker(), name="stt_worker"),
            asyncio.create_task(self._conversation_layer(), name="conversation"),
            asyncio.create_task(self._tts_layer(), name="tts"),
        ]

    async def stop(self) -> None:
        self._running = False
        await self.audio.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("[PIPELINE] All layers stopped")

    async def run_until_shutdown(self) -> None:
        """Run pipeline until Ctrl+C or signal."""
        await self.start()

        shutdown = asyncio.Event()

        def _signal():
            logger.info("[PIPELINE] Shutdown signal received")
            shutdown.set()

        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, _signal)
            loop.add_signal_handler(signal.SIGTERM, _signal)
        except NotImplementedError:
            pass

        try:
            await shutdown.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self.stop()

    # -----------------------------------------------------------------------
    # Layer 1: Audio + VAD + Turn Detection
    # -----------------------------------------------------------------------

    async def _audio_vad_layer(self) -> None:
        """
        Captures audio, runs VAD, buffers speech, detects turns.
        When a turn completes, enqueues an AudioSegment for STT.
        NEVER blocks on downstream — uses put_nowait with overflow handling.
        """
        try:
            async for frame in self.audio.frames():
                if not self._running:
                    break
                self._frame_count += 1
                await self._process_vad_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[AUDIO] Layer crashed")
        finally:
            logger.info("[AUDIO] Layer stopped (%d frames)", self._frame_count)

    async def _process_vad_frame(self, frame: bytes) -> None:
        speech = self.vad.is_speech(frame)

        # --- Barge-in with sustained speech requirement ---
        if speech and self._tts_playing:
            self._speech_frame_streak += 1
            if self._speech_frame_streak >= BARGE_IN_MIN_FRAMES:
                elapsed = (time.monotonic() - self._tts_start_time) * 1000
                if elapsed > BARGE_IN_COOLDOWN_MS:
                    logger.info("[BARGE-IN] Sustained speech (%d frames) — interrupting TTS",
                                self._speech_frame_streak)
                    await self.tts.stop()
                    self._tts_playing = False
                    self._speech_frame_streak = 0
        elif not speech:
            self._speech_frame_streak = 0

        # --- Buffer speech audio ---
        if speech:
            if not self._prev_speech:
                logger.info("[VAD] Speech start")
            self._turn.mark_speech()
            self._audio_buffer.extend(frame)
            self._speech_duration_ms += 20.0

            # Cap buffer
            if len(self._audio_buffer) > MAX_BUFFER_BYTES:
                del self._audio_buffer[:len(self._audio_buffer) - MAX_BUFFER_BYTES]

            # Feed turn manager
            proxy_words = max(1, int(self._speech_duration_ms / 200))
            self._turn.update_text(" ".join(["w"] * proxy_words))
        else:
            if self._prev_speech:
                logger.info("[VAD] Speech end (%.0fms)", self._speech_duration_ms)

        self._prev_speech = speech

        # --- Check turn boundary ---
        if self._turn.should_trigger():
            self._turn.flush()

            if self._speech_duration_ms < MIN_SPEECH_MS:
                logger.debug("[TURN] Too short (%.0fms) — discarding", self._speech_duration_ms)
                self._audio_buffer.clear()
                self._speech_duration_ms = 0.0
                return

            segment = AudioSegment(
                audio=bytes(self._audio_buffer),
                duration_ms=self._speech_duration_ms,
                timestamp=time.monotonic(),
            )
            self._audio_buffer.clear()
            self._speech_duration_ms = 0.0

            # Non-blocking enqueue — drop if STT is backed up
            try:
                self._stt_queue.put_nowait(segment)
                logger.info("[TURN] Enqueued audio segment (%.0fms)", segment.duration_ms)
            except asyncio.QueueFull:
                logger.warning("[TURN] STT queue full — dropping segment")

    # -----------------------------------------------------------------------
    # Layer 2: STT Worker
    # -----------------------------------------------------------------------

    async def _stt_worker(self) -> None:
        """
        Consumes AudioSegments, transcribes via STT router.
        Runs independently — never blocks the audio layer.
        """
        while self._running:
            try:
                segment = await asyncio.wait_for(
                    self._stt_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                latency_start = time.monotonic()
                transcript_text = await self.stt.transcribe(
                    segment.audio, sample_rate=SAMPLE_RATE
                )
                stt_latency = (time.monotonic() - latency_start) * 1000

                if not transcript_text or len(transcript_text.strip()) < 2:
                    logger.info("[STT] Empty transcript — skipping")
                    continue

                logger.info("[STT] %r (%.0fms latency)", transcript_text, stt_latency)

                transcript = Transcript(
                    text=transcript_text.strip(),
                    duration_ms=segment.duration_ms,
                    timestamp=time.monotonic(),
                )

                try:
                    self._transcript_queue.put_nowait(transcript)
                except asyncio.QueueFull:
                    logger.warning("[STT] Transcript queue full — dropping")

            except Exception:
                logger.exception("[STT] Transcription failed")

        logger.info("[STT] Worker stopped")

    # -----------------------------------------------------------------------
    # Layer 3: Conversation Engine
    # -----------------------------------------------------------------------

    async def _conversation_layer(self) -> None:
        """
        Consumes transcripts, routes through conversation engine.
        Handles shortcuts, intent detection, and LLM calls.
        """
        while self._running:
            try:
                transcript = await asyncio.wait_for(
                    self._transcript_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                latency_start = time.monotonic()
                result = await self.conversation.process(transcript.text)
                engine_latency = (time.monotonic() - latency_start) * 1000

                logger.info("[ENGINE] %r → %r (%.0fms)",
                            transcript.text, result.text, engine_latency)

                response = Response(
                    text=result.text,
                    is_shortcut=result.is_shortcut,
                    timestamp=time.monotonic(),
                )

                try:
                    self._tts_queue.put_nowait(response)
                except asyncio.QueueFull:
                    logger.warning("[ENGINE] TTS queue full — dropping response")

            except Exception:
                logger.exception("[ENGINE] Processing failed")
                # Fallback response
                try:
                    self._tts_queue.put_nowait(Response(
                        text="Sorry, could you say that again?",
                        is_shortcut=True,
                        timestamp=time.monotonic(),
                    ))
                except asyncio.QueueFull:
                    pass

        logger.info("[ENGINE] Layer stopped")

    # -----------------------------------------------------------------------
    # Layer 4: TTS Playback
    # -----------------------------------------------------------------------

    async def _tts_layer(self) -> None:
        """
        Consumes responses, plays via TTS.
        Tracks playback state for barge-in detection.
        """
        while self._running:
            try:
                response = await asyncio.wait_for(
                    self._tts_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                total_latency = (time.monotonic() - response.timestamp) * 1000
                logger.info("[TTS] Playing: %r (queued %.0fms ago)", response.text, total_latency)

                self._tts_playing = True
                self._tts_start_time = time.monotonic()

                await self.tts.speak(response.text)

            except asyncio.CancelledError:
                logger.info("[TTS] Interrupted")
            except Exception:
                logger.exception("[TTS] Playback failed")
            finally:
                self._tts_playing = False

        logger.info("[TTS] Layer stopped")
