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
import json
import logging
import signal
import time
from dataclasses import dataclass

from .audio_input import AudioInputStream, MicrophoneStream, SAMPLE_RATE, SAMPLES_PER_FRAME
from .vad import VoiceActivityDetector
from .stt.router import STTRouter
from .tts_stream import StreamingTTS, MockStreamingTTS
from .tts_sarvam import SarvamTTS
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
    stt_latency_ms: float = 0.0


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

MAX_SPEECH_SECONDS = 4
MAX_BUFFER_BYTES = SAMPLE_RATE * 2 * MAX_SPEECH_SECONDS
MIN_SPEECH_MS = 400.0
BARGE_IN_COOLDOWN_MS = 400.0
BARGE_IN_MIN_FRAMES = 25
TTS_ECHO_COOLDOWN_MS = 1000.0


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
        self.tts = tts or SarvamTTS()
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
        self._tts_end_time = 0.0
        self._vad_mute_until = 0.0
        self._barge_in_occurred = False

        # VAD + buffering state
        self._audio_buffer = bytearray()
        self._speech_duration_ms = 0.0
        self._prev_speech = False
        self._speech_frame_streak = 0
        self._frame_count = 0
        self._speech_onset_frames = 0  # debounce: frames since speech started
        self._silence_frames = 0       # debounce: frames since last speech

        # Turn manager — tuned for phone call cadence
        self._turn = TurnManager(TurnManagerConfig(
            silence_threshold_ms=450.0,
            grace_window_ms=80.0,
            min_words_short=1,
            min_words_long=2,
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

        # Agent speaks first — bilingual greeting
        opening = await self.conversation.generate_opening()
        try:
            self._tts_queue.put_nowait(Response(
                text=opening.text,
                is_shortcut=True,
                timestamp=time.monotonic(),
            ))
            logger.info("[PIPELINE] Agent opening: %r", opening.text)
        except asyncio.QueueFull:
            logger.warning("[PIPELINE] Could not enqueue opening greeting")

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
        # Skip calibration accumulation while TTS is playing or in echo
        # cooldown — speaker bleed would otherwise poison the noise floor
        # and push the speech threshold above achievable levels.
        in_tts_window = self._tts_playing or time.monotonic() < self._vad_mute_until
        raw_speech = self.vad.is_speech(frame, count_for_calibration=not in_tts_window)

        # === FULL MUTE while TTS is playing — only allow barge-in ===
        if self._tts_playing:
            if raw_speech:
                self._speech_frame_streak += 1
            else:
                self._speech_frame_streak = 0

            elapsed = (time.monotonic() - self._tts_start_time) * 1000
            if (self._speech_frame_streak >= BARGE_IN_MIN_FRAMES
                    and elapsed > BARGE_IN_COOLDOWN_MS):
                logger.info("[BARGE-IN] Sustained speech (%d frames) — interrupting TTS",
                            self._speech_frame_streak)
                await self.tts.stop()
                self._tts_playing = False
                self._barge_in_occurred = True
                self._vad_mute_until = 0.0
                self._speech_frame_streak = 0
                self._flush_pipeline_queues()
                self._reset_vad_state()
            return

        # === ECHO COOLDOWN after TTS ends naturally ===
        if time.monotonic() < self._vad_mute_until:
            if self._audio_buffer:
                logger.debug("[ECHO] Clearing %d bytes of echo audio",
                             len(self._audio_buffer))
                self._audio_buffer.clear()
                self._speech_duration_ms = 0.0
                self._turn.flush()
            self._prev_speech = False
            self._speech_onset_frames = 0
            self._silence_frames = 0
            return

        # === Normal VAD processing ===
        if raw_speech:
            self._speech_onset_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            self._speech_onset_frames = 0

        if raw_speech and self._speech_onset_frames < 3 and not self._prev_speech:
            speech = False
        elif not raw_speech and self._silence_frames < 5 and self._prev_speech:
            speech = True
        else:
            speech = raw_speech

        if speech:
            if not self._prev_speech:
                logger.info("[VAD] Speech start")
            self._turn.mark_speech()
            self._audio_buffer.extend(frame)
            self._speech_duration_ms += 20.0

            if len(self._audio_buffer) > MAX_BUFFER_BYTES:
                del self._audio_buffer[:len(self._audio_buffer) - MAX_BUFFER_BYTES]

            proxy_words = max(1, int(self._speech_duration_ms / 300))
            self._turn.update_text(" ".join(["x"] * proxy_words))
        else:
            if self._prev_speech:
                logger.info("[VAD] Speech end (%.0fms)", self._speech_duration_ms)

        self._prev_speech = speech

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

            try:
                self._stt_queue.put_nowait(segment)
                logger.info("[TURN] Enqueued audio segment (%.0fms)", segment.duration_ms)
            except asyncio.QueueFull:
                logger.warning("[TURN] STT queue full — dropping segment")

    def _reset_vad_state(self) -> None:
        self._audio_buffer.clear()
        self._speech_duration_ms = 0.0
        self._prev_speech = False
        self._speech_onset_frames = 0
        self._silence_frames = 0
        self._speech_frame_streak = 0
        self._turn.flush()

    def _flush_pipeline_queues(self) -> None:
        drained = 0
        for q in (self._tts_queue, self._stt_queue, self._transcript_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
        if drained:
            logger.info("[BARGE-IN] Drained %d items from pipeline queues", drained)

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
                language = self.conversation.state.language if self.conversation else "en"
                transcript_text = await self.stt.transcribe(
                    segment.audio, sample_rate=SAMPLE_RATE, language=language
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
                    stt_latency_ms=stt_latency,
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
        Uses streaming: sends sentence-level chunks to TTS as they arrive.
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
                first_result = None
                first_chunk_ms = 0.0
                full_text_parts: list[str] = []

                async for result in self.conversation.process_stream(transcript.text):
                    if first_result is None:
                        first_result = result
                        first_chunk_ms = (time.monotonic() - latency_start) * 1000

                    full_text_parts.append(result.text)
                    try:
                        self._tts_queue.put_nowait(Response(
                            text=result.text,
                            is_shortcut=result.is_shortcut,
                            timestamp=time.monotonic(),
                        ))
                    except asyncio.QueueFull:
                        logger.warning("[ENGINE] TTS queue full -- dropping chunk")

                engine_latency = (time.monotonic() - latency_start) * 1000
                total_latency = transcript.stt_latency_ms + engine_latency

                if first_result:
                    full_text = " ".join(full_text_parts)
                    logger.info("[ENGINE] %r -> %r (%.0fms, first_chunk=%.0fms)",
                                transcript.text, full_text, engine_latency, first_chunk_ms)

                    state = self.conversation.state
                    turn_metric = {
                        "turn_id": state.turn_count,
                        "language": state.language,
                        "lang_confidence": round(state.lang_confidence, 2),
                        "lang_source": state.lang_source,
                        "intent": first_result.intent,
                        "stage": state.stage,
                        "stt_latency_ms": round(transcript.stt_latency_ms),
                        "llm_latency_ms": round(engine_latency),
                        "first_chunk_ms": round(first_chunk_ms),
                        "response_length": len(full_text),
                        "total_latency_ms": round(total_latency),
                        "filler_used": state.filler_used,
                        "llm_cancelled": state.llm_cancelled,
                        "parallel_execution": state.parallel_execution,
                        "objection_handled": state.objection_handled,
                    }
                    logger.info("[TURN_METRIC] %s", json.dumps(turn_metric))

            except Exception:
                logger.exception("[ENGINE] Processing failed")
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

        Ensures only one utterance plays at a time: waits for speak()
        to fully complete before starting the next. Stale responses
        (queued >3s ago) are dropped to prevent pile-up.
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

            # Drop stale responses that have been sitting in queue too long
            age_ms = (time.monotonic() - response.timestamp) * 1000
            if age_ms > 3000:
                logger.warning("[TTS] Dropping stale response (%.0fms old): %r",
                               age_ms, response.text[:50])
                continue

            if self._tts_playing:
                logger.info("[TTS] Previous playback still active — stopping")
                await self.tts.stop()
                self._tts_playing = False
                await asyncio.sleep(0.1)

            try:
                logger.info("[TTS] Playing: %r (queued %.0fms ago)", response.text, age_ms)

                self._tts_playing = True
                self._tts_start_time = time.monotonic()

                if hasattr(self.tts, 'set_language') and self.conversation:
                    self.tts.set_language(self.conversation.state.language)

                await self.tts.speak(response.text)

            except asyncio.CancelledError:
                logger.info("[TTS] Interrupted")
            except Exception:
                logger.exception("[TTS] Playback failed — recovering")
                # Reset TTS state so next utterance can play
                try:
                    await self.tts.stop()
                except Exception:
                    pass
            finally:
                self._tts_playing = False
                self._tts_end_time = time.monotonic()
                if self._barge_in_occurred:
                    self._barge_in_occurred = False
                else:
                    self._vad_mute_until = (
                        time.monotonic() + TTS_ECHO_COOLDOWN_MS / 1000.0
                    )

        logger.info("[TTS] Layer stopped")
