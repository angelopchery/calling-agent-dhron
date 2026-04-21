"""
Main voice loop — orchestrates the full real-time pipeline.

    microphone → VAD → audio buffer → TurnManager
                                        ↓ (on silence trigger)
                                    STTRouter.transcribe(buffer)  [background task]
                                        ↓
                                    Shortcuts / LLM → TTS (streaming)

Audio frames are buffered during speech. When the TurnManager detects
a turn boundary, a background task handles STT→LLM→TTS so the audio
loop NEVER blocks.

Handles barge-in with debounce: if the user speaks while TTS is playing
(after a brief cooldown), playback is interrupted and the system listens.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time

from .audio_input import AudioInputStream, MicrophoneStream, SAMPLE_RATE
from .vad import VoiceActivityDetector
from .stt.router import STTRouter
from .tts_stream import StreamingTTS, MockStreamingTTS
from .turn_manager import TurnManager, TurnManagerConfig
from .conversation_engine import ConversationEngine, MockConversationEngine
from .shortcuts import check_shortcut

logger = logging.getLogger(__name__)

MAX_SPEECH_SECONDS = 7
MAX_BUFFER_BYTES = SAMPLE_RATE * 2 * MAX_SPEECH_SECONDS
MIN_SPEECH_MS_FOR_TURN = 300.0
BARGE_IN_COOLDOWN_MS = 250.0


class VoiceLoop:
    """
    Real-time voice conversation loop.

    All components are injected, making it easy to swap
    implementations without touching the loop logic.

    Key design: STT+LLM+TTS runs as a background task so the
    audio capture loop never blocks.
    """

    def __init__(
        self,
        audio_source: AudioInputStream | None = None,
        vad: VoiceActivityDetector | None = None,
        stt: STTRouter | None = None,
        tts: StreamingTTS | None = None,
        turn_manager: TurnManager | None = None,
        conversation: ConversationEngine | None = None,
    ) -> None:
        self.audio = audio_source or MicrophoneStream()
        self.vad = vad or VoiceActivityDetector()
        self.stt = stt or STTRouter()
        self.tts = tts or MockStreamingTTS()
        self.turn = turn_manager or TurnManager()
        self.conversation = conversation or MockConversationEngine(
            system_prompt="You are a helpful voice assistant."
        )

        self._tts_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._running = False
        self._prev_speech = False
        self._frame_count = 0
        self._processing_turn = False

        self._audio_buffer = bytearray()
        self._speech_duration_ms = 0.0
        self._chunk_ms = 20.0

        self._tts_start_time = 0.0

    async def run(self) -> None:
        self._running = True
        logger.info("=" * 60)
        logger.info("[LOOP] Voice loop starting — listening on microphone")
        logger.info("=" * 60)

        try:
            async for frame in self.audio.frames():
                if not self._running:
                    break
                await self._process_frame(frame)

            await self._check_turn_trigger()
        finally:
            await self._cancel_tts()
            self._running = False
            logger.info("[LOOP] Voice loop stopped (%d frames processed)", self._frame_count)

    async def stop(self) -> None:
        self._running = False
        await self.audio.stop()

    async def _process_frame(self, frame: bytes) -> None:
        self._frame_count += 1
        speech = self.vad.is_speech(frame)

        # --- Barge-in detection with cooldown ---
        if speech and self.tts.is_playing:
            elapsed_since_tts = (time.monotonic() - self._tts_start_time) * 1000
            if elapsed_since_tts > BARGE_IN_COOLDOWN_MS:
                logger.info("[BARGE-IN] User speaking during TTS — interrupting")
                await self._cancel_tts()

        # --- VAD state transitions ---
        if speech:
            if not self._prev_speech:
                logger.info("[VAD] Speech detected")
            self.turn.mark_speech()

            self._audio_buffer.extend(frame)
            self._speech_duration_ms += self._chunk_ms

            if len(self._audio_buffer) > MAX_BUFFER_BYTES:
                overflow = len(self._audio_buffer) - MAX_BUFFER_BYTES
                del self._audio_buffer[:overflow]

            proxy_words = max(1, int(self._speech_duration_ms / 200))
            proxy_text = " ".join(["speech"] * proxy_words)
            self.turn.update_text(proxy_text)

        else:
            
            if self._prev_speech:
                logger.info("[VAD] Silence detected")

        self._prev_speech = speech

        # --- Turn detection (non-blocking) ---
        await self._check_turn_trigger()

    async def _check_turn_trigger(self) -> None:
        if self._processing_turn:
            return

        if not self.turn.should_trigger():
            return

        if self._speech_duration_ms < MIN_SPEECH_MS_FOR_TURN:
            logger.info("[LOOP] Speech too short (%.0fms) — skipping turn", self._speech_duration_ms)
            self.turn.flush()
            self._audio_buffer.clear()
            self._speech_duration_ms = 0.0
            return

        self.turn.flush()

        audio_bytes = bytes(self._audio_buffer)
        buffer_duration_ms = self._speech_duration_ms
        self._audio_buffer.clear()
        self._speech_duration_ms = 0.0

        if not audio_bytes:
            return

        # Spawn background task — audio loop keeps running
        self._processing_turn = True
        self._turn_task = asyncio.create_task(
            self._process_turn(audio_bytes, buffer_duration_ms)
        )

    async def _process_turn(self, audio_bytes: bytes, duration_ms: float) -> None:
        """Background task: STT → shortcuts/LLM → TTS."""
        try:
            logger.info("[STT] Processing audio (%.0f ms)", duration_ms)

            transcript = await self.stt.transcribe(audio_bytes, sample_rate=SAMPLE_RATE)

            if not transcript or len(transcript.strip()) < 2:
                logger.info("[STT] Empty or too-short transcript — skipping")
                return

            logger.info("[USER] %r", transcript)

            # Try shortcut first (no LLM call)
            shortcut_response = check_shortcut(transcript)
            if shortcut_response:
                response = shortcut_response
                logger.info("[SHORTCUT] %r → %r", transcript, response)
            else:
                response = await self.conversation.respond(transcript)
                logger.info("[LLM] Response: %r", response)

            # Start TTS
            await self._cancel_tts()
            self._tts_start_time = time.monotonic()
            self._tts_task = asyncio.create_task(self._speak(response))

        except Exception:
            logger.exception("[LOOP] Turn processing error")
        finally:
            self._processing_turn = False

    async def _speak(self, text: str) -> None:
        try:
            await self.tts.speak(text)
        except asyncio.CancelledError:
            logger.info("[TTS] Interrupted")
        except Exception:
            logger.exception("[LOOP] TTS error")

    async def _cancel_tts(self) -> None:
        if self._tts_task and not self._tts_task.done():
            await self.tts.stop()
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
        self._tts_task = None


# ---------------------------------------------------------------------------
# Sanity test mode
# ---------------------------------------------------------------------------

async def _run_test() -> None:
    """Mic + VAD sanity test — prints energy and speech/silence status."""
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    mic = MicrophoneStream()
    vad = VoiceActivityDetector()

    logger.info("[TEST] Starting mic+VAD sanity test (Ctrl+C to stop)")
    logger.info("[TEST] Stay quiet for first 0.5s (VAD calibration)")

    frame_count = 0
    speech_frames = 0

    async for frame in mic.frames():
        frame_count += 1
        energy = float(np.abs(np.frombuffer(frame, dtype=np.int16).astype(np.float32)).mean())
        is_speech = vad.is_speech(frame)

        if is_speech:
            speech_frames += 1

        if frame_count % 5 == 0:
            bar_len = min(50, int(energy / 10))
            bar = "\u2588" * bar_len + "\u2591" * (50 - bar_len)
            status = "SPEECH" if is_speech else "silent"
            thresh_str = f"thr={vad.threshold:.1f}" if vad.calibrated else "calibrating"
            print(
                f"\r  energy={energy:6.1f} {bar} {status:7s} ({thresh_str})",
                end="", flush=True,
            )

        if frame_count > 0 and frame_count % 500 == 0:
            print()
            logger.info(
                "[TEST] %d frames, %d speech (%.0f%%)",
                frame_count, speech_frames,
                100 * speech_frames / frame_count if frame_count else 0,
            )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here to allow conditional LLM engine based on env
    import os
    from .conversation_engine import MockConversationEngine

    # Use real LLM if API key available, else mock
    engine: ConversationEngine
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        from .conversation_engine_llm import LLMConversationEngine
        engine = LLMConversationEngine()
        logger.info("[LOOP] Using LLM conversation engine (OpenAI)")
    else:
        engine = MockConversationEngine(system_prompt="You are a helpful voice assistant.")
        logger.info("[LOOP] Using mock conversation engine (no OPENAI_API_KEY)")

    loop = VoiceLoop(conversation=engine)

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("[LOOP] Shutdown signal received")
        shutdown_event.set()

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _signal_handler)
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        pass

    logger.info("[LOOP] Speak into your microphone (Ctrl+C to stop)")
    logger.info("[LOOP] First ~0.5s is VAD calibration — stay quiet")

    try:
        run_task = asyncio.create_task(loop.run())
        stop_task = asyncio.create_task(shutdown_event.wait())
        done, _ = await asyncio.wait(
            [run_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        await loop.stop()
        if run_task not in done:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    except KeyboardInterrupt:
        logger.info("[LOOP] Interrupted by user")
        await loop.stop()


def main() -> None:
    try:
        if "--test" in sys.argv:
            asyncio.run(_run_test())
        else:
            asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
