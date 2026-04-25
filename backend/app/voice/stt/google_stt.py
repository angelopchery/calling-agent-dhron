"""
Google Cloud Speech-to-Text provider (batch + streaming).

Authentication
--------------
Uses Application Default Credentials (ADC) only — NO API keys.
Run once on the host:

    gcloud auth application-default login
    gcloud config set project <YOUR_PROJECT_ID>
    gcloud services enable speech.googleapis.com

The official ``google-cloud-speech`` SDK reads credentials from
``GOOGLE_APPLICATION_CREDENTIALS`` or ADC automatically — no key
material is read or stored by this module.

This module exposes two classes:

* :class:`GoogleSTT` — batch provider implementing the existing
  :class:`STTProvider` interface (used by :class:`STTRouter` and the
  segmented pipeline). Backed by ``client.recognize()``.

* :class:`GoogleStreamingSTT` — chunked provider implementing the
  :class:`StreamingSTT` interface. Backed by
  ``client.streaming_recognize()`` with ``interim_results=True`` for
  low-latency partials. Supports mid-call language switching by
  restarting the stream on :meth:`reset` / :meth:`set_language`.

Both classes accept the same language hints used elsewhere in the
project ("en", "hi", "gu") and map them to BCP-47 ("en-IN", "hi-IN",
"gu-IN").
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Iterable, Iterator

import numpy as np
from google.api_core import exceptions as gax_exceptions
from google.cloud import speech

from .base import STTProvider, STTResult
from ..stt_stream import STTPartial, StreamingSTT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TIMEOUT_S = 10.0
# Single attempt by default. The router (`stt/router.py`) retries on its own
# for transport exceptions, and Google's responses are deterministic — there's
# no value in two internal retries that would also fire on 4xx config errors.
MAX_RETRIES = 1
_STREAM_END_SENTINEL = b""

# Adaptive gain: compute the multiplier per utterance to bring peak amplitude
# to TARGET_BOOSTED_PEAK (~75% of int16 max), capped at MAX_AUTO_GAIN so we
# never amplify pure silence/noise infinitely. This handles both:
#   - quiet utterances (raw_peak ~800 → boost up to 10x)
#   - loud utterances (raw_peak ~22000 → minimal boost, no clipping)
# Real-world testing showed fixed gain failed at both ends of the range.
TARGET_BOOSTED_PEAK = 24000
MAX_AUTO_GAIN = 10.0
DEFAULT_STT_GAIN = MAX_AUTO_GAIN  # back-compat alias used by streaming path


def _apply_adaptive_gain(
    audio_bytes: bytes, max_gain: float = MAX_AUTO_GAIN
) -> tuple[bytes, int, int, float]:
    """Boost PCM bytes toward TARGET_BOOSTED_PEAK without clipping.

    Returns (boosted_bytes, raw_peak, boosted_peak, gain_applied).
    """
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return audio_bytes, 0, 0, 1.0
    raw_peak = int(np.abs(samples).max())
    if raw_peak == 0:
        return audio_bytes, 0, 0, 1.0
    desired = TARGET_BOOSTED_PEAK / raw_peak
    gain = max(1.0, min(desired, max_gain))
    if gain == 1.0:
        return audio_bytes, raw_peak, raw_peak, 1.0
    boosted = np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    return boosted.tobytes(), raw_peak, int(np.abs(boosted).max()), gain


def _apply_gain(audio_bytes: bytes, gain: float) -> tuple[bytes, int, int]:
    """Fixed-gain boost (used by the streaming path; kept for back-compat)."""
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    raw_peak = int(np.abs(samples).max()) if samples.size else 0
    if gain == 1.0:
        return audio_bytes, raw_peak, raw_peak
    boosted = np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    return boosted.tobytes(), raw_peak, int(np.abs(boosted).max())


# Multilingual recognition: Google accepts up to 3 alternative language codes
# per call. Primary is the conversation's locked language; alternatives let
# Google return text in another locale when the audio doesn't match the
# primary. This is what makes Gujarati transcribable while the conversation
# is locked to Hindi (or vice versa) without us re-sending the audio.
_LANGUAGE_ALTERNATIVES: dict[str, list[str]] = {
    "en-IN": ["hi-IN", "gu-IN"],
    "hi-IN": ["gu-IN", "en-IN"],
    "gu-IN": ["hi-IN", "en-IN"],
}


def _alternative_languages(primary_bcp47: str) -> list[str]:
    return _LANGUAGE_ALTERNATIVES.get(primary_bcp47, [])


# Per-language model selection. Google's `latest_long` enhanced model does
# not support every locale — notably gu-IN (Gujarati) returns
#   400 Invalid recognition 'config': The requested model is currently not
#   supported for language : gu-IN.
# For unsupported languages we fall back to the `default` (legacy) model
# without `use_enhanced`. Quality is lower but it works. When Google's model
# coverage expands, just remove the language from this map.
_LANGUAGE_MODEL_OVERRIDES: dict[str, tuple[str, bool]] = {
    "gu-IN": ("default", False),
}


def _model_for_language(bcp47: str, default_model: str) -> tuple[str, bool]:
    """Returns (model_name, use_enhanced) appropriate for the language."""
    return _LANGUAGE_MODEL_OVERRIDES.get(bcp47, (default_model, True))

# Map short language hints to Google BCP-47 codes.
_LANGUAGE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
}


def _to_bcp47(language: str) -> str:
    """Normalize an internal language hint to a Google BCP-47 code."""
    if not language:
        return "en-IN"
    if "-" in language:
        return language
    return _LANGUAGE_MAP.get(language.lower(), f"{language}-IN")


def _validate_audio(audio_bytes: bytes, sample_rate: int) -> None:
    """Cheap pre-flight check; raises ValueError on bad input."""
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"GoogleSTT expects {DEFAULT_SAMPLE_RATE} Hz LINEAR16, got {sample_rate} Hz"
        )
    if not audio_bytes:
        raise ValueError("GoogleSTT received empty audio buffer")
    if len(audio_bytes) % 2 != 0:
        raise ValueError("GoogleSTT received an odd-length buffer (not LINEAR16)")


# Domain phrase hints — Google biases recognition toward these terms,
# which is critical when audio is noisy (open-air mic, no headset). Mirrors
# the canonical vocabulary in stt/post_processor.py so the STT layer and
# the post-processor agree on what the agent expects to hear.
_DOMAIN_PHRASES: list[str] = [
    # Brand
    "Pramukh", "Pramukh Group",
    # Cities (canonical from post_processor._CANONICAL_CITIES)
    "Surat", "Vapi", "Silvassa", "Dadra",
    # Property type / sizing
    "BHK", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK",
    "bedroom", "apartment", "flat", "property",
    # Action vocab
    "site visit", "booking", "book",
    # Common Hindi (Devanagari)
    "हाँ", "हां", "नहीं", "ठीक है", "कौन", "कौन हैं", "बात",
    "मुझे", "चाहिए", "प्रॉपर्टी", "सूरत",
    # Common Gujarati (in Gujarati script — gives Google a strong gu-IN
    # signal so alternative_language_codes actually fires when the caller
    # code-switches to Gujarati without explicitly asking for it)
    "હા", "ના", "સારું", "મને", "જોઈએ", "છે", "મા", "બાત",
    "પ્રોપર્ટી", "સૂરત", "વાપી", "સિલ્વાસા",
]

_DOMAIN_BOOST = 15.0  # 0–20 range; 15 is moderate-strong bias.


def _domain_speech_contexts() -> list[speech.SpeechContext]:
    return [speech.SpeechContext(phrases=_DOMAIN_PHRASES, boost=_DOMAIN_BOOST)]


# ---------------------------------------------------------------------------
# Batch provider
# ---------------------------------------------------------------------------


class GoogleSTT(STTProvider):
    """
    Batch Google Cloud Speech-to-Text provider.

    Used by :class:`STTRouter` for the VAD-segmented pipeline path.
    Each call invokes ``recognize()`` on a complete utterance in a
    worker thread (so the async event loop is never blocked).
    """

    def __init__(
        self,
        model: str = "latest_long",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: speech.SpeechClient | None = None,
        gain: float = DEFAULT_STT_GAIN,
    ) -> None:
        self._model = model
        self._gain = float(gain)
        self._timeout_s = timeout_s
        # ADC: SpeechClient() picks up application-default credentials.
        self._client = client or speech.SpeechClient()

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: str = "en",
    ) -> STTResult:
        _validate_audio(audio_bytes, sample_rate)

        bcp47 = _to_bcp47(language)
        alt_langs = _alternative_languages(bcp47)
        model, use_enhanced = _model_for_language(bcp47, self._model)
        duration_ms = len(audio_bytes) / (sample_rate * 2) * 1000
        audio_bytes, raw_peak, boosted_peak, applied_gain = _apply_adaptive_gain(
            audio_bytes, max_gain=self._gain
        )
        logger.info(
            "[STT:Google] Sending %d bytes (%.0fms) lang=%s+%s model=%s "
            "gain=%.1fx raw_peak=%d boosted_peak=%d",
            len(audio_bytes), duration_ms, bcp47,
            ",".join(alt_langs) if alt_langs else "none",
            model, applied_gain, raw_peak, boosted_peak,
        )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            audio_channel_count=1,
            language_code=bcp47,
            alternative_language_codes=alt_langs,
            model=model,
            use_enhanced=use_enhanced,
            enable_automatic_punctuation=True,
            max_alternatives=1,
            speech_contexts=_domain_speech_contexts(),
        )
        audio = speech.RecognitionAudio(content=audio_bytes)

        response = None
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self._client.recognize,
                    config=config,
                    audio=audio,
                    timeout=self._timeout_s,
                )
                last_exc = None
                break
            except gax_exceptions.GoogleAPICallError as exc:
                last_exc = exc
                logger.warning(
                    "[STT:Google] recognize() attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, exc,
                )
                # 4xx errors are deterministic config issues — retry won't
                # help. Bail immediately to avoid burning seconds on each
                # bad turn.
                if isinstance(exc, gax_exceptions.ClientError):
                    break

        if response is None:
            assert last_exc is not None
            raise last_exc

        if not response.results:
            logger.info("[STT:Google] No results")
            return STTResult(text="", confidence=0.0)

        # Take the highest-confidence alternative across results.
        text_parts: list[str] = []
        confidences: list[float] = []
        detected_langs: set[str] = set()
        for result in response.results:
            if not result.alternatives:
                continue
            best = result.alternatives[0]
            text_parts.append(best.transcript.strip())
            # Google returns 0.0 confidence when not measured; keep 1.0 as
            # the optimistic fallback so the router doesn't filter blindly.
            confidences.append(best.confidence if best.confidence else 1.0)
            # Surface which language Google actually picked (primary or alt).
            lang_picked = getattr(result, "language_code", "") or ""
            if lang_picked:
                detected_langs.add(lang_picked)

        transcript = " ".join(p for p in text_parts if p).strip()
        confidence = min(confidences) if confidences else 0.0
        lang_str = ",".join(sorted(detected_langs)) if detected_langs else bcp47
        logger.info(
            "[STT:Google] Transcript: %r (confidence=%.2f detected=%s)",
            transcript, confidence, lang_str,
        )
        return STTResult(text=transcript, confidence=confidence)


# ---------------------------------------------------------------------------
# Streaming provider
# ---------------------------------------------------------------------------


class GoogleStreamingSTT(StreamingSTT):
    """
    Streaming Google Cloud Speech-to-Text provider.

    Implements the project's :class:`StreamingSTT` interface used by the
    real-time pipeline. Audio chunks fed via :meth:`process_audio` are
    forwarded to a background thread that drives the synchronous
    ``streaming_recognize`` gRPC call. Partial transcripts (interim and
    final) are surfaced back to the async pipeline through an
    :class:`asyncio.Queue`.

    Mid-call language switching:
        Call :meth:`set_language` (or :meth:`reset`) on a turn boundary
        to tear down the current stream and start a fresh one with the
        new ``language_code``. Google's streaming endpoint cannot change
        language inside an open stream — restarting is the supported
        path and it is sub-second on the same client.

    Barge-in:
        Interim partials are emitted as soon as the recognizer returns
        them (typically within ~100–200ms of the first voiced frame),
        so upstream callers can cancel TTS the moment a partial appears.
    """

    # ~5 minutes is the hard cap on a single Google streaming call; rotate
    # well before then so an active utterance is never severed mid-word.
    _STREAM_RESTART_AFTER_S = 240.0

    def __init__(
        self,
        language: str = "en",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        model: str = "latest_long",
        single_utterance: bool = False,
        client: speech.SpeechClient | None = None,
        gain: float = DEFAULT_STT_GAIN,
    ) -> None:
        self._language = language
        self._sample_rate = sample_rate
        self._model = model
        self._single_utterance = single_utterance
        self._client = client or speech.SpeechClient()
        self._gain = float(gain)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_q: queue.Queue[bytes] | None = None
        self._partials_q: asyncio.Queue[STTPartial] | None = None
        self._worker: threading.Thread | None = None
        self._started = False
        self._stream_started_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        self._partials_q = asyncio.Queue()
        self._spawn_stream()
        self._started = True
        logger.info("[STT:Google] Streaming session started lang=%s", _to_bcp47(self._language))

    async def stop(self) -> None:
        if not self._started:
            return
        self._terminate_stream()
        self._started = False
        logger.info("[STT:Google] Streaming session stopped")

    async def process_audio(self, chunk: bytes) -> STTPartial | None:
        """
        Feed one audio chunk into the live recognizer.

        Returns the next available partial (interim or final) without
        waiting — partials produced after this call are surfaced on
        subsequent invocations or via :meth:`partials`.
        """
        if not self._started:
            await self.start()

        # Lazy-rotate the stream before Google's hard limit.
        if (
            self._stream_started_at
            and time.monotonic() - self._stream_started_at > self._STREAM_RESTART_AFTER_S
        ):
            logger.info("[STT:Google] Rotating streaming session (idle restart)")
            self._terminate_stream()
            self._spawn_stream()

        if chunk and self._audio_q is not None:
            boosted_chunk, _, _ = _apply_gain(chunk, self._gain)
            try:
                self._audio_q.put_nowait(boosted_chunk)
            except queue.Full:
                logger.warning("[STT:Google] Audio queue full — dropping chunk")

        # Return at most one partial per call without blocking the loop.
        if self._partials_q and not self._partials_q.empty():
            return self._partials_q.get_nowait()
        return None

    async def reset(self) -> None:
        """End the current utterance — restart the recognizer cleanly."""
        if self._started:
            self._terminate_stream()
            self._spawn_stream()
            logger.info("[STT:Google] Stream reset for new utterance")

    def set_language(self, language: str) -> None:
        """
        Switch language for subsequent audio.

        The change takes effect on the next :meth:`reset` or stream
        rotation. Call this from the conversation layer when language
        detection flips, then ``await stt.reset()`` on the next turn
        boundary so a fresh stream picks up the new ``language_code``.
        """
        if language == self._language:
            return
        logger.info(
            "[STT:Google] Language switch: %s → %s (effective on next reset)",
            self._language, language,
        )
        self._language = language

    async def partials(self):
        """Async iterator over emitted partials — useful for tests."""
        if not self._partials_q:
            return
        while True:
            yield await self._partials_q.get()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_stream(self) -> None:
        self._audio_q = queue.Queue(maxsize=1024)
        self._stream_started_at = time.monotonic()

        primary_lang = _to_bcp47(self._language)
        model, use_enhanced = _model_for_language(primary_lang, self._model)
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._sample_rate,
                audio_channel_count=1,
                language_code=primary_lang,
                alternative_language_codes=_alternative_languages(primary_lang),
                model=model,
                use_enhanced=use_enhanced,
                enable_automatic_punctuation=True,
                max_alternatives=1,
                speech_contexts=_domain_speech_contexts(),
            ),
            interim_results=True,
            single_utterance=self._single_utterance,
        )

        self._worker = threading.Thread(
            target=self._run_stream,
            args=(streaming_config, self._audio_q),
            name="GoogleStreamingSTT",
            daemon=True,
        )
        self._worker.start()

    def _terminate_stream(self) -> None:
        if self._audio_q is not None:
            try:
                self._audio_q.put_nowait(_STREAM_END_SENTINEL)
            except queue.Full:
                pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._worker = None
        self._audio_q = None
        self._stream_started_at = 0.0

    def _run_stream(
        self,
        streaming_config: speech.StreamingRecognitionConfig,
        audio_q: queue.Queue[bytes],
    ) -> None:
        """Blocking gRPC loop — runs in its own thread."""

        def request_iter() -> Iterator[speech.StreamingRecognizeRequest]:
            while True:
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    # Heartbeat; keep waiting unless the loop has stopped.
                    if not self._started:
                        return
                    continue
                if chunk == _STREAM_END_SENTINEL:
                    return
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        try:
            responses = self._client.streaming_recognize(
                config=streaming_config,
                requests=request_iter(),
            )
            for response in responses:
                self._dispatch_responses(response)
        except gax_exceptions.OutOfRange:
            # Hit Google's max stream duration — caller will rotate.
            logger.info("[STT:Google] Stream hit duration limit — restart on next chunk")
        except gax_exceptions.GoogleAPICallError as exc:
            logger.warning("[STT:Google] streaming_recognize error: %s", exc)
        except Exception:
            logger.exception("[STT:Google] streaming_recognize crashed")

    def _dispatch_responses(self, response: speech.StreamingRecognizeResponse) -> None:
        if not response.results:
            return
        for result in response.results:
            if not result.alternatives:
                continue
            best = result.alternatives[0]
            text = best.transcript.strip()
            if not text:
                continue
            partial = STTPartial(
                text=text,
                is_final=bool(result.is_final),
                confidence=best.confidence if best.confidence else 1.0,
            )
            self._enqueue_partial(partial)

    def _enqueue_partial(self, partial: STTPartial) -> None:
        if self._loop is None or self._partials_q is None:
            return
        self._loop.call_soon_threadsafe(self._partials_q.put_nowait, partial)


# ---------------------------------------------------------------------------
# Helper for ad-hoc batch use
# ---------------------------------------------------------------------------


async def transcribe_segments(
    segments: Iterable[bytes],
    language: str = "en",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[STTResult]:
    """Convenience helper: batch-transcribe a list of PCM16 segments."""
    stt = GoogleSTT()
    return [
        await stt.transcribe(seg, sample_rate=sample_rate, language=language)
        for seg in segments
    ]
