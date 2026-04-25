"""
Streaming Speech-to-Text abstraction.

The base class defines the async interface. `PlaceholderSTT` provides
a real-audio-driven stand-in that tracks actual speech duration — no
fake phrases, no canned text. It exists solely to give the TurnManager
word-count signals proportional to real speech timing.

Replace with GoogleStreamingSTT (or any provider) by subclassing.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class STTPartial:
    """A single partial transcript update."""
    text: str
    is_final: bool = False
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.monotonic)


class StreamingSTT(ABC):
    """Async streaming STT interface."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @abstractmethod
    async def process_audio(self, chunk: bytes) -> STTPartial | None:
        """Feed an audio chunk; returns a partial transcript if available."""
        ...

    async def reset(self) -> None:
        """Reset internal state for a new utterance."""
        pass


class PlaceholderSTT(StreamingSTT):
    """
    Placeholder STT driven by real speech timing.

    The main loop only calls process_audio() on frames the VAD has
    already classified as speech. This class simply counts those calls
    to track cumulative speech duration and emits growing placeholder
    text at ~1 word per 200ms (natural speech cadence of ~5 words/sec).

    This gives the TurnManager real duration-based word counts without
    any fake transcription. When Google streaming STT is wired in, this class
    is swapped out — the StreamingSTT interface stays identical.
    """

    def __init__(self, ms_per_word: float = 200.0, chunk_ms: float = 20.0) -> None:
        self._ms_per_word = ms_per_word
        self._chunk_ms = chunk_ms
        self._speech_ms = 0.0
        self._last_emitted_words = 0

    async def process_audio(self, chunk: bytes) -> STTPartial | None:
        self._speech_ms += self._chunk_ms
        target_words = max(1, int(self._speech_ms / self._ms_per_word))

        if target_words <= self._last_emitted_words:
            return None

        self._last_emitted_words = target_words
        dur = self._speech_ms / 1000
        text = f"[speech {dur:.1f}s]"

        # Pad with extra tokens so TurnManager sees enough words.
        # TurnManager triggers at min_words_long=5, so we need 5 tokens
        # after ~1 second of speech.
        padding = ["detected", "audio", "active", "input", "stream", "live", "user"]
        extra = padding[: max(0, target_words - 1)]
        if extra:
            text = text + " " + " ".join(extra)

        partial = STTPartial(text=text, is_final=False)
        logger.info("[STT] %r (speech=%.0fms, words=%d)", text, self._speech_ms, target_words)
        return partial

    async def reset(self) -> None:
        logger.info("[STT] Reset — turn had %.0fms of speech", self._speech_ms)
        self._speech_ms = 0.0
        self._last_emitted_words = 0
