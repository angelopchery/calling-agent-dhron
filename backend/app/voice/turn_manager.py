"""
Intelligent turn-taking manager.

Decides when the user has finished speaking and the system should respond.
Uses BOTH silence duration AND semantic completeness — never triggers on
silence alone for short fragments.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TurnManagerConfig:
    silence_threshold_ms: float = 600.0
    grace_window_ms: float = 150.0
    min_words_short: int = 3
    min_words_long: int = 5
    sentence_end_chars: str = ".?!,"


class TurnManager:
    """
    Accumulates STT partials and decides when a user turn is complete.

    The trigger fires only when:
      1. Silence exceeds `silence_threshold_ms`, AND
      2. The accumulated text is semantically complete:
         - ends with sentence-ending punctuation, OR
         - word count >= min_words_short with punctuation, OR
         - word count >= min_words_long (regardless of punctuation)
      3. An additional `grace_window_ms` has elapsed after conditions
         1+2 are first met (absorbs trailing speech fragments).
    """

    def __init__(self, config: TurnManagerConfig | None = None) -> None:
        self.config = config or TurnManagerConfig()
        self._text_buffer: str = ""
        self._last_speech_time: float = 0.0
        self._last_text_update_time: float = 0.0
        self._trigger_candidate_time: float | None = None
        self._triggered = False

    @property
    def text(self) -> str:
        return self._text_buffer.strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    def update_text(self, partial: str) -> None:
        """Update buffer with latest STT partial (replaces previous partial)."""
        self._text_buffer = partial
        self._last_text_update_time = time.monotonic()
        self._trigger_candidate_time = None
        logger.debug("[TURN] Buffer updated: %r (%d words)", self.text, self.word_count)

    def mark_speech(self) -> None:
        """Call on every frame classified as speech by VAD."""
        self._last_speech_time = time.monotonic()
        self._trigger_candidate_time = None

    def silence_duration_ms(self) -> float:
        if self._last_speech_time == 0.0:
            return 0.0
        return (time.monotonic() - self._last_speech_time) * 1000

    def is_semantically_complete(self) -> bool:
        text = self.text
        if not text:
            return False

        ends_with_punctuation = text[-1] in self.config.sentence_end_chars
        wc = self.word_count

        if wc >= self.config.min_words_long:
            return True
        if wc >= self.config.min_words_short and ends_with_punctuation:
            return True
        if ends_with_punctuation and wc >= 2:
            return True

        return False

    def should_trigger(self) -> bool:
        """
        Returns True exactly once when the turn is considered complete.
        After triggering, call flush() before the next turn.
        """
        if self._triggered:
            return False

        if not self.text:
            return False

        silence_ok = self.silence_duration_ms() >= self.config.silence_threshold_ms
        semantic_ok = self.is_semantically_complete()

        if not (silence_ok and semantic_ok):
            self._trigger_candidate_time = None
            return False

        now = time.monotonic()
        if self._trigger_candidate_time is None:
            self._trigger_candidate_time = now
            logger.debug("[TURN] Trigger candidate started (grace window)")
            return False

        grace_elapsed = (now - self._trigger_candidate_time) * 1000
        if grace_elapsed < self.config.grace_window_ms:
            return False

        logger.info(
            "[TURN] Triggering — silence=%.0fms, words=%d, text=%r",
            self.silence_duration_ms(),
            self.word_count,
            self.text,
        )
        self._triggered = True
        return True

    def flush(self) -> str:
        """Return accumulated text and reset for next turn."""
        text = self.text
        self._text_buffer = ""
        self._trigger_candidate_time = None
        self._triggered = False
        self._last_speech_time = 0.0
        self._last_text_update_time = 0.0
        logger.info("[TURN] Flushed: %r", text)
        return text
