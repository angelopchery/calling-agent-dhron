"""
STT Router — tries primary provider, falls back to secondary on failure.

Includes retry logic, minimum-confidence filtering, transcript validation,
and hallucination filtering.
"""

from __future__ import annotations

import re
import logging
from collections import Counter

from .base import STTProvider, STTResult
from .sarvam import SarvamSTT
from .openai_whisper import OpenAIWhisperSTT
from .post_processor import post_process_transcript

logger = logging.getLogger(__name__)

MIN_TRANSCRIPT_LENGTH = 2
MAX_RETRIES = 1
MIN_CONFIDENCE = 0.4

_FILLER_WORDS = {"um", "uh", "like", "you know"}
_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _FILLER_WORDS) + r")\b",
    re.IGNORECASE,
)

_HALLUCINATIONS = [
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "please subscribe",
    "thanks for listening",
    "youtube originals",
    "youtube",
]

_DEVANAGARI_HALLUCINATIONS = [
    "प्रस्तुत",
    "यूट्यूब",
    "ओरिजिनल्स",
    "सबस्क्राइब",
    "परमओत",
]

_REPETITION_THRESHOLD = 0.6

_REPEATED_CHAR_RE = re.compile(r"^(.{1,3}[- ]?)\1{2,}[-]?$")

_GARBAGE_SINGLE_WORDS = {
    "shoe", "care", "shoes", "show", "sure",
}


def validate_transcript(text: str) -> str:
    """
    Clean and validate an STT transcript. Returns empty string to reject.

    Steps: normalize -> strip fillers -> reject short -> reject hallucinations
    -> reject excessive repetition.
    """
    if not text:
        return ""

    # Normalize
    cleaned = text.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip filler words
    cleaned = _FILLER_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Length check
    if len(cleaned) < 2:
        logger.info("[STT-FILTER] reason=too_short text=%r", text)
        return ""

    # Hallucination filter
    for hallucination in _HALLUCINATIONS:
        if hallucination in cleaned:
            logger.info("[STT-FILTER] reason=hallucination text=%r match=%r", text, hallucination)
            return ""

    # Repeated character/syllable patterns like "s-s-s-s-s-" or "na na na"
    if _REPEATED_CHAR_RE.match(cleaned):
        logger.info("[STT-FILTER] reason=repeated_pattern text=%r", text)
        return ""

    # Devanagari hallucination filter
    for hallucination in _DEVANAGARI_HALLUCINATIONS:
        if hallucination in text:
            logger.info("[STT-FILTER] reason=devanagari_hallucination text=%r match=%r",
                        text, hallucination)
            return ""

    # Repetition check: if >60% of words are the same word
    words = cleaned.split()
    if len(words) >= 3:
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(words) > _REPETITION_THRESHOLD:
            logger.info("[STT-FILTER] reason=repetition text=%r ratio=%.2f",
                        text, most_common_count / len(words))
            return ""

    # Single-word garbage: common misrecognitions from noise
    if len(words) == 1 and cleaned in _GARBAGE_SINGLE_WORDS:
        logger.info("[STT-FILTER] reason=garbage_single_word text=%r", text)
        return ""

    return cleaned


class STTRouter:
    """
    Multi-provider STT with automatic fallback.

    Tries the primary provider (Sarvam) with up to MAX_RETRIES retries.
    If the primary fails or returns an empty/too-short transcript, falls
    back to the secondary provider (OpenAI Whisper).

    Both providers are optional — if API keys are missing, transcribe()
    returns an empty string with a warning.
    """

    def __init__(
        self,
        primary: STTProvider | None = None,
        fallback: STTProvider | None = None,
        min_length: int = MIN_TRANSCRIPT_LENGTH,
        max_retries: int = MAX_RETRIES,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self._primary = primary or SarvamSTT()
        self._fallback = fallback or OpenAIWhisperSTT()
        self._min_length = min_length
        self._max_retries = max_retries
        self._min_confidence = min_confidence

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16_000,
        language: str = "en",
    ) -> str:
        duration_ms = len(audio_bytes) / (sample_rate * 2) * 1000
        logger.info("[STT] Processing audio (%.0f ms)", duration_ms)

        # --- Try primary with retries ---
        text = await self._try_provider(
            self._primary, "primary", audio_bytes, sample_rate, language,
        )
        if text:
            return text

        # --- Fallback ---
        logger.info("[STT] Primary failed or empty → using fallback")
        text = await self._try_provider(
            self._fallback, "fallback", audio_bytes, sample_rate, language,
        )
        if text:
            return text

        logger.warning("[STT] Both providers failed — returning empty transcript")
        return ""

    def _post_process(self, text: str, language: str) -> str:
        if not text:
            return text
        return post_process_transcript(text, language)

    async def _try_provider(
        self,
        provider: STTProvider,
        label: str,
        audio_bytes: bytes,
        sample_rate: int,
        language: str,
    ) -> str:
        for attempt in range(1, self._max_retries + 2):
            try:
                result = await provider.transcribe(audio_bytes, sample_rate, language)
                text = result.text
                confidence = result.confidence

                if confidence < self._min_confidence:
                    logger.info(
                        "[STT-FILTER] reason=low_confidence score=%.2f text=%r provider=%s attempt=%d",
                        confidence, text, label, attempt,
                    )
                    continue

                if len(text) < self._min_length:
                    logger.info(
                        "[STT] %s returned short transcript %r (attempt %d) -- treating as empty",
                        label, text, attempt,
                    )
                else:
                    validated = validate_transcript(text)
                    if validated:
                        return self._post_process(validated, language)
                    logger.info("[STT] %s transcript rejected by validation (attempt %d)", label, attempt)
            except Exception as exc:
                logger.warning(
                    "[STT] %s attempt %d failed: %s", label, attempt, exc,
                )
            if attempt <= self._max_retries:
                logger.info("[STT] Retrying %s (attempt %d/%d)", label, attempt + 1, self._max_retries + 1)

        return ""
