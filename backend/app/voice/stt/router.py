"""
STT Router — tries primary provider, falls back to secondary on failure.

Includes retry logic and minimum-confidence filtering (rejects transcripts
that are too short to be meaningful).
"""

from __future__ import annotations

import logging

from .base import STTProvider
from .sarvam import SarvamSTT
from .openai_whisper import OpenAIWhisperSTT

logger = logging.getLogger(__name__)

MIN_TRANSCRIPT_LENGTH = 2
MAX_RETRIES = 1


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
    ) -> None:
        self._primary = primary or SarvamSTT()
        self._fallback = fallback or OpenAIWhisperSTT()
        self._min_length = min_length
        self._max_retries = max_retries

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
                text = await provider.transcribe(audio_bytes, sample_rate, language)
                if len(text) >= self._min_length:
                    return text
                logger.info(
                    "[STT] %s returned short transcript %r (attempt %d) — treating as empty",
                    label, text, attempt,
                )
            except Exception as exc:
                logger.warning(
                    "[STT] %s attempt %d failed: %s", label, attempt, exc,
                )
            if attempt <= self._max_retries:
                logger.info("[STT] Retrying %s (attempt %d/%d)", label, attempt + 1, self._max_retries + 1)

        return ""
