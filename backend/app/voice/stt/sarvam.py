"""
Sarvam AI Speech-to-Text provider.

Uses the Sarvam REST API with multipart WAV upload.
Requires SARVAMAI_API_KEY environment variable.
"""

from __future__ import annotations

import os
import logging

import httpx

from .base import STTProvider, STTResult, pcm_to_wav

logger = logging.getLogger(__name__)

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TIMEOUT = 10.0


class SarvamSTT(STTProvider):
    """
    Sarvam AI STT provider.

    Converts PCM audio to WAV, uploads via multipart POST, and extracts
    the transcript from the JSON response.

    Uses a persistent httpx client for connection reuse.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = SARVAM_API_URL,
        timeout: float = SARVAM_TIMEOUT,
        model: str = "saarika:v2.5",
    ) -> None:
        self._api_key = api_key or os.environ.get("SARVAMAI_API_KEY", "")
        self._api_url = api_url
        self._timeout = timeout
        self._model = model
        self._client: httpx.AsyncClient | None = None

        if not self._api_key:
            logger.warning("[STT:Sarvam] SARVAMAI_API_KEY not set — calls will fail")

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16_000,
        language: str = "en",
    ) -> STTResult:
        wav_data = pcm_to_wav(audio_bytes, sample_rate=sample_rate)
        duration_ms = len(audio_bytes) // (sample_rate * 2) * 1000
        logger.info("[STT:Sarvam] Sending %d bytes (%.0fms) for transcription", len(wav_data), duration_ms)

        # Language mapping: "en" → "en-IN", "hi" → "hi-IN"
        lang_code = language if "-" in language else f"{language}-IN"

        client = self._get_client()

        try:
            response = await client.post(
                self._api_url,
                headers={"api-subscription-key": self._api_key},
                data={
                    "language_code": lang_code,
                    "model": self._model,
                },
                files={"file": ("audio.wav", wav_data, "audio/wav")},
            )
        except httpx.TimeoutException as exc:
            logger.warning("[STT:Sarvam] Request timed out: %s", exc)
            raise

        if response.status_code != 200:
            logger.error(
                "[STT:Sarvam] HTTP %d — body: %s",
                response.status_code,
                response.text[:500],
            )
            response.raise_for_status()

        data = response.json()
        transcript = data.get("transcript", "").strip()
        confidence = float(data.get("confidence", 1.0))
        logger.info("[STT:Sarvam] Transcript: %r (confidence=%.2f)", transcript, confidence)
        return STTResult(text=transcript, confidence=confidence)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
