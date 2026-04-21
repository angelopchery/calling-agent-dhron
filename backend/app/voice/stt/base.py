"""
Base interface for STT providers.

All providers receive a raw PCM16 mono audio buffer and return a
transcript string. The `pcm_to_wav` helper converts raw bytes into
a valid WAV file in memory — required by most HTTP-based STT APIs.
"""

from __future__ import annotations

import io
import wave
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """STT transcription result with confidence score."""
    text: str
    confidence: float = 1.0


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM16 bytes into a complete WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class STTProvider(ABC):
    """
    Abstract STT provider.

    Subclasses implement `transcribe()` which takes raw PCM16 mono audio
    bytes and returns the transcript as a plain string. Implementations
    handle WAV conversion, HTTP transport, and response parsing internally.
    """

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16_000,
        language: str = "en",
    ) -> STTResult:
        """
        Transcribe raw PCM16 audio to text.

        Parameters
        ----------
        audio_bytes : bytes
            Raw PCM16 mono audio.
        sample_rate : int
            Sample rate of the audio.
        language : str
            BCP-47 language code hint.

        Returns
        -------
        STTResult
            Transcript text and confidence score.
        """
        ...
