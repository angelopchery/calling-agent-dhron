"""
Voice Activity Detection.

Energy-based VAD using numpy-vectorized mean-absolute computation.
Auto-calibrates from ambient noise during the first 500ms, then
uses a simple energy > threshold comparison — no complex state machine.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Classifies a PCM16 audio frame as speech or silence.

    Auto-calibration: during the first `calibration_frames` frames,
    measures the ambient noise floor and sets threshold above it.
    After calibration, every frame is a simple energy comparison.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (for logging only).
    energy_threshold : float | None
        Fixed threshold. If None, auto-calibrates from ambient noise.
    calibration_frames : int
        Frames for noise floor measurement. 25 frames = 500ms at 20ms/frame.
    calibration_multiplier : float
        Threshold = max(noise_p95 * multiplier, minimum_threshold).
    minimum_threshold : float
        Absolute floor for the threshold to prevent near-zero thresholds
        on very quiet mics.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float | None = None,
        calibration_frames: int = 25,
        calibration_multiplier: float = 3.5,
        minimum_threshold: float = 3.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._threshold: float = energy_threshold or 0.0
        self._minimum_threshold = minimum_threshold
        self._calibration_multiplier = calibration_multiplier

        # Calibration state
        self._calibrating = energy_threshold is None
        self._calibration_energies: list[float] = []
        self._calibration_target = calibration_frames
        self._calibrated = not self._calibrating

        self._log_counter = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def is_speech(self, frame: bytes) -> bool:
        energy = self._compute_energy(frame)

        # Periodic debug log every 250 frames (~5 seconds)
        self._log_counter += 1
        if self._log_counter % 250 == 1:
            logger.info(
                "[VAD] energy=%.1f threshold=%.1f calibrated=%s",
                energy, self._threshold, self._calibrated,
            )

        # Auto-calibration phase: collect noise samples, always return False
        if not self._calibrated:
            self._calibration_energies.append(energy)
            if len(self._calibration_energies) >= self._calibration_target:
                self._finish_calibration()
            return False

        return energy > self._threshold

    def _compute_energy(self, frame: bytes) -> float:
        """Mean absolute amplitude of PCM16 frame."""
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return float(np.abs(audio).mean())

    def _finish_calibration(self) -> None:
        arr = np.array(self._calibration_energies)
        noise_mean = float(arr.mean())
        noise_p95 = float(np.percentile(arr, 95))
        noise_max = float(arr.max())

        self._threshold = max(
            noise_p95 * self._calibration_multiplier,
            self._minimum_threshold,
        )
        self._calibrated = True

        logger.info(
            "[VAD] Calibrated in %d frames (%.0fms) — "
            "noise: mean=%.2f p95=%.2f max=%.2f → threshold=%.1f",
            len(self._calibration_energies),
            len(self._calibration_energies) * 20,
            noise_mean,
            noise_p95,
            noise_max,
            self._threshold,
        )
