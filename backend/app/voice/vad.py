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
        calibration_frames: int = 30,
        calibration_multiplier: float = 4.0,
        minimum_threshold: float = 5.0,
        maximum_threshold: float = 1500.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._threshold: float = energy_threshold or 0.0
        self._minimum_threshold = minimum_threshold
        self._maximum_threshold = maximum_threshold
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

    def is_speech(self, frame: bytes, count_for_calibration: bool = True) -> bool:
        energy = self._compute_energy(frame)

        # Periodic debug log every 250 frames (~5 seconds)
        self._log_counter += 1
        if self._log_counter % 250 == 1:
            logger.info(
                "[VAD] energy=%.1f threshold=%.1f calibrated=%s",
                energy, self._threshold, self._calibrated,
            )

        # Auto-calibration phase: collect noise samples, always return False.
        # Pipeline passes count_for_calibration=False during TTS playback /
        # echo cooldown so speaker bleed doesn't poison the noise floor.
        if not self._calibrated:
            if count_for_calibration:
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
        # Robust noise floor: use only the quietest half of the calibration
        # frames. A transient burst (door, breath, brief speech, residual
        # TTS echo trail) would otherwise pull p95 high enough to set the
        # threshold above achievable speech levels — making the agent deaf
        # for the rest of the session.
        quiet_half = np.sort(arr)[: max(1, len(arr) // 2)]
        noise_mean = float(quiet_half.mean())
        noise_p95 = float(np.percentile(quiet_half, 95))
        noise_max = float(quiet_half.max())

        raw_threshold = noise_p95 * self._calibration_multiplier
        capped = raw_threshold > self._maximum_threshold
        self._threshold = max(
            min(raw_threshold, self._maximum_threshold),
            self._minimum_threshold,
        )
        self._calibrated = True
        if capped:
            logger.warning(
                "[VAD] Calibration noise too high — threshold capped at %.1f "
                "(uncapped would be %.1f). Environment may be unusually loud.",
                self._maximum_threshold, raw_threshold,
            )

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
