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
        on very quiet mics. Must be high enough that ambient room noise
        amplified by mic preamp gain does not consistently cross it.
    maximum_threshold : float
        Absolute ceiling for the threshold. If calibration captures user
        speech / button-press transients / mic warmup noise, p95 * mult
        can blow past this; the cap keeps the agent from going deaf.
    barge_in_multiplier : float
        Barge-in threshold = max(threshold * barge_in_multiplier,
        barge_in_minimum). Stricter than the speech threshold so TTS
        residual / ambient noise can't masquerade as a sustained interrupt.
    barge_in_minimum : float
        Absolute floor for the barge-in threshold.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float | None = None,
        calibration_frames: int = 25,
        calibration_multiplier: float = 4.0,
        minimum_threshold: float = 150.0,
        maximum_threshold: float = 1500.0,
        barge_in_multiplier: float = 2.5,
        barge_in_minimum: float = 600.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._threshold: float = energy_threshold or 0.0
        self._minimum_threshold = minimum_threshold
        self._maximum_threshold = maximum_threshold
        self._calibration_multiplier = calibration_multiplier
        self._barge_in_multiplier = barge_in_multiplier
        self._barge_in_minimum = barge_in_minimum

        # Calibration state
        self._calibrating = energy_threshold is None
        self._calibration_energies: list[float] = []
        self._calibration_target = calibration_frames
        self._calibrated = not self._calibrating

        self._last_energy: float = 0.0
        self._log_counter = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def barge_in_threshold(self) -> float:
        return max(self._threshold * self._barge_in_multiplier, self._barge_in_minimum)

    @property
    def last_energy(self) -> float:
        return self._last_energy

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def is_speech(self, frame: bytes) -> bool:
        energy = self._compute_energy(frame)
        self._last_energy = energy

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
        noise_max = float(arr.max())

        # Quiet-half calibration: sort frames by energy and use only the
        # quietest 50% to compute p95. Robust to bursts during the calibration
        # window (button-click transients, the user starting to speak, brief
        # background noise) — those land in the loud half and get discarded.
        sorted_arr = np.sort(arr)
        quiet_half = sorted_arr[: max(1, len(sorted_arr) // 2)]
        noise_p95 = float(np.percentile(quiet_half, 95))

        # Apply multiplier with min floor, then cap at maximum_threshold so a
        # pathologically loud calibration window can't make the agent deaf.
        raw = noise_p95 * self._calibration_multiplier
        capped = min(raw, self._maximum_threshold)
        self._threshold = max(capped, self._minimum_threshold)

        if raw > self._maximum_threshold:
            logger.warning(
                "[VAD] Calibration noise too high (p95=%.1f * %.1f = %.1f) — "
                "capped at %.1f. User likely spoke or made noise during "
                "calibration window.",
                noise_p95, self._calibration_multiplier, raw,
                self._maximum_threshold,
            )

        self._calibrated = True

        logger.info(
            "[VAD] Calibrated in %d frames (%.0fms) — "
            "noise (quietest half): mean=%.2f p95=%.2f, max(all)=%.2f → "
            "threshold=%.1f barge_in=%.1f",
            len(self._calibration_energies),
            len(self._calibration_energies) * 20,
            float(quiet_half.mean()),
            noise_p95,
            noise_max,
            self._threshold,
            self.barge_in_threshold,
        )
