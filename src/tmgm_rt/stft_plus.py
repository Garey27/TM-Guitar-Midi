from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .config import FrontendConfig


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray:
    return 440.0 * np.power(2.0, (np.asarray(midi) - 69.0) / 12.0)


class CausalSTFTPlus:
    """Strictly causal, block-size invariant STFT+ feature extractor."""

    pitch_feature_names = (
        "fundamental",
        "harmonic_salience",
        "local_contrast",
        "subharmonic_margin",
        "positive_flux",
        "fast_slow_attack",
    )
    global_feature_names = ("rms", "broadband_flux", "spectral_centroid")

    def __init__(self, config: FrontendConfig):
        self.config = config
        if config.fft_size <= config.hop_size:
            raise ValueError("fft_size must exceed hop_size")
        self.window = np.hanning(config.fft_size).astype(np.float32)
        self.ring = np.zeros(config.fft_size, dtype=np.float32)
        self.previous_salience = np.zeros(config.note_count, dtype=np.float32)
        self.ema_salience = np.zeros(config.note_count, dtype=np.float32)
        self.previous_contrast = np.zeros(config.note_count, dtype=np.float32)
        self.ema_contrast = np.zeros(config.note_count, dtype=np.float32)
        self.previous_magnitude = np.zeros(config.fft_size // 2 + 1, dtype=np.float32)

        pitches = np.arange(config.midi_min, config.midi_max + 1)
        fundamentals = midi_to_hz(pitches)
        harmonic_numbers = np.arange(1, config.harmonics + 1, dtype=np.float32)
        harmonic_hz = fundamentals[:, None] * harmonic_numbers[None, :]
        self.harmonic_bins = harmonic_hz * config.fft_size / config.sample_rate
        self.harmonic_weights = (1.0 / np.sqrt(harmonic_numbers)).astype(np.float32)
        self.harmonic_weights /= self.harmonic_weights.sum()
        contrast_offset = float(
            getattr(config, "contrast_offset_semitones", 0.5)
        )
        if getattr(config, "harmonic_local_contrast", False):
            contrast_hz = harmonic_hz
        else:
            contrast_hz = fundamentals
        self.side_low_bins = (
            contrast_hz
            * (2.0 ** (-contrast_offset / 12.0))
            * config.fft_size
            / config.sample_rate
        )
        self.side_high_bins = (
            contrast_hz
            * (2.0 ** (contrast_offset / 12.0))
            * config.fft_size
            / config.sample_rate
        )
        self.subharmonic_bins = fundamentals * 0.5 * config.fft_size / config.sample_rate
        self.frequency_axis = np.fft.rfftfreq(config.fft_size, 1.0 / config.sample_rate).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        """Return the streaming frontend to its deterministic initial state."""
        self.ring.fill(0.0)
        self.write_index = 0
        self.sample_count = 0
        self.next_frame_sample = 1
        self.previous_salience.fill(0.0)
        self.ema_salience.fill(0.0)
        # Corrected local contrast is centered at 0.5 when both the harmonic
        # and its side references are silent. Starting the attack state at the
        # same neutral value avoids manufacturing a transient on reset.
        self.previous_contrast.fill(0.5)
        self.ema_contrast.fill(0.5)
        self.previous_magnitude.fill(0.0)

    @property
    def feature_count(self) -> int:
        pitch_groups = len(self.pitch_feature_names)
        if self.config.expose_harmonic_local_profile:
            pitch_groups += self.config.harmonics
        if self.config.contrast_attack_features:
            pitch_groups += 2
        return pitch_groups * self.config.note_count + len(self.global_feature_names)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for group in self.pitch_feature_names:
            names.extend(
                f"{group}:midi_{pitch}"
                for pitch in range(self.config.midi_min, self.config.midi_max + 1)
            )
        if self.config.expose_harmonic_local_profile:
            for harmonic in range(1, self.config.harmonics + 1):
                names.extend(
                    f"harmonic_local_profile_h{harmonic}:midi_{pitch}"
                    for pitch in range(
                        self.config.midi_min, self.config.midi_max + 1
                    )
                )
        if self.config.contrast_attack_features:
            for group in ("contrast_positive_flux", "contrast_fast_slow_attack"):
                names.extend(
                    f"{group}:midi_{pitch}"
                    for pitch in range(
                        self.config.midi_min, self.config.midi_max + 1
                    )
                )
        names.extend(self.global_feature_names)
        return names

    @staticmethod
    def _sample_bins(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        maximum = values.shape[0] - 1
        clipped = np.clip(positions, 0.0, float(maximum))
        lower = np.floor(clipped).astype(np.int32)
        upper = np.minimum(lower + 1, maximum)
        fraction = clipped - lower
        result = values[lower] * (1.0 - fraction) + values[upper] * fraction
        return np.where(positions <= maximum, result, 0.0).astype(np.float32)

    @staticmethod
    def _unit_db(values: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
        db = 20.0 * np.log10(np.maximum(values, 1.0e-7))
        return np.clip((db - floor_db) / -floor_db, 0.0, 1.0).astype(np.float32)

    def _ordered_frame(self) -> np.ndarray:
        if self.write_index == 0:
            return self.ring.copy()
        return np.concatenate(
            (self.ring[self.write_index :], self.ring[: self.write_index])
        )

    def _append(self, samples: np.ndarray) -> None:
        count = samples.size
        if count >= self.config.fft_size:
            self.ring[:] = samples[-self.config.fft_size :]
            self.write_index = 0
            return
        first = min(count, self.config.fft_size - self.write_index)
        self.ring[self.write_index : self.write_index + first] = samples[:first]
        remaining = count - first
        if remaining:
            self.ring[:remaining] = samples[first:]
        self.write_index = (self.write_index + count) % self.config.fft_size

    def _frame_features(self) -> np.ndarray:
        frame = self._ordered_frame()
        magnitude = np.abs(np.fft.rfft(frame * self.window)).astype(np.float32)
        magnitude /= max(float(self.window.sum()) * 0.5, 1.0)

        harmonic = self._sample_bins(magnitude, self.harmonic_bins)
        fundamental = self._unit_db(harmonic[:, 0])
        harmonic_unit = self._unit_db(harmonic)
        salience = harmonic_unit @ self.harmonic_weights
        side_unit = 0.5 * (
            self._unit_db(self._sample_bins(magnitude, self.side_low_bins))
            + self._unit_db(self._sample_bins(magnitude, self.side_high_bins))
        )
        side = (
            side_unit @ self.harmonic_weights
            if side_unit.ndim == 2
            else side_unit
        )
        contrast = np.clip(salience - side + 0.5, 0.0, 1.0)
        harmonic_local_profile: np.ndarray | None = None
        if self.config.expose_harmonic_local_profile:
            if side_unit.ndim != 2 or side_unit.shape != harmonic_unit.shape:
                raise AssertionError(
                    "harmonic local profile requires per-harmonic side references"
                )
            harmonic_local_profile = np.clip(
                harmonic_unit - side_unit + 0.5, 0.0, 1.0
            ).astype(np.float32, copy=False)
        subharmonic = self._unit_db(
            self._sample_bins(magnitude, self.subharmonic_bins)
        )
        subharmonic_margin = np.clip(salience - subharmonic + 0.5, 0.0, 1.0)
        positive_flux = np.clip(salience - self.previous_salience, 0.0, 1.0)
        self.ema_salience += self.config.ema_alpha * (
            salience - self.ema_salience
        )
        fast_slow = np.clip(salience - self.ema_salience + 0.5, 0.0, 1.0)
        contrast_positive_flux: np.ndarray | None = None
        contrast_fast_slow: np.ndarray | None = None
        if self.config.contrast_attack_features:
            contrast_positive_flux = np.clip(
                contrast - self.previous_contrast, 0.0, 1.0
            ).astype(np.float32, copy=False)
            self.ema_contrast += self.config.ema_alpha * (
                contrast - self.ema_contrast
            )
            contrast_fast_slow = np.clip(
                contrast - self.ema_contrast + 0.5, 0.0, 1.0
            ).astype(np.float32, copy=False)

        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        rms_unit = float(self._unit_db(np.asarray([rms]))[0])
        broadband_flux = float(
            np.mean(np.maximum(magnitude - self.previous_magnitude, 0.0))
        )
        broadband_flux_unit = float(np.clip(broadband_flux * 40.0, 0.0, 1.0))
        magnitude_sum = float(magnitude.sum())
        centroid = (
            float(np.dot(magnitude, self.frequency_axis)) / magnitude_sum
            if magnitude_sum > 1.0e-9
            else 0.0
        )
        centroid_unit = float(np.clip(centroid / (self.config.sample_rate * 0.5), 0.0, 1.0))

        self.previous_salience = salience.astype(np.float32, copy=True)
        if self.config.contrast_attack_features:
            self.previous_contrast = contrast.astype(np.float32, copy=True)
        self.previous_magnitude = magnitude
        global_features = np.asarray(
            [rms_unit, broadband_flux_unit, centroid_unit], dtype=np.float32
        )
        if not (
            self.config.expose_harmonic_local_profile
            or self.config.contrast_attack_features
        ):
            # Preserve the established default concatenation exactly. Existing
            # native binarizers depend on both the values and their byte order.
            return np.concatenate(
                (
                    fundamental,
                    salience,
                    contrast,
                    subharmonic_margin,
                    positive_flux,
                    fast_slow,
                    global_features,
                )
            ).astype(np.float32, copy=False)

        feature_parts = [
            fundamental,
            salience,
            contrast,
            subharmonic_margin,
            positive_flux,
            fast_slow,
        ]
        if harmonic_local_profile is not None:
            # Feature names are harmonic-major, then ascending MIDI pitch.
            feature_parts.append(harmonic_local_profile.T.reshape(-1))
        if contrast_positive_flux is not None and contrast_fast_slow is not None:
            feature_parts.extend((contrast_positive_flux, contrast_fast_slow))
        feature_parts.append(global_features)
        return np.concatenate(
            feature_parts
        ).astype(np.float32, copy=False)

    def push(self, samples: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        output: list[np.ndarray] = []
        offset = 0
        while offset < samples.size:
            needed = self.next_frame_sample - self.sample_count
            take = min(needed, samples.size - offset)
            self._append(samples[offset : offset + take])
            self.sample_count += take
            offset += take
            if self.sample_count == self.next_frame_sample:
                output.append(self._frame_features())
                self.next_frame_sample += self.config.hop_size
        if not output:
            return np.empty((0, self.feature_count), dtype=np.float32)
        return np.stack(output)

    def describe(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "feature_count": self.feature_count,
            "feature_names": self.feature_names(),
            "strictly_causal": True,
            "lookahead_frames": 0,
        }


def extract_stft_plus(samples: np.ndarray, config: FrontendConfig) -> np.ndarray:
    return CausalSTFTPlus(config).push(samples)
