from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 22_050
    hop_size: int = 256
    fft_size: int = 2_048
    midi_min: int = 40
    midi_max: int = 88
    harmonics: int = 6
    ema_alpha: float = 0.08
    # Legacy contrast subtracts only a fundamental-side reference from a
    # six-harmonic salience average. The corrected mode measures both sides of
    # every harmonic before applying the same harmonic weights.
    harmonic_local_contrast: bool = False
    contrast_offset_semitones: float = 0.5
    # Optional ablation channels. They are off by default so the established
    # STFT+ feature matrix remains byte-for-byte compatible with existing
    # binarizers and native models.
    expose_harmonic_local_profile: bool = False
    contrast_attack_features: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "sample_rate": self.sample_rate,
            "hop_size": self.hop_size,
            "fft_size": self.fft_size,
            "midi_min": self.midi_min,
            "midi_max": self.midi_max,
            "harmonics": self.harmonics,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        boolean_fields = {
            "harmonic_local_contrast": self.harmonic_local_contrast,
            "expose_harmonic_local_profile": self.expose_harmonic_local_profile,
            "contrast_attack_features": self.contrast_attack_features,
        }
        for name, value in boolean_fields.items():
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_size <= 0:
            raise ValueError("hop_size must be positive")
        if self.fft_size <= self.hop_size:
            raise ValueError("fft_size must exceed hop_size")
        if not 0 <= self.midi_min <= self.midi_max <= 127:
            raise ValueError("MIDI range must be within 0..127")
        if self.harmonics <= 0:
            raise ValueError("harmonics must be positive")
        if (
            isinstance(self.ema_alpha, bool)
            or not isinstance(self.ema_alpha, (int, float))
            or not math.isfinite(float(self.ema_alpha))
            or not 0.0 < float(self.ema_alpha) <= 1.0
        ):
            raise ValueError("ema_alpha must be finite and in the interval (0, 1]")
        if (
            isinstance(self.contrast_offset_semitones, bool)
            or not isinstance(self.contrast_offset_semitones, (int, float))
            or not math.isfinite(float(self.contrast_offset_semitones))
            or self.contrast_offset_semitones <= 0.0
        ):
            raise ValueError("contrast_offset_semitones must be positive")
        if (
            self.expose_harmonic_local_profile
            or self.contrast_attack_features
        ) and not self.harmonic_local_contrast:
            raise ValueError(
                "harmonic profile/contrast attack features require "
                "harmonic_local_contrast"
            )

    @property
    def note_count(self) -> int:
        return self.midi_max - self.midi_min + 1

    @property
    def frame_seconds(self) -> float:
        return self.hop_size / self.sample_rate

    @property
    def history_seconds(self) -> float:
        return self.fft_size / self.sample_rate


@dataclass(frozen=True)
class ContextConfig:
    delays: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)

    def __post_init__(self) -> None:
        if not self.delays or self.delays[0] != 0:
            raise ValueError("context delays must start with zero")
        if tuple(sorted(set(self.delays))) != self.delays:
            raise ValueError("context delays must be sorted and unique")


@dataclass(frozen=True)
class TargetConfig:
    activity_outputs: bool = True
    onset_outputs: bool = True
    onset_width_frames: int = 2
    onset_delay_frames: int = 0

    def __post_init__(self) -> None:
        if self.onset_width_frames <= 0:
            raise ValueError("onset_width_frames must be positive")
        if self.onset_delay_frames < 0:
            raise ValueError("onset_delay_frames cannot be negative")


DEFAULT_FRONTEND = FrontendConfig()
DEFAULT_CONTEXT = ContextConfig()
DEFAULT_TARGETS = TargetConfig()
