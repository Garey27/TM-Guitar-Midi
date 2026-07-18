from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio_mono_channel_zero(
    path: str | Path, target_sample_rate: int
) -> np.ndarray:
    """Load exactly channel 0, matching NeuralNoteBatch's channel policy."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(samples[:, 0], dtype=np.float32)
    if sample_rate == target_sample_rate:
        return mono

    ratio = Fraction(target_sample_rate, int(sample_rate)).limit_denominator()
    converted = resample_poly(mono, ratio.numerator, ratio.denominator)
    return np.ascontiguousarray(converted, dtype=np.float32)
