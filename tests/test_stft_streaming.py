import hashlib

import numpy as np
import pytest

from tmgm_rt.config import ContextConfig, FrontendConfig
from tmgm_rt.context import StreamingContext, stack_causal_context
from tmgm_rt.stft_plus import CausalSTFTPlus, midi_to_hz


def test_stft_plus_is_block_size_invariant():
    config = FrontendConfig()
    seconds = 0.35
    time = np.arange(round(seconds * config.sample_rate)) / config.sample_rate
    audio = (
        0.25 * np.sin(2 * np.pi * 110.0 * time)
        + 0.15 * np.sin(2 * np.pi * 164.81 * time)
        + 0.10 * np.sin(2 * np.pi * 220.0 * time)
    ).astype(np.float32)

    reference = CausalSTFTPlus(config).push(audio)
    streaming = CausalSTFTPlus(config)
    chunks = []
    first = 0
    block_sizes = [1, 17, 64, 257, 511, 33]
    block_index = 0
    while first < audio.size:
        count = block_sizes[block_index % len(block_sizes)]
        chunks.append(streaming.push(audio[first : first + count]))
        first += count
        block_index += 1
    candidate = np.concatenate(chunks)
    assert candidate.shape == reference.shape
    np.testing.assert_allclose(candidate, reference, rtol=0, atol=1e-7)
    assert reference.shape[0] == int(np.ceil(audio.size / config.hop_size))
    # Golden output from the pre-ablation frontend. Optional channels must not
    # perturb a default model's continuous matrix or native feature identity.
    assert hashlib.sha256(reference.tobytes()).hexdigest() == (
        "57ac4df39d10dc80d40cdee9ccc54b5aeb7cc7b8cf74705436e21c1d666c07b9"
    )


def test_streaming_context_matches_batch():
    rng = np.random.default_rng(3)
    features = rng.random((50, 7), dtype=np.float32)
    config = ContextConfig()
    reference = stack_causal_context(features, config)
    context = StreamingContext(features.shape[1], config)
    candidate = np.stack([context.push(frame) for frame in features])
    np.testing.assert_array_equal(candidate, reference)


def test_harmonic_local_contrast_uses_a_side_reference_per_harmonic():
    legacy = CausalSTFTPlus(FrontendConfig())
    corrected = CausalSTFTPlus(
        FrontendConfig(
            harmonic_local_contrast=True,
            contrast_offset_semitones=1.5,
        )
    )

    assert legacy.side_low_bins.shape == (legacy.config.note_count,)
    assert corrected.side_low_bins.shape == corrected.harmonic_bins.shape
    assert corrected.side_high_bins.shape == corrected.harmonic_bins.shape

    time = np.arange(4096, dtype=np.float32) / corrected.config.sample_rate
    audio = (
        0.35 * np.sin(2.0 * np.pi * 82.4069 * time)
        + 0.22 * np.sin(2.0 * np.pi * 164.8138 * time)
        + 0.14 * np.sin(2.0 * np.pi * 247.2207 * time)
    ).astype(np.float32)
    legacy_features = legacy.push(audio)
    corrected_features = corrected.push(audio)
    assert corrected_features.shape == legacy_features.shape
    assert np.isfinite(corrected_features).all()

    note_count = corrected.config.note_count
    contrast_slice = slice(2 * note_count, 3 * note_count)
    assert not np.array_equal(
        corrected_features[:, contrast_slice],
        legacy_features[:, contrast_slice],
    )


def test_optional_feature_dimensions_names_and_description_are_deterministic():
    config = FrontendConfig(
        sample_rate=8_000,
        hop_size=16,
        fft_size=64,
        midi_min=40,
        midi_max=41,
        harmonics=3,
        harmonic_local_contrast=True,
        contrast_offset_semitones=1.5,
        expose_harmonic_local_profile=True,
        contrast_attack_features=True,
    )
    frontend = CausalSTFTPlus(config)
    names = frontend.feature_names()

    # Six established pitch groups, three harmonic-profile groups, two attack
    # groups, and the three globals: (6 + 3 + 2) * 2 + 3 = 25.
    assert frontend.feature_count == 25
    assert len(names) == frontend.feature_count
    assert names[12:18] == [
        "harmonic_local_profile_h1:midi_40",
        "harmonic_local_profile_h1:midi_41",
        "harmonic_local_profile_h2:midi_40",
        "harmonic_local_profile_h2:midi_41",
        "harmonic_local_profile_h3:midi_40",
        "harmonic_local_profile_h3:midi_41",
    ]
    assert names[18:22] == [
        "contrast_positive_flux:midi_40",
        "contrast_positive_flux:midi_41",
        "contrast_fast_slow_attack:midi_40",
        "contrast_fast_slow_attack:midi_41",
    ]
    assert names[-3:] == ["rms", "broadband_flux", "spectral_centroid"]
    assert frontend.describe() == {
        "config": {
            "sample_rate": 8_000,
            "hop_size": 16,
            "fft_size": 64,
            "midi_min": 40,
            "midi_max": 41,
            "harmonics": 3,
            "ema_alpha": 0.08,
            "harmonic_local_contrast": True,
            "contrast_offset_semitones": 1.5,
            "expose_harmonic_local_profile": True,
            "contrast_attack_features": True,
        },
        "feature_count": 25,
        "feature_names": names,
        "strictly_causal": True,
        "lookahead_frames": 0,
    }
    silent = frontend.push(np.zeros(1, dtype=np.float32))[0]
    np.testing.assert_array_equal(silent[18:20], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(
        silent[20:22], np.full(2, 0.5, dtype=np.float32)
    )


def test_harmonic_profile_exposes_true_stack_vs_overtone_only_spectrum():
    config = FrontendConfig(
        midi_min=40,
        midi_max=40,
        harmonics=6,
        harmonic_local_contrast=True,
        contrast_offset_semitones=1.5,
        expose_harmonic_local_profile=True,
    )
    sample_count = 4 * config.fft_size
    time = np.arange(sample_count, dtype=np.float32) / config.sample_rate
    fundamental = float(midi_to_hz(40))

    true_stack = sum(
        (0.18 / np.sqrt(harmonic))
        * np.sin(2.0 * np.pi * fundamental * harmonic * time)
        for harmonic in range(1, 7)
    ).astype(np.float32)
    overtone_only = sum(
        (0.18 / np.sqrt(harmonic))
        * np.sin(2.0 * np.pi * fundamental * harmonic * time)
        for harmonic in (2, 4, 6)
    ).astype(np.float32)

    true_features = CausalSTFTPlus(config).push(true_stack)[-1]
    overtone_features = CausalSTFTPlus(config).push(overtone_only)[-1]
    # With one pitch the six profile channels follow the six established pitch
    # channels. Even harmonics match closely; missing odd harmonics remain
    # separately observable instead of disappearing inside one aggregate.
    true_profile = true_features[6:12]
    overtone_profile = overtone_features[6:12]
    assert true_profile.shape == (6,)
    np.testing.assert_allclose(
        true_profile[[1, 3, 5]], overtone_profile[[1, 3, 5]], atol=0.01
    )
    assert np.all(true_profile[[0, 2, 4]] > overtone_profile[[0, 2, 4]])
    assert float(np.linalg.norm(true_profile - overtone_profile)) > 0.5


def test_optional_frontend_is_block_invariant_and_reset_restores_all_state():
    config = FrontendConfig(
        harmonic_local_contrast=True,
        contrast_offset_semitones=1.5,
        expose_harmonic_local_profile=True,
        contrast_attack_features=True,
    )
    time = np.arange(5_777, dtype=np.float32) / config.sample_rate
    audio = (
        0.3 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.16 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.08 * np.sin(2.0 * np.pi * 329.63 * time)
    ).astype(np.float32)
    reference = CausalSTFTPlus(config).push(audio)

    streaming = CausalSTFTPlus(config)
    parts: list[np.ndarray] = []
    first = 0
    sizes = (1, 31, 256, 7, 509, 64)
    while first < audio.size:
        size = sizes[len(parts) % len(sizes)]
        parts.append(streaming.push(audio[first : first + size]))
        first += size
    candidate = np.concatenate(parts)
    np.testing.assert_array_equal(candidate, reference)

    streaming.push(audio[:733])
    streaming.reset()
    reset_candidate = streaming.push(audio)
    np.testing.assert_array_equal(reset_candidate, reference)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": True},
        {"hop_size": 0},
        {"fft_size": 256, "hop_size": 256},
        {"midi_min": 50, "midi_max": 49},
        {"harmonics": 0},
        {"ema_alpha": float("nan")},
        {"harmonic_local_contrast": 1},
        {"contrast_offset_semitones": float("inf")},
        {"expose_harmonic_local_profile": True},
        {"contrast_attack_features": True},
    ],
)
def test_frontend_config_rejects_invalid_or_ambiguous_ablation_state(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FrontendConfig(**kwargs)
