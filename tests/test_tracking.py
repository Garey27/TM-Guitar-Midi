import numpy as np

from tmgm_rt.tracking import (
    PolyphonicTracker,
    SpectralEvidence,
    TrackerConfig,
    midi_to_hz,
)


def _fade_envelope(sample_count: int, sample_rate: int, fade_ms: float = 8.0):
    envelope = np.ones(sample_count, dtype=np.float32)
    fade = min(round(fade_ms * sample_rate / 1_000.0), sample_count // 2)
    if fade:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade)) ** 2
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    return envelope


def _tone(
    pitch: int,
    seconds: float,
    config: TrackerConfig,
    *,
    amplitude: float = 0.25,
    harmonic_count: int = 1,
) -> np.ndarray:
    sample_count = round(seconds * config.sample_rate)
    time = np.arange(sample_count, dtype=np.float64) / config.sample_rate
    fundamental = float(midi_to_hz(pitch))
    audio = np.zeros(sample_count, dtype=np.float64)
    for harmonic in range(1, harmonic_count + 1):
        audio += (amplitude / harmonic) * np.sin(
            2.0 * np.pi * fundamental * harmonic * time
        )
    audio *= _fade_envelope(sample_count, config.sample_rate)
    return audio.astype(np.float32)


def _with_silence(signal: np.ndarray, config: TrackerConfig) -> np.ndarray:
    lead = np.zeros(round(0.12 * config.sample_rate), dtype=np.float32)
    tail = np.zeros(round(0.35 * config.sample_rate), dtype=np.float32)
    return np.concatenate((lead, signal, tail))


def _note_ons(events):
    return [event for event in events if event.kind == "note_on"]


def test_tracker_is_block_size_invariant():
    config = TrackerConfig()
    first = _tone(45, 0.35, config, amplitude=0.22, harmonic_count=6)
    second = sum(
        (_tone(pitch, 0.42, config, amplitude=0.16) for pitch in (45, 52, 57)),
        start=np.zeros(round(0.42 * config.sample_rate), dtype=np.float32),
    )
    audio = np.concatenate(
        (
            np.zeros(700, dtype=np.float32),
            first,
            np.zeros(1_300, dtype=np.float32),
            second,
            np.zeros(9_000, dtype=np.float32),
        )
    )

    reference = PolyphonicTracker(config).push(audio)
    streaming = PolyphonicTracker(config)
    candidate = []
    offset = 0
    sizes = (1, 17, 64, 255, 511, 7, 1_033)
    block_index = 0
    while offset < audio.size:
        size = sizes[block_index % len(sizes)]
        candidate.extend(streaming.push(audio[offset : offset + size]))
        offset += size
        block_index += 1
    assert candidate == reference


def test_frontend_frame_grid_matches_frozen_tm_contract():
    config = TrackerConfig()
    samples = np.zeros(3 * config.hop_size, dtype=np.float32)
    evidence = PolyphonicTracker(config).frontend.push(samples)
    assert evidence.sample_indices.tolist() == [
        1,
        1 + config.hop_size,
        1 + 2 * config.hop_size,
    ]


def test_pure_tone_maps_to_one_pitch():
    config = TrackerConfig()
    events = PolyphonicTracker(config).push(
        _with_silence(_tone(45, 0.70, config), config)
    )
    assert [event.pitch for event in _note_ons(events)] == [45]


def test_polyphonic_chord_keeps_independent_notes():
    config = TrackerConfig()
    chord = sum(
        (_tone(pitch, 0.75, config, amplitude=0.20) for pitch in (45, 52, 57)),
        start=np.zeros(round(0.75 * config.sample_rate), dtype=np.float32),
    )
    events = PolyphonicTracker(config).push(_with_silence(chord, config))
    pitches = {event.pitch for event in _note_ons(events)}
    assert pitches == {45, 52, 57}


def test_lower_fundamental_owns_its_overtones():
    config = TrackerConfig()
    guitar_like = _tone(45, 0.75, config, amplitude=0.28, harmonic_count=8)
    events = PolyphonicTracker(config).push(_with_silence(guitar_like, config))
    assert [event.pitch for event in _note_ons(events)] == [45]


def test_new_acoustic_attack_retriggers_an_active_pitch():
    config = TrackerConfig(retrigger_refractory_frames=4)
    pluck = _tone(52, 0.34, config, amplitude=0.24, harmonic_count=4)
    gap = np.zeros(round(0.045 * config.sample_rate), dtype=np.float32)
    audio = _with_silence(np.concatenate((pluck, gap, pluck)), config)
    events = PolyphonicTracker(config).push(audio)
    pitch_events = [event for event in events if event.pitch == 52]
    assert [event.kind for event in pitch_events[:3]] == [
        "note_on",
        "note_off",
        "note_on",
    ]
    assert len([event for event in pitch_events if event.kind == "note_on"]) == 2


def test_velocity_is_monotonic_with_acoustic_attack_amplitude():
    config = TrackerConfig()

    def velocity(amplitude: float) -> int:
        audio = _with_silence(
            _tone(52, 0.60, config, amplitude=amplitude, harmonic_count=3),
            config,
        )
        note_ons = _note_ons(PolyphonicTracker(config).push(audio))
        return next(event.velocity for event in note_ons if event.pitch == 52)

    quiet = velocity(0.055)
    loud = velocity(0.30)
    assert 1 <= quiet < loud <= 127


def test_tm_confidence_cannot_change_velocity():
    config = TrackerConfig(attack_frames=1, tm_activity_weight=1.0)
    note_count = config.note_count
    activity = np.zeros((1, note_count), dtype=np.float32)
    onset = np.zeros_like(activity)
    attack = np.zeros_like(activity)
    attack[0, 52 - config.midi_min] = 0.08
    evidence = SpectralEvidence(
        sample_indices=np.asarray([config.hop_size], dtype=np.int64),
        activity=activity,
        onset=onset,
        attack_energy=attack,
        fundamental_amplitude=attack.copy(),
    )
    low_tm = np.zeros_like(activity)
    high_tm = np.zeros_like(activity)
    low_tm[0, 52 - config.midi_min] = config.activity_on
    high_tm[0, 52 - config.midi_min] = 1.0

    low_event = PolyphonicTracker(config).process_evidence(
        evidence, tm_activity=low_tm
    )[0]
    high_event = PolyphonicTracker(config).process_evidence(
        evidence, tm_activity=high_tm
    )[0]
    assert low_event.pitch == high_event.pitch == 52
    assert low_event.velocity == high_event.velocity


def test_tm_activity_cannot_create_a_track_without_acoustic_attack():
    config = TrackerConfig(attack_frames=1, tm_activity_weight=1.0)
    note_count = config.note_count
    zeros = np.zeros((1, note_count), dtype=np.float32)
    evidence = SpectralEvidence(
        sample_indices=np.asarray([config.hop_size], dtype=np.int64),
        activity=zeros.copy(),
        onset=zeros.copy(),
        attack_energy=zeros.copy(),
        fundamental_amplitude=zeros.copy(),
    )
    tm_activity = zeros.copy()
    tm_activity[0, 52 - config.midi_min] = 1.0
    assert (
        PolyphonicTracker(config).process_evidence(
            evidence, tm_activity=tm_activity
        )
        == []
    )


def test_tm_onset_cannot_retrigger_without_a_new_acoustic_attack():
    config = TrackerConfig(
        attack_frames=1,
        tm_activity_weight=1.0,
        tm_onset_weight=1.0,
        retrigger_refractory_frames=0,
        attack_memory_frames=1,
        global_attack_memory_frames=1,
    )
    note_count = config.note_count
    frame_count = 4
    activity = np.zeros((frame_count, note_count), dtype=np.float32)
    onset = np.zeros_like(activity)
    attack = np.zeros_like(activity)
    pitch_index = 52 - config.midi_min
    activity[:, pitch_index] = 1.0
    onset[2, pitch_index] = 1.0
    attack[0, pitch_index] = 0.08
    evidence = SpectralEvidence(
        sample_indices=np.arange(1, frame_count + 1, dtype=np.int64)
        * config.hop_size,
        activity=np.zeros_like(activity),
        onset=np.zeros_like(activity),
        attack_energy=attack,
        fundamental_amplitude=attack.copy(),
    )
    events = PolyphonicTracker(config).process_evidence(
        evidence, tm_activity=activity, tm_onset=onset
    )
    assert [event.kind for event in events] == ["note_on"]


def test_recent_tm_onset_needs_a_real_global_chord_attack():
    config = TrackerConfig(
        attack_frames=2,
        tm_activity_weight=1.0,
        tm_onset_weight=1.0,
    )
    note_count = config.note_count
    activity = np.zeros((2, note_count), dtype=np.float32)
    onset = np.zeros_like(activity)
    pitch_index = 52 - config.midi_min
    activity[:, pitch_index] = 1.0
    onset[0, pitch_index] = 1.0
    attack = np.zeros_like(activity)
    attack[0, 45 - config.midi_min] = 0.08
    evidence = SpectralEvidence(
        sample_indices=np.asarray(
            [config.hop_size, 2 * config.hop_size], dtype=np.int64
        ),
        activity=np.zeros_like(activity),
        onset=np.zeros_like(activity),
        attack_energy=attack,
        fundamental_amplitude=attack.copy(),
    )
    events = PolyphonicTracker(config).process_evidence(
        evidence, tm_activity=activity, tm_onset=onset
    )
    assert [(event.kind, event.pitch) for event in events] == [
        ("note_on", 52)
    ]
    assert events[0].velocity > 1
