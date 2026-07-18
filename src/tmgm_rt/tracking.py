"""Strictly-causal polyphonic guitar tracker with optional TM evidence.

This module is deliberately independent of the training and VST code.  It is
a clean-room DSP baseline: two causal spectral resolutions produce acoustic
evidence, harmonic energy is assigned to lower fundamentals where possible,
and every MIDI pitch owns an independent note state machine.  A Tsetlin
Machine may refine the activity/onset evidence later, but it never contributes
to velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def midi_to_hz(midi: np.ndarray | float | int) -> np.ndarray:
    """Convert MIDI note numbers to equal-tempered frequencies."""

    return 440.0 * np.power(2.0, (np.asarray(midi) - 69.0) / 12.0)


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration shared by the spectral frontend and note decoder."""

    # The TM-stable contracts and NeuralNote teacher grid both run at 22.05 kHz.
    # 512/4096 here have the same 23.2/185.8 ms spans as 1024/8192 at 44.1 kHz,
    # while keeping the tracker and TM evidence on one 256-sample frame grid.
    sample_rate: int = 22_050
    hop_size: int = 256
    short_fft_size: int = 512
    long_fft_size: int = 4_096
    midi_min: int = 40
    midi_max: int = 88
    harmonics: int = 11
    harmonic_decay: float = 1.0
    ownership_strength: float = 1.05
    ownership_tolerance_cents: float = 18.0
    acoustic_floor: float = 0.0015
    acoustic_reference: float = 0.10
    attack_reference: float = 0.035
    slow_attack_alpha: float = 0.08
    # Frozen 0/1 TM decisions are the primary pitch classifier when supplied.
    # Acoustic evidence still owns physical attack admission, state and
    # velocity. A fractional blend is a separate calibration experiment and
    # must not silently replace the frozen operating points.
    tm_activity_weight: float = 1.0
    tm_onset_weight: float = 1.0
    activity_on: float = 0.30
    activity_off: float = 0.16
    onset_on: float = 0.32
    retrigger_on: float = 0.42
    attack_frames: int = 2
    release_frames: int = 4
    retrigger_refractory_frames: int = 6
    attack_memory_frames: int = 12
    global_attack_memory_frames: int = 3
    new_note_attack_floor: float = 0.0020
    retrigger_attack_floor: float = 0.0025
    global_attack_floor: float = 0.015
    global_velocity_mix: float = 0.50
    velocity_floor: float = 0.0015
    velocity_reference: float = 0.16

    def __post_init__(self) -> None:
        integer_positive = (
            "sample_rate",
            "hop_size",
            "short_fft_size",
            "long_fft_size",
            "harmonics",
            "attack_frames",
            "release_frames",
            "attack_memory_frames",
            "global_attack_memory_frames",
        )
        for name in integer_positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.short_fft_size > self.long_fft_size:
            raise ValueError("short_fft_size cannot exceed long_fft_size")
        if self.hop_size > self.short_fft_size:
            raise ValueError("hop_size cannot exceed short_fft_size")
        if self.midi_min > self.midi_max:
            raise ValueError("midi_min cannot exceed midi_max")
        if self.retrigger_refractory_frames < 0:
            raise ValueError("retrigger_refractory_frames cannot be negative")
        for name in (
            "slow_attack_alpha",
            "tm_activity_weight",
            "tm_onset_weight",
            "activity_on",
            "activity_off",
            "onset_on",
            "retrigger_on",
            "global_velocity_mix",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.activity_off >= self.activity_on:
            raise ValueError("activity_off must be lower than activity_on")
        for name in (
            "harmonic_decay",
            "ownership_strength",
            "ownership_tolerance_cents",
            "acoustic_floor",
            "acoustic_reference",
            "attack_reference",
            "velocity_floor",
            "velocity_reference",
            "new_note_attack_floor",
            "retrigger_attack_floor",
            "global_attack_floor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.acoustic_reference <= self.acoustic_floor:
            raise ValueError("acoustic_reference must exceed acoustic_floor")
        if self.velocity_reference <= self.velocity_floor:
            raise ValueError("velocity_reference must exceed velocity_floor")

    @property
    def note_count(self) -> int:
        return self.midi_max - self.midi_min + 1


@dataclass(frozen=True)
class SpectralEvidence:
    """A batch of causal per-frame acoustic measurements."""

    sample_indices: np.ndarray
    activity: np.ndarray
    onset: np.ndarray
    attack_energy: np.ndarray
    fundamental_amplitude: np.ndarray

    @classmethod
    def empty(cls, note_count: int) -> "SpectralEvidence":
        frame_pitch = np.empty((0, note_count), dtype=np.float32)
        return cls(
            sample_indices=np.empty(0, dtype=np.int64),
            activity=frame_pitch.copy(),
            onset=frame_pitch.copy(),
            attack_energy=frame_pitch.copy(),
            fundamental_amplitude=frame_pitch.copy(),
        )

    @property
    def frame_count(self) -> int:
        return int(self.sample_indices.shape[0])


@dataclass(frozen=True)
class NoteEvent:
    """A sample-accurate event emitted at a causal analysis boundary."""

    kind: Literal["note_on", "note_off"]
    pitch: int
    velocity: int
    sample_index: int
    frame_index: int


class CausalDualResolutionFrontend:
    """Short/long spectral frontend with no future samples or lookahead."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._ring = np.zeros(config.long_fft_size, dtype=np.float32)
        self._write_index = 0
        self._sample_count = 0
        # Match the frozen TM frontend exactly: emit the first zero-padded,
        # strictly causal frame after sample 1, then advance by hop_size.  A
        # first frame at hop_size would add one gratuitous 11.6 ms delay and
        # leave the TM score grid one row longer than the acoustic grid.
        self._next_frame_sample = 1
        self._short_window = np.hanning(config.short_fft_size).astype(np.float32)
        self._long_window = np.hanning(config.long_fft_size).astype(np.float32)
        self._short_scale = 2.0 / max(float(self._short_window.sum()), 1.0)
        self._long_scale = 2.0 / max(float(self._long_window.sum()), 1.0)

        pitches = np.arange(config.midi_min, config.midi_max + 1)
        self._frequencies = midi_to_hz(pitches).astype(np.float64)
        harmonic_numbers = np.arange(1, config.harmonics + 1, dtype=np.float64)
        self._harmonic_frequencies = (
            self._frequencies[:, None] * harmonic_numbers[None, :]
        )
        self._harmonic_weights = np.power(
            harmonic_numbers, -config.harmonic_decay
        )
        self._harmonic_weights /= self._harmonic_weights.sum()
        self._ownership_links = self._build_ownership_links()
        self._slow_short = np.zeros(config.note_count, dtype=np.float32)

    def reset(self) -> None:
        self._ring.fill(0.0)
        self._write_index = 0
        self._sample_count = 0
        self._next_frame_sample = 1
        self._slow_short.fill(0.0)

    def _build_ownership_links(self) -> list[list[list[tuple[int, int]]]]:
        """Precompute lower fundamentals able to explain each harmonic bin."""

        result: list[list[list[tuple[int, int]]]] = []
        tolerance = self.config.ownership_tolerance_cents
        for pitch_index, pitch_harmonics in enumerate(self._harmonic_frequencies):
            pitch_links: list[list[tuple[int, int]]] = []
            for target_frequency in pitch_harmonics:
                links: list[tuple[int, int]] = []
                for owner_index in range(pitch_index):
                    ratio = target_frequency / self._frequencies[owner_index]
                    owner_harmonic = int(round(float(ratio)))
                    if not 2 <= owner_harmonic <= self.config.harmonics:
                        continue
                    cents = abs(1_200.0 * np.log2(ratio / owner_harmonic))
                    if cents <= tolerance:
                        links.append((owner_index, owner_harmonic))
                pitch_links.append(links)
            result.append(pitch_links)
        return result

    def _append(self, samples: np.ndarray) -> None:
        count = int(samples.size)
        if count >= self.config.long_fft_size:
            self._ring[:] = samples[-self.config.long_fft_size :]
            self._write_index = 0
            return
        first = min(count, self.config.long_fft_size - self._write_index)
        self._ring[self._write_index : self._write_index + first] = samples[:first]
        remaining = count - first
        if remaining:
            self._ring[:remaining] = samples[first:]
        self._write_index = (
            self._write_index + count
        ) % self.config.long_fft_size

    def _ordered_long_frame(self) -> np.ndarray:
        if self._write_index == 0:
            return self._ring.copy()
        return np.concatenate(
            (self._ring[self._write_index :], self._ring[: self._write_index])
        )

    @staticmethod
    def _sample_bins(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        maximum = values.shape[0] - 1
        clipped = np.clip(positions, 0.0, float(maximum))
        lower = np.floor(clipped).astype(np.int32)
        upper = np.minimum(lower + 1, maximum)
        fraction = clipped - lower
        sampled = values[lower] * (1.0 - fraction) + values[upper] * fraction
        return np.where(positions <= maximum, sampled, 0.0).astype(np.float32)

    def _harmonic_amplitudes(
        self, frame: np.ndarray, window: np.ndarray, scale: float
    ) -> np.ndarray:
        magnitude = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        magnitude *= scale
        positions = (
            self._harmonic_frequencies * frame.shape[0] / self.config.sample_rate
        )
        return self._sample_bins(magnitude, positions)

    def _reject_harmonic_owners(self, amplitudes: np.ndarray) -> np.ndarray:
        """Remove energy predictable from real lower fundamentals.

        This is local attribution, not a global top-k decision.  If an octave
        is genuinely played as well, the energy above the lower note's
        predicted harmonic envelope remains available to the octave pitch.
        """

        residual = amplitudes.copy()
        fundamentals = amplitudes[:, 0]
        decay = self.config.harmonic_decay
        strength = self.config.ownership_strength
        for pitch_index, pitch_links in enumerate(self._ownership_links):
            for harmonic_index, links in enumerate(pitch_links):
                explained = 0.0
                for owner_index, owner_harmonic in links:
                    expected = (
                        fundamentals[owner_index]
                        / (float(owner_harmonic) ** decay)
                        * strength
                    )
                    measured_owner_harmonic = amplitudes[
                        owner_index, owner_harmonic - 1
                    ]
                    explained += min(float(measured_owner_harmonic), expected)
                residual[pitch_index, harmonic_index] = max(
                    float(amplitudes[pitch_index, harmonic_index]) - explained,
                    0.0,
                )
        return residual

    @staticmethod
    def _pitch_tuning_mask(fundamentals: np.ndarray) -> np.ndarray:
        """Attribute a local spectral peak to its nearest semitone centres.

        The mask compares only immediate neighbours, so simultaneous adjacent
        notes of similar strength survive.  Broad leakage into a weaker
        neighbour does not become a second pitch.  This is a local frequency
        attribution step and never limits global polyphony.
        """

        neighbours = np.zeros_like(fundamentals)
        if fundamentals.size > 1:
            neighbours[1:] = np.maximum(neighbours[1:], fundamentals[:-1])
            neighbours[:-1] = np.maximum(neighbours[:-1], fundamentals[1:])
        ratio = fundamentals / np.maximum(neighbours, 1.0e-8)
        # A partial low-frequency window initially looks like a broad plateau;
        # wait until a semitone centre is a real local maximum before allowing
        # it to start a note.  This avoids the common +/-1-semitone fan-out at
        # attacks without imposing any total-note or chord-size constraint.
        return np.clip((ratio - 1.02) / 0.18, 0.0, 1.0).astype(np.float32)

    def _compress(self, values: np.ndarray) -> np.ndarray:
        floor = self.config.acoustic_floor
        span = self.config.acoustic_reference - floor
        positive = np.maximum(values - floor, 0.0)
        return (1.0 - np.exp(-positive / span)).astype(np.float32)

    def _frame_evidence(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        long_frame = self._ordered_long_frame()
        short_frame = long_frame[-self.config.short_fft_size :]
        long_raw = self._harmonic_amplitudes(
            long_frame, self._long_window, self._long_scale
        )
        short_raw = self._harmonic_amplitudes(
            short_frame, self._short_window, self._short_scale
        )
        long_residual = self._reject_harmonic_owners(long_raw)
        short_residual = self._reject_harmonic_owners(short_raw)

        # A long window has enough frequency resolution to decide which MIDI
        # centre owns the fundamental.  Reuse that attribution for the short
        # attack window, whose job is timing rather than low-frequency pitch.
        tuning_mask = self._pitch_tuning_mask(long_raw[:, 0])
        long_residual[:, 0] *= tuning_mask
        short_residual[:, 0] *= tuning_mask

        long_unit = self._compress(long_residual)
        short_unit = self._compress(short_residual)
        long_harmonic = long_unit @ self._harmonic_weights
        short_harmonic = short_unit @ self._harmonic_weights
        long_pitch = 0.72 * long_unit[:, 0] + 0.28 * long_harmonic
        short_pitch = 0.72 * short_unit[:, 0] + 0.28 * short_harmonic
        activity = np.clip(
            0.78 * long_pitch + 0.22 * short_pitch, 0.0, 1.0
        ).astype(np.float32)

        short_acoustic = (
            0.78 * short_residual[:, 0]
            + 0.22 * (short_residual @ self._harmonic_weights)
        ).astype(np.float32)
        attack_energy = np.maximum(short_acoustic - self._slow_short, 0.0)
        self._slow_short += self.config.slow_attack_alpha * (
            short_acoustic - self._slow_short
        )
        onset = (
            1.0
            - np.exp(
                -np.maximum(attack_energy - self.config.acoustic_floor, 0.0)
                / self.config.attack_reference
            )
        ).astype(np.float32)
        return (
            activity,
            onset,
            attack_energy.astype(np.float32, copy=False),
            long_residual[:, 0].astype(np.float32, copy=False),
        )

    def push(self, samples: np.ndarray) -> SpectralEvidence:
        samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        sample_indices: list[int] = []
        activities: list[np.ndarray] = []
        onsets: list[np.ndarray] = []
        attacks: list[np.ndarray] = []
        fundamentals: list[np.ndarray] = []
        offset = 0
        while offset < samples.size:
            needed = self._next_frame_sample - self._sample_count
            take = min(needed, samples.size - offset)
            self._append(samples[offset : offset + take])
            offset += take
            self._sample_count += take
            if self._sample_count == self._next_frame_sample:
                activity, onset, attack, fundamental = self._frame_evidence()
                sample_indices.append(self._sample_count)
                activities.append(activity)
                onsets.append(onset)
                attacks.append(attack)
                fundamentals.append(fundamental)
                self._next_frame_sample += self.config.hop_size
        if not sample_indices:
            return SpectralEvidence.empty(self.config.note_count)
        return SpectralEvidence(
            sample_indices=np.asarray(sample_indices, dtype=np.int64),
            activity=np.stack(activities),
            onset=np.stack(onsets),
            attack_energy=np.stack(attacks),
            fundamental_amplitude=np.stack(fundamentals),
        )


@dataclass
class _PitchState:
    active: bool = False
    attack_count: int = 0
    release_count: int = 0
    refractory: int = 0
    onset_above: bool = False
    pending_attack: float = 0.0
    pending_attack_age: int = 0
    pending_onset_age: int = 1_000_000


class PolyphonicTracker:
    """Independent note-state decoder around causal acoustic evidence."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self.frontend = CausalDualResolutionFrontend(config)
        self._states = [_PitchState() for _ in range(config.note_count)]
        self._frame_index = 0
        self._global_attack = 0.0
        self._global_attack_age = config.global_attack_memory_frames + 1

    def reset(self) -> None:
        self.frontend.reset()
        self._states = [_PitchState() for _ in range(self.config.note_count)]
        self._frame_index = 0
        self._global_attack = 0.0
        self._global_attack_age = self.config.global_attack_memory_frames + 1

    def _normalized_optional(
        self,
        values: np.ndarray | None,
        frame_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        result = np.asarray(values, dtype=np.float32)
        expected = (frame_count, self.config.note_count)
        if result.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {result.shape}")
        if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
            raise ValueError(f"{name} must contain finite normalized values")
        return result

    @staticmethod
    def _fuse(
        acoustic: np.ndarray, tm: np.ndarray | None, weight: float
    ) -> np.ndarray:
        if tm is None or weight == 0.0:
            return acoustic
        return ((1.0 - weight) * acoustic + weight * tm).astype(np.float32)

    def _velocity(self, attack_energy: float) -> int:
        """Map acoustic attack energy only to MIDI velocity."""

        floor = self.config.velocity_floor
        span = self.config.velocity_reference - floor
        unit = 1.0 - np.exp(-max(attack_energy - floor, 0.0) / span)
        return int(np.clip(round(1.0 + 126.0 * np.sqrt(unit)), 1, 127))

    def _event_attack_energy(self, pitch_attack: float) -> float:
        """Keep chord-note velocity physical when local attribution is weak.

        The strongest simultaneous acoustic attack supplies only a floor; a
        pitch's own stronger attack remains untouched.  This compensates for
        a high chord tone being partially assigned to a lower fundamental,
        without ever consulting TM confidence.
        """

        return max(
            pitch_attack,
            self.config.global_velocity_mix * self._global_attack,
        )

    def process_evidence(
        self,
        evidence: SpectralEvidence,
        *,
        tm_activity: np.ndarray | None = None,
        tm_onset: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        """Decode frontend evidence, optionally refined by normalized TM scores."""

        frame_count = evidence.frame_count
        if evidence.activity.shape != (frame_count, self.config.note_count):
            raise ValueError("evidence activity shape does not match tracker config")
        tm_activity_array = self._normalized_optional(
            tm_activity, frame_count, "tm_activity"
        )
        tm_onset_array = self._normalized_optional(tm_onset, frame_count, "tm_onset")
        activity = self._fuse(
            evidence.activity,
            tm_activity_array,
            self.config.tm_activity_weight,
        )
        onset = self._fuse(
            evidence.onset, tm_onset_array, self.config.tm_onset_weight
        )

        events: list[NoteEvent] = []
        for row in range(frame_count):
            sample_index = int(evidence.sample_indices[row])
            frame_global_attack = float(np.max(evidence.attack_energy[row]))
            if frame_global_attack >= self._global_attack:
                self._global_attack = frame_global_attack
                self._global_attack_age = 0
            else:
                self._global_attack_age += 1
                if (
                    self._global_attack_age
                    > self.config.global_attack_memory_frames
                ):
                    self._global_attack = 0.0
            global_attack_is_fresh = (
                self._global_attack_age
                <= self.config.global_attack_memory_frames
                and self._global_attack >= self.config.global_attack_floor
            )
            for note_index, state in enumerate(self._states):
                pitch = self.config.midi_min + note_index
                acoustic_attack = float(evidence.attack_energy[row, note_index])
                if acoustic_attack > state.pending_attack:
                    state.pending_attack = acoustic_attack
                    state.pending_attack_age = 0
                else:
                    state.pending_attack_age += 1
                    if state.pending_attack_age > self.config.attack_memory_frames:
                        state.pending_attack = 0.0

                onset_above = float(onset[row, note_index]) >= (
                    self.config.retrigger_on if state.active else self.config.onset_on
                )
                rising_onset = onset_above and not state.onset_above
                state.onset_above = onset_above
                if rising_onset:
                    state.pending_onset_age = 0
                else:
                    state.pending_onset_age += 1
                if state.refractory > 0:
                    state.refractory -= 1

                note_activity = float(activity[row, note_index])
                if not state.active:
                    if note_activity >= self.config.activity_on:
                        state.attack_count = min(
                            state.attack_count + 1, self.config.attack_frames
                        )
                    else:
                        state.attack_count = 0
                    # Persistent activity is not enough to create a track.  A
                    # real note must also have a recent local attack, or a TM
                    # onset coincident with a real broadband/chord attack. This
                    # keeps onset creation tied to physical attack evidence.
                    # It stops a flickering classifier from resurrecting a decaying
                    # string without imposing any chord/polyphony limit.
                    attack_is_fresh = (
                        state.pending_attack_age
                        <= self.config.attack_memory_frames
                    )
                    onset_is_fresh = (
                        state.pending_onset_age
                        <= self.config.attack_memory_frames
                    )
                    if (
                        state.attack_count >= self.config.attack_frames
                        and (
                            (onset_is_fresh and global_attack_is_fresh)
                            or (
                                attack_is_fresh
                                and state.pending_attack
                                >= self.config.new_note_attack_floor
                            )
                        )
                    ):
                        events.append(
                            NoteEvent(
                                kind="note_on",
                                pitch=pitch,
                                velocity=self._velocity(
                                    self._event_attack_energy(
                                        state.pending_attack
                                    )
                                ),
                                sample_index=sample_index,
                                frame_index=self._frame_index,
                            )
                        )
                        state.active = True
                        state.attack_count = 0
                        state.release_count = 0
                        state.refractory = self.config.retrigger_refractory_frames
                        state.pending_attack = 0.0
                        state.pending_attack_age = 0
                        state.pending_onset_age = 1_000_000
                else:
                    # TM onset is evidence for *which* pitch was attacked; it
                    # cannot manufacture a physical retrigger by itself.  A
                    # fresh acoustic attack is required as a second, entirely
                    # independent condition.  Velocity is derived from that
                    # same acoustic quantity below, never from TM confidence.
                    attack_is_fresh = (
                        state.pending_attack_age
                        <= self.config.attack_memory_frames
                    )
                    if (
                        rising_onset
                        and state.refractory == 0
                        and (
                            (
                                attack_is_fresh
                                and state.pending_attack
                                >= self.config.retrigger_attack_floor
                            )
                            or global_attack_is_fresh
                        )
                    ):
                        events.append(
                            NoteEvent(
                                kind="note_off",
                                pitch=pitch,
                                velocity=0,
                                sample_index=sample_index,
                                frame_index=self._frame_index,
                            )
                        )
                        events.append(
                            NoteEvent(
                                kind="note_on",
                                pitch=pitch,
                                velocity=self._velocity(
                                    self._event_attack_energy(
                                        state.pending_attack
                                    )
                                ),
                                sample_index=sample_index,
                                frame_index=self._frame_index,
                            )
                        )
                        state.refractory = self.config.retrigger_refractory_frames
                        state.pending_attack = 0.0
                        state.pending_attack_age = 0
                        state.pending_onset_age = 1_000_000
                        state.release_count = 0
                    elif note_activity < self.config.activity_off:
                        state.release_count += 1
                        if state.release_count >= self.config.release_frames:
                            events.append(
                                NoteEvent(
                                    kind="note_off",
                                    pitch=pitch,
                                    velocity=0,
                                    sample_index=sample_index,
                                    frame_index=self._frame_index,
                                )
                            )
                            state.active = False
                            state.attack_count = 0
                            state.release_count = 0
                            state.refractory = 0
                            state.pending_onset_age = 1_000_000
                    else:
                        state.release_count = 0
            self._frame_index += 1
        return events

    def push(
        self,
        samples: np.ndarray,
        *,
        tm_activity: np.ndarray | None = None,
        tm_onset: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        """Process an arbitrary audio block and return new note events."""

        evidence = self.frontend.push(samples)
        return self.process_evidence(
            evidence, tm_activity=tm_activity, tm_onset=tm_onset
        )

    @property
    def active_pitches(self) -> tuple[int, ...]:
        return tuple(
            self.config.midi_min + index
            for index, state in enumerate(self._states)
            if state.active
        )


__all__ = [
    "CausalDualResolutionFrontend",
    "NoteEvent",
    "PolyphonicTracker",
    "SpectralEvidence",
    "TrackerConfig",
    "midi_to_hz",
]
