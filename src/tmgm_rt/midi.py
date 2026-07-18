from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import mido
import numpy as np


@dataclass(frozen=True)
class NoteStateConfig:
    """Causal per-pitch state machine; no chord or global-polyphony rules."""

    attack_frames: int = 2
    release_frames: int = 4
    retrigger_refractory_frames: int = 6

    def __post_init__(self) -> None:
        if self.attack_frames < 1:
            raise ValueError("attack_frames must be at least one")
        if self.release_frames < 1:
            raise ValueError("release_frames must be at least one")
        if self.retrigger_refractory_frames < 0:
            raise ValueError("retrigger refractory must be non-negative")


def stabilize_frame_predictions(
    prediction: np.ndarray,
    note_count: int,
    config: NoteStateConfig = NoteStateConfig(),
) -> np.ndarray:
    """Turn flickering frame decisions into causal note states.

    Each pitch is independent. A note needs consecutive activity frames to
    start, survives short negative gaps, and retriggers only on a new onset
    rising edge outside the refractory interval.
    """
    values = np.asarray(prediction)
    if values.ndim != 2 or values.shape[1] < note_count:
        raise ValueError("prediction shape does not contain the activity head")
    activity = values[:, :note_count].astype(bool)
    onset = (
        values[:, note_count : 2 * note_count].astype(bool)
        if values.shape[1] >= 2 * note_count
        else np.zeros_like(activity)
    )
    result = np.zeros((values.shape[0], 2 * note_count), dtype=np.uint32)
    active = np.zeros(note_count, dtype=bool)
    attack_count = np.zeros(note_count, dtype=np.int32)
    release_count = np.zeros(note_count, dtype=np.int32)
    refractory = np.zeros(note_count, dtype=np.int32)
    previous_onset = np.zeros(note_count, dtype=bool)

    for frame in range(values.shape[0]):
        refractory = np.maximum(refractory - 1, 0)
        rising_onset = np.logical_and(onset[frame], np.logical_not(previous_onset))
        for note in range(note_count):
            if active[note]:
                if activity[frame, note]:
                    release_count[note] = 0
                else:
                    release_count[note] += 1
                    if release_count[note] >= config.release_frames:
                        active[note] = False
                        release_count[note] = 0
                        attack_count[note] = 0
                if active[note] and rising_onset[note] and refractory[note] == 0:
                    result[frame, note_count + note] = 1
                    refractory[note] = config.retrigger_refractory_frames
            else:
                if activity[frame, note]:
                    attack_count[note] += 1
                else:
                    attack_count[note] = 0
                if attack_count[note] >= config.attack_frames:
                    active[note] = True
                    attack_count[note] = 0
                    release_count[note] = 0
                    refractory[note] = config.retrigger_refractory_frames
                    result[frame, note_count + note] = 1
            if active[note]:
                result[frame, note] = 1
        previous_onset = onset[frame]
    return result


def write_frame_predictions(
    path: str | Path,
    prediction: np.ndarray,
    midi_min: int,
    note_count: int,
    frame_seconds: float,
    velocity: int = 80,
) -> None:
    activity = prediction[:, :note_count].astype(bool)
    has_onset = prediction.shape[1] >= 2 * note_count
    onset = (
        prediction[:, note_count : 2 * note_count].astype(bool)
        if has_onset
        else np.zeros_like(activity)
    )
    events: list[tuple[int, int, int, int]] = []
    active = np.zeros(note_count, dtype=bool)
    previous_onset = np.zeros(note_count, dtype=bool)
    ticks_per_second = 960.0
    for frame in range(prediction.shape[0]):
        tick = round(frame * frame_seconds * ticks_per_second)
        for note in range(note_count):
            if active[note] and not activity[frame, note]:
                events.append((tick, 0, midi_min + note, 0))
                active[note] = False
            onset_rising = onset[frame, note] and not previous_onset[note]
            if onset_rising and active[note]:
                events.append((tick, 0, midi_min + note, 0))
                active[note] = False
            if activity[frame, note] and not active[note]:
                events.append((tick, 1, midi_min + note, velocity))
                active[note] = True
        previous_onset = onset[frame]
    final_tick = round(prediction.shape[0] * frame_seconds * ticks_per_second)
    for note in np.flatnonzero(active):
        events.append((final_tick, 0, midi_min + int(note), 0))
    events.sort(key=lambda event: (event[0], event[1]))

    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    previous_tick = 0
    for tick, is_on, pitch, event_velocity in events:
        delta = tick - previous_tick
        previous_tick = tick
        if is_on:
            track.append(
                mido.Message(
                    "note_on", note=pitch, velocity=event_velocity, time=delta
                )
            )
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=max(0, final_tick - previous_tick)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def write_teacher_events(
    path: str | Path,
    events_path: str | Path,
    midi_min: int,
    midi_max: int,
) -> None:
    """Write one valid MIDI note per decoded NeuralNote TSV event."""
    events: list[tuple[int, int, int, int]] = []
    ticks_per_second = 960.0
    final_tick = 0
    with Path(events_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pitch = int(row["pitch"])
            if pitch < midi_min or pitch > midi_max:
                continue
            start_tick = max(0, round(float(row["start_sec"]) * ticks_per_second))
            end_tick = max(start_tick + 1, round(float(row["end_sec"]) * ticks_per_second))
            velocity = int(np.clip(round(float(row["amplitude"]) * 127.0), 1, 127))
            events.append((start_tick, 1, pitch, velocity))
            events.append((end_tick, 0, pitch, 0))
            final_tick = max(final_tick, end_tick)
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    previous_tick = 0
    for tick, is_on, pitch, velocity in events:
        delta = tick - previous_tick
        previous_tick = tick
        if is_on:
            track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=max(0, final_tick - previous_tick)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
