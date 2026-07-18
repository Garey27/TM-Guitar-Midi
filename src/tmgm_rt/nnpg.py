from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import struct

import numpy as np


_HEADER = struct.Struct("<4sIIIQIIIIIIfff d I Q Q Q 41s 119s")


@dataclass(frozen=True)
class PosteriorHeader:
    version: int
    header_bytes: int
    frame_bytes: int
    frame_count: int
    note_bins: int
    onset_bins: int
    contour_bins: int
    hop_size: int
    sample_rate: int
    quantization_levels: int
    note_sensitivity: float
    split_sensitivity: float
    minimum_note_duration_ms: float
    source_sample_rate: float
    source_channels: int
    source_samples: int
    source_bytes: int
    source_fnv1a64: int
    neuralnote_commit: str


@dataclass(frozen=True)
class Posteriorgram:
    header: PosteriorHeader
    notes: np.ndarray
    onsets: np.ndarray
    contours: np.ndarray


@dataclass(frozen=True)
class TeacherTargets:
    activity: np.ndarray
    onset: np.ndarray
    velocity: np.ndarray


def read_nnpg(path: str | Path) -> Posteriorgram:
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(_HEADER.size)
        if len(raw) != _HEADER.size:
            raise ValueError(f"truncated NNPG header: {path}")
        values = _HEADER.unpack(raw)
        if values[0] != b"NNPG":
            raise ValueError(f"bad NNPG magic: {path}")
        if values[1] != 1 or values[2] != 256:
            raise ValueError(f"unsupported NNPG version/header: {values[1:3]}")
        header = PosteriorHeader(
            version=values[1],
            header_bytes=values[2],
            frame_bytes=values[3],
            frame_count=values[4],
            note_bins=values[5],
            onset_bins=values[6],
            contour_bins=values[7],
            hop_size=values[8],
            sample_rate=values[9],
            quantization_levels=values[10],
            note_sensitivity=values[11],
            split_sensitivity=values[12],
            minimum_note_duration_ms=values[13],
            source_sample_rate=values[14],
            source_channels=values[15],
            source_samples=values[16],
            source_bytes=values[17],
            source_fnv1a64=values[18],
            neuralnote_commit=values[19].split(b"\0", 1)[0].decode("ascii"),
        )
        if header.frame_bytes != (
            header.note_bins + header.onset_bins + header.contour_bins
        ):
            raise ValueError("NNPG frame dimensions disagree")
        stream.seek(header.header_bytes)
        payload = np.fromfile(stream, dtype=np.uint8)

    expected = header.frame_count * header.frame_bytes
    if payload.size != expected:
        raise ValueError(
            f"NNPG payload size {payload.size} != expected {expected}: {path}"
        )
    frames = payload.reshape(header.frame_count, header.frame_bytes).astype(
        np.float32
    ) / float(header.quantization_levels)
    note_end = header.note_bins
    onset_end = note_end + header.onset_bins
    return Posteriorgram(
        header=header,
        notes=frames[:, :note_end],
        onsets=frames[:, note_end:onset_end],
        contours=frames[:, onset_end:],
    )


def event_targets(
    events_path: str | Path,
    frame_count: int,
    midi_min: int,
    midi_max: int,
    onset_width_frames: int = 2,
    onset_delay_frames: int = 0,
) -> TeacherTargets:
    if onset_width_frames <= 0:
        raise ValueError("onset_width_frames must be positive")
    if onset_delay_frames < 0:
        raise ValueError("onset_delay_frames cannot be negative")
    note_count = midi_max - midi_min + 1
    activity = np.zeros((frame_count, note_count), dtype=np.uint32)
    onset = np.zeros_like(activity)
    velocity = np.zeros((frame_count, note_count), dtype=np.float32)
    with Path(events_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pitch = int(row["pitch"])
            if pitch < midi_min or pitch > midi_max:
                continue
            column = pitch - midi_min
            start = max(0, min(frame_count, int(row["start_frame"])))
            end = max(start, min(frame_count, int(row["end_frame"])))
            amplitude = float(row["amplitude"])
            activity[start:end, column] = 1
            velocity[start:end, column] = np.maximum(
                velocity[start:end, column], amplitude
            )
            onset_start = min(frame_count, start + onset_delay_frames)
            onset[
                onset_start : min(frame_count, onset_start + onset_width_frames),
                column,
            ] = 1
    return TeacherTargets(activity=activity, onset=onset, velocity=velocity)


def guitar_teacher_slice(
    posterior: Posteriorgram, midi_min: int, midi_max: int
) -> tuple[np.ndarray, np.ndarray]:
    # Basic Pitch output bin zero is MIDI 21.
    begin = midi_min - 21
    end = midi_max - 21 + 1
    if begin < 0 or end > posterior.header.note_bins:
        raise ValueError("requested MIDI range is outside NeuralNote output")
    return posterior.notes[:, begin:end], posterior.onsets[:, begin:end]
