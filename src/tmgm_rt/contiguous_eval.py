from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .native_dataset import NativeDatasetHeader, read_native_dataset_header
from .native_score_eval import ScoreMetadata, _score_batches


MANIFEST_SCHEMA = "tmgm-contiguous-eval-manifest-v1"
RESULT_SCHEMA = "tmgm-contiguous-temporal-eval-v1"


@dataclass(frozen=True)
class FeatureArtifact:
    dataset: Path
    metadata: Path


@dataclass(frozen=True)
class ManifestTrack:
    key: str
    source: str
    corpus_split: str
    evaluation_role: str
    track_id: str
    group: str
    wav: Path
    events: Path
    wav_sha256: str
    events_sha256: str
    feature_sets: dict[str, FeatureArtifact]


@dataclass(frozen=True)
class TeacherEvent:
    pitch: int
    start: int
    end: int
    active_previous_frame: bool
    retrigger: bool
    chord_onset: bool


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    result = value.lower()
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return result


def _resolve_manifest_path(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> list[ManifestTrack]:
    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid contiguous-eval manifest JSON: {manifest_path}") from error
    if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported contiguous-eval manifest schema: {manifest_path}")
    rows = raw.get("tracks")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest must contain a non-empty tracks list: {manifest_path}")

    result: list[ManifestTrack] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"tracks[{index}] must be an object")
        strings: dict[str, str] = {}
        for field in (
            "key",
            "source",
            "corpus_split",
            "evaluation_role",
            "id",
            "group",
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"tracks[{index}].{field} must be a non-empty string")
            strings[field] = value
        if strings["key"] in seen:
            raise ValueError(f"duplicate track key {strings['key']!r}")
        seen.add(strings["key"])

        features_value = row.get("feature_sets")
        if not isinstance(features_value, dict) or not features_value:
            raise ValueError(f"tracks[{index}].feature_sets must be a non-empty object")
        feature_sets: dict[str, FeatureArtifact] = {}
        for feature_name, feature_value in features_value.items():
            if not isinstance(feature_name, str) or not feature_name:
                raise ValueError(f"tracks[{index}] has an invalid feature-set name")
            if not isinstance(feature_value, dict):
                raise ValueError(
                    f"tracks[{index}].feature_sets.{feature_name} must be an object"
                )
            dataset = _resolve_manifest_path(
                feature_value.get("dataset"),
                manifest_path.parent,
                f"tracks[{index}].feature_sets.{feature_name}.dataset",
            )
            metadata = _resolve_manifest_path(
                feature_value.get("metadata"),
                manifest_path.parent,
                f"tracks[{index}].feature_sets.{feature_name}.metadata",
            )
            feature_sets[feature_name] = FeatureArtifact(dataset, metadata)

        result.append(
            ManifestTrack(
                key=strings["key"],
                source=strings["source"],
                corpus_split=strings["corpus_split"],
                evaluation_role=strings["evaluation_role"],
                track_id=strings["id"],
                group=strings["group"],
                wav=_resolve_manifest_path(
                    row.get("wav"), manifest_path.parent, f"tracks[{index}].wav"
                ),
                events=_resolve_manifest_path(
                    row.get("events"), manifest_path.parent, f"tracks[{index}].events"
                ),
                wav_sha256=_require_sha256(
                    row.get("wav_sha256"), f"tracks[{index}].wav_sha256"
                ),
                events_sha256=_require_sha256(
                    row.get("events_sha256"), f"tracks[{index}].events_sha256"
                ),
                feature_sets=feature_sets,
            )
        )
    return result


def _require_file_identity(path: Path, expected_sha256: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {actual}: {path}"
        )


def _validate_export_metadata(
    track: ManifestTrack, feature: FeatureArtifact, header: NativeDatasetHeader
) -> dict[str, Any]:
    if not feature.dataset.is_file():
        raise FileNotFoundError(f"native dataset does not exist: {feature.dataset}")
    if not feature.metadata.is_file():
        raise FileNotFoundError(f"native dataset metadata does not exist: {feature.metadata}")
    try:
        metadata = json.loads(feature.metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid native dataset metadata: {feature.metadata}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"native dataset metadata must be an object: {feature.metadata}")
    if metadata.get("schema") != "tmgm-native-wav-inference-v1":
        raise ValueError(f"dataset is not a causal WAV inference export: {feature.metadata}")
    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, dict):
        raise ValueError(f"native export metadata has no input identity: {feature.metadata}")
    input_path = input_metadata.get("path")
    if not isinstance(input_path, str) or Path(input_path).resolve() != track.wav:
        raise ValueError(f"native export input path disagrees with manifest: {feature.metadata}")
    if input_metadata.get("sha256") != track.wav_sha256:
        raise ValueError(f"native export input SHA-256 disagrees with manifest: {feature.metadata}")

    exported_header = metadata.get("header")
    if not isinstance(exported_header, dict):
        raise ValueError(f"native export metadata has no header: {feature.metadata}")
    for field, expected in (
        ("frame_count", header.frame_count),
        ("feature_count", header.feature_count),
        ("note_count", header.note_count),
        ("midi_min", header.midi_min),
        ("midi_max", header.midi_max),
        ("sample_rate", header.sample_rate),
        ("hop_size", header.hop_size),
    ):
        if exported_header.get(field) != expected:
            raise ValueError(
                f"native export metadata {field} disagrees with dataset: {feature.metadata}"
            )
    samples = metadata.get("resampled_mono_samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError(f"invalid resampled_mono_samples: {feature.metadata}")
    expected_frames = (samples + header.hop_size - 1) // header.hop_size
    if expected_frames != header.frame_count:
        raise ValueError(
            f"native export length mismatch: {samples} samples imply {expected_frames} "
            f"frames, dataset has {header.frame_count}: {feature.metadata}"
        )
    causality = metadata.get("causality")
    if causality != {
        "all_frames_in_source_order": True,
        "lookahead_frames": 0,
        "strictly_causal": True,
    }:
        raise ValueError(f"native export is not declared strictly causal: {feature.metadata}")
    return metadata


def _validate_score_metadata(
    metadata: ScoreMetadata,
    header: NativeDatasetHeader,
    expected_head: str,
    score_path: Path,
) -> None:
    if metadata.head != expected_head:
        raise ValueError(
            f"expected {expected_head} scores, got {metadata.head}: {score_path}"
        )
    actual = (
        metadata.frames,
        metadata.outputs,
        metadata.midi_min,
        metadata.sample_rate,
        metadata.hop_size,
    )
    expected = (
        header.frame_count,
        header.note_count,
        header.midi_min,
        header.sample_rate,
        header.hop_size,
    )
    if actual != expected:
        raise ValueError(
            f"score metadata does not match contiguous dataset; got {actual}, "
            f"expected {expected}: {score_path}"
        )


def load_score_predictions(
    path: str | Path,
    header: NativeDatasetHeader,
    expected_head: str,
    *,
    batch_rows: int = 4096,
) -> tuple[np.ndarray, ScoreMetadata]:
    score_path = Path(path).resolve()
    batches: list[np.ndarray] = []
    metadata: ScoreMetadata | None = None
    for current, _, prediction in _score_batches(score_path, batch_rows=batch_rows):
        if metadata is None:
            metadata = current
            _validate_score_metadata(metadata, header, expected_head, score_path)
        elif current != metadata:
            raise AssertionError("score metadata changed while streaming")
        batches.append(prediction)
    if metadata is None:
        raise ValueError(f"empty score file: {score_path}")
    result = np.concatenate(batches, axis=0)
    if result.shape != (header.frame_count, header.note_count):
        raise AssertionError("validated score rows have an unexpected shape")
    return result, metadata


def _read_teacher_rows(path: Path) -> list[tuple[int, int, int]]:
    required = {"pitch", "start_frame", "end_frame"}
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"teacher events are missing {sorted(required)}: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                pitch = int(row["pitch"])
                start = int(row["start_frame"])
                end = int(row["end_frame"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid teacher event at line {line}: {path}") from error
            if start < 0 or end < start:
                raise ValueError(f"invalid teacher interval at line {line}: {path}")
            rows.append((pitch, start, end))
    return rows


def build_teacher_targets(
    events_path: str | Path,
    *,
    frame_count: int,
    midi_min: int,
    midi_max: int,
    onset_delay_frames: int,
    onset_width_frames: int,
    retrigger_silence_frames: int,
    chord_window_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[TeacherEvent]]:
    if onset_delay_frames < 0:
        raise ValueError("onset_delay_frames cannot be negative")
    if onset_width_frames <= 0:
        raise ValueError("onset_width_frames must be positive")
    if retrigger_silence_frames < 0:
        raise ValueError("retrigger_silence_frames cannot be negative")
    if chord_window_frames < 0:
        raise ValueError("chord_window_frames cannot be negative")
    note_count = midi_max - midi_min + 1
    activity = np.zeros((frame_count, note_count), dtype=np.uint8)
    onset = np.zeros_like(activity)
    retained: list[tuple[int, int, int]] = []
    for pitch, raw_start, raw_end in _read_teacher_rows(Path(events_path)):
        if pitch < midi_min or pitch > midi_max:
            continue
        if raw_start >= frame_count:
            continue
        start = raw_start
        end = min(frame_count, raw_end)
        column = pitch - midi_min
        activity[start:end, column] = 1
        onset_start = min(frame_count, start + onset_delay_frames)
        onset[onset_start : min(frame_count, onset_start + onset_width_frames), column] = 1
        retained.append((pitch, start, end))

    retained.sort(key=lambda row: (row[1], row[0], row[2]))
    starts = np.asarray([row[1] for row in retained], dtype=np.int64)
    pitches = np.asarray([row[0] for row in retained], dtype=np.int16)
    previous_end: dict[int, int] = {}
    events: list[TeacherEvent] = []
    for index, (pitch, start, end) in enumerate(retained):
        prior_end = previous_end.get(pitch)
        active_previous = start > 0 and bool(activity[start - 1, pitch - midi_min])
        gap = None if prior_end is None else start - prior_end
        retrigger = active_previous or (
            gap is not None and gap < retrigger_silence_frames
        )
        chord = bool(
            np.any(
                (np.abs(starts - start) <= chord_window_frames)
                & (pitches != pitch)
            )
        )
        events.append(
            TeacherEvent(
                pitch=pitch,
                start=start,
                end=end,
                active_previous_frame=active_previous,
                retrigger=retrigger,
                chord_onset=chord,
            )
        )
        previous_end[pitch] = max(end, prior_end if prior_end is not None else end)
    return activity, onset, events


def _binary_counts(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    frame_mask: np.ndarray | None = None,
    note_mask: np.ndarray | None = None,
) -> dict[str, int]:
    target_bool = np.asarray(target, dtype=bool)
    prediction_bool = np.asarray(prediction, dtype=bool)
    if target_bool.ndim != 2 or prediction_bool.shape != target_bool.shape:
        raise ValueError("target and prediction must have equal two-dimensional shapes")
    if frame_mask is not None:
        mask = np.asarray(frame_mask, dtype=bool)
        if mask.shape != (target_bool.shape[0],):
            raise ValueError("frame mask has the wrong shape")
        target_bool = target_bool[mask]
        prediction_bool = prediction_bool[mask]
    if note_mask is not None:
        mask = np.asarray(note_mask, dtype=bool)
        if mask.shape != (target_bool.shape[1],):
            raise ValueError("note mask has the wrong shape")
        target_bool = target_bool[:, mask]
        prediction_bool = prediction_bool[:, mask]
    return {
        "frames": int(target_bool.shape[0]),
        "cells": int(target_bool.size),
        "true_positives": int(np.logical_and(target_bool, prediction_bool).sum()),
        "false_positives": int(np.logical_and(~target_bool, prediction_bool).sum()),
        "false_negatives": int(np.logical_and(target_bool, ~prediction_bool).sum()),
        "predicted_positives": int(prediction_bool.sum()),
        "target_positives": int(target_bool.sum()),
    }


def _with_binary_rates(counts: dict[str, int]) -> dict[str, int | float]:
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_mean_polyphony": counts["predicted_positives"]
        / max(counts["frames"], 1),
        "target_mean_polyphony": counts["target_positives"]
        / max(counts["frames"], 1),
    }


def _extract_prediction_events(
    prediction: np.ndarray, midi_min: int
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for column in range(prediction.shape[1]):
        active = np.asarray(prediction[:, column], dtype=bool)
        starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
        result.extend((midi_min + column, int(frame)) for frame in starts)
    result.sort(key=lambda item: (item[0], item[1]))
    return result


def _match_events(
    target_events: list[tuple[int, int, int]],
    predicted_events: list[tuple[int, int]],
    tolerance: int,
) -> tuple[set[int], set[int]]:
    if tolerance < 0:
        raise ValueError("event tolerance cannot be negative")
    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()
    pitches = sorted(
        {event[0] for event in target_events} | {event[0] for event in predicted_events}
    )
    for pitch in pitches:
        targets = sorted(
            ((frame, index) for index, (p, frame, _) in enumerate(target_events) if p == pitch)
        )
        predictions = sorted(
            ((frame, index) for index, (p, frame) in enumerate(predicted_events) if p == pitch)
        )
        target_index = 0
        prediction_index = 0
        while target_index < len(targets) and prediction_index < len(predictions):
            target_frame, target_original = targets[target_index]
            predicted_frame, prediction_original = predictions[prediction_index]
            if predicted_frame < target_frame - tolerance:
                prediction_index += 1
            elif predicted_frame > target_frame + tolerance:
                target_index += 1
            else:
                matched_targets.add(target_original)
                matched_predictions.add(prediction_original)
                target_index += 1
                prediction_index += 1
    return matched_targets, matched_predictions


def _event_metrics(
    teacher_events: list[TeacherEvent],
    predicted_events: list[tuple[int, int]],
    *,
    delay_frames: int,
    tolerances: Iterable[int],
    midi_min: int,
    low_midi_max: int,
) -> dict[str, Any]:
    target = [
        (event.pitch, event.start + delay_frames, index)
        for index, event in enumerate(teacher_events)
    ]
    subsets = {
        "low_notes": {i for i, event in enumerate(teacher_events) if event.pitch <= low_midi_max},
        "retrigger": {i for i, event in enumerate(teacher_events) if event.retrigger},
        "active_previous_frame": {
            i for i, event in enumerate(teacher_events) if event.active_previous_frame
        },
        "chord_onset": {i for i, event in enumerate(teacher_events) if event.chord_onset},
        "single_onset": {i for i, event in enumerate(teacher_events) if not event.chord_onset},
    }
    results: dict[str, Any] = {}
    for tolerance in sorted(set(int(value) for value in tolerances) | {0}):
        matched_target, matched_prediction = _match_events(
            target, predicted_events, tolerance
        )
        tp = len(matched_target)
        fp = len(predicted_events) - len(matched_prediction)
        fn = len(target) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        target_subsets = {}
        for name, indices in subsets.items():
            matched = len(indices & matched_target)
            target_subsets[name] = {
                "targets": len(indices),
                "matched_targets": matched,
                "recall": matched / max(len(indices), 1),
            }

        low_target_indices = subsets["low_notes"]
        low_predictions = [
            event for event in predicted_events if midi_min <= event[0] <= low_midi_max
        ]
        low_targets = [target[index] for index in sorted(low_target_indices)]
        low_matched_targets, low_matched_predictions = _match_events(
            low_targets, low_predictions, tolerance
        )
        low_tp = len(low_matched_targets)
        low_fp = len(low_predictions) - len(low_matched_predictions)
        low_fn = len(low_targets) - low_tp
        low_precision = low_tp / max(low_tp + low_fp, 1)
        low_recall = low_tp / max(low_tp + low_fn, 1)
        low_f1 = 2.0 * low_precision * low_recall / max(
            low_precision + low_recall, 1.0e-12
        )
        results[str(tolerance)] = {
            "tolerance_frames": tolerance,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "target_events": len(target),
            "predicted_events": len(predicted_events),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "low_notes_full": {
                "true_positives": low_tp,
                "false_positives": low_fp,
                "false_negatives": low_fn,
                "target_events": len(low_targets),
                "predicted_events": len(low_predictions),
                "precision": low_precision,
                "recall": low_recall,
                "f1": low_f1,
            },
            "target_subsets": target_subsets,
        }
    return results


def _frame_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    midi_min: int,
    low_midi_max: int,
) -> dict[str, Any]:
    polyphony = target.sum(axis=1)
    note_numbers = np.arange(midi_min, midi_min + target.shape[1])
    low_mask = note_numbers <= low_midi_max
    by_polyphony: dict[str, Any] = {}
    categories = {
        "0": polyphony == 0,
        "1": polyphony == 1,
        "2": polyphony == 2,
        "3": polyphony == 3,
        "4+": polyphony >= 4,
    }
    for name, mask in categories.items():
        by_polyphony[name] = _with_binary_rates(
            _binary_counts(target, prediction, frame_mask=mask)
        )
    return {
        "all": _with_binary_rates(_binary_counts(target, prediction)),
        "single_note_frames": _with_binary_rates(
            _binary_counts(target, prediction, frame_mask=polyphony == 1)
        ),
        "polyphonic_frames": _with_binary_rates(
            _binary_counts(target, prediction, frame_mask=polyphony >= 2)
        ),
        "low_notes_midi_40_59": _with_binary_rates(
            _binary_counts(target, prediction, note_mask=low_mask)
        ),
        "by_target_polyphony": by_polyphony,
    }


def evaluate_track(
    track: ManifestTrack,
    feature_set: str,
    activity_score_path: str | Path,
    onset_score_path: str | Path,
    *,
    training_onset_delay_frames: int = 2,
    onset_width_frames: int = 3,
    target_aligned_tolerances: Iterable[int] = (2, 3, 4),
    wall_clock_tolerances: Iterable[int] = (2, 3, 4, 6),
    retrigger_silence_frames: int = 3,
    chord_window_frames: int = 3,
    low_midi_max: int = 59,
    batch_rows: int = 4096,
) -> dict[str, Any]:
    if feature_set not in track.feature_sets:
        raise ValueError(f"track {track.key!r} has no feature set {feature_set!r}")
    _require_file_identity(track.wav, track.wav_sha256, "manifest WAV")
    _require_file_identity(track.events, track.events_sha256, "teacher events")
    feature = track.feature_sets[feature_set]
    header = read_native_dataset_header(feature.dataset)
    _validate_export_metadata(track, feature, header)
    if low_midi_max < header.midi_min:
        raise ValueError("low_midi_max must not be below the model MIDI range")
    low_midi_max = min(low_midi_max, header.midi_max)

    activity_prediction, activity_metadata = load_score_predictions(
        activity_score_path,
        header,
        "activity",
        batch_rows=batch_rows,
    )
    onset_prediction, onset_metadata = load_score_predictions(
        onset_score_path,
        header,
        "onset",
        batch_rows=batch_rows,
    )
    activity_target, onset_target, teacher_events = build_teacher_targets(
        track.events,
        frame_count=header.frame_count,
        midi_min=header.midi_min,
        midi_max=header.midi_max,
        onset_delay_frames=training_onset_delay_frames,
        onset_width_frames=onset_width_frames,
        retrigger_silence_frames=retrigger_silence_frames,
        chord_window_frames=chord_window_frames,
    )
    predicted_events = _extract_prediction_events(onset_prediction, header.midi_min)
    return {
        "key": track.key,
        "source": track.source,
        "corpus_split": track.corpus_split,
        "evaluation_role": track.evaluation_role,
        "id": track.track_id,
        "group": track.group,
        "frames": header.frame_count,
        "seconds": header.frame_count * header.hop_size / header.sample_rate,
        "midi_min": header.midi_min,
        "midi_max": header.midi_max,
        "score_thresholds": {
            "activity": activity_metadata.threshold,
            "onset": onset_metadata.threshold,
        },
        "teacher_events": len(teacher_events),
        "predicted_onset_events": len(predicted_events),
        "activity": _frame_metrics(
            activity_target,
            activity_prediction,
            midi_min=header.midi_min,
            low_midi_max=low_midi_max,
        ),
        "onset": {
            "target_aligned": {
                "description": "teacher onset shifted by the training-label delay",
                "delay_frames": training_onset_delay_frames,
                "frame_exact": _frame_metrics(
                    onset_target,
                    onset_prediction,
                    midi_min=header.midi_min,
                    low_midi_max=low_midi_max,
                ),
                "events": _event_metrics(
                    teacher_events,
                    predicted_events,
                    delay_frames=training_onset_delay_frames,
                    tolerances=target_aligned_tolerances,
                    midi_min=header.midi_min,
                    low_midi_max=low_midi_max,
                ),
            },
            "wall_clock": {
                "description": "predicted onset against the unshifted teacher event",
                "delay_frames": 0,
                "events": _event_metrics(
                    teacher_events,
                    predicted_events,
                    delay_frames=0,
                    tolerances=wall_clock_tolerances,
                    midi_min=header.midi_min,
                    low_midi_max=low_midi_max,
                ),
            },
        },
    }


def _sum_binary_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "frames",
        "cells",
        "true_positives",
        "false_positives",
        "false_negatives",
        "predicted_positives",
        "target_positives",
    )
    counts = {field: sum(int(value[field]) for value in values) for field in fields}
    return _with_binary_rates(counts)


def _sum_frame_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        name: _sum_binary_metrics([value[name] for value in values])
        for name in (
            "all",
            "single_note_frames",
            "polyphonic_frames",
            "low_notes_midi_40_59",
        )
    }
    result["by_target_polyphony"] = {
        category: _sum_binary_metrics(
            [value["by_target_polyphony"][category] for value in values]
        )
        for category in ("0", "1", "2", "3", "4+")
    }
    return result


def _sum_event_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(values[0], key=int)
    if any(sorted(value, key=int) != keys for value in values[1:]):
        raise ValueError("tracks have inconsistent event tolerance sets")
    result: dict[str, Any] = {}
    for key in keys:
        rows = [value[key] for value in values]
        tp = sum(int(row["true_positives"]) for row in rows)
        fp = sum(int(row["false_positives"]) for row in rows)
        fn = sum(int(row["false_negatives"]) for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        low_rows = [row["low_notes_full"] for row in rows]
        low_tp = sum(int(row["true_positives"]) for row in low_rows)
        low_fp = sum(int(row["false_positives"]) for row in low_rows)
        low_fn = sum(int(row["false_negatives"]) for row in low_rows)
        low_precision = low_tp / max(low_tp + low_fp, 1)
        low_recall = low_tp / max(low_tp + low_fn, 1)
        low_f1 = 2.0 * low_precision * low_recall / max(
            low_precision + low_recall, 1.0e-12
        )
        subset_names = rows[0]["target_subsets"].keys()
        subsets: dict[str, Any] = {}
        for name in subset_names:
            targets = sum(int(row["target_subsets"][name]["targets"]) for row in rows)
            matched = sum(
                int(row["target_subsets"][name]["matched_targets"]) for row in rows
            )
            subsets[name] = {
                "targets": targets,
                "matched_targets": matched,
                "recall": matched / max(targets, 1),
            }
        result[key] = {
            "tolerance_frames": int(key),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "target_events": sum(int(row["target_events"]) for row in rows),
            "predicted_events": sum(int(row["predicted_events"]) for row in rows),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "low_notes_full": {
                "true_positives": low_tp,
                "false_positives": low_fp,
                "false_negatives": low_fn,
                "target_events": sum(int(row["target_events"]) for row in low_rows),
                "predicted_events": sum(int(row["predicted_events"]) for row in low_rows),
                "precision": low_precision,
                "recall": low_recall,
                "f1": low_f1,
            },
            "target_subsets": subsets,
        }
    return result


def _aggregate_tracks(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = [track["onset"]["target_aligned"] for track in tracks]
    wall_clock = [track["onset"]["wall_clock"] for track in tracks]
    return {
        "tracks": len(tracks),
        "frames": sum(int(track["frames"]) for track in tracks),
        "seconds": sum(float(track["seconds"]) for track in tracks),
        "teacher_events": sum(int(track["teacher_events"]) for track in tracks),
        "predicted_onset_events": sum(
            int(track["predicted_onset_events"]) for track in tracks
        ),
        "activity": _sum_frame_metrics([track["activity"] for track in tracks]),
        "onset": {
            "target_aligned": {
                "delay_frames": aligned[0]["delay_frames"],
                "frame_exact": _sum_frame_metrics(
                    [value["frame_exact"] for value in aligned]
                ),
                "events": _sum_event_metrics([value["events"] for value in aligned]),
            },
            "wall_clock": {
                "delay_frames": 0,
                "events": _sum_event_metrics(
                    [value["events"] for value in wall_clock]
                ),
            },
        },
    }


def evaluate_manifest_scores(
    manifest_path: str | Path,
    feature_set: str,
    scores_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    tracks = load_manifest(manifest_path)
    root = Path(scores_root).resolve()
    track_results = []
    for track in tracks:
        track_results.append(
            evaluate_track(
                track,
                feature_set,
                root / f"{track.key}.activity.tsv",
                root / f"{track.key}.onset.tsv",
                **kwargs,
            )
        )
    by_source = {
        source: _aggregate_tracks(
            [track for track in track_results if track["source"] == source]
        )
        for source in sorted({track["source"] for track in track_results})
    }
    return {
        "format": RESULT_SCHEMA,
        "manifest": str(Path(manifest_path).resolve()),
        "feature_set": feature_set,
        "scores_root": str(root),
        "configuration": {
            key: value
            for key, value in kwargs.items()
            if key != "batch_rows"
        },
        "aggregate": _aggregate_tracks(track_results),
        "by_source": by_source,
        "tracks": {track["key"]: track for track in track_results},
    }
