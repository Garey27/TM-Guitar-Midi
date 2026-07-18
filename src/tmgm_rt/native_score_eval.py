from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tmgm_rt.native_dataset import NativeDatasetHeader, read_native_dataset_header


SCORE_MAGIC = "#TMGM_SCORES_V1"


@dataclass
class BinaryMetricAccumulator:
    frames: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    predicted_positives: int = 0
    target_positives: int = 0

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target_bool = np.asarray(target, dtype=bool)
        prediction_bool = np.asarray(prediction, dtype=bool)
        if target_bool.ndim != 2 or prediction_bool.shape != target_bool.shape:
            raise ValueError("target/prediction batches must have equal 2D shapes")
        self.frames += int(target_bool.shape[0])
        self.true_positives += int(np.logical_and(target_bool, prediction_bool).sum())
        self.false_positives += int(
            np.logical_and(np.logical_not(target_bool), prediction_bool).sum()
        )
        self.false_negatives += int(
            np.logical_and(target_bool, np.logical_not(prediction_bool)).sum()
        )
        self.predicted_positives += int(prediction_bool.sum())
        self.target_positives += int(target_bool.sum())

    def metrics(self) -> dict[str, int | float]:
        precision = self.true_positives / max(
            self.true_positives + self.false_positives, 1
        )
        recall = self.true_positives / max(
            self.true_positives + self.false_negatives, 1
        )
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        return {
            "frames": self.frames,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_mean_polyphony": self.predicted_positives
            / max(self.frames, 1),
            "target_mean_polyphony": self.target_positives / max(self.frames, 1),
        }


@dataclass(frozen=True)
class ScoreMetadata:
    head: str
    frames: int
    outputs: int
    midi_min: int
    sample_rate: int
    hop_size: int
    threshold: int
    raw: dict[str, str]


@dataclass(frozen=True)
class TrackRange:
    first: int
    last: int
    source: str


def _parse_metadata(values: dict[str, str], path: Path) -> ScoreMetadata:
    required = (
        "head",
        "frames",
        "outputs",
        "midi_min",
        "sample_rate",
        "hop_size",
        "threshold",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"score metadata is missing {missing}: {path}")
    head = values["head"]
    if head not in {"activity", "onset"}:
        raise ValueError(f"unsupported score head {head!r}: {path}")
    try:
        return ScoreMetadata(
            head=head,
            frames=int(values["frames"]),
            outputs=int(values["outputs"]),
            midi_min=int(values["midi_min"]),
            sample_rate=int(values["sample_rate"]),
            hop_size=int(values["hop_size"]),
            threshold=int(values["threshold"]),
            raw=dict(values),
        )
    except ValueError as error:
        raise ValueError(f"non-integer score metadata value: {path}") from error


def _score_batches(
    path: Path, *, batch_rows: int
) -> Iterator[tuple[ScoreMetadata, int, np.ndarray]]:
    """Yield predicted bits without retaining the score matrix in memory."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    metadata_values: dict[str, str] = {}
    metadata: ScoreMetadata | None = None
    header_seen = False
    magic_seen = False
    expected_frame = 0
    prediction_rows: list[np.ndarray] = []
    batch_first = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#"):
                if header_seen:
                    raise ValueError(
                        f"metadata after score header at line {line_number}: {path}"
                    )
                if line == SCORE_MAGIC:
                    magic_seen = True
                    continue
                text = line[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    metadata_values[key] = value
                continue

            if not header_seen:
                if not magic_seen:
                    raise ValueError(f"missing {SCORE_MAGIC} marker: {path}")
                metadata = _parse_metadata(metadata_values, path)
                columns = line.split("\t")
                expected_scores = [
                    f"score_{note}"
                    for note in range(
                        metadata.midi_min, metadata.midi_min + metadata.outputs
                    )
                ]
                expected_predictions = [
                    f"pred_{note}"
                    for note in range(
                        metadata.midi_min, metadata.midi_min + metadata.outputs
                    )
                ]
                if columns != ["frame", *expected_scores, *expected_predictions]:
                    raise ValueError(f"unexpected score TSV columns: {path}")
                header_seen = True
                continue

            assert metadata is not None
            values = np.fromstring(line, sep="\t", dtype=np.int64)
            expected_columns = 1 + 2 * metadata.outputs
            if values.size != expected_columns:
                raise ValueError(
                    f"score row {line_number} has {values.size} columns, "
                    f"expected {expected_columns}: {path}"
                )
            if int(values[0]) != expected_frame:
                raise ValueError(
                    f"score frame index {int(values[0])} is not expected "
                    f"{expected_frame}: {path}"
                )
            prediction = values[1 + metadata.outputs :]
            if not np.logical_or(prediction == 0, prediction == 1).all():
                raise ValueError(f"non-binary prediction at frame {expected_frame}: {path}")
            prediction_rows.append(prediction.astype(np.uint8, copy=False))
            expected_frame += 1

            if len(prediction_rows) == batch_rows:
                yield metadata, batch_first, np.stack(prediction_rows)
                batch_first = expected_frame
                prediction_rows.clear()

    if not header_seen or metadata is None:
        raise ValueError(f"empty native score file: {path}")
    if prediction_rows:
        yield metadata, batch_first, np.stack(prediction_rows)
    if expected_frame != metadata.frames:
        raise ValueError(
            f"score row count {expected_frame} disagrees with frames={metadata.frames}: {path}"
        )


def _load_track_ranges(sidecar_path: Path, frame_count: int) -> list[TrackRange]:
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    selected = metadata.get("selected_tracks")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"sidecar has no selected_tracks list: {sidecar_path}")

    result: list[TrackRange] = []
    first = 0
    for index, track in enumerate(selected):
        if not isinstance(track, dict):
            raise ValueError(f"selected_tracks[{index}] is not an object: {sidecar_path}")
        source = track.get("source")
        rows = track.get("rows")
        if not isinstance(source, str) or not source:
            raise ValueError(f"selected_tracks[{index}] has invalid source: {sidecar_path}")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError(f"selected_tracks[{index}] has invalid rows: {sidecar_path}")
        result.append(TrackRange(first=first, last=first + rows, source=source))
        first += rows
    if first != frame_count:
        raise ValueError(
            f"selected_tracks rows sum to {first}, expected {frame_count}: {sidecar_path}"
        )
    return result


def _validate_score_metadata(
    metadata: ScoreMetadata, header: NativeDatasetHeader, score_path: Path
) -> None:
    expected = (
        header.frame_count,
        header.note_count,
        header.midi_min,
        header.sample_rate,
        header.hop_size,
    )
    actual = (
        metadata.frames,
        metadata.outputs,
        metadata.midi_min,
        metadata.sample_rate,
        metadata.hop_size,
    )
    if actual != expected:
        raise ValueError(
            f"score metadata does not match dataset; got {actual}, expected {expected}: "
            f"{score_path}"
        )


def _unpack_label_batch(words: np.ndarray, note_count: int) -> np.ndarray:
    contiguous = np.ascontiguousarray(words, dtype="<u8")
    bytes_view = contiguous.view(np.uint8).reshape(contiguous.shape[0], -1)
    return np.unpackbits(bytes_view, axis=1, bitorder="little")[:, :note_count]


def evaluate_score_file(
    dataset_path: str | Path,
    score_path: str | Path,
    sidecar_path: str | Path,
    *,
    batch_rows: int = 4096,
) -> dict[str, object]:
    dataset_path = Path(dataset_path)
    score_path = Path(score_path)
    sidecar_path = Path(sidecar_path)
    header = read_native_dataset_header(dataset_path)
    track_ranges = _load_track_ranges(sidecar_path, header.frame_count)

    global_metrics = BinaryMetricAccumulator()
    source_metrics: dict[str, BinaryMetricAccumulator] = {}
    label_words: np.memmap | None = None
    metadata: ScoreMetadata | None = None
    track_index = 0

    for current_metadata, first, prediction in _score_batches(
        score_path, batch_rows=batch_rows
    ):
        if metadata is None:
            metadata = current_metadata
            _validate_score_metadata(metadata, header, score_path)
            label_offset = (
                header.activity_offset
                if metadata.head == "activity"
                else header.onset_offset
            )
            label_words = np.memmap(
                dataset_path,
                dtype="<u8",
                mode="r",
                offset=label_offset,
                shape=(header.frame_count, header.label_words_per_row),
            )
        elif current_metadata != metadata:
            raise AssertionError("score metadata changed while streaming")

        assert label_words is not None
        last = first + prediction.shape[0]
        target = _unpack_label_batch(label_words[first:last], header.note_count)
        global_metrics.update(target, prediction)

        cursor = first
        while cursor < last:
            while track_index < len(track_ranges) and cursor >= track_ranges[track_index].last:
                track_index += 1
            if track_index == len(track_ranges):
                raise AssertionError("score batch extends beyond selected track ranges")
            track = track_ranges[track_index]
            segment_last = min(last, track.last)
            local_first = cursor - first
            local_last = segment_last - first
            accumulator = source_metrics.setdefault(
                track.source, BinaryMetricAccumulator()
            )
            accumulator.update(
                target[local_first:local_last], prediction[local_first:local_last]
            )
            cursor = segment_last

    if metadata is None:
        raise ValueError(f"empty native score file: {score_path}")
    return {
        "head": metadata.head,
        "threshold": metadata.threshold,
        "aggregate": global_metrics.metrics(),
        "by_source": {
            source: accumulator.metrics()
            for source, accumulator in sorted(source_metrics.items())
        },
    }


def evaluate_score_files(
    dataset_path: str | Path,
    score_paths: Iterable[str | Path],
    sidecar_path: str | Path,
    *,
    batch_rows: int = 4096,
) -> dict[str, object]:
    dataset_path = Path(dataset_path)
    sidecar_path = Path(sidecar_path)
    heads: dict[str, object] = {}
    score_files: dict[str, str] = {}
    for path_value in score_paths:
        path = Path(path_value)
        result = evaluate_score_file(
            dataset_path, path, sidecar_path, batch_rows=batch_rows
        )
        head = str(result["head"])
        if head in heads:
            raise ValueError(f"duplicate {head} score head: {path}")
        score_files[head] = str(path.resolve())
        heads[head] = result
    if not heads:
        raise ValueError("at least one score file is required")
    header = read_native_dataset_header(dataset_path)
    return {
        "format": "TMGM_NATIVE_SCORE_EVAL_V1",
        "dataset": str(dataset_path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
        "score_files": score_files,
        "frames": header.frame_count,
        "outputs": header.note_count,
        "heads": heads,
    }
