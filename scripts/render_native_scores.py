from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tmgm_rt.metrics import polyphonic_metrics
from tmgm_rt.midi import (
    NoteStateConfig,
    stabilize_frame_predictions,
    write_frame_predictions,
    write_teacher_events,
)
from tmgm_rt.native_dataset import read_native_dataset, unpack_binary_rows


def read_scores(path: Path) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    metadata: dict[str, str] = {}
    rows: list[list[str]] = []
    header: list[str] | None = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                text = line[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    metadata[key] = value
                continue
            values = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = values
            else:
                rows.append(values)
    if header is None or not rows:
        raise ValueError(f"empty native score file: {path}")
    outputs = int(metadata["outputs"])
    array = np.asarray(rows, dtype=np.int32)
    if array.shape != (int(metadata["frames"]), 1 + 2 * outputs):
        raise ValueError(f"score matrix shape disagrees with metadata: {path}")
    if not np.array_equal(array[:, 0], np.arange(array.shape[0])):
        raise ValueError(f"score frame indices are not contiguous: {path}")
    return metadata, array[:, 1 : 1 + outputs], array[:, 1 + outputs :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--onset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-events", type=Path)
    parser.add_argument("--source-wav", type=Path)
    args = parser.parse_args()

    dataset = read_native_dataset(args.dataset)
    activity_meta, _, activity = read_scores(args.activity)
    onset_meta, _, onset = read_scores(args.onset)
    if activity_meta["head"] != "activity" or onset_meta["head"] != "onset":
        raise ValueError("activity/onset score files were passed in the wrong order")
    note_count = dataset.header.note_count
    if activity.shape != onset.shape or activity.shape[1] != note_count:
        raise ValueError("native activity/onset dimensions disagree")

    raw = np.concatenate((activity, onset), axis=1).astype(np.uint32)
    stable = stabilize_frame_predictions(raw, note_count, NoteStateConfig())
    truth = np.concatenate(
        (
            unpack_binary_rows(dataset.activity_words, note_count),
            unpack_binary_rows(dataset.onset_words, note_count),
        ),
        axis=1,
    ).astype(np.uint32)
    metrics = polyphonic_metrics(truth, raw, note_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_seconds = dataset.header.hop_size / dataset.header.sample_rate
    write_frame_predictions(
        args.output_dir / "native-cuda-raw.mid",
        raw,
        dataset.header.midi_min,
        note_count,
        frame_seconds,
    )
    write_frame_predictions(
        args.output_dir / "native-cuda-stable.mid",
        stable,
        dataset.header.midi_min,
        note_count,
        frame_seconds,
    )
    if args.teacher_events is not None:
        write_teacher_events(
            args.output_dir / "neuralnote.mid",
            args.teacher_events,
            dataset.header.midi_min,
            dataset.header.midi_max,
        )
    if args.source_wav is not None:
        (args.output_dir / "source-wav.txt").write_text(
            str(args.source_wav), encoding="utf-8"
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
