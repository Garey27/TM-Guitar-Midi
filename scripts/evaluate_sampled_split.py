from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tmgm_rt.dataset import build_split
from tmgm_rt.metrics import polyphonic_metrics
from tmgm_rt.model import (
    calibrate_event_threshold,
    calibrate_head_threshold,
    load_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tracks", type=int, required=True)
    parser.add_argument("--frames-per-track", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    data = build_split(
        args.corpus,
        args.teacher_root,
        args.split,
        args.tracks,
        args.frames_per_track,
        bundle.frontend,
        bundle.context,
        bundle.targets,
        args.seed,
    )
    scores = bundle.predict_scores(data.features)
    prediction = (
        scores >= bundle.output_thresholds[None, :]
    ).astype(np.uint32)
    note_count = bundle.frontend.note_count
    fixed = polyphonic_metrics(data.targets, prediction, note_count)

    activity_threshold = calibrate_head_threshold(
        scores[:, :note_count], data.targets[:, :note_count], maximum_ratio=1.25
    )
    onset_threshold, _ = calibrate_event_threshold(
        scores[:, note_count : 2 * note_count],
        data.targets[:, note_count : 2 * note_count],
        maximum_ratio=3.0,
        radius=4,
    )
    oracle_thresholds = np.concatenate(
        (
            np.full(note_count, activity_threshold, dtype=np.float32),
            np.full(note_count, onset_threshold, dtype=np.float32),
        )
    )
    oracle_prediction = (scores >= oracle_thresholds[None, :]).astype(np.uint32)
    result = {
        "split": args.split,
        "tracks": args.tracks,
        "frames_per_track": args.frames_per_track,
        "rows": int(data.features.shape[0]),
        "category_counts": data.category_counts,
        "fixed_validation_thresholds": fixed,
        "diagnostic_train_oracle_thresholds": polyphonic_metrics(
            data.targets, oracle_prediction, note_count
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
