from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import build_split, read_corpus
from .metrics import binary_metrics, polyphonic_metrics
from .model import (
    EnsembleBundle,
    calibrate_event_threshold,
    load_bundle,
)


def _selection_signature(entries) -> list[dict[str, str]]:
    return [
        {"source": entry.source, "id": entry.identifier, "group": entry.group}
        for entry in entries
    ]


def recalibrate_ensemble(args: Any) -> int:
    value = load_bundle(args.model)
    if not isinstance(value, EnsembleBundle) or value.onset is None:
        raise TypeError("recalibration requires an activity+onset EnsembleBundle")
    bundle = value
    selected = read_corpus(
        args.corpus, "validation", args.validation_tracks, args.seed
    )
    expected = bundle.metadata.get("selected_validation")
    actual = _selection_signature(selected)
    if expected is not None and expected != actual:
        raise ValueError(
            "validation selection differs from training artifact; refusing leakage-prone recalibration"
        )

    validation = build_split(
        args.corpus,
        args.teacher_root,
        "validation",
        args.validation_tracks,
        args.frames_per_track,
        bundle.frontend,
        bundle.context,
        bundle.targets,
        args.seed,
    )
    binary = bundle.binarizer.transform(validation.features)
    note_count = bundle.frontend.note_count
    activity_truth = validation.targets[:, :note_count]
    onset_truth = validation.targets[:, note_count : 2 * note_count]
    activity_scores = bundle.activity.predict_scores(binary)
    member_scores = bundle.onset.predict_member_scores(binary)

    candidates: list[dict[str, Any]] = []
    for reducer in ("mean", "top2_mean", "max"):
        bundle.onset.reducer = reducer
        onset_scores = bundle.onset.reduce_member_scores(member_scores)
        threshold, tolerant = calibrate_event_threshold(
            onset_scores,
            onset_truth,
            maximum_ratio=args.maximum_onset_ratio,
            minimum_ratio=args.minimum_onset_ratio,
            radius=args.tolerance_frames,
        )
        prediction = onset_scores >= threshold
        exact = binary_metrics(onset_truth, prediction)
        candidates.append(
            {
                "reducer": reducer,
                "threshold": threshold,
                "exact": exact,
                "tolerant": tolerant,
                "predicted_mean_onsets": float(prediction.sum(axis=1).mean()),
                "teacher_mean_onsets": float(onset_truth.sum(axis=1).mean()),
            }
        )
    selected_candidate = max(
        candidates,
        key=lambda row: (
            row["tolerant"]["f1"],
            row["exact"]["f1"],
            -abs(
                row["predicted_mean_onsets"] - row["teacher_mean_onsets"]
            ),
        ),
    )
    bundle.onset.reducer = selected_candidate["reducer"]
    onset_threshold = float(selected_candidate["threshold"])
    bundle.output_thresholds[note_count : 2 * note_count] = onset_threshold

    onset_scores = bundle.onset.reduce_member_scores(member_scores)
    combined_scores = np.concatenate((activity_scores, onset_scores), axis=1)
    prediction = (
        combined_scores >= bundle.output_thresholds[None, :]
    ).astype(np.uint32)
    combined_metrics = polyphonic_metrics(
        validation.targets, prediction, note_count
    )
    calibration = {
        "validation_tracks": args.validation_tracks,
        "frames_per_track": args.frames_per_track,
        "seed": args.seed,
        "tolerance_frames": args.tolerance_frames,
        "minimum_onset_ratio": args.minimum_onset_ratio,
        "maximum_onset_ratio": args.maximum_onset_ratio,
        "candidates": candidates,
        "selected": selected_candidate,
        "combined_validation_metrics": combined_metrics,
    }
    bundle.metadata["artifact_schema"] = 3
    bundle.metadata["onset_fusion"] = selected_candidate["reducer"]
    bundle.metadata["fusion_recalibration"] = calibration
    bundle.save(args.output)
    report_path = Path(args.output).with_suffix(".json")
    report_path.write_text(
        json.dumps(bundle.metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(calibration, indent=2))
    print(f"saved {args.output}")
    return 0
