from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from tmgm_rt.binarize import QuantileThermometer
from tmgm_rt.config import ContextConfig, FrontendConfig, TargetConfig
from tmgm_rt.dataset import build_track_examples, read_corpus
from tmgm_rt.metrics import binary_metrics, polyphonic_metrics, tolerant_event_metrics
from tmgm_rt.midi import (
    NoteStateConfig,
    stabilize_frame_predictions,
    write_frame_predictions,
    write_teacher_events,
)
from tmgm_rt.model import (
    EnsembleBundle,
    TMConfig,
    TMHeadEnsemble,
    calibrate_event_threshold,
    calibrate_head_threshold,
    create_model,
    make_head_member,
)


def write_status(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def onset_training_indices(
    targets: np.ndarray, note_count: int, rows: int, seed: int
) -> np.ndarray:
    activity = targets[:, :note_count].sum(axis=1)
    onset = targets[:, note_count : 2 * note_count].sum(axis=1)
    groups = [
        (np.flatnonzero(onset > 0), 0.55),
        (np.flatnonzero((onset == 0) & (activity > 0)), 0.35),
        (np.flatnonzero(activity == 0), 0.10),
    ]
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    remaining = rows
    for index, (candidates, fraction) in enumerate(groups):
        if candidates.size == 0:
            continue
        count = remaining if index == len(groups) - 1 else int(round(rows * fraction))
        count = min(count, remaining)
        selected.append(rng.choice(candidates, size=count, replace=True))
        remaining -= count
    if remaining:
        selected.append(rng.choice(targets.shape[0], size=remaining, replace=True))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result.astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--source", default="guitarset")
    parser.add_argument("--id", default="00_Jazz1-130-D_comp_mix")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clauses", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--max-literals", type=int, default=64)
    parser.add_argument("--onset-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--platform", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--midi-dir", type=Path, required=True)
    args = parser.parse_args()

    entries = read_corpus(args.corpus, "train")
    entry = next(
        candidate
        for candidate in entries
        if candidate.source == args.source and candidate.identifier == args.id
    )
    frontend = FrontendConfig()
    context = ContextConfig()
    targets = TargetConfig()
    continuous, truth, _, _, categories = build_track_examples(
        entry,
        args.teacher_root,
        frontend,
        context,
        targets,
        frames_per_track=10_000_000,
        seed=args.seed,
        balanced_sampling=False,
    )
    binarizer = QuantileThermometer()
    binary = binarizer.fit_transform(continuous)
    note_count = frontend.note_count
    activity_truth = truth[:, :note_count]
    onset_truth = truth[:, note_count : 2 * note_count]
    onset_indices = onset_training_indices(
        truth, note_count, args.onset_rows, args.seed
    )
    onset_binary = np.ascontiguousarray(binary[onset_indices])
    onset_train_truth = np.ascontiguousarray(onset_truth[onset_indices])
    status_path = args.output.with_suffix(".status.json")
    status = {
        "intentional_overfit": True,
        "source": entry.source,
        "id": entry.identifier,
        "frames": int(binary.shape[0]),
        "binary_features": int(binary.shape[1]),
        "onset_training_rows": int(onset_binary.shape[0]),
        "clauses": args.clauses,
        "epochs": args.epochs,
        "activity": [],
        "onset": [],
    }
    write_status(status_path, status)

    activity_config = TMConfig(
        clauses=args.clauses,
        threshold=args.threshold,
        specificity=5.0,
        negative_samples=8.0,
        max_included_literals=args.max_literals,
        clause_drop=0.0,
        literal_drop=0.0,
        seed=args.seed,
        platform=args.platform,
    )
    onset_config = TMConfig(
        clauses=args.clauses,
        threshold=args.threshold,
        specificity=4.0,
        negative_samples=4.0,
        max_included_literals=args.max_literals,
        clause_drop=0.0,
        literal_drop=0.0,
        seed=args.seed + 10_000,
        platform=args.platform,
    )

    activity_model = create_model(activity_config)
    for epoch in range(args.epochs):
        started = time.perf_counter()
        activity_model.fit(binary, activity_truth, shuffle=True)
        _, scores = activity_model.predict(binary, return_class_sums=True)
        threshold = calibrate_head_threshold(scores, activity_truth, 1.5)
        metrics = binary_metrics(activity_truth, scores >= threshold)
        row = {"epoch": epoch + 1, "seconds": time.perf_counter() - started, **metrics}
        status["activity"].append(row)
        write_status(status_path, status)
        print(json.dumps({"head": "activity", **row}), flush=True)

    onset_model = create_model(onset_config)
    for epoch in range(args.epochs):
        started = time.perf_counter()
        onset_model.fit(onset_binary, onset_train_truth, shuffle=True)
        _, scores = onset_model.predict(binary, return_class_sums=True)
        threshold, tolerant = calibrate_event_threshold(
            scores,
            onset_truth,
            maximum_ratio=4.0,
            minimum_ratio=0.25,
            radius=4,
        )
        exact = binary_metrics(onset_truth, scores >= threshold)
        row = {
            "epoch": epoch + 1,
            "seconds": time.perf_counter() - started,
            "exact_f1": exact["f1"],
            "tolerant_f1": tolerant["f1"],
        }
        status["onset"].append(row)
        write_status(status_path, status)
        print(json.dumps({"head": "onset", **row}), flush=True)

    _, activity_raw_scores = activity_model.predict(binary, return_class_sums=True)
    _, onset_raw_scores = onset_model.predict(binary, return_class_sums=True)
    activity_member = make_head_member(
        activity_model,
        activity_config,
        activity_raw_scores,
        activity_truth,
        maximum_ratio=1.5,
    )
    onset_member = make_head_member(
        onset_model,
        onset_config,
        onset_raw_scores,
        onset_truth,
        maximum_ratio=4.0,
    )
    activity = TMHeadEnsemble(
        "activity", [activity_member], np.ones(1, dtype=np.float32), "mean"
    )
    onset = TMHeadEnsemble(
        "onset", [onset_member], np.ones(1, dtype=np.float32), "max"
    )
    activity_scores = activity.predict_scores(binary)
    onset_scores = onset.predict_scores(binary)
    activity_threshold = calibrate_head_threshold(
        activity_scores, activity_truth, maximum_ratio=1.5
    )
    onset_threshold, _ = calibrate_event_threshold(
        onset_scores,
        onset_truth,
        maximum_ratio=4.0,
        minimum_ratio=0.25,
        radius=4,
    )
    thresholds = np.concatenate(
        (
            np.full(note_count, activity_threshold, dtype=np.float32),
            np.full(note_count, onset_threshold, dtype=np.float32),
        )
    )
    scores = np.concatenate((activity_scores, onset_scores), axis=1)
    raw_prediction = (scores >= thresholds[None, :]).astype(np.uint32)
    stable_prediction = stabilize_frame_predictions(
        raw_prediction, note_count, NoteStateConfig()
    )
    final_metrics = polyphonic_metrics(truth, raw_prediction, note_count)
    metadata = {
        "artifact_schema": 3,
        "intentional_overfit": True,
        "source": entry.source,
        "id": entry.identifier,
        "input": str(entry.input_path),
        "frontend": asdict(frontend),
        "context": asdict(context),
        "targets": asdict(targets),
        "activity_tm": asdict(activity_config),
        "onset_tm": asdict(onset_config),
        "category_counts": categories,
        "final_metrics": final_metrics,
        "epoch_status": str(status_path),
    }
    bundle = EnsembleBundle(
        frontend,
        context,
        targets,
        binarizer,
        activity,
        onset,
        thresholds,
        metadata,
    )
    bundle.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    args.midi_dir.mkdir(parents=True, exist_ok=True)
    write_frame_predictions(
        args.midi_dir / "tm-overfit-raw.mid",
        raw_prediction,
        frontend.midi_min,
        note_count,
        frontend.frame_seconds,
    )
    write_frame_predictions(
        args.midi_dir / "tm-overfit-stable.mid",
        stable_prediction,
        frontend.midi_min,
        note_count,
        frontend.frame_seconds,
    )
    write_teacher_events(
        args.midi_dir / "neuralnote.mid",
        (args.teacher_root / entry.output_relative).with_suffix(".events.tsv"),
        frontend.midi_min,
        frontend.midi_max,
    )
    (args.midi_dir / "source-wav.txt").write_text(
        str(entry.input_path), encoding="utf-8"
    )
    print(json.dumps(final_metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
