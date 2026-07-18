from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .binarize import QuantileThermometer
from .config import ContextConfig, FrontendConfig, TargetConfig
from .dataset import build_split, read_corpus
from .metrics import polyphonic_metrics
from .model import (
    EnsembleBundle,
    HeadEnsembleMember,
    TMConfig,
    TMHeadEnsemble,
    bundle_metadata,
    calibrate_event_threshold,
    calibrate_head_threshold,
    create_model,
    make_head_member,
)


def _diverse_config(
    base: TMConfig, member_index: int, member_count: int, head_seed_offset: int
) -> TMConfig:
    # Seeds provide bootstrap-like stochastic diversity; modest s/q variation
    # changes clause granularity without making the ablation incomparable.
    position = 0.0
    if member_count > 1:
        position = -1.0 + 2.0 * member_index / (member_count - 1)
    return replace(
        base,
        specificity=max(1.5, base.specificity * (1.0 + 0.25 * position)),
        negative_samples=max(
            1.0, base.negative_samples * (1.0 - 0.20 * position)
        ),
        seed=base.seed + head_seed_offset + 9_973 * member_index,
    )


def _member_report(member: HeadEnsembleMember) -> dict[str, Any]:
    return {
        "tm": asdict(member.tm_config),
        "score_center": member.score_center,
        "score_scale": member.score_scale,
        "validation_f1": member.validation_f1,
    }


def _diversity_report(
    ensemble: TMHeadEnsemble, binary_features: np.ndarray
) -> dict[str, float]:
    scores = ensemble.predict_member_scores(binary_features)
    correlations: list[float] = []
    disagreements: list[float] = []
    for first in range(scores.shape[0]):
        for second in range(first + 1, scores.shape[0]):
            a = scores[first].ravel()
            b = scores[second].ravel()
            if float(a.std()) > 1.0e-9 and float(b.std()) > 1.0e-9:
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
            disagreements.append(float(np.mean((a >= 0.0) != (b >= 0.0))))
    return {
        "pairwise_margin_correlation": float(np.mean(correlations))
        if correlations
        else 1.0,
        "pairwise_binary_disagreement": float(np.mean(disagreements))
        if disagreements
        else 0.0,
    }


def _train_head(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    base_config: TMConfig,
    member_count: int,
    epochs: int,
    maximum_ratio: float,
    seed_offset: int,
    reducer: str = "mean",
) -> tuple[TMHeadEnsemble, list[dict[str, Any]]]:
    members: list[HeadEnsembleMember] = []
    reports: list[dict[str, Any]] = []
    for member_index in range(member_count):
        config = _diverse_config(
            base_config, member_index, member_count, seed_offset
        )
        model = create_model(config)
        started = time.perf_counter()
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            model.fit(x_train, y_train, shuffle=True)
            print(
                f"[{name} {member_index + 1}/{member_count}] "
                f"epoch {epoch + 1}/{epochs} "
                f"{time.perf_counter() - epoch_started:.2f}s"
            )
        _, validation_scores = model.predict(
            x_validation, return_class_sums=True
        )
        member = make_head_member(
            model,
            config,
            np.asarray(validation_scores, dtype=np.float32),
            y_validation,
            maximum_ratio,
        )
        report = {
            "member": member_index + 1,
            "seconds": time.perf_counter() - started,
            **_member_report(member),
        }
        print(json.dumps({"head": name, **report}, sort_keys=True))
        members.append(member)
        reports.append(report)
    weights = np.full(member_count, 1.0 / member_count, dtype=np.float32)
    return (
        TMHeadEnsemble(
            name=name, members=members, weights=weights, reducer=reducer
        ),
        reports,
    )


def train_ensemble(args: Any) -> int:
    frontend = FrontendConfig()
    context = ContextConfig()
    targets = TargetConfig()
    started = time.perf_counter()

    train = build_split(
        args.corpus,
        args.teacher_root,
        "train",
        args.train_tracks,
        args.frames_per_track,
        frontend,
        context,
        targets,
        args.seed,
    )
    validation = build_split(
        args.corpus,
        args.teacher_root,
        "validation",
        args.validation_tracks,
        args.frames_per_track,
        frontend,
        context,
        targets,
        args.seed,
    )
    print(
        f"continuous train={train.features.shape} "
        f"validation={validation.features.shape}"
    )
    binarizer = QuantileThermometer()
    x_train = binarizer.fit_transform(train.features)
    x_validation = binarizer.transform(validation.features)
    print(
        f"binary train={x_train.shape} "
        f"kept_literals={int(binarizer.keep_columns.sum())}"
    )

    note_count = frontend.note_count
    activity_train = train.targets[:, :note_count]
    activity_validation = validation.targets[:, :note_count]
    onset_train = train.targets[:, note_count : 2 * note_count]
    onset_validation = validation.targets[:, note_count : 2 * note_count]

    activity_config = TMConfig(
        clauses=args.activity_clauses,
        threshold=args.activity_threshold,
        specificity=args.activity_specificity,
        negative_samples=args.activity_negative_samples,
        max_included_literals=args.max_literals,
        seed=args.seed,
        platform=args.platform,
    )
    onset_config = TMConfig(
        clauses=args.onset_clauses,
        threshold=args.onset_threshold,
        specificity=args.onset_specificity,
        negative_samples=args.onset_negative_samples,
        max_included_literals=args.max_literals,
        seed=args.seed,
        platform=args.platform,
    )
    activity, activity_reports = _train_head(
        "activity",
        x_train,
        activity_train,
        x_validation,
        activity_validation,
        activity_config,
        args.activity_members,
        args.epochs,
        maximum_ratio=1.25,
        seed_offset=10_000,
    )
    onset, onset_reports = _train_head(
        "onset",
        x_train,
        onset_train,
        x_validation,
        onset_validation,
        onset_config,
        args.onset_members,
        args.epochs,
        maximum_ratio=1.5,
        seed_offset=20_000,
        reducer=args.onset_fusion,
    )

    activity_scores = activity.predict_scores(x_validation)
    onset_scores = onset.predict_scores(x_validation)
    activity_threshold = calibrate_head_threshold(
        activity_scores, activity_validation, maximum_ratio=1.25
    )
    onset_threshold, onset_calibration = calibrate_event_threshold(
        onset_scores,
        onset_validation,
        maximum_ratio=args.maximum_onset_ratio,
        minimum_ratio=args.minimum_onset_ratio,
        radius=args.onset_tolerance_frames,
    )
    thresholds = np.concatenate(
        (
            np.full(note_count, activity_threshold, dtype=np.float32),
            np.full(note_count, onset_threshold, dtype=np.float32),
        )
    )
    validation_scores = np.concatenate(
        (activity_scores, onset_scores), axis=1
    )
    validation_prediction = (
        validation_scores >= thresholds[None, :]
    ).astype(np.uint32)
    validation_metrics = polyphonic_metrics(
        validation.targets, validation_prediction, note_count
    )
    validation_metrics["activity_member_mean_f1"] = float(
        np.mean([member.validation_f1 for member in activity.members])
    )
    validation_metrics["activity_member_best_f1"] = float(
        np.max([member.validation_f1 for member in activity.members])
    )
    validation_metrics["onset_member_mean_f1"] = float(
        np.mean([member.validation_f1 for member in onset.members])
    )
    validation_metrics["onset_member_best_f1"] = float(
        np.max([member.validation_f1 for member in onset.members])
    )
    validation_metrics["activity_ensemble_gain"] = (
        validation_metrics["activity_f1"]
        - validation_metrics["activity_member_mean_f1"]
    )
    validation_metrics["onset_ensemble_gain"] = (
        validation_metrics["onset_f1"]
        - validation_metrics["onset_member_mean_f1"]
    )
    validation_metrics["activity_ensemble_gain_vs_best"] = (
        validation_metrics["activity_f1"]
        - validation_metrics["activity_member_best_f1"]
    )
    validation_metrics["onset_ensemble_gain_vs_best"] = (
        validation_metrics["onset_f1"]
        - validation_metrics["onset_member_best_f1"]
    )
    diversity = {
        "activity": _diversity_report(activity, x_validation),
        "onset": _diversity_report(onset, x_validation),
    }

    selected_train = read_corpus(
        args.corpus, "train", args.train_tracks, args.seed
    )
    selected_validation = read_corpus(
        args.corpus, "validation", args.validation_tracks, args.seed
    )
    metadata = bundle_metadata(frontend, context, targets, activity_config)
    metadata.update(
        {
            "artifact_schema": 2,
            "backend": "TMU.experimental.separate_head_ensemble",
            "score_space": "calibrated_class_sum_margin_v1",
            "fusion": "validation_normalized_clipped_margin",
            "activity_fusion": "mean",
            "onset_fusion": args.onset_fusion,
            "onset_calibration": {
                "tolerance_frames": args.onset_tolerance_frames,
                "minimum_ratio": args.minimum_onset_ratio,
                "maximum_ratio": args.maximum_onset_ratio,
                "metrics": onset_calibration,
            },
            "margin_clip": 4.0,
            "tmu_commit": "5d6d9da7d3e8c3a15e40f93b94ec882db518c57c",
            "corpus": str(Path(args.corpus).resolve()),
            "teacher_root": str(Path(args.teacher_root).resolve()),
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_validation.shape),
            "train_categories": train.category_counts,
            "validation_categories": validation.category_counts,
            "selected_train": [
                {
                    "source": entry.source,
                    "id": entry.identifier,
                    "group": entry.group,
                }
                for entry in selected_train
            ],
            "selected_validation": [
                {
                    "source": entry.source,
                    "id": entry.identifier,
                    "group": entry.group,
                }
                for entry in selected_validation
            ],
            "activity_members": activity_reports,
            "onset_members": onset_reports,
            "validation_metrics": validation_metrics,
            "diversity": diversity,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    bundle = EnsembleBundle(
        frontend=frontend,
        context=context,
        targets=targets,
        binarizer=binarizer,
        activity=activity,
        onset=onset,
        output_thresholds=thresholds,
        metadata=metadata,
    )
    bundle.save(args.output)
    report_path = Path(args.output).with_suffix(".json")
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(validation_metrics, indent=2))
    print(f"saved {args.output}")
    return 0
