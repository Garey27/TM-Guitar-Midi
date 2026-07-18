from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .audio import load_audio_mono_channel_zero
from .binarize import QuantileThermometer
from .config import ContextConfig, FrontendConfig, TargetConfig
from .context import stack_causal_context
from .dataset import build_split
from .metrics import polyphonic_metrics
from .midi import NoteStateConfig, stabilize_frame_predictions, write_frame_predictions
from .nnpg import event_targets, read_nnpg
from .stft_plus import extract_stft_plus


def _inspect_nnpg(args: argparse.Namespace) -> int:
    posterior = read_nnpg(args.path)
    payload = asdict(posterior.header)
    payload.update(
        {
            "notes_min_max": [
                float(posterior.notes.min()),
                float(posterior.notes.max()),
            ],
            "onsets_min_max": [
                float(posterior.onsets.min()),
                float(posterior.onsets.max()),
            ],
        }
    )
    print(json.dumps(payload, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    from .model import (
        ModelBundle,
        TMConfig,
        bundle_metadata,
        calibrate_thresholds,
        create_model,
    )

    frontend = FrontendConfig()
    context = ContextConfig()
    targets = TargetConfig()
    tm_config = TMConfig(
        clauses=args.clauses,
        threshold=args.threshold,
        specificity=args.specificity,
        negative_samples=args.negative_samples,
        max_included_literals=args.max_literals,
        seed=args.seed,
        platform=args.platform,
    )
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
        f"continuous train={train.features.shape} validation={validation.features.shape}"
    )
    binarizer = QuantileThermometer()
    x_train = binarizer.fit_transform(train.features)
    x_validation = binarizer.transform(validation.features)
    print(
        f"binary train={x_train.shape} kept_literals={int(binarizer.keep_columns.sum())}"
    )

    model = create_model(tm_config)
    epoch_reports: list[dict[str, object]] = []
    thresholds = np.zeros(train.targets.shape[1], dtype=np.float32)
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.fit(x_train, train.targets, shuffle=True)
        _, scores = model.predict(x_validation, return_class_sums=True)
        thresholds = calibrate_thresholds(scores, validation.targets, frontend.note_count)
        prediction = (scores >= thresholds[None, :]).astype(np.uint32)
        metrics = polyphonic_metrics(
            validation.targets, prediction, frontend.note_count
        )
        report = {
            "epoch": epoch + 1,
            "seconds": time.perf_counter() - epoch_started,
            **metrics,
        }
        epoch_reports.append(report)
        print(json.dumps(report, sort_keys=True))

    metadata = bundle_metadata(frontend, context, targets, tm_config)
    metadata.update(
        {
            "tmu_commit": "5d6d9da7d3e8c3a15e40f93b94ec882db518c57c",
            "corpus": str(Path(args.corpus).resolve()),
            "teacher_root": str(Path(args.teacher_root).resolve()),
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_validation.shape),
            "train_categories": train.category_counts,
            "validation_categories": validation.category_counts,
            "epoch_reports": epoch_reports,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    bundle = ModelBundle(
        frontend=frontend,
        context=context,
        targets=targets,
        tm_config=tm_config,
        binarizer=binarizer,
        model=model,
        output_thresholds=thresholds,
        metadata=metadata,
    )
    bundle.save(args.output)
    report_path = Path(args.output).with_suffix(".json")
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.output}")
    return 0


def _train_ensemble(args: argparse.Namespace) -> int:
    from .ensemble_train import train_ensemble

    return train_ensemble(args)


def _transcribe(args: argparse.Namespace) -> int:
    from .model import load_bundle

    bundle = load_bundle(args.model)
    audio = load_audio_mono_channel_zero(args.input, bundle.frontend.sample_rate)
    started = time.perf_counter()
    spectral = extract_stft_plus(audio, bundle.frontend)
    continuous = stack_causal_context(spectral, bundle.context)
    scores = bundle.predict_scores(continuous)
    raw_prediction = (
        scores >= bundle.output_thresholds[None, :]
    ).astype(np.uint32)
    decoder = NoteStateConfig(
        attack_frames=args.attack_frames,
        release_frames=args.release_frames,
        retrigger_refractory_frames=args.retrigger_refractory_frames,
    )
    prediction = (
        raw_prediction
        if args.raw_decoder
        else stabilize_frame_predictions(
            raw_prediction, bundle.frontend.note_count, decoder
        )
    )
    write_frame_predictions(
        args.output,
        prediction,
        bundle.frontend.midi_min,
        bundle.frontend.note_count,
        bundle.frontend.frame_seconds,
    )
    np.savez_compressed(
        Path(args.output).with_suffix(".scores.npz"),
        scores=scores,
        raw_prediction=raw_prediction,
        prediction=prediction,
        thresholds=bundle.output_thresholds,
        decoder=np.asarray(
            [
                decoder.attack_frames,
                decoder.release_frames,
                decoder.retrigger_refractory_frames,
            ],
            dtype=np.int32,
        ),
    )
    elapsed = time.perf_counter() - started
    duration = audio.size / bundle.frontend.sample_rate
    print(
        json.dumps(
            {
                "audio_seconds": duration,
                "elapsed_seconds": elapsed,
                "realtime_factor": duration / max(elapsed, 1.0e-9),
                "frames": int(prediction.shape[0]),
                "note_on_candidates": int(
                    prediction[:, bundle.frontend.note_count :].sum()
                ),
                "raw_note_on_candidates": int(
                    raw_prediction[:, bundle.frontend.note_count :].sum()
                ),
                "decoder": "raw" if args.raw_decoder else asdict(decoder),
            },
            indent=2,
        )
    )
    return 0


def _evaluate_teacher(args: argparse.Namespace) -> int:
    from .model import load_bundle

    bundle = load_bundle(args.model)
    posterior = read_nnpg(args.nnpg)
    teacher = event_targets(
        args.events,
        posterior.header.frame_count,
        bundle.frontend.midi_min,
        bundle.frontend.midi_max,
        bundle.targets.onset_width_frames,
        bundle.targets.onset_delay_frames,
    )
    truth = np.concatenate((teacher.activity, teacher.onset), axis=1)
    audio = load_audio_mono_channel_zero(args.input, bundle.frontend.sample_rate)
    started = time.perf_counter()
    spectral = extract_stft_plus(audio, bundle.frontend)
    continuous = stack_causal_context(spectral, bundle.context)
    frame_count = min(continuous.shape[0], truth.shape[0])
    raw_prediction = bundle.predict(continuous[:frame_count])
    decoder = NoteStateConfig(
        attack_frames=args.attack_frames,
        release_frames=args.release_frames,
        retrigger_refractory_frames=args.retrigger_refractory_frames,
    )
    prediction = (
        raw_prediction
        if args.raw_decoder
        else stabilize_frame_predictions(
            raw_prediction, bundle.frontend.note_count, decoder
        )
    )
    metrics = polyphonic_metrics(
        truth[:frame_count], prediction, bundle.frontend.note_count
    )
    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "frames": int(frame_count),
            "audio_seconds": float(audio.size / bundle.frontend.sample_rate),
            "elapsed_seconds": elapsed,
            "realtime_factor": float(
                (audio.size / bundle.frontend.sample_rate) / max(elapsed, 1.0e-9)
            ),
        }
    )
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


def _evaluate_corpus(args: argparse.Namespace) -> int:
    from .corpus_eval import evaluate_corpus

    return evaluate_corpus(args)


def _recalibrate_ensemble(args: argparse.Namespace) -> int:
    from .fusion_calibrate import recalibrate_ensemble

    return recalibrate_ensemble(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmgm-rt")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-nnpg")
    inspect.add_argument("path", type=Path)
    inspect.set_defaults(handler=_inspect_nnpg)

    train = commands.add_parser("train")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--teacher-root", type=Path, required=True)
    train.add_argument("--train-tracks", type=int, default=4)
    train.add_argument("--validation-tracks", type=int, default=2)
    train.add_argument("--frames-per-track", type=int, default=800)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--clauses", type=int, default=128)
    train.add_argument("--threshold", type=int, default=64)
    train.add_argument("--specificity", type=float, default=5.0)
    train.add_argument("--negative-samples", type=float, default=8.0)
    train.add_argument("--max-literals", type=int, default=32)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--platform", choices=("CPU", "CUDA"), default="CPU")
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(handler=_train)

    ensemble = commands.add_parser("train-ensemble")
    ensemble.add_argument("--corpus", type=Path, required=True)
    ensemble.add_argument("--teacher-root", type=Path, required=True)
    ensemble.add_argument("--train-tracks", type=int, default=12)
    ensemble.add_argument("--validation-tracks", type=int, default=6)
    ensemble.add_argument("--frames-per-track", type=int, default=800)
    ensemble.add_argument("--epochs", type=int, default=4)
    ensemble.add_argument("--activity-members", type=int, default=3)
    ensemble.add_argument("--onset-members", type=int, default=3)
    ensemble.add_argument("--activity-clauses", type=int, default=128)
    ensemble.add_argument("--onset-clauses", type=int, default=128)
    ensemble.add_argument("--activity-threshold", type=int, default=64)
    ensemble.add_argument("--onset-threshold", type=int, default=64)
    ensemble.add_argument("--activity-specificity", type=float, default=5.0)
    ensemble.add_argument("--onset-specificity", type=float, default=4.0)
    ensemble.add_argument(
        "--activity-negative-samples", type=float, default=8.0
    )
    ensemble.add_argument("--onset-negative-samples", type=float, default=4.0)
    ensemble.add_argument(
        "--onset-fusion",
        choices=("mean", "top2_mean", "max"),
        default="top2_mean",
    )
    ensemble.add_argument("--onset-tolerance-frames", type=int, default=4)
    ensemble.add_argument("--minimum-onset-ratio", type=float, default=0.5)
    ensemble.add_argument("--maximum-onset-ratio", type=float, default=3.0)
    ensemble.add_argument("--max-literals", type=int, default=32)
    ensemble.add_argument("--seed", type=int, default=42)
    ensemble.add_argument("--platform", choices=("CPU", "CUDA"), default="CPU")
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.set_defaults(handler=_train_ensemble)

    transcribe = commands.add_parser("transcribe")
    transcribe.add_argument("--model", type=Path, required=True)
    transcribe.add_argument("--input", type=Path, required=True)
    transcribe.add_argument("--output", type=Path, required=True)
    transcribe.add_argument("--raw-decoder", action="store_true")
    transcribe.add_argument("--attack-frames", type=int, default=2)
    transcribe.add_argument("--release-frames", type=int, default=4)
    transcribe.add_argument(
        "--retrigger-refractory-frames", type=int, default=6
    )
    transcribe.set_defaults(handler=_transcribe)

    evaluate = commands.add_parser("evaluate-teacher")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--nnpg", type=Path, required=True)
    evaluate.add_argument("--events", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--raw-decoder", action="store_true")
    evaluate.add_argument("--attack-frames", type=int, default=2)
    evaluate.add_argument("--release-frames", type=int, default=4)
    evaluate.add_argument(
        "--retrigger-refractory-frames", type=int, default=6
    )
    evaluate.set_defaults(handler=_evaluate_teacher)

    evaluate_set = commands.add_parser("evaluate-corpus")
    evaluate_set.add_argument("--model", type=Path, required=True)
    evaluate_set.add_argument("--corpus", type=Path, required=True)
    evaluate_set.add_argument("--teacher-root", type=Path, required=True)
    evaluate_set.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    evaluate_set.add_argument("--tracks", type=int, default=8)
    evaluate_set.add_argument("--seed", type=int, default=20260717)
    evaluate_set.add_argument("--output", type=Path)
    evaluate_set.add_argument("--raw-decoder", action="store_true")
    evaluate_set.add_argument("--attack-frames", type=int, default=2)
    evaluate_set.add_argument("--release-frames", type=int, default=4)
    evaluate_set.add_argument(
        "--retrigger-refractory-frames", type=int, default=6
    )
    evaluate_set.set_defaults(handler=_evaluate_corpus)

    recalibrate = commands.add_parser("recalibrate-ensemble")
    recalibrate.add_argument("--model", type=Path, required=True)
    recalibrate.add_argument("--corpus", type=Path, required=True)
    recalibrate.add_argument("--teacher-root", type=Path, required=True)
    recalibrate.add_argument("--validation-tracks", type=int, default=6)
    recalibrate.add_argument("--frames-per-track", type=int, default=800)
    recalibrate.add_argument("--seed", type=int, default=42)
    recalibrate.add_argument("--tolerance-frames", type=int, default=4)
    recalibrate.add_argument("--minimum-onset-ratio", type=float, default=0.5)
    recalibrate.add_argument("--maximum-onset-ratio", type=float, default=3.0)
    recalibrate.add_argument("--output", type=Path, required=True)
    recalibrate.set_defaults(handler=_recalibrate_ensemble)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
