from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .audio import load_audio_mono_channel_zero
from .context import stack_causal_context
from .dataset import CorpusEntry, read_corpus
from .metrics import polyphonic_metrics
from .midi import NoteStateConfig, stabilize_frame_predictions
from .model import load_bundle
from .nnpg import event_targets, read_nnpg
from .stft_plus import extract_stft_plus


def _evaluate_entry(
    bundle: Any,
    entry: CorpusEntry,
    teacher_root: Path,
    decoder: NoteStateConfig | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float, float]:
    teacher_base = teacher_root / entry.output_relative
    posterior = read_nnpg(teacher_base.with_suffix(".nnpg"))
    teacher = event_targets(
        teacher_base.with_suffix(".events.tsv"),
        posterior.header.frame_count,
        bundle.frontend.midi_min,
        bundle.frontend.midi_max,
        bundle.targets.onset_width_frames,
        bundle.targets.onset_delay_frames,
    )
    truth = np.concatenate((teacher.activity, teacher.onset), axis=1)
    audio = load_audio_mono_channel_zero(
        entry.input_path, bundle.frontend.sample_rate
    )
    started = time.perf_counter()
    spectral = extract_stft_plus(audio, bundle.frontend)
    continuous = stack_causal_context(spectral, bundle.context)
    frame_count = min(continuous.shape[0], truth.shape[0])
    prediction = bundle.predict(continuous[:frame_count])
    if decoder is not None:
        prediction = stabilize_frame_predictions(
            prediction, bundle.frontend.note_count, decoder
        )
    elapsed = time.perf_counter() - started
    truth = truth[:frame_count]
    metrics = polyphonic_metrics(
        truth, prediction, bundle.frontend.note_count
    )
    duration = float(audio.size / bundle.frontend.sample_rate)
    report: dict[str, Any] = {
        "source": entry.source,
        "id": entry.identifier,
        "group": entry.group,
        "frames": int(frame_count),
        "audio_seconds": duration,
        "elapsed_seconds": elapsed,
        "realtime_factor": duration / max(elapsed, 1.0e-9),
        **metrics,
    }
    return report, truth, prediction, duration, elapsed


def evaluate_corpus(args: Any) -> int:
    bundle = load_bundle(args.model)
    entries = read_corpus(args.corpus, args.split, args.tracks, args.seed)
    teacher_root = Path(args.teacher_root)
    track_reports: list[dict[str, Any]] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    by_source: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    total_duration = 0.0
    total_elapsed = 0.0
    decoder = (
        None
        if args.raw_decoder
        else NoteStateConfig(
            attack_frames=args.attack_frames,
            release_frames=args.release_frames,
            retrigger_refractory_frames=args.retrigger_refractory_frames,
        )
    )
    for index, entry in enumerate(entries):
        print(
            f"[{args.split} {index + 1}/{len(entries)}] "
            f"{entry.source}/{entry.identifier}"
        )
        report, truth, prediction, duration, elapsed = _evaluate_entry(
            bundle, entry, teacher_root, decoder
        )
        print(
            json.dumps(
                {
                    "activity_f1": report["activity_f1"],
                    "onset_tolerance_4f_f1": report[
                        "onset_tolerance_4f_f1"
                    ],
                    "chord_frame_pitch_recall": report[
                        "chord_frame_pitch_recall"
                    ],
                    "realtime_factor": report["realtime_factor"],
                },
                sort_keys=True,
            )
        )
        track_reports.append(report)
        truths.append(truth)
        predictions.append(prediction)
        source_truth, source_prediction = by_source.setdefault(
            entry.source, ([], [])
        )
        source_truth.append(truth)
        source_prediction.append(prediction)
        total_duration += duration
        total_elapsed += elapsed

    metric_keys = [
        "activity_precision",
        "activity_recall",
        "activity_f1",
        "onset_precision",
        "onset_recall",
        "onset_f1",
        "onset_tolerance_4f_precision",
        "onset_tolerance_4f_recall",
        "onset_tolerance_4f_f1",
        "chord_frame_pitch_recall",
        "chord_frame_complete_rate",
    ]
    macro = {
        key: float(np.mean([report[key] for report in track_reports]))
        for key in metric_keys
    }
    micro = polyphonic_metrics(
        np.concatenate(truths),
        np.concatenate(predictions),
        bundle.frontend.note_count,
    )
    source_metrics = {
        source: polyphonic_metrics(
            np.concatenate(source_truth),
            np.concatenate(source_prediction),
            bundle.frontend.note_count,
        )
        for source, (source_truth, source_prediction) in by_source.items()
    }
    result = {
        "model": str(Path(args.model).resolve()),
        "split": args.split,
        "seed": args.seed,
        "track_count": len(entries),
        "selected_tracks": [
            {"source": e.source, "id": e.identifier, "group": e.group}
            for e in entries
        ],
        "macro": macro,
        "micro": micro,
        "by_source": source_metrics,
        "audio_seconds": total_duration,
        "elapsed_seconds": total_elapsed,
        "realtime_factor": total_duration / max(total_elapsed, 1.0e-9),
        "decoder": "raw"
        if decoder is None
        else {
            "attack_frames": decoder.attack_frames,
            "release_frames": decoder.release_frames,
            "retrigger_refractory_frames": decoder.retrigger_refractory_frames,
        },
        "tracks": track_reports,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0
