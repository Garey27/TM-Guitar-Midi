from __future__ import annotations

"""Render the clean-room polyphonic tracker over frozen TM complex-6 scores.

The activity and onset scores are the already frozen strict-cap16-v3 TM
ensemble outputs.  This script performs no training, calibration, threshold
search, or TCN inference.  It adds the causal dual-resolution acoustic
frontend/state machine from :mod:`tmgm_rt.tracking`, writes its sample-timed
``NoteEvent`` objects to MIDI, and compares note onsets with the NeuralNote
teacher using one pitch-exact match per onset.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Sequence

import mido
import numpy as np

from tmgm_rt.audio import load_audio_mono_channel_zero
from tmgm_rt.native_score_ensemble import LoadedScores, load_score_file
from tmgm_rt.tracking import NoteEvent, PolyphonicTracker, TrackerConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "configs/complex-dataset-selected-tracks-20260718.json"
DEFAULT_SOURCE_ROOT = (
    ROOT / "artifacts/complex-dataset-listening-20260718-strict-cap16-v3"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/tm-tracker-complex6-20260718-cleanroom-v1"
DEFAULT_GAIN_OUTPUT_ROOT = (
    ROOT / "artifacts/tm-tracker-complex6-20260718-cleanroom-inputgain-v2"
)
EXPECTED_ACTIVITY_THRESHOLD = -169
EXPECTED_ONSET_THRESHOLD = -492
EXPECTED_OUTPUTS = 49
EXPECTED_MIDI_MIN = 40
EXPECTED_SAMPLE_RATE = 22_050
EXPECTED_HOP_SIZE = 256
EXPECTED_QUANTIZATION = 1_024
TICKS_PER_SECOND = 960.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: str | Path) -> dict[str, Any]:
    value = Path(path).resolve()
    return {
        "path": str(value),
        "bytes": value.stat().st_size,
        "sha256": sha256_file(value),
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ensemble_quantization(scores: LoadedScores, path: Path) -> int:
    artifact_value = scores.metadata.raw.get("ensemble_artifact")
    if not artifact_value:
        raise ValueError(f"score TSV has no ensemble_artifact: {path}")
    artifact_path = Path(artifact_value)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("format") != "TMGM_NATIVE_SCORE_ENSEMBLE_V1":
        raise ValueError(f"score TSV does not reference a native ensemble: {path}")
    quantization = int(artifact.get("quantization", 0))
    if quantization != EXPECTED_QUANTIZATION:
        raise ValueError(
            f"unexpected ensemble quantization {quantization}: {artifact_path}"
        )
    if int(artifact["ensemble_threshold"]) != scores.metadata.threshold:
        raise ValueError(
            f"TSV/artifact threshold mismatch: {path} versus {artifact_path}"
        )
    return quantization


def validate_score_pair(
    activity: LoadedScores,
    activity_path: Path,
    onset: LoadedScores,
    onset_path: Path,
) -> int:
    expected = {
        "frames": activity.metadata.frames,
        "outputs": EXPECTED_OUTPUTS,
        "midi_min": EXPECTED_MIDI_MIN,
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "hop_size": EXPECTED_HOP_SIZE,
    }
    for head, scores, path, threshold in (
        ("activity", activity, activity_path, EXPECTED_ACTIVITY_THRESHOLD),
        ("onset", onset, onset_path, EXPECTED_ONSET_THRESHOLD),
    ):
        metadata = scores.metadata
        if metadata.head != head:
            raise ValueError(f"expected {head} scores, got {metadata.head}: {path}")
        for name, value in expected.items():
            if getattr(metadata, name) != value:
                raise ValueError(
                    f"{head} {name}={getattr(metadata, name)} differs from "
                    f"frozen geometry {value}: {path}"
                )
        if metadata.threshold != threshold:
            raise ValueError(
                f"{head} threshold {metadata.threshold} is not frozen {threshold}: {path}"
            )
    activity_quantization = _ensemble_quantization(activity, activity_path)
    onset_quantization = _ensemble_quantization(onset, onset_path)
    if activity_quantization != onset_quantization:
        raise ValueError("activity/onset ensemble quantization differs")
    return activity_quantization


def score_decision_evidence(scores: LoadedScores, quantization: int) -> np.ndarray:
    """Return the frozen ensemble decision as normalized 0/1 TM evidence.

    ``quantization`` is validated against the frozen ensemble artifact even
    though the deployed decision only needs its calibrated integer threshold.
    A sigmoid temperature would be a new score calibration; it is deliberately
    not invented or fitted on complex-6 in this parity benchmark.
    """

    if quantization != EXPECTED_QUANTIZATION:
        raise ValueError(f"unexpected frozen ensemble quantization: {quantization}")
    return (scores.scores >= scores.metadata.threshold).astype(np.float32)


def write_note_events_midi(
    path: str | Path,
    events: Sequence[NoteEvent],
    *,
    sample_rate: int,
    final_sample: int,
) -> dict[str, Any]:
    """Write sample-timed tracker events while preserving acoustic velocity."""

    ordered = sorted(
        events,
        key=lambda event: (
            event.sample_index,
            0 if event.kind == "note_off" else 1,
            event.pitch,
        ),
    )
    active: set[int] = set()
    midi_events: list[tuple[int, int, int, int]] = []
    note_ons = 0
    maximum_polyphony = 0
    velocities: list[int] = []
    for event in ordered:
        tick = max(0, round(event.sample_index / sample_rate * TICKS_PER_SECOND))
        if event.kind == "note_off":
            if event.pitch in active:
                midi_events.append((tick, 0, event.pitch, 0))
                active.remove(event.pitch)
            continue
        if event.pitch in active:
            # The tracker normally emits an explicit NoteOff before retrigger;
            # keep the MIDI valid even if a future state implementation does not.
            midi_events.append((tick, 0, event.pitch, 0))
            active.remove(event.pitch)
        velocity = int(np.clip(event.velocity, 1, 127))
        midi_events.append((tick, 1, event.pitch, velocity))
        active.add(event.pitch)
        note_ons += 1
        velocities.append(velocity)
        maximum_polyphony = max(maximum_polyphony, len(active))

    final_tick = max(0, round(final_sample / sample_rate * TICKS_PER_SECOND))
    for pitch in sorted(active):
        midi_events.append((final_tick, 0, pitch, 0))
    midi_events.sort(key=lambda item: (item[0], item[1], item[2]))

    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    previous_tick = 0
    for tick, is_on, pitch, velocity in midi_events:
        delta = tick - previous_tick
        previous_tick = tick
        if is_on:
            track.append(
                mido.Message("note_on", note=pitch, velocity=velocity, time=delta)
            )
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    track.append(
        mido.MetaMessage("end_of_track", time=max(0, final_tick - previous_tick))
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.save(destination)

    velocity_array = np.asarray(velocities, dtype=np.float64)
    return {
        "note_on_count": note_ons,
        "note_off_count": note_ons,
        "max_polyphony": maximum_polyphony,
        "velocity": {
            "minimum": int(velocity_array.min()) if velocities else None,
            "maximum": int(velocity_array.max()) if velocities else None,
            "mean": float(velocity_array.mean()) if velocities else None,
            "median": float(np.median(velocity_array)) if velocities else None,
            "distinct": len(set(velocities)),
            "source": "acoustic attack energy only; no TM confidence multiplier",
        },
    }


def write_note_events_tsv(
    path: str | Path, events: Sequence[NoteEvent], sample_rate: int
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["kind", "pitch", "velocity", "sample_index", "seconds", "frame_index"]
        )
        for event in events:
            writer.writerow(
                [
                    event.kind,
                    event.pitch,
                    event.velocity,
                    event.sample_index,
                    f"{event.sample_index / sample_rate:.9f}",
                    event.frame_index,
                ]
            )


def teacher_onsets(
    path: str | Path, *, midi_min: int, midi_max: int
) -> list[tuple[float, int]]:
    result: list[tuple[float, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pitch = int(row["pitch"])
            if midi_min <= pitch <= midi_max:
                result.append((float(row["start_sec"]), pitch))
    result.sort()
    return result


def tracker_onsets(
    events: Iterable[NoteEvent], sample_rate: int
) -> list[tuple[float, int]]:
    return sorted(
        (event.sample_index / sample_rate, event.pitch)
        for event in events
        if event.kind == "note_on"
    )


def midi_note_onsets(path: str | Path) -> tuple[list[tuple[float, int]], int]:
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError(f"asynchronous type-2 MIDI is unsupported: {path}")
    tempo = 500_000
    seconds = 0.0
    result: list[tuple[float, int]] = []
    active: dict[tuple[int, int], int] = {}
    maximum_polyphony = 0
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if message.type not in {"note_on", "note_off"}:
            continue
        key = (int(getattr(message, "channel", 0)), int(message.note))
        is_on = message.type == "note_on" and int(message.velocity) > 0
        if is_on:
            result.append((seconds, int(message.note)))
            active[key] = active.get(key, 0) + 1
            maximum_polyphony = max(maximum_polyphony, sum(active.values()))
        elif key in active:
            active[key] -= 1
            if active[key] <= 0:
                del active[key]
    return result, maximum_polyphony


def match_onsets(
    predicted: Sequence[tuple[float, int]],
    teacher: Sequence[tuple[float, int]],
    tolerance_seconds: float,
) -> dict[str, Any]:
    """Maximum-cardinality chronological matching, independently per pitch."""

    predicted_by_pitch: dict[int, list[float]] = {}
    teacher_by_pitch: dict[int, list[float]] = {}
    for seconds, pitch in predicted:
        predicted_by_pitch.setdefault(pitch, []).append(seconds)
    for seconds, pitch in teacher:
        teacher_by_pitch.setdefault(pitch, []).append(seconds)

    true_positives = 0
    timing_errors: list[float] = []
    for pitch in sorted(set(predicted_by_pitch) | set(teacher_by_pitch)):
        left = sorted(predicted_by_pitch.get(pitch, []))
        right = sorted(teacher_by_pitch.get(pitch, []))
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            delta = left[i] - right[j]
            if delta < -tolerance_seconds:
                i += 1
            elif delta > tolerance_seconds:
                j += 1
            else:
                true_positives += 1
                timing_errors.append(delta)
                i += 1
                j += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(teacher) - true_positives
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(teacher) if teacher else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors_ms = np.asarray(timing_errors, dtype=np.float64) * 1_000.0
    return {
        "tolerance_ms": tolerance_seconds * 1_000.0,
        "pitch_exact": True,
        "one_to_one": True,
        "predicted_onsets": len(predicted),
        "teacher_onsets": len(teacher),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_timing_error_ms": {
            "mean_signed": float(errors_ms.mean()) if timing_errors else None,
            "mean_absolute": float(np.abs(errors_ms).mean()) if timing_errors else None,
            "median_absolute": float(np.median(np.abs(errors_ms))) if timing_errors else None,
        },
    }


def run_tracker(
    audio: np.ndarray,
    activity_probability: np.ndarray,
    onset_probability: np.ndarray,
    config: TrackerConfig,
    block_samples: int,
) -> tuple[list[NoteEvent], dict[str, Any]]:
    tracker = PolyphonicTracker(config)
    events: list[NoteEvent] = []
    score_cursor = 0
    frontend_seconds = 0.0
    decoder_seconds = 0.0
    for first in range(0, int(audio.size), block_samples):
        block = audio[first : first + block_samples]
        started = time.perf_counter()
        evidence = tracker.frontend.push(block)
        frontend_seconds += time.perf_counter() - started
        count = evidence.frame_count
        if score_cursor + count > activity_probability.shape[0]:
            raise ValueError("acoustic frontend produced more frames than frozen scores")
        started = time.perf_counter()
        events.extend(
            tracker.process_evidence(
                evidence,
                tm_activity=activity_probability[score_cursor : score_cursor + count],
                tm_onset=onset_probability[score_cursor : score_cursor + count],
            )
        )
        decoder_seconds += time.perf_counter() - started
        score_cursor += count
    # The frozen TM/STFT+ grid starts at sample 1, then advances by one hop:
    # 1, 257, 513, ... .  It therefore emits ceil(samples / hop) frames.
    expected = (int(audio.size) + config.hop_size - 1) // config.hop_size
    if score_cursor != expected:
        raise AssertionError(
            f"tracker emitted {score_cursor} frames, expected ceil(audio/hop)={expected}"
        )
    return events, {
        "audio_samples": int(audio.size),
        "tracker_frames": score_cursor,
        "score_frames": int(activity_probability.shape[0]),
        "unused_terminal_score_frames": int(activity_probability.shape[0] - score_cursor),
        "frontend_seconds": frontend_seconds,
        "decoder_seconds": decoder_seconds,
        "tracker_seconds": frontend_seconds + decoder_seconds,
        "block_samples": block_samples,
    }


def render_track(
    track: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    config: TrackerConfig,
    block_samples: int,
    tolerance_seconds: float,
    target_input_p99: float | None,
) -> dict[str, Any]:
    package_id = str(track["package_id"])
    track_output = output_root / package_id
    track_output.mkdir(parents=True, exist_ok=True)
    wav_path = Path(track["wav"])
    teacher_events_path = Path(track["teacher_events"])
    score_root = source_root / "scores" / package_id
    activity_path = score_root / "activity-final.tsv"
    onset_path = score_root / "onset-primary.tsv"

    score_started = time.perf_counter()
    activity = load_score_file(activity_path)
    onset = load_score_file(onset_path)
    score_load_seconds = time.perf_counter() - score_started
    quantization = validate_score_pair(activity, activity_path, onset, onset_path)
    tm_activity = score_decision_evidence(activity, quantization)
    tm_onset = score_decision_evidence(onset, quantization)

    load_started = time.perf_counter()
    audio = load_audio_mono_channel_zero(wav_path, config.sample_rate)
    audio_load_resample_seconds = time.perf_counter() - load_started
    duration_seconds = float(audio.size) / config.sample_rate
    absolute_audio = np.abs(audio)
    raw_p99 = float(np.quantile(absolute_audio, 0.99))
    raw_peak = float(absolute_audio.max(initial=0.0))
    minimum_gain = 10.0 ** (-24.0 / 20.0)
    maximum_gain = 10.0 ** (24.0 / 20.0)
    desired_gain = (
        1.0
        if target_input_p99 is None
        else target_input_p99 / max(raw_p99, np.finfo(np.float32).tiny)
    )
    input_gain = float(np.clip(desired_gain, minimum_gain, maximum_gain))
    tracker_audio = (
        audio
        if input_gain == 1.0
        else np.ascontiguousarray(audio * np.float32(input_gain), dtype=np.float32)
    )
    input_gain_db = 20.0 * float(np.log10(input_gain))

    tracker_started = time.perf_counter()
    events, timing = run_tracker(
        tracker_audio, tm_activity, tm_onset, config, block_samples
    )
    tracker_wall_seconds = time.perf_counter() - tracker_started
    # ``run_tracker`` separately accumulates frontend and decoder timings.
    # Wall time is authoritative for the realtime factor.
    timing["tracker_wall_seconds"] = tracker_wall_seconds
    timing["tracker_x_realtime"] = (
        duration_seconds / tracker_wall_seconds if tracker_wall_seconds else None
    )

    midi_path = track_output / "tm-cleanroom-tracker.mid"
    events_path = track_output / "tm-cleanroom-tracker.events.tsv"
    write_started = time.perf_counter()
    midi_stats = write_note_events_midi(
        midi_path,
        events,
        sample_rate=config.sample_rate,
        final_sample=int(audio.size),
    )
    write_note_events_tsv(events_path, events, config.sample_rate)
    midi_write_seconds = time.perf_counter() - write_started

    teacher = teacher_onsets(
        teacher_events_path, midi_min=config.midi_min, midi_max=config.midi_max
    )
    current_match = match_onsets(
        tracker_onsets(events, config.sample_rate), teacher, tolerance_seconds
    )

    copied: dict[str, Any] = {}
    teacher_midi_source = source_root / "listening" / package_id / "neuralnote.mid"
    if teacher_midi_source.is_file():
        teacher_midi = track_output / "neuralnote-teacher.mid"
        shutil.copy2(teacher_midi_source, teacher_midi)
        copied["teacher_midi"] = file_ref(teacher_midi)

    stable_source = source_root / "listening" / package_id / "tm-stable.mid"
    baseline: dict[str, Any] | None = None
    if stable_source.is_file():
        stable_midi = track_output / "tm-stable-legacy-decoder.mid"
        shutil.copy2(stable_source, stable_midi)
        stable_onsets, stable_polyphony = midi_note_onsets(stable_midi)
        baseline = {
            "midi": file_ref(stable_midi),
            "note_on_count": len(stable_onsets),
            "max_polyphony": stable_polyphony,
            "teacher_onset_match": match_onsets(
                stable_onsets, teacher, tolerance_seconds
            ),
            "decoder": "previous frozen tm-stable state decoder",
        }

    end_to_end_seconds = (
        score_load_seconds
        + audio_load_resample_seconds
        + tracker_wall_seconds
        + midi_write_seconds
    )
    checksums: dict[str, Any] = {
        "wav": file_ref(wav_path),
        "teacher_events": file_ref(teacher_events_path),
        "activity_scores": file_ref(activity_path),
        "onset_scores": file_ref(onset_path),
        "output_midi": file_ref(midi_path),
        "output_events": file_ref(events_path),
        **copied,
    }
    return {
        "package_id": package_id,
        "source": track["source"],
        "split": track["split"],
        "duration_seconds": duration_seconds,
        "input_gain": {
            "raw_absolute_p99": raw_p99,
            "raw_absolute_peak": raw_peak,
            "target_absolute_p99": target_input_p99,
            "linear": input_gain,
            "db": input_gain_db,
            "clamp_db": [-24.0, 24.0],
            "clamped": not np.isclose(input_gain, desired_gain),
            "post_gain_absolute_p99": raw_p99 * input_gain,
            "post_gain_absolute_peak": raw_peak * input_gain,
            "scope": (
                "acoustic frontend/state/velocity only; frozen TM score TSV is unchanged"
            ),
        },
        "frames": timing,
        "runtime": {
            "score_tsv_load_seconds": score_load_seconds,
            "audio_load_resample_seconds": audio_load_resample_seconds,
            "midi_and_event_write_seconds": midi_write_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "end_to_end_x_realtime": (
                duration_seconds / end_to_end_seconds if end_to_end_seconds else None
            ),
            "scope_note": (
                "frozen score TSV loading is included, but TM model inference was "
                "precomputed; tracker_x_realtime measures spectral frontend plus decoder"
            ),
        },
        "tracker": {
            **midi_stats,
            "teacher_onset_match": current_match,
            "midi": file_ref(midi_path),
            "events_tsv": file_ref(events_path),
        },
        "previous_tm_stable": baseline,
        "checksums": checksums,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render clean-room spectral tracker + frozen strict-cap16-v3 TM scores "
            "on the selected complex six tracks."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--block-samples", type=int, default=1_024)
    parser.add_argument("--tolerance-ms", type=float, default=80.0)
    parser.add_argument(
        "--target-input-p99",
        type=float,
        help=(
            "optional fixed acoustic input-gain policy; scale each track so its "
            "absolute-amplitude p99 reaches this value, with gain clamped to ±24 dB"
        ),
    )
    parser.add_argument(
        "--track",
        action="append",
        default=[],
        help="optional package_id filter; repeat to render more than one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.block_samples <= 0:
        raise ValueError("--block-samples must be positive")
    if not np.isfinite(args.tolerance_ms) or args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms must be finite and non-negative")
    if args.target_input_p99 is not None and (
        not np.isfinite(args.target_input_p99) or args.target_input_p99 <= 0.0
    ):
        raise ValueError("--target-input-p99 must be finite and positive")

    selection_path = args.selection.resolve()
    source_root = args.source_root.resolve()
    default_output = (
        DEFAULT_GAIN_OUTPUT_ROOT
        if args.target_input_p99 is not None
        else DEFAULT_OUTPUT_ROOT
    )
    output_root = (args.output_root or default_output).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "tmgm-complex-dataset-selection-v1":
        raise ValueError(f"unsupported track selection: {selection_path}")
    tracks = list(selection["tracks"])
    if args.track:
        requested = set(args.track)
        available = {str(track["package_id"]) for track in tracks}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown --track package IDs: {missing}")
        tracks = [track for track in tracks if str(track["package_id"]) in requested]
    if not tracks:
        raise ValueError("selection contains no requested tracks")

    config = TrackerConfig()
    if (
        config.sample_rate != EXPECTED_SAMPLE_RATE
        or config.hop_size != EXPECTED_HOP_SIZE
        or config.note_count != EXPECTED_OUTPUTS
        or config.midi_min != EXPECTED_MIDI_MIN
        or config.tm_activity_weight != 1.0
        or config.tm_onset_weight != 1.0
        or config.global_velocity_mix != 0.5
    ):
        raise ValueError(
            "tracking defaults no longer match the benchmark/frozen TM contract"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for index, track in enumerate(tracks, start=1):
        package_id = str(track["package_id"])
        print(f"[{index}/{len(tracks)}] {package_id}", flush=True)
        result = render_track(
            track,
            source_root=source_root,
            output_root=output_root,
            config=config,
            block_samples=args.block_samples,
            tolerance_seconds=args.tolerance_ms / 1_000.0,
            target_input_p99=args.target_input_p99,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "note_on_count": result["tracker"]["note_on_count"],
                    "f1": result["tracker"]["teacher_onset_match"]["f1"],
                    "max_polyphony": result["tracker"]["max_polyphony"],
                    "x_realtime": result["frames"]["tracker_x_realtime"],
                }
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - started
    total_duration = sum(float(item["duration_seconds"]) for item in results)
    aggregate_predicted = sum(
        int(item["tracker"]["teacher_onset_match"]["predicted_onsets"])
        for item in results
    )
    aggregate_teacher = sum(
        int(item["tracker"]["teacher_onset_match"]["teacher_onsets"])
        for item in results
    )
    aggregate_tp = sum(
        int(item["tracker"]["teacher_onset_match"]["true_positives"])
        for item in results
    )
    aggregate_precision = aggregate_tp / aggregate_predicted if aggregate_predicted else 0.0
    aggregate_recall = aggregate_tp / aggregate_teacher if aggregate_teacher else 0.0
    aggregate_f1 = (
        2.0 * aggregate_precision * aggregate_recall
        / (aggregate_precision + aggregate_recall)
        if aggregate_precision + aggregate_recall
        else 0.0
    )
    baseline_rows = [
        item["previous_tm_stable"]
        for item in results
        if item["previous_tm_stable"] is not None
    ]
    baseline_predicted = sum(
        int(item["teacher_onset_match"]["predicted_onsets"])
        for item in baseline_rows
    )
    baseline_teacher = sum(
        int(item["teacher_onset_match"]["teacher_onsets"])
        for item in baseline_rows
    )
    baseline_tp = sum(
        int(item["teacher_onset_match"]["true_positives"])
        for item in baseline_rows
    )
    baseline_precision = baseline_tp / baseline_predicted if baseline_predicted else 0.0
    baseline_recall = baseline_tp / baseline_teacher if baseline_teacher else 0.0
    baseline_f1 = (
        2.0 * baseline_precision * baseline_recall
        / (baseline_precision + baseline_recall)
        if baseline_precision + baseline_recall
        else 0.0
    )
    tracker_seconds = sum(float(item["frames"]["tracker_wall_seconds"]) for item in results)
    metrics = {
        "schema": "tmgm-tm-cleanroom-tracker-complex6-v1",
        "model_policy": {
            "family": "Tsetlin Machine only",
            "tcn_used": False,
            "activity": "frozen strict-cap16-v3 seven-member ensemble, threshold -169",
            "onset": "frozen strict-cap16-v3 ten-member ensemble, threshold -492",
            "threshold_or_model_fitting_on_complex6": False,
            "score_to_evidence": (
                "float32(fused_quantized_score >= frozen ensemble threshold); "
                "identical to frozen pred_40..88 columns"
            ),
        },
        "tracker_config": asdict(config),
        "input_gain_policy": {
            "enabled": args.target_input_p99 is not None,
            "target_absolute_p99": args.target_input_p99,
            "gain_clamp_db": [-24.0, 24.0],
            "labels_or_tm_scores_used": False,
            "applies_to": "acoustic frontend/state/velocity only",
        },
        "selection": file_ref(selection_path),
        "source_artifact": file_ref(source_root / "manifest.json"),
        "match_contract": {
            "pitch_exact": True,
            "one_to_one": True,
            "tolerance_ms": args.tolerance_ms,
            "teacher": "NeuralNote events TSV restricted to MIDI 40..88",
        },
        "aggregate": {
            "tracks": len(results),
            "audio_duration_seconds": total_duration,
            "tracker_seconds": tracker_seconds,
            "tracker_x_realtime": total_duration / tracker_seconds if tracker_seconds else None,
            "script_wall_seconds": elapsed,
            "predicted_note_ons": aggregate_predicted,
            "teacher_note_ons": aggregate_teacher,
            "matched_note_ons": aggregate_tp,
            "precision": aggregate_precision,
            "recall": aggregate_recall,
            "f1": aggregate_f1,
            "maximum_polyphony": max(
                int(item["tracker"]["max_polyphony"]) for item in results
            ),
        },
        "previous_tm_stable_aggregate": {
            "tracks": len(baseline_rows),
            "predicted_note_ons": baseline_predicted,
            "teacher_note_ons": baseline_teacher,
            "matched_note_ons": baseline_tp,
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1,
            "maximum_polyphony": max(
                (
                    int(item["max_polyphony"])
                    for item in baseline_rows
                ),
                default=0,
            ),
        },
        "comparison_to_previous_tm_stable": {
            "predicted_note_on_delta": aggregate_predicted - baseline_predicted,
            "matched_note_on_delta": aggregate_tp - baseline_tp,
            "precision_delta": aggregate_precision - baseline_precision,
            "recall_delta": aggregate_recall - baseline_recall,
            "f1_delta": aggregate_f1 - baseline_f1,
            "interpretation": (
                "positive precision delta with negative recall delta means the "
                "physical attack gate removed both false and true onsets"
            ),
        },
        "tracks": {str(item["package_id"]): item for item in results},
    }
    metrics_path = output_root / "metrics.json"
    write_json(metrics_path, metrics)

    readme_path = output_root / "README.md"
    readme_path.write_text(
        "# TM-only clean-room tracker: complex-6\n\n"
        "Each track directory contains `tm-cleanroom-tracker.mid` (the new "
        "spectral physical gate/state machine over frozen strict-cap16-v3 TM "
        "decisions), `tm-stable-legacy-decoder.mid` (the previous decoder), "
        "and `neuralnote-teacher.mid`. The `.events.tsv` file preserves the "
        "new tracker events at their causal sample indices and acoustic MIDI "
        "velocities.\n\n"
        "No TCN, training, score calibration, or threshold fitting is run by "
        "this renderer. TM activity/onset decisions use the frozen -169/-492 "
        "thresholds. Acoustic evidence gates physical attacks/retriggers and "
        "sets velocity without a classifier-confidence multiplier.\n\n"
        + (
            f"This run applies a fixed target absolute-amplitude p99 of "
            f"{args.target_input_p99:g} to the acoustic path only, with gain "
            "clamped to ±24 dB. Per-track gain is recorded in `metrics.json`; "
            "the frozen TM scores remain unchanged.\n\n"
            if args.target_input_p99 is not None
            else "This run uses raw dataset gain (0 dB input gain).\n\n"
        )
        + f"Across {len(results)} tracks, the new tracker has pitch-exact "
        f"±{args.tolerance_ms:g} ms onset P/R/F1 "
        f"{aggregate_precision:.4f}/{aggregate_recall:.4f}/{aggregate_f1:.4f}; "
        "the previous decoder has "
        f"{baseline_precision:.4f}/{baseline_recall:.4f}/{baseline_f1:.4f}. "
        "See `metrics.json` for per-track counts, timing, checksums, and the "
        "comparison contract.\n",
        encoding="utf-8",
    )

    checksum_paths = [metrics_path, readme_path, selection_path]
    checksum_paths.extend(sorted(output_root.rglob("*.mid")))
    checksum_paths.extend(sorted(output_root.rglob("*.tsv")))
    checksum_rows = [
        f"{sha256_file(path)}  "
        + (
            path.relative_to(output_root).as_posix()
            if path.is_relative_to(output_root)
            else str(path)
        )
        for path in checksum_paths
    ]
    (output_root / "checksums.sha256").write_text(
        "\n".join(checksum_rows) + "\n", encoding="ascii"
    )
    print(json.dumps({"metrics": str(metrics_path), "aggregate": metrics["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
