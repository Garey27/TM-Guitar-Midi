from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tmgm_rt.contiguous_eval import (
    MANIFEST_SCHEMA,
    evaluate_manifest_scores,
)
from tmgm_rt.native_dataset import read_native_dataset_header, write_native_dataset


def _write_scores(path: Path, head: str, prediction: np.ndarray) -> None:
    midi_min = 40
    outputs = prediction.shape[1]
    columns = ["frame"]
    columns += [f"score_{pitch}" for pitch in range(midi_min, midi_min + outputs)]
    columns += [f"pred_{pitch}" for pitch in range(midi_min, midi_min + outputs)]
    lines = [
        "#TMGM_SCORES_V1",
        f"#head={head}",
        f"#frames={prediction.shape[0]}",
        f"#outputs={outputs}",
        f"#midi_min={midi_min}",
        "#sample_rate=22050",
        "#hop_size=256",
        "#threshold=7",
        "\t".join(columns),
    ]
    for frame, row in enumerate(prediction):
        lines.append(
            "\t".join(
                [str(frame), *(["0"] * outputs), *(str(int(value)) for value in row)]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    wav = tmp_path / "track.wav"
    wav.write_bytes(b"deterministic-wave-fixture")
    events = tmp_path / "track.events.tsv"
    with events.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("pitch", "start_frame", "end_frame"))
        writer.writerows(
            (
                (40, 1, 4),
                (41, 1, 3),
                (40, 4, 6),
                (42, 5, 8),
            )
        )

    activity = np.zeros((8, 3), dtype=np.uint8)
    activity[1:6, 0] = 1
    activity[1:3, 1] = 1
    activity[5:8, 2] = 1
    dataset = tmp_path / "track.tmgd"
    write_native_dataset(
        dataset,
        np.zeros((8, 5), dtype=np.uint8),
        np.zeros_like(activity),
        np.zeros_like(activity),
        np.zeros(0, dtype=np.uint32),
        midi_min=40,
        sample_rate=22050,
        hop_size=256,
        seed=0,
    )
    header = read_native_dataset_header(dataset)
    metadata = Path(str(dataset) + ".json")
    metadata.write_text(
        json.dumps(
            {
                "schema": "tmgm-native-wav-inference-v1",
                "input": {
                    "path": str(wav.resolve()),
                    "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                },
                "header": {
                    "frame_count": header.frame_count,
                    "feature_count": header.feature_count,
                    "note_count": header.note_count,
                    "midi_min": header.midi_min,
                    "midi_max": header.midi_max,
                    "sample_rate": header.sample_rate,
                    "hop_size": header.hop_size,
                },
                "resampled_mono_samples": header.frame_count * header.hop_size,
                "causality": {
                    "all_frames_in_source_order": True,
                    "lookahead_frames": 0,
                    "strictly_causal": True,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "tracks": [
                    {
                        "key": "fixture",
                        "source": "test-source",
                        "corpus_split": "test",
                        "evaluation_role": "test",
                        "id": "fixture-id",
                        "group": "fixture-group",
                        "wav": str(wav.resolve()),
                        "events": str(events.resolve()),
                        "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                        "events_sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
                        "feature_sets": {
                            "plain": {
                                "dataset": str(dataset.resolve()),
                                "metadata": str(metadata.resolve()),
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scores = tmp_path / "scores"
    scores.mkdir()
    _write_scores(scores / "fixture.activity.tsv", "activity", activity)
    onset = np.zeros_like(activity)
    onset[3, 0] = 1
    onset[6, 0] = 1
    onset[3, 1] = 1
    onset[5, 2] = 1
    _write_scores(scores / "fixture.onset.tsv", "onset", onset)
    return manifest, scores, metadata, events


def test_contiguous_eval_separates_training_alignment_from_wall_clock_and_subsets(
    tmp_path: Path,
):
    manifest, scores, _, _ = _fixture(tmp_path)
    result = evaluate_manifest_scores(
        manifest,
        "plain",
        scores,
        training_onset_delay_frames=2,
        onset_width_frames=1,
        target_aligned_tolerances=(2, 3, 4),
        wall_clock_tolerances=(2, 3, 4, 6),
        retrigger_silence_frames=3,
        chord_window_frames=0,
        low_midi_max=42,
        batch_rows=2,
    )

    aggregate = result["aggregate"]
    assert aggregate["frames"] == 8
    assert aggregate["activity"]["all"]["f1"] == 1.0
    assert aggregate["activity"]["polyphonic_frames"]["frames"] == 3
    aligned = aggregate["onset"]["target_aligned"]["events"]
    wall_clock = aggregate["onset"]["wall_clock"]["events"]
    assert aligned["0"]["true_positives"] == 3
    assert aligned["2"]["true_positives"] == 4
    assert wall_clock["0"]["true_positives"] == 1
    assert wall_clock["2"]["true_positives"] == 4
    assert aligned["0"]["target_subsets"]["retrigger"] == {
        "targets": 1,
        "matched_targets": 1,
        "recall": 1.0,
    }
    assert aligned["0"]["target_subsets"]["chord_onset"]["targets"] == 2
    assert aligned["0"]["target_subsets"]["single_onset"]["targets"] == 2


def test_contiguous_eval_rejects_score_length_mismatch(tmp_path: Path):
    manifest, scores, _, _ = _fixture(tmp_path)
    _write_scores(
        scores / "fixture.activity.tsv",
        "activity",
        np.zeros((7, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="score metadata does not match"):
        evaluate_manifest_scores(manifest, "plain", scores)


def test_contiguous_eval_rejects_native_export_input_metadata_mismatch(tmp_path: Path):
    manifest, scores, metadata, _ = _fixture(tmp_path)
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    raw["input"]["sha256"] = "0" * 64
    metadata.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="input SHA-256 disagrees"):
        evaluate_manifest_scores(manifest, "plain", scores)


def test_contiguous_eval_rejects_changed_teacher_events(tmp_path: Path):
    manifest, scores, _, events = _fixture(tmp_path)
    events.write_text(events.read_text(encoding="utf-8") + "42\t7\t8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher events SHA-256 mismatch"):
        evaluate_manifest_scores(manifest, "plain", scores)
