from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tmgm_rt.native_dataset import write_native_dataset
from tmgm_rt.native_score_eval import evaluate_score_files


def _write_scores(
    path: Path,
    *,
    head: str,
    prediction: np.ndarray,
    midi_min: int = 40,
) -> None:
    outputs = prediction.shape[1]
    columns = ["frame"]
    columns.extend(f"score_{note}" for note in range(midi_min, midi_min + outputs))
    columns.extend(f"pred_{note}" for note in range(midi_min, midi_min + outputs))
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
        scores = ["0"] * outputs
        lines.append(
            "\t".join([str(frame), *scores, *(str(int(value)) for value in row)])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    activity = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    onset = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    dataset = tmp_path / "validation.tmgd"
    write_native_dataset(
        dataset,
        np.zeros((6, 5), dtype=np.uint8),
        activity,
        onset,
        np.arange(6, dtype=np.uint32),
        midi_min=40,
        sample_rate=22050,
        hop_size=256,
        seed=19,
    )
    sidecar = Path(str(dataset) + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "selected_tracks": [
                    {"source": "alpha", "rows": 2},
                    {"source": "beta", "rows": 3},
                    {"source": "alpha", "rows": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    return dataset, sidecar, activity, onset


def test_evaluate_native_scores_streams_both_heads_and_sources(tmp_path: Path):
    dataset, sidecar, activity, onset = _fixture(tmp_path)
    activity_prediction = activity.copy()
    activity_prediction[1] = [0, 0, 1]
    activity_prediction[5] = [1, 0, 0]
    onset_prediction = onset.copy()
    onset_prediction[3] = [0, 1, 0]

    activity_scores = tmp_path / "activity.tsv"
    onset_scores = tmp_path / "onset.tsv"
    _write_scores(activity_scores, head="activity", prediction=activity_prediction)
    _write_scores(onset_scores, head="onset", prediction=onset_prediction)

    result = evaluate_score_files(
        dataset,
        [activity_scores, onset_scores],
        sidecar,
        batch_rows=2,
    )

    assert result["frames"] == 6
    activity_result = result["heads"]["activity"]
    assert activity_result["aggregate"]["true_positives"] == 5
    assert activity_result["aggregate"]["false_positives"] == 1
    assert activity_result["aggregate"]["false_negatives"] == 2
    assert activity_result["aggregate"]["precision"] == pytest.approx(5 / 6)
    assert activity_result["aggregate"]["recall"] == pytest.approx(5 / 7)
    assert activity_result["aggregate"]["predicted_mean_polyphony"] == 1.0
    assert activity_result["aggregate"]["target_mean_polyphony"] == pytest.approx(7 / 6)
    assert activity_result["by_source"]["alpha"]["frames"] == 3
    assert activity_result["by_source"]["beta"]["frames"] == 3
    assert activity_result["by_source"]["alpha"]["false_negatives"] == 2

    onset_result = result["heads"]["onset"]
    assert onset_result["aggregate"]["true_positives"] == 3
    assert onset_result["aggregate"]["false_positives"] == 1
    assert onset_result["aggregate"]["false_negatives"] == 1
    assert onset_result["by_source"]["alpha"]["f1"] == 1.0
    assert onset_result["by_source"]["beta"]["f1"] == pytest.approx(0.5)


def test_evaluate_native_scores_rejects_sidecar_row_mismatch(tmp_path: Path):
    dataset, sidecar, activity, _ = _fixture(tmp_path)
    sidecar.write_text(
        json.dumps({"selected_tracks": [{"source": "alpha", "rows": 5}]}),
        encoding="utf-8",
    )
    scores = tmp_path / "activity.tsv"
    _write_scores(scores, head="activity", prediction=activity)

    with pytest.raises(ValueError, match="rows sum"):
        evaluate_score_files(dataset, [scores], sidecar, batch_rows=2)


def test_evaluate_native_scores_rejects_noncontiguous_frames(tmp_path: Path):
    dataset, sidecar, activity, _ = _fixture(tmp_path)
    scores = tmp_path / "activity.tsv"
    _write_scores(scores, head="activity", prediction=activity)
    text = scores.read_text(encoding="utf-8")
    text = text.replace("\n1\t", "\n9\t", 1)
    scores.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="not expected"):
        evaluate_score_files(dataset, [scores], sidecar, batch_rows=2)
