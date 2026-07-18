from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tmgm_rt.native_dataset import write_native_dataset
from tmgm_rt.native_score_ensemble import (
    ARTIFACT_FORMAT,
    MemberSpec,
    apply_score_ensemble,
    calibrate_common_threshold,
    estimate_robust_scale,
    fit_score_ensemble,
    load_score_file,
    parse_member_spec,
)


def _dataset(
    path: Path,
    activity: np.ndarray,
    onset: np.ndarray,
    *,
    sample_rate: int = 22050,
) -> Path:
    write_native_dataset(
        path,
        np.zeros((activity.shape[0], 7), dtype=np.uint8),
        activity,
        onset,
        np.arange(activity.shape[0], dtype=np.uint32),
        midi_min=40,
        sample_rate=sample_rate,
        hop_size=256,
        seed=23,
    )
    return path


def _scores(
    path: Path,
    values: np.ndarray,
    *,
    head: str,
    threshold: int,
    sample_rate: int = 22050,
    member_id: str | None = None,
) -> Path:
    values = np.asarray(values, dtype=np.int32)
    notes = range(40, 40 + values.shape[1])
    lines = [
        "#TMGM_SCORES_V1",
        f"#head={head}",
        f"#frames={values.shape[0]}",
        f"#outputs={values.shape[1]}",
        "#midi_min=40",
        f"#sample_rate={sample_rate}",
        "#hop_size=256",
        f"#threshold={threshold}",
    ]
    if member_id is not None:
        lines.append(f"#member_id={member_id}")
    lines.append(
        "\t".join(
            [
                "frame",
                *(f"score_{note}" for note in notes),
                *(f"pred_{note}" for note in notes),
            ]
        )
    )
    for frame, row in enumerate(values):
        prediction = row >= threshold
        lines.append(
            "\t".join(
                [
                    str(frame),
                    *(str(int(value)) for value in row),
                    *(str(int(value)) for value in prediction),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, list[MemberSpec]]:
    onset = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    activity = np.maximum.accumulate(onset, axis=0)
    dataset = _dataset(tmp_path / "validation.tmgd", activity, onset)
    base = np.asarray(
        [
            [8, -4, -8],
            [-5, 7, -6],
            [-4, -7, -5],
            [6, -5, 8],
            [-7, -4, -6],
            [-5, 6, -4],
        ],
        dtype=np.int32,
    )
    first = _scores(
        tmp_path / "first.tsv",
        base + 10,
        head="onset",
        threshold=10,
        member_id="first",
    )
    second = _scores(
        tmp_path / "second.tsv",
        base * 10 - 30,
        head="onset",
        threshold=-30,
        member_id="second",
    )
    third_values = base.copy()
    third_values[2, 2] = 12
    third = _scores(
        tmp_path / "third.tsv",
        third_values + 100,
        head="onset",
        threshold=100,
        member_id="third",
    )
    return dataset, [
        MemberSpec("first", first),
        MemberSpec("second", second),
        MemberSpec("third", third),
    ]


def test_fit_and_apply_ensemble_with_explicit_member_order(tmp_path: Path):
    dataset, members = _fixture(tmp_path)
    artifact_path = tmp_path / "ensemble.json"
    artifact = fit_score_ensemble(dataset, members, artifact_path)

    assert artifact["format"] == ARTIFACT_FORMAT
    assert artifact["head"] == "onset"
    assert set(artifact["calibration"]["candidates"]) == {
        "mean",
        "top2_mean",
        "max",
    }
    assert artifact["fusion"] in {"mean", "top2_mean", "max"}
    assert [member["id"] for member in artifact["members"]] == [
        "first",
        "second",
        "third",
    ]
    assert all(member["robust_scale"] > 0 for member in artifact["members"])
    assert json.loads(artifact_path.read_text(encoding="utf-8"))["format"] == (
        ARTIFACT_FORMAT
    )

    # Apply is allowed to use another frame count, as a WAV inference TMGD does.
    inference_activity = np.zeros((3, 3), dtype=np.uint8)
    inference_onset = np.zeros_like(inference_activity)
    inference = _dataset(
        tmp_path / "wav-inference.tmgd", inference_activity, inference_onset
    )
    apply_members: list[MemberSpec] = []
    for member, fitted in zip(members, artifact["members"], strict=True):
        source = load_score_file(member.path).scores[:3]
        target = _scores(
            tmp_path / f"wav-{member.identifier}.tsv",
            source,
            head="onset",
            threshold=int(fitted["threshold"]),
            member_id=member.identifier,
        )
        apply_members.append(MemberSpec(member.identifier, target))

    output = tmp_path / "ensemble-scores.tsv"
    summary = apply_score_ensemble(
        artifact_path, inference, apply_members, output
    )
    loaded = load_score_file(output)
    assert loaded.metadata.head == "onset"
    assert loaded.metadata.frames == 3
    assert loaded.metadata.threshold == artifact["ensemble_threshold"]
    assert loaded.scores.shape == (3, 3)
    assert summary["output"] == str(output.resolve())


def test_apply_rejects_swapped_member_identity_even_when_shapes_match(tmp_path: Path):
    dataset, members = _fixture(tmp_path)
    artifact_path = tmp_path / "ensemble.json"
    fit_score_ensemble(dataset, members, artifact_path)

    swapped = [members[1], members[0], members[2]]
    with pytest.raises(ValueError, match="member order/identity"):
        apply_score_ensemble(artifact_path, dataset, swapped, tmp_path / "bad.tsv")


def test_fit_rejects_timebase_mismatch(tmp_path: Path):
    dataset, members = _fixture(tmp_path)
    bad_values = load_score_file(members[1].path).scores
    bad_path = _scores(
        tmp_path / "wrong-rate.tsv",
        bad_values,
        head="onset",
        threshold=-30,
        sample_rate=44100,
        member_id="second",
    )
    bad_members = [members[0], MemberSpec("second", bad_path), members[2]]
    with pytest.raises(ValueError, match="does not match dataset"):
        fit_score_ensemble(dataset, bad_members, tmp_path / "bad.json")


def test_score_loader_rejects_frame_and_pitch_order(tmp_path: Path):
    dataset, members = _fixture(tmp_path)
    del dataset
    text = members[0].path.read_text(encoding="utf-8")
    wrong_frame = tmp_path / "wrong-frame.tsv"
    wrong_frame.write_text(text.replace("\n1\t", "\n9\t", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="is not expected"):
        load_score_file(wrong_frame)

    wrong_pitch = tmp_path / "wrong-pitch.tsv"
    wrong_pitch.write_text(
        text.replace("score_40", "score_41", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected score TSV columns"):
        load_score_file(wrong_pitch)


def test_robust_normalization_is_scale_invariant_and_guard_is_enforced():
    centered = np.asarray(
        [[-7, -4, -2, 1], [3, 6, 9, 12]], dtype=np.int32
    )
    first_scale = estimate_robust_scale(centered)
    second_scale = estimate_robust_scale(centered * 10)
    assert second_scale == pytest.approx(first_scale * 10)

    scores = np.asarray([[9, 8, 7], [6, 5, 4]], dtype=np.int32)
    truth = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=bool)
    result = calibrate_common_threshold(scores, truth, 1.0)
    assert result["predicted_mean_polyphony"] <= result["target_mean_polyphony"]
    assert result["threshold"] == 9
    assert result["f1"] == 1.0


def test_member_parser_requires_explicit_identity():
    parsed = parse_member_spec("model-a=C:/scores/member.tsv")
    assert parsed.identifier == "model-a"
    assert parsed.path == Path("C:/scores/member.tsv")
    with pytest.raises(ValueError, match="ID=path"):
        parse_member_spec("member.tsv")
