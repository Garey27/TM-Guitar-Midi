from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from scripts.run_native_ablation import (
    RunnerPaths,
    _check_or_write_config,
    _model_ready,
    _resolved_run_config,
    build_evaluate_command,
    build_predict_command,
    build_train_command,
    load_experiments,
    main,
)
from tmgm_rt.native_dataset import write_native_dataset


def _write_mock_model(
    path: Path,
    *,
    version: int = 2,
    flags: int = 0x03,
    hard_probability: float = 0.0,
    feature_fingerprint: bytes | None = None,
) -> bytearray:
    feature_count = 1
    output_count = 1
    clause_count = 1
    state_bits = 8
    literal_count = 2
    literal_words = 1
    ta_bytes = clause_count * state_bits * literal_words * 4
    weight_bytes = output_count * clause_count * 4
    file_bytes = 256 + ta_bytes + weight_bytes
    header = bytearray(256)
    header[:8] = b"TMGMMOD\0"
    struct.pack_into("<IIII", header, 8, version, 256, 32, flags)
    struct.pack_into(
        "<IIIIIII", header, 24, 2, state_bits, feature_count,
        output_count, clause_count, literal_count, literal_words,
    )
    struct.pack_into(
        "<iiIfff", header, 52, 1, 0, literal_count, 2.0, 0.0, 1.0
    )
    struct.pack_into("<iiIII", header, 88, 40, 40, 1, 22_050, 256)
    struct.pack_into("<f", header, 108, hard_probability)
    struct.pack_into(
        "<QQQQQQ", header, 112, 256, ta_bytes, 256 + ta_bytes,
        weight_bytes, ta_bytes + weight_bytes, file_bytes,
    )
    if version == 3 and feature_fingerprint is not None:
        header[192:224] = feature_fingerprint
    value = header + bytes(ta_bytes + weight_bytes)
    if version == 3:
        value[160:192] = hashlib.sha256(value).digest()
    path.write_bytes(value)
    return value


def test_model_ready_accepts_v1_transitional_and_v2(tmp_path: Path):
    path = tmp_path / "model.tmgmmod"
    _write_mock_model(path, version=1)
    assert _model_ready(path, allow_legacy_feature_contract=True)
    _write_mock_model(path, version=1, flags=0x07, hard_probability=0.0)
    assert _model_ready(path, allow_legacy_feature_contract=True)
    _write_mock_model(path, version=2, flags=0x07, hard_probability=0.1)
    assert _model_ready(path, allow_legacy_feature_contract=True)
    _write_mock_model(path, version=2, flags=0x0F, hard_probability=1.0)
    assert _model_ready(path, allow_legacy_feature_contract=True)
    assert not _model_ready(path)


def test_model_ready_v3_requires_authenticated_exact_fingerprint(tmp_path: Path):
    path = tmp_path / "model.tmgmmod"
    fingerprint = bytes([0x42]) * 32
    _write_mock_model(path, version=3, feature_fingerprint=fingerprint)
    assert _model_ready(path, fingerprint)
    assert not _model_ready(path, bytes([0x43]) * 32)

    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    assert not _model_ready(path, fingerprint)


@pytest.mark.parametrize(
    ("version", "flags", "probability"),
    [
        (3, 0x03, 0.0),
        (2, 0x07, 0.0),
        (2, 0x0B, 0.0),
        (2, 0x13, 0.0),
    ],
)
def test_model_ready_rejects_unsupported_or_inconsistent_headers(
    tmp_path: Path, version: int, flags: int, probability: float
):
    path = tmp_path / "model.tmgmmod"
    _write_mock_model(
        path, version=version, flags=flags, hard_probability=probability
    )
    assert not _model_ready(path, allow_legacy_feature_contract=True)


def test_model_ready_rejects_truncation_and_wrong_declared_size(tmp_path: Path):
    path = tmp_path / "model.tmgmmod"
    value = _write_mock_model(path)
    path.write_bytes(value[:-1])
    assert not _model_ready(path, allow_legacy_feature_contract=True)

    value = _write_mock_model(path)
    struct.pack_into("<Q", value, 152, len(value) + 1)
    path.write_bytes(value)
    assert not _model_ready(path, allow_legacy_feature_contract=True)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "epochs": 12,
                    "validation_patience": 4,
                    "activity": {"specificity": 6.5},
                },
                "experiments": [
                    {
                        "name": "c512-generalize",
                        "clauses": 512,
                        "threshold": 256,
                        "max_literals": 32,
                        "activity": {"negative_samples": 6, "seed": 101},
                        "onset": {
                            "specificity": 3.5,
                            "negative_samples": 3,
                            "seed": 202,
                            "onset_sustain_hard_negatives": True,
                            "onset_sustain_hard_negative_weight_only": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_resolution_and_command_generation(tmp_path: Path):
    spec = load_experiments(_manifest(tmp_path / "ablations.json"))[0]

    assert spec.name == "c512-generalize"
    assert spec.heads["activity"] == {
        "epochs": 12,
        "clauses": 512,
        "threshold": 256,
        "specificity": 6.5,
        "negative_samples": 6.0,
        "max_literals": 32,
        "samples_per_launch": 128,
        "validation_patience": 4,
        "seed": 101,
        "onset_sustain_hard_negatives": False,
        "onset_sustain_hard_negative_probability": 1.0,
        "onset_sustain_hard_negative_weight_only": False,
        "rotate_output_update_order": False,
    }
    assert spec.heads["onset"]["specificity"] == 3.5
    assert spec.heads["onset"]["negative_samples"] == 3.0
    assert spec.heads["onset"]["onset_sustain_hard_negatives"] is True
    assert spec.heads["onset"]["onset_sustain_hard_negative_weight_only"] is True

    train = build_train_command(
        "train.exe",
        "train.tmgd",
        "validation.tmgd",
        "activity",
        spec.heads["activity"],
        "train.partial.tsv",
        "model.partial.tmgmmod",
    )
    assert train == [
        "train.exe",
        "train.tmgd",
        "--validation",
        "validation.tmgd",
        "--head",
        "activity",
        "--epochs",
        "12",
        "--clauses",
        "512",
        "--threshold",
        "256",
        "--specificity",
        "6.5",
        "--negative-samples",
        "6",
        "--max-literals",
        "32",
        "--samples-per-launch",
        "128",
        "--validation-patience",
        "4",
        "--seed",
        "101",
        "--output",
        "train.partial.tsv",
        "--model",
        "model.partial.tmgmmod",
    ]
    assert build_predict_command(
        "predict.exe", "validation.tmgd", "model.tmgmmod", "scores.tsv"
    ) == [
        "predict.exe",
        "validation.tmgd",
        "model.tmgmmod",
        "--output",
        "scores.tsv",
    ]
    assert build_predict_command(
        "predict.exe",
        "validation.tmgd",
        "legacy.tmgmmod",
        "scores.tsv",
        allow_legacy_feature_contract=True,
    )[-1] == "--allow-legacy-feature-contract"
    legacy_train = build_train_command(
        "train.exe",
        "train.tmgd",
        "validation.tmgd",
        "activity",
        spec.heads["activity"],
        "train.tsv",
        "model.tmgmmod",
        allow_legacy_feature_contract=True,
    )
    assert "--allow-legacy-feature-contract" in legacy_train
    onset_train = build_train_command(
        "train.exe",
        "train.tmgd",
        "validation.tmgd",
        "onset",
        spec.heads["onset"],
        "train.tsv",
        "model.tmgmmod",
    )
    hard_option = "--onset-sustain-hard-negative-probability"
    assert hard_option in onset_train
    assert onset_train[onset_train.index(hard_option) + 1] == "1"
    assert "--onset-sustain-hard-negative-weight-only" in onset_train
    assert "--rotate-output-update-order" not in train
    rotated_parameters = dict(spec.heads["activity"])
    rotated_parameters["rotate_output_update_order"] = True
    rotated_train = build_train_command(
        "tmgm_train_order_ablation.exe",
        "train.tmgd",
        "validation.tmgd",
        "activity",
        rotated_parameters,
        "train.tsv",
        "model.tmgmmod",
    )
    rotate_index = rotated_train.index("--rotate-output-update-order")
    assert rotated_train[rotate_index + 1] == "--output"
    assert build_evaluate_command(
        "python.exe",
        "evaluate.py",
        "validation.tmgd",
        ["activity.tsv", "onset.tsv"],
        "metrics.json",
    ) == [
        "python.exe",
        "evaluate.py",
        "--dataset",
        "validation.tmgd",
        "--scores",
        "activity.tsv",
        "onset.tsv",
        "--output",
        "metrics.json",
    ]


def test_dry_run_prints_both_heads_without_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config = _manifest(tmp_path / "ablations.json")
    output_root = tmp_path / "not-created"

    result = main(
        [
            "--config",
            str(config),
            "--train-dataset",
            str(tmp_path / "future-train.tmgd"),
            "--validation-dataset",
            str(tmp_path / "future-validation.tmgd"),
            "--output-root",
            str(output_root),
            "--train-exe",
            str(tmp_path / "future-train.exe"),
            "--predict-exe",
            str(tmp_path / "future-predict.exe"),
            "--dry-run",
        ]
    )

    printed = capsys.readouterr().out
    assert result == 0
    assert "EXPERIMENT c512-generalize" in printed
    assert "--head activity" in printed
    assert "--head onset" in printed
    assert "--validation-patience 4" in printed
    assert "evaluate_native_scores.py" in printed
    assert not output_root.exists()


def test_manifest_rejects_zero_patience(tmp_path: Path):
    manifest = {
        "defaults": {"validation_patience": 0},
        "experiments": [{"name": "bad"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_patience"):
        load_experiments(path)


def test_run_config_binds_fingerprint_and_rejects_same_width_resume(
    tmp_path: Path,
):
    spec = load_experiments(_manifest(tmp_path / "ablations.json"))[0]
    common = dict(
        train_dataset=tmp_path / "train.tmgd",
        validation_dataset=tmp_path / "validation.tmgd",
        output_root=tmp_path / "out",
        train_executable=tmp_path / "train.exe",
        predict_executable=tmp_path / "predict.exe",
        python_executable=tmp_path / "python.exe",
        evaluator=tmp_path / "evaluate.py",
    )
    first = RunnerPaths(
        **common, feature_fingerprint_sha256="42" * 32
    )
    second = RunnerPaths(
        **common, feature_fingerprint_sha256="43" * 32
    )
    path = tmp_path / "run-config.json"
    _check_or_write_config(path, _resolved_run_config(spec, first), force=False)
    with pytest.raises(RuntimeError, match="resolved config changed"):
        _check_or_write_config(
            path, _resolved_run_config(spec, second), force=False
        )


def _tiny_dataset(path: Path, fingerprint: bytes | None) -> None:
    write_native_dataset(
        path,
        np.asarray([[1], [0]], dtype=np.uint8),
        np.asarray([[1], [0]], dtype=np.uint8),
        np.asarray([[1], [0]], dtype=np.uint8),
        np.asarray([0, 1], dtype=np.uint32),
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=7,
        feature_fingerprint_sha256=fingerprint,
    )


def test_dataset_contract_default_closed_and_dry_run_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config = _manifest(tmp_path / "ablations.json")
    train = tmp_path / "train.tmgd"
    validation = tmp_path / "validation.tmgd"
    _tiny_dataset(train, None)
    _tiny_dataset(validation, None)
    arguments = [
        "--config", str(config),
        "--train-dataset", str(train),
        "--validation-dataset", str(validation),
        "--output-root", str(tmp_path / "out"),
        "--dry-run",
    ]
    with pytest.raises(ValueError, match="explicit.*legacy"):
        main(arguments)
    assert main([*arguments, "--allow-legacy-feature-contract"]) == 0
    assert "--allow-legacy-feature-contract" in capsys.readouterr().out

    fingerprint = bytes([0x55]) * 32
    _tiny_dataset(train, fingerprint)
    _tiny_dataset(validation, fingerprint)
    assert main(arguments) == 0
    assert fingerprint.hex() in capsys.readouterr().out

    _tiny_dataset(validation, bytes([0x56]) * 32)
    with pytest.raises(ValueError, match="fingerprints differ"):
        main(arguments)
