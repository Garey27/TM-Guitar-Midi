from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

import tmgm_rt.offline_listening as listening
from tmgm_rt.config import ContextConfig, FrontendConfig
from tmgm_rt.feature_semantics import binary_feature_semantics
from tmgm_rt.native_dataset import write_native_dataset
from tmgm_rt.stft_plus import CausalSTFTPlus


def _sha(path: Path) -> str:
    return listening.sha256_file(path)


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _score(path: Path, *, head: str, frames: int, outputs: int) -> None:
    notes = range(40, 40 + outputs)
    lines = [
        "#TMGM_SCORES_V1",
        f"#head={head}",
        f"#frames={frames}",
        f"#outputs={outputs}",
        "#midi_min=40",
        "#sample_rate=8000",
        "#hop_size=16",
        "#threshold=0",
        "\t".join(
            [
                "frame",
                *(f"score_{note}" for note in notes),
                *(f"pred_{note}" for note in notes),
            ]
        ),
    ]
    for frame in range(frames):
        scores = [1 if (frame + note) % 3 == 0 else -1 for note in range(outputs)]
        lines.append(
            "\t".join(
                [
                    str(frame),
                    *(str(value) for value in scores),
                    *(str(int(value >= 0)) for value in scores),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, str, int]:
    frontend = FrontendConfig(
        sample_rate=8000,
        hop_size=16,
        fft_size=64,
        midi_min=40,
        midi_max=42,
        harmonics=2,
    )
    context = ContextConfig(delays=(0, 1))
    continuous = CausalSTFTPlus(frontend).feature_count * len(context.delays)
    semantics = binary_feature_semantics(
        frontend,
        context,
        binarizer_sha256="1" * 64,
        binarizer_signature="fixture-encoder",
        continuous_feature_count=continuous,
        binary_feature_count=7,
    )
    fingerprint = semantics["fingerprint_sha256"]
    source = tmp_path / "source.wav"
    source.write_bytes(b"not-a-real-wave-but-content-addressed")
    dataset = tmp_path / "track.tmgd"
    frames = 5
    labels = np.zeros((frames, 3), dtype=np.uint8)
    header = write_native_dataset(
        dataset,
        np.zeros((frames, 7), dtype=np.uint8),
        labels,
        labels,
        np.empty(0, dtype=np.uint32),
        midi_min=40,
        sample_rate=8000,
        hop_size=16,
        seed=0,
        feature_fingerprint_sha256=fingerprint,
    )
    metadata = {
        "frontend": asdict(frontend),
        "context": asdict(context),
        "continuous_feature_count": continuous,
        "kept_binary_features": 7,
        "binarizer": {
            "sha256": "1" * 64,
            "signature": "fixture-encoder",
        },
        "feature_semantics": semantics,
        "header": {
            "frame_count": header.frame_count,
            "feature_count": header.feature_count,
            "note_count": header.note_count,
            "midi_min": header.midi_min,
            "midi_max": header.midi_max,
            "sample_rate": header.sample_rate,
            "hop_size": header.hop_size,
            "payload_sha256": header.payload_sha256.hex(),
        },
        "input": {"path": str(source), "sha256": _sha(source)},
        "output": {"path": str(dataset), "sha256": _sha(dataset)},
    }
    metadata_path = tmp_path / "track.tmgd.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    predictor = tmp_path / "predictor.exe"
    predictor.write_bytes(b"fixture predictor")
    activity = tmp_path / "activity.tmgmmod"
    activity.write_bytes(b"activity model")
    onset = tmp_path / "onset.tmgmmod"
    onset.write_bytes(b"onset model")
    reference = tmp_path / "training.tmgd.json"
    reference.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = {
        "schema": listening.MANIFEST_SCHEMA,
        "predictor": {"executable": _ref(predictor), "arguments": []},
        "cache_root": str(tmp_path / "cache"),
        "output_root": str(tmp_path / "output"),
        "feature_banks": {
            "plain": {
                "fingerprint_sha256": fingerprint,
                "allow_legacy_zero_fingerprint": False,
                "provenance": {
                    "reference_metadata": _ref(reference),
                    "binarizer_sha256": "1" * 64,
                },
            }
        },
        "tracks": [
            {
                "id": "track",
                "source_wav": _ref(source),
                "feature_banks": {
                    "plain": {
                        "dataset": _ref(dataset),
                        "metadata": _ref(metadata_path),
                    }
                },
            }
        ],
        "models": [
            {
                "id": "activity",
                "head": "activity",
                "feature_bank": "plain",
                "feature_fingerprint_sha256": fingerprint,
                "model": _ref(activity),
                "allow_legacy_zero_fingerprint": False,
                "provenance": {"training_metadata": _ref(reference)},
            },
            {
                "id": "onset",
                "head": "onset",
                "feature_bank": "plain",
                "feature_fingerprint_sha256": fingerprint,
                "model": _ref(onset),
                "allow_legacy_zero_fingerprint": False,
                "provenance": {"training_metadata": _ref(reference)},
            },
        ],
        "renders": [
            {
                "id": "single",
                "tracks": ["track"],
                "activity": {"member": "activity"},
                "onset": {"member": "onset"},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, fingerprint, frames


def _patch_contracts(monkeypatch, fingerprint: str) -> None:
    monkeypatch.setattr(
        listening,
        "_dataset_contract",
        lambda path, allow_legacy: listening.DatasetContract(
            2, 7, 3, 40, 42, 8000, 16, fingerprint, False, "2" * 64
        ),
    )

    def model_contract(path: Path, allow_legacy: bool) -> listening.ModelContract:
        head = "activity" if "activity" in path.name else "onset"
        return listening.ModelContract(
            3, head, 7, 3, 40, 42, 8000, 16, fingerprint, False, "3" * 64
        )

    monkeypatch.setattr(listening, "_model_contract", model_contract)


def test_dry_run_validates_and_never_creates_cache_or_outputs(tmp_path, monkeypatch):
    manifest, fingerprint, _ = _fixture(tmp_path)
    _patch_contracts(monkeypatch, fingerprint)
    result = listening.run_listening_manifest(manifest, dry_run=True)
    assert result["dry_run"] is True
    assert len(result["predictions"]) == 2
    assert {row["head"] for row in result["predictions"]} == {"activity", "onset"}
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "output").exists()


def test_embedded_same_width_wrong_bank_fails_closed(tmp_path, monkeypatch):
    manifest, fingerprint, _ = _fixture(tmp_path)
    _patch_contracts(monkeypatch, fingerprint)
    original = listening._model_contract

    def wrong(path: Path, allow_legacy: bool):
        value = original(path, allow_legacy)
        if "onset" in path.name:
            return listening.ModelContract(
                value.format_version,
                value.head,
                value.feature_count,
                value.outputs,
                value.midi_min,
                value.midi_max,
                value.sample_rate,
                value.hop_size,
                "f" * 64,
                value.legacy,
                value.checksum_sha256,
            )
        return value

    monkeypatch.setattr(listening, "_model_contract", wrong)
    try:
        listening.build_listening_plan(manifest)
    except ValueError as error:
        assert "embedded fingerprint" in str(error)
    else:
        raise AssertionError("same-width wrong-bank model was accepted")


def test_prediction_cache_and_raw_stable_render_are_reproducible(tmp_path, monkeypatch):
    manifest, fingerprint, frames = _fixture(tmp_path)
    _patch_contracts(monkeypatch, fingerprint)
    calls: list[list[str]] = []

    def runner(command):
        calls.append(list(command))
        model = Path(command[-3])
        head = "activity" if "activity" in model.name else "onset"
        _score(Path(command[-1]), head=head, frames=frames, outputs=3)

    first = listening.run_listening_manifest(manifest, process_runner=runner)
    assert first["cache_misses"] == 2
    assert len(calls) == 2
    result = first["renders"]["single"]["track"]
    assert Path(result["raw_midi"]["path"]).is_file()
    assert Path(result["stable_midi"]["path"]).is_file()

    calls.clear()
    second = listening.run_listening_manifest(manifest, process_runner=runner)
    assert second["cache_hits"] == 2
    assert calls == []


def test_cache_key_is_content_addressed():
    base = dict(
        predictor_sha256="1" * 64,
        predictor_arguments=(),
        model_sha256="2" * 64,
        dataset_sha256="3" * 64,
    )
    first = listening.prediction_cache_key(**base)
    assert first == listening.prediction_cache_key(**base)
    assert first != listening.prediction_cache_key(
        **{**base, "dataset_sha256": "4" * 64}
    )
