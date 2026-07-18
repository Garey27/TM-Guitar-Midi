from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tmgm_rt.native_dataset import read_native_dataset, write_native_dataset
from tmgm_rt.native_repack import REPACK_SCHEMA, repack_native_corpus, repack_native_split


def _header_metadata(header):
    return {
        **asdict(header),
        "payload_sha256": header.payload_sha256.hex(),
        "feature_fingerprint_sha256": (
            header.feature_fingerprint_sha256.hex()
        ),
    }


def _write_sidecar(
    path: Path,
    header,
    *,
    split: str,
    feature_source: bool,
    track_id: str = "track-1",
    sampling_name: str = "natural",
):
    continuous_feature_count = (6 * header.note_count + 3) * 2
    metadata = {
        "format": "TMGMDAT",
        "version": 1,
        "export_schema": 1,
        "export_signature": f"{split}-{'features' if feature_source else 'labels'}",
        "split": split,
        "sampling": {
            "frame_sampling_policy": sampling_name,
            "onset_training_indices": {
                "name": "natural_order",
                "rows": header.onset_index_count,
            },
        },
        "source_track_counts": {"fixture": 1},
        "source_row_counts": {"fixture": header.frame_count},
        "category_counts": {"natural_uniform": header.frame_count},
        "track_count": 1,
        "rows": header.frame_count,
        "continuous_feature_count": continuous_feature_count,
        "kept_binary_features": header.feature_count,
        "binarizer": {
            "signature": "fixture-binarizer",
            "sha256": "1" * 64,
            "quantiles": [0.5, 0.8],
            "thresholds_shape": [continuous_feature_count, 2],
            "raw_thermometer_literals": continuous_feature_count * 2,
            "kept_binary_features": header.feature_count,
        },
        "frontend": {
            "sample_rate": header.sample_rate,
            "hop_size": header.hop_size,
            "fft_size": 64,
            "midi_min": header.midi_min,
            "midi_max": header.midi_max,
            "harmonics": 2,
            "ema_alpha": 0.08,
            "harmonic_local_contrast": feature_source,
            "contrast_offset_semitones": 1.5 if feature_source else 0.5,
        },
        "context": {"delays": [0, 1]},
        "targets": {
            "activity_outputs": True,
            "onset_outputs": True,
            "onset_width_frames": 3,
            "onset_delay_frames": 2 if feature_source else 3,
        },
        "selected_tracks": [
            {
                "source": "fixture",
                "id": track_id,
                "group": "group-1",
                "rows": header.frame_count,
                "cache_signature": (
                    "feature-cache" if feature_source else "label-cache"
                ),
            }
        ],
        "header": _header_metadata(header),
        "file_bytes": path.stat().st_size,
    }
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _make_split(root: Path, split: str):
    root.mkdir(parents=True, exist_ok=True)
    feature_root = root / "features"
    label_root = root / "labels"
    feature_root.mkdir(exist_ok=True)
    label_root.mkdir(exist_ok=True)
    feature_path = feature_root / f"{split}.tmgd"
    label_path = label_root / f"{split}.tmgd"
    rows = 5
    feature_values = np.asarray(
        [
            [1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    ignored_label_features = 1 - feature_values
    activity = np.asarray(
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
        dtype=np.uint8,
    )
    feature_onset = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
        dtype=np.uint8,
    )
    label_onset = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
        dtype=np.uint8,
    )
    indices = np.asarray([0, 2, 4], dtype=np.uint32)
    feature_header = write_native_dataset(
        feature_path,
        feature_values,
        activity,
        feature_onset,
        indices,
        midi_min=40,
        sample_rate=8_000,
        hop_size=16,
        seed=42,
    )
    label_header = write_native_dataset(
        label_path,
        ignored_label_features,
        activity,
        label_onset,
        indices,
        midi_min=40,
        sample_rate=8_000,
        hop_size=16,
        seed=42,
    )
    feature_metadata = _write_sidecar(
        feature_path, feature_header, split=split, feature_source=True
    )
    label_metadata = _write_sidecar(
        label_path, label_header, split=split, feature_source=False
    )
    return {
        "feature_path": feature_path,
        "label_path": label_path,
        "feature_values": feature_values,
        "activity": activity,
        "label_onset": label_onset,
        "indices": indices,
        "feature_metadata": feature_metadata,
        "label_metadata": label_metadata,
        "rows": rows,
    }


def _install_binarizer(feature_root: Path, splits: list[dict]):
    path = feature_root / "global-quantile-thermometer.npz"
    path.write_bytes(b"fixture quantile thermometer")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "schema": 1,
        "signature": "fixture-binarizer",
        "sha256": sha256,
        "train_rows": 10,
        "continuous_feature_count": 42,
        "quantiles": [0.5, 0.8],
        "raw_thermometer_literals": 84,
        "kept_binary_features": 7,
        "file_bytes": path.stat().st_size,
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    for split in splits:
        sidecar_path = split["feature_path"].with_suffix(".tmgd.json")
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        metadata["binarizer"]["sha256"] = sha256
        sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_repack_native_corpus_copies_verified_sections_and_rebuilds_metadata(
    tmp_path: Path,
):
    train = _make_split(tmp_path, "train")
    validation = _make_split(tmp_path, "validation")
    _install_binarizer(train["feature_path"].parent, [train, validation])
    output_root = tmp_path / "output"

    results = repack_native_corpus(
        train["feature_path"].parent,
        train["label_path"].parent,
        output_root,
    )
    assert [result.path.name for result in results] == [
        "train.tmgd",
        "validation.tmgd",
    ]
    for expected, result in zip((train, validation), results, strict=True):
        dataset = read_native_dataset(result.path)
        np.testing.assert_array_equal(
            dataset.feature_words, read_native_dataset(expected["feature_path"]).feature_words
        )
        np.testing.assert_array_equal(
            dataset.activity_words, read_native_dataset(expected["label_path"]).activity_words
        )
        np.testing.assert_array_equal(
            dataset.onset_words, read_native_dataset(expected["label_path"]).onset_words
        )
        np.testing.assert_array_equal(dataset.onset_indices, expected["indices"])

        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        assert metadata["frontend"]["harmonic_local_contrast"] is True
        assert metadata["targets"]["onset_delay_frames"] == 3
        assert metadata["repack"]["schema"] == REPACK_SCHEMA
        assert all(metadata["repack"]["verification"].values())
        assert metadata["selected_tracks"][0]["feature_cache_signature"] == (
            "feature-cache"
        )
        assert metadata["selected_tracks"][0]["label_cache_signature"] == (
            "label-cache"
        )
        assert hashlib.sha256(result.path.read_bytes()).hexdigest() == (
            metadata["file_sha256"]
        )

    copied_binarizer = output_root / "global-quantile-thermometer.npz"
    binarizer_metadata = json.loads(
        copied_binarizer.with_suffix(".npz.json").read_text(encoding="utf-8")
    )
    assert binarizer_metadata["provenance"]["schema"] == REPACK_SCHEMA
    assert hashlib.sha256(copied_binarizer.read_bytes()).hexdigest() == (
        binarizer_metadata["sha256"]
    )

    # A repacked sidecar is a valid provenance source for another repack; the
    # synthetic cache_signature and complete lineage must survive.
    rerepacked = tmp_path / "rerepacked-train.tmgd"
    rerepack_result = repack_native_split(
        output_root / "train.tmgd",
        train["label_path"],
        rerepacked,
        split="train",
    )
    rerepack_metadata = json.loads(
        rerepack_result.metadata_path.read_text(encoding="utf-8")
    )
    track = rerepack_metadata["selected_tracks"][0]
    assert isinstance(track["cache_signature"], str)
    assert "feature-cache" in track["cache_lineage"]
    assert "label-cache" in track["cache_lineage"]


def test_repack_rejects_nonidentical_activity(tmp_path: Path):
    fixture = _make_split(tmp_path, "train")
    label = read_native_dataset(fixture["label_path"])
    changed = fixture["activity"].copy()
    changed[0, 0] = 0
    header = write_native_dataset(
        fixture["label_path"],
        fixture["feature_values"],
        changed,
        fixture["label_onset"],
        fixture["indices"],
        midi_min=40,
        sample_rate=8_000,
        hop_size=16,
        seed=42,
    )
    _write_sidecar(
        fixture["label_path"], header, split="train", feature_source=False
    )
    assert label.header.frame_count == header.frame_count

    with pytest.raises(ValueError, match="activity sections are not byte-identical"):
        repack_native_split(
            fixture["feature_path"],
            fixture["label_path"],
            tmp_path / "bad-activity.tmgd",
            split="train",
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("track", "ordered selected_tracks disagree"),
        ("sampling", "sampling is not identical"),
    ],
)
def test_repack_rejects_metadata_row_alignment_mismatch(
    tmp_path: Path, change: str, message: str
):
    fixture = _make_split(tmp_path, "train")
    sidecar = fixture["label_path"].with_suffix(".tmgd.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if change == "track":
        metadata["selected_tracks"][0]["id"] = "another-track"
    else:
        metadata["sampling"]["frame_sampling_policy"] = "balanced"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        repack_native_split(
            fixture["feature_path"],
            fixture["label_path"],
            tmp_path / "bad-alignment.tmgd",
            split="train",
        )


def test_repack_rejects_header_timebase_mismatch(tmp_path: Path):
    fixture = _make_split(tmp_path, "train")
    header = write_native_dataset(
        fixture["label_path"],
        fixture["feature_values"],
        fixture["activity"],
        fixture["label_onset"],
        fixture["indices"],
        midi_min=40,
        sample_rate=8_000,
        hop_size=32,
        seed=42,
    )
    _write_sidecar(
        fixture["label_path"], header, split="train", feature_source=False
    )
    with pytest.raises(ValueError, match="headers disagree on hop_size"):
        repack_native_split(
            fixture["feature_path"],
            fixture["label_path"],
            tmp_path / "bad-timebase.tmgd",
            split="train",
        )


def test_repack_rejects_corrupt_input_payload(tmp_path: Path):
    fixture = _make_split(tmp_path, "train")
    with fixture["feature_path"].open("r+b") as stream:
        stream.seek(256)
        original = stream.read(1)
        stream.seek(256)
        stream.write(bytes([original[0] ^ 1]))

    with pytest.raises(ValueError, match="feature source payload checksum mismatch"):
        repack_native_split(
            fixture["feature_path"],
            fixture["label_path"],
            tmp_path / "corrupt.tmgd",
            split="train",
        )


def test_repack_rejects_sampling_indices_mismatch(tmp_path: Path):
    fixture = _make_split(tmp_path, "train")
    changed_indices = np.asarray([1, 2, 4], dtype=np.uint32)
    header = write_native_dataset(
        fixture["label_path"],
        fixture["feature_values"],
        fixture["activity"],
        fixture["label_onset"],
        changed_indices,
        midi_min=40,
        sample_rate=8_000,
        hop_size=16,
        seed=42,
    )
    _write_sidecar(
        fixture["label_path"], header, split="train", feature_source=False
    )
    with pytest.raises(ValueError, match="onset index sections are not byte-identical"):
        repack_native_split(
            fixture["feature_path"],
            fixture["label_path"],
            tmp_path / "bad-indices.tmgd",
            split="train",
        )


def test_repack_canonicalizes_missing_flags_and_allows_feature_only_ablation(
    tmp_path: Path,
):
    fixture = _make_split(tmp_path, "train")
    feature_sidecar = fixture["feature_path"].with_suffix(".tmgd.json")
    label_sidecar = fixture["label_path"].with_suffix(".tmgd.json")
    feature_metadata = json.loads(feature_sidecar.read_text(encoding="utf-8"))
    label_metadata = json.loads(label_sidecar.read_text(encoding="utf-8"))
    feature_metadata["frontend"]["expose_harmonic_local_profile"] = True
    feature_metadata["continuous_feature_count"] = 54
    label_metadata["frontend"].pop("expose_harmonic_local_profile", None)
    label_metadata["frontend"].pop("contrast_attack_features", None)
    feature_sidecar.write_text(json.dumps(feature_metadata), encoding="utf-8")
    label_sidecar.write_text(json.dumps(label_metadata), encoding="utf-8")

    result = repack_native_split(
        fixture["feature_path"],
        fixture["label_path"],
        tmp_path / "feature-only-profile.tmgd",
        split="train",
    )
    output_metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert output_metadata["frontend"]["expose_harmonic_local_profile"] is True
    assert output_metadata["frontend"]["contrast_attack_features"] is False


def test_repack_rejects_target_relabel_for_balanced_sampling(tmp_path: Path):
    fixture = _make_split(tmp_path, "train")
    for path in (fixture["feature_path"], fixture["label_path"]):
        sidecar = path.with_suffix(".tmgd.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["sampling"]["frame_sampling_policy"] = "balanced"
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="supported only for natural sampling"):
        repack_native_split(
            fixture["feature_path"],
            fixture["label_path"],
            tmp_path / "balanced-relabel.tmgd",
            split="train",
        )
