from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tmgm_rt.config import ContextConfig, FrontendConfig
from tmgm_rt.feature_contract import inspect_dataset_contract
from tmgm_rt.native_contract_upgrade import upgrade_native_feature_contract
from tmgm_rt.native_dataset import write_native_dataset
from tmgm_rt.stft_plus import CausalSTFTPlus


def _header_json(header) -> dict[str, object]:
    return {
        **asdict(header),
        "payload_sha256": header.payload_sha256.hex(),
        "feature_fingerprint_sha256": header.feature_fingerprint_sha256.hex(),
    }


def _legacy_corpus(root: Path) -> None:
    frontend = FrontendConfig()
    context = ContextConfig(delays=(0,))
    continuous = CausalSTFTPlus(frontend).feature_count
    binarizer = root / "global-quantile-thermometer.npz"
    np.savez(
        binarizer,
        quantiles=np.asarray([0.5], dtype=np.float64),
        thresholds=np.zeros((continuous, 1), dtype=np.float32),
        keep_columns=np.ones(continuous, dtype=np.bool_),
    )
    binarizer_sha = hashlib.sha256(binarizer.read_bytes()).hexdigest()
    binarizer_signature = "ab" * 32
    (root / "global-quantile-thermometer.npz.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "signature": binarizer_signature,
                "sha256": binarizer_sha,
                "train_rows": 2,
                "continuous_feature_count": continuous,
                "quantiles": [0.5],
                "raw_thermometer_literals": continuous,
                "kept_binary_features": continuous,
                "file_bytes": binarizer.stat().st_size,
            }
        ),
        encoding="utf-8",
    )

    legacy_frontend = asdict(frontend)
    for key in (
        "harmonic_local_contrast",
        "contrast_offset_semitones",
        "expose_harmonic_local_profile",
        "contrast_attack_features",
    ):
        legacy_frontend.pop(key)
    features = np.zeros((2, continuous), dtype=np.uint8)
    features[0, 0] = 1
    labels = np.zeros((2, frontend.note_count), dtype=np.uint8)
    labels[0, 0] = 1
    for split in ("train", "validation"):
        path = root / f"{split}.tmgd"
        header = write_native_dataset(
            path,
            features,
            labels,
            labels,
            np.asarray([0, 1], dtype=np.uint32),
            midi_min=frontend.midi_min,
            sample_rate=frontend.sample_rate,
            hop_size=frontend.hop_size,
            seed=42,
        )
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(
                {
                    "format": "TMGMDAT",
                    "version": 1,
                    "export_schema": 1,
                    "export_signature": ("11" if split == "train" else "22")
                    * 32,
                    "split": split,
                    "continuous_feature_count": continuous,
                    "kept_binary_features": continuous,
                    "binarizer": {
                        "sha256": binarizer_sha,
                        "signature": binarizer_signature,
                    },
                    "frontend": legacy_frontend,
                    "context": asdict(context),
                    "header": _header_json(header),
                    "file_bytes": path.stat().st_size,
                }
            ),
            encoding="utf-8",
        )


def test_upgrade_is_verified_payload_preserving_and_idempotent(tmp_path: Path) -> None:
    _legacy_corpus(tmp_path)
    payload_hashes = {
        split: inspect_dataset_contract(
            tmp_path / f"{split}.tmgd", allow_legacy=True
        ).checksum_sha256
        for split in ("train", "validation")
    }
    dry_run = upgrade_native_feature_contract(tmp_path, dry_run=True)
    assert dry_run["dry_run"] is True
    assert inspect_dataset_contract(
        tmp_path / "train.tmgd", allow_legacy=True
    ).legacy

    result = upgrade_native_feature_contract(tmp_path)
    assert result["payload_rewritten"] is False
    for split in ("train", "validation"):
        contract = inspect_dataset_contract(tmp_path / f"{split}.tmgd")
        assert contract.format_version == 2 and not contract.legacy
        assert contract.checksum_sha256 == payload_hashes[split]
        assert contract.feature_fingerprint_sha256 == result[
            "feature_fingerprint_sha256"
        ]

    tracked = sorted(tmp_path.glob("*"))
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
        if path.is_file()
    }
    second = upgrade_native_feature_contract(tmp_path)
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
        if path.is_file()
    }
    assert second["upgrade_id"] == result["upgrade_id"]
    assert second_hashes == first_hashes
