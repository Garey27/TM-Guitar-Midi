from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np
import pytest

from tmgm_rt.feature_contract import (
    inspect_dataset_contract,
    inspect_model_contract,
    validate_dataset_model_contract,
)
from tmgm_rt.native_dataset import write_native_dataset


def _write_model(path: Path, fingerprint: bytes) -> None:
    features = 3
    outputs = 2
    clauses = 2
    state_bits = 8
    literal_count = features * 2
    literal_words = 1
    ta = np.zeros((clauses, state_bits, literal_words), dtype="<u4")
    weights = np.zeros((outputs, clauses), dtype="<i4")
    ta_bytes = ta.nbytes
    weight_bytes = weights.nbytes
    file_bytes = 256 + ta_bytes + weight_bytes
    header = bytearray(256)
    header[:8] = b"TMGMMOD\0"
    struct.pack_into("<IIII", header, 8, 3, 256, 32, 3)
    struct.pack_into(
        "<IIIIIII",
        header,
        24,
        1,
        state_bits,
        features,
        outputs,
        clauses,
        literal_count,
        literal_words,
    )
    struct.pack_into("<iiIfffIQ", header, 52, 8, 0, 6, 4.0, 1.0, 1.0, 1, 7)
    struct.pack_into("<iiIII", header, 88, 40, 41, 1, 22_050, 256)
    struct.pack_into(
        "<QQQQQQ",
        header,
        112,
        256,
        ta_bytes,
        256 + ta_bytes,
        weight_bytes,
        ta_bytes + weight_bytes,
        file_bytes,
    )
    header[192:224] = fingerprint
    raw = header + ta.tobytes() + weights.tobytes()
    raw[160:192] = hashlib.sha256(raw).digest()
    path.write_bytes(raw)


def _write_dataset(path: Path, fingerprint: bytes | None) -> None:
    features = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    activity = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    onset = activity.copy()
    write_native_dataset(
        path,
        features,
        activity,
        onset,
        np.asarray([0, 1], dtype=np.uint32),
        midi_min=40,
        sample_rate=22_050,
        hop_size=256,
        seed=7,
        feature_fingerprint_sha256=fingerprint,
    )


def test_contract_inspection_and_exact_match(tmp_path: Path) -> None:
    fingerprint = bytes([0x42]) * 32
    dataset_path = tmp_path / "data.tmgd"
    model_path = tmp_path / "activity.tmgmmod"
    _write_dataset(dataset_path, fingerprint)
    _write_model(model_path, fingerprint)

    dataset = inspect_dataset_contract(dataset_path)
    model = inspect_model_contract(model_path)
    assert dataset.format_version == 2 and dataset.head is None
    assert model.format_version == 3 and model.head == "activity"
    assert dataset.feature_fingerprint_sha256 == fingerprint.hex()
    assert validate_dataset_model_contract(dataset, model) == (dataset, model)


def test_same_width_mismatch_and_legacy_are_fail_closed(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.tmgd"
    model_path = tmp_path / "activity.tmgmmod"
    _write_dataset(dataset_path, bytes([0x42]) * 32)
    _write_model(model_path, bytes([0x43]) * 32)
    with pytest.raises(ValueError, match="fingerprint differs"):
        validate_dataset_model_contract(dataset_path, model_path)

    legacy_dataset = tmp_path / "legacy.tmgd"
    _write_dataset(legacy_dataset, None)
    with pytest.raises(ValueError, match="allow_legacy"):
        inspect_dataset_contract(legacy_dataset)
    inspected = inspect_dataset_contract(legacy_dataset, allow_legacy=True)
    assert inspected.legacy and inspected.feature_fingerprint_sha256 is None


def test_contract_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    fingerprint = bytes([0x42]) * 32
    model_path = tmp_path / "activity.tmgmmod"
    _write_model(model_path, fingerprint)
    raw = bytearray(model_path.read_bytes())
    raw[192] ^= 1
    model_path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_model_contract(model_path)
