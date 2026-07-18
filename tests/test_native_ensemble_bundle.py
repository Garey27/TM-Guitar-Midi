from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tmgm_rt.native_ensemble_bundle import (
    BUNDLE_CHECKSUM_BYTES,
    BUNDLE_CHECKSUM_OFFSET,
    BUNDLE_HEADER_BYTES,
    BUNDLE_MAGIC,
    MEMBER_DESCRIPTOR_BYTES,
    ModelSpec,
    export_ensemble_bundle,
    parse_model_spec,
)


_TEST_FEATURE_BASIS = {
    "schema": "tmgm-binary-feature-semantics-test-v1",
    "ordered_feature_names_sha256": "11" * 32,
}
_TEST_FEATURE_FINGERPRINT = hashlib.sha256(
    json.dumps(
        _TEST_FEATURE_BASIS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).digest()


def _write_model(
    path: Path,
    *,
    head: int,
    score_threshold: int,
    weights: np.ndarray | None = None,
    feature_fingerprint: bytes = _TEST_FEATURE_FINGERPRINT,
) -> None:
    feature_count = 3
    output_count = 2
    clause_count = 3
    state_bits = 8
    literal_count = feature_count * 2
    literal_words = 1
    ta = np.zeros((clause_count, state_bits, literal_words), dtype="<u4")
    ta[0, -1, 0] = (1 << 0) | (1 << 4)
    # TMGMMOD/native CUDA masks action padding; exporter must not turn this
    # deliberately dirty padding bit into an out-of-range sparse literal.
    ta[1, -1, 0] = 1 << 31
    ta[2, -1, 0] = (1 << 2) | (1 << 3)
    if weights is None:
        weights = np.asarray([[5, 100, -7], [-2, 50, 9]], dtype="<i4")
    else:
        weights = np.asarray(weights, dtype="<i4")
    ta_bytes = ta.nbytes
    weight_bytes = weights.nbytes
    file_bytes = 256 + ta_bytes + weight_bytes

    header = bytearray(256)
    header[:8] = b"TMGMMOD\0"
    version = 3 if any(feature_fingerprint) else 2
    struct.pack_into("<IIII", header, 8, version, 256, 32, 0x03)
    struct.pack_into(
        "<IIIIIII",
        header,
        24,
        head,
        state_bits,
        feature_count,
        output_count,
        clause_count,
        literal_count,
        literal_words,
    )
    struct.pack_into(
        "<iiIfffIQ",
        header,
        52,
        32,
        score_threshold,
        literal_count,
        4.0,
        1.0,
        1.0,
        1,
        17,
    )
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
    if version == 3:
        header[192:224] = feature_fingerprint
    raw = header + ta.tobytes(order="C") + weights.tobytes(order="C")
    raw[160:192] = hashlib.sha256(raw).digest()
    path.write_bytes(raw)


def _artifact(
    path: Path,
    *,
    head: str,
    identifiers: tuple[str, ...],
    thresholds: tuple[int, ...],
    scales: tuple[float, ...],
    ensemble_threshold: int,
    feature_count: int = 3,
) -> Path:
    value = {
        "format": "TMGM_NATIVE_SCORE_ENSEMBLE_V1",
        "head": head,
        "fusion": "mean",
        "quantization": 1024,
        "ensemble_threshold": ensemble_threshold,
        "member_order_sha256": hashlib.sha256(
            "\0".join(identifiers).encode("utf-8")
        ).hexdigest(),
        "members": [
            {"id": identifier, "threshold": threshold, "robust_scale": scale}
            for identifier, threshold, scale in zip(
                identifiers, thresholds, scales, strict=True
            )
        ],
        "fit_dataset": {
            "feature_count": feature_count,
            "outputs": 2,
            "midi_min": 40,
            "midi_max": 41,
            "sample_rate": 22_050,
            "hop_size": 256,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(
    tmp_path: Path,
) -> tuple[Path, list[ModelSpec], Path, list[ModelSpec], Path]:
    a1 = tmp_path / "a1.tmgmmod"
    a2 = tmp_path / "a2.tmgmmod"
    q1 = tmp_path / "q1.tmgmmod"
    q2 = tmp_path / "q2.tmgmmod"
    _write_model(a1, head=1, score_threshold=9)
    _write_model(a2, head=1, score_threshold=10)
    _write_model(q1, head=2, score_threshold=11)
    _write_model(q2, head=2, score_threshold=13)
    feature_source = tmp_path / "train.tmgd.json"
    feature_source.write_text(
        json.dumps(
            {
                "feature_semantics": {
                    **_TEST_FEATURE_BASIS,
                    "fingerprint_sha256": _TEST_FEATURE_FINGERPRINT.hex(),
                },
                "header": {
                    "feature_fingerprint_sha256": (
                        _TEST_FEATURE_FINGERPRINT.hex()
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        _artifact(
            tmp_path / "activity.json",
            head="activity",
            identifiers=("a1", "a2"),
            thresholds=(8, 12),
            scales=(2.0, 4.0),
            ensemble_threshold=-100,
        ),
        [ModelSpec("a1", a1), ModelSpec("a2", a2)],
        _artifact(
            tmp_path / "onset.json",
            head="onset",
            identifiers=("q1", "q2"),
            thresholds=(11, 13),
            scales=(7.25, 3.5),
            ensemble_threshold=-386,
        ),
        [ModelSpec("q1", q1), ModelSpec("q2", q2)],
        feature_source,
    )


def _descriptor(raw: bytes, index: int) -> bytes:
    offset = BUNDLE_HEADER_BYTES + index * MEMBER_DESCRIPTOR_BYTES
    return raw[offset : offset + MEMBER_DESCRIPTOR_BYTES]


def test_parse_model_spec_requires_stable_explicit_id():
    assert parse_model_spec("q05=model.tmgmmod") == ModelSpec(
        "q05", Path("model.tmgmmod")
    )
    with pytest.raises(ValueError, match="ID=path"):
        parse_model_spec("model.tmgmmod")
    with pytest.raises(ValueError, match="model ID"):
        parse_model_spec("bad id=model.tmgmmod")


def test_bundle_export_is_deterministic_sparse_and_checksummed(tmp_path: Path):
    activity_artifact, activity, onset_artifact, onset, feature_source = _fixture(
        tmp_path
    )
    first = tmp_path / "first.tmgmbundle"
    second = tmp_path / "second.tmgmbundle"
    result = export_ensemble_bundle(
        activity_artifact,
        activity,
        onset_artifact,
        onset,
        feature_source,
        first,
    )
    export_ensemble_bundle(
        activity_artifact,
        activity,
        onset_artifact,
        onset,
        feature_source,
        second,
    )
    raw = first.read_bytes()

    assert raw == second.read_bytes()
    assert raw[:8] == BUNDLE_MAGIC
    assert result["bytes"] == len(raw)
    assert result["activity_members"] == ["a1", "a2"]
    assert result["onset_members"] == ["q1", "q2"]
    stored = raw[
        BUNDLE_CHECKSUM_OFFSET : BUNDLE_CHECKSUM_OFFSET + BUNDLE_CHECKSUM_BYTES
    ]
    canonical = bytearray(raw)
    canonical[
        BUNDLE_CHECKSUM_OFFSET : BUNDLE_CHECKSUM_OFFSET + BUNDLE_CHECKSUM_BYTES
    ] = bytes(BUNDLE_CHECKSUM_BYTES)
    assert stored == hashlib.sha256(canonical).digest()
    assert stored.hex() == result["checksum_sha256"]

    version, header_bytes, descriptor_bytes = struct.unpack_from("<III", raw, 8)
    assert (version, header_bytes, descriptor_bytes) == (2, 256, 192)
    assert struct.unpack_from("<III", raw, 24) == (4, 2, 2)
    assert struct.unpack_from("<Ii", raw, 64) == (1024, -100)
    assert struct.unpack_from("<Ii", raw, 76) == (1024, -386)
    assert raw[192:224] == _TEST_FEATURE_FINGERPRINT

    activity_descriptor = _descriptor(raw, 0)
    assert activity_descriptor[:64].rstrip(b"\0") == b"a1"
    assert struct.unpack_from("<I", activity_descriptor, 64)[0] == 1
    # Ensemble calibration is authoritative and may differ from model metadata.
    assert struct.unpack_from("<i", activity_descriptor, 72)[0] == 8
    assert struct.unpack_from("<f", activity_descriptor, 76)[0] == 2.0

    onset_descriptor = _descriptor(raw, 2)
    assert onset_descriptor[:64].rstrip(b"\0") == b"q1"
    assert struct.unpack_from("<I", onset_descriptor, 64)[0] == 2
    assert struct.unpack_from("<i", onset_descriptor, 72)[0] == 11
    assert struct.unpack_from("<f", onset_descriptor, 76)[0] == np.float32(7.25)
    included = struct.unpack_from("<I", onset_descriptor, 96)[0]
    assert included == 4
    offsets_offset, offsets_bytes, literals_offset, literals_bytes = (
        struct.unpack_from("<QQQQ", onset_descriptor, 136)
    )
    assert offsets_bytes == 16
    assert literals_bytes == 8
    assert np.frombuffer(raw, dtype="<u4", count=4, offset=offsets_offset).tolist() == [
        0,
        2,
        2,
        4,
    ]
    assert np.frombuffer(raw, dtype="<u2", count=4, offset=literals_offset).tolist() == [
        0,
        4,
        2,
        3,
    ]


def test_bundle_rejects_member_order_geometry_and_damaged_source(tmp_path: Path):
    activity_artifact, activity, onset_artifact, onset, feature_source = _fixture(
        tmp_path
    )
    output = tmp_path / "bundle.tmgmbundle"
    with pytest.raises(ValueError, match="order/identity"):
        export_ensemble_bundle(
            activity_artifact,
            activity,
            onset_artifact,
            list(reversed(onset)),
            feature_source,
            output,
        )

    wrong_geometry = _artifact(
        tmp_path / "wrong-geometry.json",
        head="activity",
        identifiers=("a1", "a2"),
        thresholds=(8, 12),
        scales=(2.0, 4.0),
        ensemble_threshold=-100,
        feature_count=4,
    )
    with pytest.raises(ValueError, match="geometries differ"):
        export_ensemble_bundle(
            wrong_geometry,
            activity,
            onset_artifact,
            onset,
            feature_source,
            output,
        )

    damaged = bytearray(onset[0].path.read_bytes())
    damaged[-1] ^= 1
    onset[0].path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_ensemble_bundle(
            activity_artifact,
            activity,
            onset_artifact,
            onset,
            feature_source,
            output,
        )


def test_bundle_rejects_weight_that_cannot_be_stored_losslessly(tmp_path: Path):
    activity_artifact, activity, onset_artifact, onset, feature_source = _fixture(
        tmp_path
    )
    oversized = np.asarray([[5, 100, -7], [-2, 50, 40_000]], dtype="<i4")
    _write_model(onset[1].path, head=2, score_threshold=13, weights=oversized)
    with pytest.raises(ValueError, match="int16"):
        export_ensemble_bundle(
            activity_artifact,
            activity,
            onset_artifact,
            onset,
            feature_source,
            tmp_path / "bundle.tmgmbundle",
        )


def test_bundle_rejects_same_width_feature_semantics_mismatch(tmp_path: Path):
    activity_artifact, activity, onset_artifact, onset, feature_source = _fixture(
        tmp_path
    )
    _write_model(
        onset[1].path,
        head=2,
        score_threshold=13,
        feature_fingerprint=bytes([0x77]) * 32,
    )
    with pytest.raises(ValueError, match="fingerprints differ"):
        export_ensemble_bundle(
            activity_artifact,
            activity,
            onset_artifact,
            onset,
            feature_source,
            tmp_path / "mismatch.tmgmbundle",
        )


def test_legacy_bundle_export_requires_explicit_opt_in(tmp_path: Path):
    activity_artifact, activity, onset_artifact, onset, feature_source = _fixture(
        tmp_path
    )
    for spec in [*activity, *onset]:
        _write_model(
            spec.path,
            head=1 if spec in activity else 2,
            score_threshold=9,
            feature_fingerprint=bytes(32),
        )
    with pytest.raises(ValueError, match="explicit legacy opt-in"):
        export_ensemble_bundle(
            activity_artifact,
            activity,
            onset_artifact,
            onset,
            feature_source,
            tmp_path / "legacy-rejected.tmgmbundle",
        )
    result = export_ensemble_bundle(
        activity_artifact,
        activity,
        onset_artifact,
        onset,
        feature_source,
        tmp_path / "legacy-audit.tmgmbundle",
        allow_legacy_feature_contract=True,
    )
    assert result["format_version"] == 1
    assert result["legacy_feature_contract"] is True
