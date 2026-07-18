from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any

from .native_dataset import (
    FEATURE_FINGERPRINT_BYTES,
    FEATURE_FINGERPRINT_OFFSET,
    HEADER_BYTES as DATASET_HEADER_BYTES,
    LEGACY_VERSION as DATASET_LEGACY_VERSION,
    MAGIC as DATASET_MAGIC,
    VERSION as DATASET_VERSION,
    read_native_dataset_header,
)


MODEL_MAGIC = b"TMGMMOD\0"
MODEL_HEADER_BYTES = 256
MODEL_LEGACY_VERSIONS = frozenset({1, 2})
MODEL_VERSION = 3
MODEL_CHECKSUM_OFFSET = 160
MODEL_CHECKSUM_BYTES = 32
MODEL_FEATURE_FINGERPRINT_OFFSET = 192


@dataclass(frozen=True)
class FeatureContract:
    path: Path
    artifact_kind: str
    format_version: int
    head: str | None
    feature_count: int
    outputs: int
    midi_min: int
    midi_max: int
    sample_rate: int
    hop_size: int
    feature_fingerprint_sha256: str | None
    legacy: bool
    checksum_sha256: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def _require_nonlegacy(contract: FeatureContract, allow_legacy: bool) -> None:
    if contract.legacy and not allow_legacy:
        raise ValueError(
            f"{contract.artifact_kind} artifact has no authenticated "
            "feature-semantics fingerprint; pass allow_legacy=True only for "
            "explicit audit use"
        )


def _stream_dataset_payload_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(DATASET_HEADER_BYTES)
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def inspect_dataset_contract(
    path_value: str | Path, *, allow_legacy: bool = False
) -> FeatureContract:
    path = Path(path_value)
    with path.open("rb") as stream:
        raw_header = stream.read(DATASET_HEADER_BYTES)
    if len(raw_header) != DATASET_HEADER_BYTES:
        raise ValueError(f"truncated TMGD dataset header: {path}")
    if raw_header[:8] != DATASET_MAGIC:
        raise ValueError(f"invalid TMGD dataset magic: {path}")
    format_version = struct.unpack_from("<I", raw_header, 8)[0]
    if format_version not in {DATASET_LEGACY_VERSION, DATASET_VERSION}:
        raise ValueError(f"unsupported TMGD dataset version {format_version}: {path}")

    header = read_native_dataset_header(path)
    actual_payload_checksum = _stream_dataset_payload_sha256(path)
    if actual_payload_checksum != header.payload_sha256:
        raise ValueError(f"TMGD dataset payload checksum mismatch: {path}")

    fingerprint = header.feature_fingerprint_sha256
    legacy = format_version == DATASET_LEGACY_VERSION or not any(fingerprint)
    contract = FeatureContract(
        path=path,
        artifact_kind="dataset",
        format_version=format_version,
        head=None,
        feature_count=header.feature_count,
        outputs=header.note_count,
        midi_min=header.midi_min,
        midi_max=header.midi_max,
        sample_rate=header.sample_rate,
        hop_size=header.hop_size,
        feature_fingerprint_sha256=None if legacy else fingerprint.hex(),
        legacy=legacy,
        checksum_sha256=header.payload_sha256.hex(),
    )
    _require_nonlegacy(contract, allow_legacy)
    return contract


def _model_checksum(path: Path, expected_size: int) -> tuple[bytes, bytes]:
    digest = hashlib.sha256()
    stored = b""
    bytes_seen = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            mutable = bytearray(chunk)
            begin = bytes_seen
            end = begin + len(mutable)
            checksum_begin = MODEL_CHECKSUM_OFFSET
            checksum_end = checksum_begin + MODEL_CHECKSUM_BYTES
            overlap_begin = max(begin, checksum_begin)
            overlap_end = min(end, checksum_end)
            if overlap_begin < overlap_end:
                local_begin = overlap_begin - begin
                local_end = overlap_end - begin
                stored += bytes(mutable[local_begin:local_end])
                mutable[local_begin:local_end] = bytes(local_end - local_begin)
            digest.update(mutable)
            bytes_seen = end
    if bytes_seen != expected_size or len(stored) != MODEL_CHECKSUM_BYTES:
        raise ValueError(f"truncated TMGMMOD model: {path}")
    return digest.digest(), stored


def inspect_model_contract(
    path_value: str | Path, *, allow_legacy: bool = False
) -> FeatureContract:
    path = Path(path_value)
    with path.open("rb") as stream:
        header = stream.read(MODEL_HEADER_BYTES)
    if len(header) != MODEL_HEADER_BYTES:
        raise ValueError(f"truncated TMGMMOD model header: {path}")
    if header[:8] != MODEL_MAGIC:
        raise ValueError(f"invalid TMGMMOD model magic: {path}")

    format_version, header_bytes, word_bits, flags = struct.unpack_from(
        "<IIII", header, 8
    )
    if format_version not in MODEL_LEGACY_VERSIONS | {MODEL_VERSION}:
        raise ValueError(f"unsupported TMGMMOD model version {format_version}: {path}")
    if header_bytes != MODEL_HEADER_BYTES or word_bits != 32:
        raise ValueError(f"unsupported TMGMMOD header or word size: {path}")
    if flags & ~0x0F:
        raise ValueError(f"unknown TMGMMOD flags: {path}")

    head_value = struct.unpack_from("<I", header, 24)[0]
    if head_value not in {1, 2}:
        raise ValueError(f"unknown TMGMMOD target head: {path}")
    head = "activity" if head_value == 1 else "onset"
    (
        state_bits,
        feature_count,
        outputs,
        clause_count,
        literal_count,
        literal_word_count,
    ) = struct.unpack_from("<IIIIII", header, 28)
    midi_min, midi_max = struct.unpack_from("<ii", header, 88)
    sample_rate, hop_size = struct.unpack_from("<II", header, 100)
    (
        ta_offset,
        ta_bytes,
        weights_offset,
        weights_bytes,
        payload_bytes,
        file_bytes,
    ) = struct.unpack_from("<QQQQQQ", header, 112)
    if (
        not 2 <= state_bits <= 16
        or feature_count <= 0
        or outputs <= 0
        or clause_count <= 0
    ):
        raise ValueError(f"invalid TMGMMOD dimensions: {path}")
    expected_literal_count = feature_count * 2
    expected_literal_words = (expected_literal_count + 31) // 32
    expected_ta_bytes = clause_count * state_bits * expected_literal_words * 4
    expected_weights_bytes = outputs * clause_count * 4
    if (
        literal_count != expected_literal_count
        or literal_word_count != expected_literal_words
        or ta_offset != MODEL_HEADER_BYTES
        or ta_bytes != expected_ta_bytes
        or weights_offset != ta_offset + ta_bytes
        or weights_bytes != expected_weights_bytes
        or payload_bytes != ta_bytes + weights_bytes
        or file_bytes != MODEL_HEADER_BYTES + payload_bytes
    ):
        raise ValueError(f"TMGMMOD payload layout is inconsistent: {path}")
    if midi_min < 0 or midi_max > 127 or midi_max - midi_min + 1 != outputs:
        raise ValueError(f"TMGMMOD MIDI range disagrees with outputs: {path}")
    if sample_rate <= 0 or hop_size <= 0 or file_bytes != path.stat().st_size:
        raise ValueError(f"invalid TMGMMOD timebase or file size: {path}")

    fingerprint = bytes(
        header[
            MODEL_FEATURE_FINGERPRINT_OFFSET :
            MODEL_FEATURE_FINGERPRINT_OFFSET + FEATURE_FINGERPRINT_BYTES
        ]
    )
    if format_version == MODEL_VERSION:
        if not any(fingerprint):
            raise ValueError(f"TMGMMOD v3 feature fingerprint is zero: {path}")
        if any(header[224:]):
            raise ValueError(f"TMGMMOD v3 reserved header bytes are non-zero: {path}")
        legacy = False
    else:
        if any(header[MODEL_FEATURE_FINGERPRINT_OFFSET:]):
            raise ValueError(f"legacy TMGMMOD reserved header bytes are non-zero: {path}")
        fingerprint = bytes(FEATURE_FINGERPRINT_BYTES)
        legacy = True

    actual_checksum, stored_checksum = _model_checksum(path, file_bytes)
    if actual_checksum != stored_checksum:
        raise ValueError(f"TMGMMOD checksum mismatch: {path}")
    contract = FeatureContract(
        path=path,
        artifact_kind="model",
        format_version=format_version,
        head=head,
        feature_count=feature_count,
        outputs=outputs,
        midi_min=midi_min,
        midi_max=midi_max,
        sample_rate=sample_rate,
        hop_size=hop_size,
        feature_fingerprint_sha256=None if legacy else fingerprint.hex(),
        legacy=legacy,
        checksum_sha256=stored_checksum.hex(),
    )
    _require_nonlegacy(contract, allow_legacy)
    return contract


def validate_dataset_model_contract(
    dataset: str | Path | FeatureContract,
    model: str | Path | FeatureContract,
    *,
    allow_legacy: bool = False,
) -> tuple[FeatureContract, FeatureContract]:
    dataset_contract = (
        dataset
        if isinstance(dataset, FeatureContract)
        else inspect_dataset_contract(dataset, allow_legacy=allow_legacy)
    )
    model_contract = (
        model
        if isinstance(model, FeatureContract)
        else inspect_model_contract(model, allow_legacy=allow_legacy)
    )
    if dataset_contract.artifact_kind != "dataset":
        raise ValueError("first feature contract is not a dataset")
    if model_contract.artifact_kind != "model":
        raise ValueError("second feature contract is not a model")
    _require_nonlegacy(dataset_contract, allow_legacy)
    _require_nonlegacy(model_contract, allow_legacy)

    fields = (
        "feature_count",
        "outputs",
        "midi_min",
        "midi_max",
        "sample_rate",
        "hop_size",
    )
    for field in fields:
        if getattr(dataset_contract, field) != getattr(model_contract, field):
            raise ValueError(f"dataset/model {field} differs")
    if not dataset_contract.legacy and not model_contract.legacy:
        if (
            dataset_contract.feature_fingerprint_sha256
            != model_contract.feature_fingerprint_sha256
        ):
            raise ValueError("dataset/model feature-semantics fingerprint differs")
    return dataset_contract, model_contract
