from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Sequence

import numpy as np


BUNDLE_MAGIC = b"TMGMBND\0"
BUNDLE_LEGACY_FORMAT_VERSION = 1
BUNDLE_FORMAT_VERSION = 2
BUNDLE_HEADER_BYTES = 256
MEMBER_DESCRIPTOR_BYTES = 192
BUNDLE_CHECKSUM_OFFSET = 224
BUNDLE_CHECKSUM_BYTES = 32
MODEL_MAGIC = b"TMGMMOD\0"
MODEL_HEADER_BYTES = 256
MODEL_CHECKSUM_OFFSET = 160
MODEL_CHECKSUM_BYTES = 32
MODEL_FEATURE_FINGERPRINT_OFFSET = 192
MODEL_FEATURE_FINGERPRINT_BYTES = 32
FUSION_MEAN = 1
HEAD_ACTIVITY = 1
HEAD_ONSET = 2
WEIGHT_BITS = 16

_MEMBER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_I16 = np.iinfo(np.int16)
_I32 = np.iinfo(np.int32)


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    path: Path


@dataclass(frozen=True)
class ParsedModel:
    path: Path
    format_version: int
    head: int
    state_bits: int
    feature_count: int
    output_count: int
    clause_count: int
    literal_count: int
    literal_word_count: int
    score_threshold: int
    midi_min: int
    midi_max: int
    sample_rate: int
    hop_size: int
    source_sha256: bytes
    feature_fingerprint_sha256: bytes
    clause_offsets: np.ndarray
    literal_ids: np.ndarray
    weights: np.ndarray


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        raise ValueError(f"model must use ID=path syntax: {value!r}")
    identifier, path_text = value.split("=", 1)
    if not _MEMBER_ID.fullmatch(identifier):
        raise ValueError(
            "model ID must be 1-64 ASCII letters, digits, '.', '_' or '-'"
        )
    if not path_text:
        raise ValueError(f"model {identifier!r} has an empty path")
    return ModelSpec(identifier=identifier, path=Path(path_text))


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _model_error(path: Path, message: str) -> ValueError:
    return ValueError(f"invalid TMGMMOD model ({path}): {message}")


def _validate_model_checksum(path: Path, raw: bytes) -> None:
    stored = raw[
        MODEL_CHECKSUM_OFFSET : MODEL_CHECKSUM_OFFSET + MODEL_CHECKSUM_BYTES
    ]
    canonical = bytearray(raw)
    canonical[
        MODEL_CHECKSUM_OFFSET : MODEL_CHECKSUM_OFFSET + MODEL_CHECKSUM_BYTES
    ] = bytes(MODEL_CHECKSUM_BYTES)
    if hashlib.sha256(canonical).digest() != stored:
        raise _model_error(path, "checksum mismatch")


def _parse_model(path_value: str | Path) -> ParsedModel:
    path = Path(path_value)
    raw = path.read_bytes()
    if len(raw) < MODEL_HEADER_BYTES:
        raise _model_error(path, "file is truncated")
    if raw[:8] != MODEL_MAGIC:
        raise _model_error(path, "wrong magic")

    version, header_bytes, word_bits, flags = struct.unpack_from("<IIII", raw, 8)
    if version not in {1, 2, 3}:
        raise _model_error(path, f"unsupported version {version}")
    if header_bytes != MODEL_HEADER_BYTES:
        raise _model_error(path, "wrong header size")
    if word_bits != 32:
        raise _model_error(path, "unsupported packed word size")
    if flags & ~0x0F:
        raise _model_error(path, "unknown flags")

    (
        head,
        state_bits,
        feature_count,
        output_count,
        clause_count,
        literal_count,
        literal_word_count,
    ) = struct.unpack_from("<IIIIIII", raw, 24)
    if head not in {HEAD_ACTIVITY, HEAD_ONSET}:
        raise _model_error(path, "unknown head")
    if not 2 <= state_bits <= 16:
        raise _model_error(path, "state_bits must be in [2, 16]")
    if feature_count <= 0 or output_count <= 0 or clause_count <= 0:
        raise _model_error(path, "model dimensions must be positive")
    if feature_count > 32_768:
        raise _model_error(
            path, "feature count exceeds inference bundle uint16 literal capacity"
        )
    if literal_count != feature_count * 2:
        raise _model_error(path, "literal count disagrees with feature count")
    if literal_word_count != (literal_count + 31) // 32:
        raise _model_error(path, "literal word count is inconsistent")

    score_threshold = struct.unpack_from("<i", raw, 56)[0]
    midi_min, midi_max = struct.unpack_from("<ii", raw, 88)
    sample_rate, hop_size = struct.unpack_from("<II", raw, 100)
    ta_offset, ta_bytes, weights_offset, weights_bytes, payload_bytes, file_bytes = (
        struct.unpack_from("<QQQQQQ", raw, 112)
    )
    expected_ta_bytes = clause_count * state_bits * literal_word_count * 4
    expected_weights_bytes = output_count * clause_count * 4
    if midi_min < 0 or midi_max > 127 or midi_max - midi_min + 1 != output_count:
        raise _model_error(path, "MIDI range disagrees with output count")
    if sample_rate <= 0 or hop_size <= 0:
        raise _model_error(path, "invalid audio timebase")
    if (
        ta_offset != MODEL_HEADER_BYTES
        or ta_bytes != expected_ta_bytes
        or weights_offset != ta_offset + ta_bytes
        or weights_bytes != expected_weights_bytes
        or payload_bytes != ta_bytes + weights_bytes
        or file_bytes != MODEL_HEADER_BYTES + payload_bytes
        or file_bytes != len(raw)
    ):
        raise _model_error(path, "payload layout disagrees with dimensions")
    feature_fingerprint = bytes(
        raw[
            MODEL_FEATURE_FINGERPRINT_OFFSET :
            MODEL_FEATURE_FINGERPRINT_OFFSET + MODEL_FEATURE_FINGERPRINT_BYTES
        ]
    )
    if version == 3:
        if not any(feature_fingerprint):
            raise _model_error(path, "v3 feature-semantics fingerprint is zero")
        if any(raw[224:MODEL_HEADER_BYTES]):
            raise _model_error(path, "reserved header bytes are non-zero")
    else:
        if any(raw[MODEL_FEATURE_FINGERPRINT_OFFSET:MODEL_HEADER_BYTES]):
            raise _model_error(path, "legacy reserved header bytes are non-zero")
        feature_fingerprint = bytes(MODEL_FEATURE_FINGERPRINT_BYTES)
    _validate_model_checksum(path, raw)

    ta_words = np.frombuffer(
        raw,
        dtype="<u4",
        count=expected_ta_bytes // 4,
        offset=ta_offset,
    )
    source_weights = np.frombuffer(
        raw,
        dtype="<i4",
        count=expected_weights_bytes // 4,
        offset=weights_offset,
    ).reshape(output_count, clause_count)
    if np.any(source_weights < _I16.min) or np.any(source_weights > _I16.max):
        raise _model_error(path, "weight does not fit inference bundle int16")

    offsets = np.empty(clause_count + 1, dtype="<u4")
    literals: list[int] = []
    for clause in range(clause_count):
        offsets[clause] = len(literals)
        action_base = (
            clause * state_bits + (state_bits - 1)
        ) * literal_word_count
        for word_index in range(literal_word_count):
            word = int(ta_words[action_base + word_index])
            if word_index + 1 == literal_word_count and literal_count % 32:
                word &= (1 << (literal_count % 32)) - 1
            while word:
                bit = (word & -word).bit_length() - 1
                literals.append(word_index * 32 + bit)
                word &= word - 1
    offsets[clause_count] = len(literals)
    literal_ids = np.asarray(literals, dtype="<u2")
    weights = np.asarray(source_weights.T, dtype="<i2", order="C").reshape(-1)
    return ParsedModel(
        path=path,
        format_version=version,
        head=head,
        state_bits=state_bits,
        feature_count=feature_count,
        output_count=output_count,
        clause_count=clause_count,
        literal_count=literal_count,
        literal_word_count=literal_word_count,
        score_threshold=score_threshold,
        midi_min=midi_min,
        midi_max=midi_max,
        sample_rate=sample_rate,
        hop_size=hop_size,
        source_sha256=hashlib.sha256(raw).digest(),
        feature_fingerprint_sha256=feature_fingerprint,
        clause_offsets=offsets,
        literal_ids=literal_ids,
        weights=weights,
    )


def _load_ensemble_artifact(path: Path, expected_head: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != (
        "TMGM_NATIVE_SCORE_ENSEMBLE_V1"
    ):
        raise ValueError(f"unsupported score ensemble artifact: {path}")
    if value.get("head") != expected_head:
        raise ValueError(
            f"bundle {expected_head} artifact has head={value.get('head')!r}"
        )
    if value.get("fusion") != "mean":
        raise ValueError("bundle v1 supports exact mean fusion only")
    quantization = value.get("quantization")
    threshold = value.get("ensemble_threshold")
    if (
        not isinstance(quantization, int)
        or isinstance(quantization, bool)
        or not 1 <= quantization <= 1_000_000
    ):
        raise ValueError("ensemble quantization must be in [1, 1000000]")
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or not _I32.min <= threshold <= _I32.max
    ):
        raise ValueError("ensemble threshold must fit int32")
    members = value.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"{expected_head} ensemble needs at least one member")
    identifiers: list[str] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError(f"ensemble member {index} is not an object")
        identifier = member.get("id")
        member_threshold = member.get("threshold")
        scale = member.get("robust_scale")
        if not isinstance(identifier, str) or not _MEMBER_ID.fullmatch(identifier):
            raise ValueError(f"ensemble member {index} has an invalid ID")
        if (
            not isinstance(member_threshold, int)
            or isinstance(member_threshold, bool)
            or not _I32.min <= member_threshold <= _I32.max
        ):
            raise ValueError(f"ensemble member {identifier!r} has invalid threshold")
        if (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise ValueError(f"ensemble member {identifier!r} has invalid scale")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ensemble member IDs must be unique")
    order_digest = hashlib.sha256("\0".join(identifiers).encode("utf-8")).hexdigest()
    if value.get("member_order_sha256") != order_digest:
        raise ValueError("ensemble member order checksum mismatch")
    fit = value.get("fit_dataset")
    geometry_fields = (
        "feature_count",
        "outputs",
        "midi_min",
        "midi_max",
        "sample_rate",
        "hop_size",
    )
    if not isinstance(fit, dict) or any(
        not isinstance(fit.get(field), int) or isinstance(fit.get(field), bool)
        for field in geometry_fields
    ):
        raise ValueError("ensemble fit dataset geometry is invalid")
    return value


def _same_geometry(model: ParsedModel) -> tuple[int, ...]:
    return (
        model.feature_count,
        model.output_count,
        model.midi_min,
        model.midi_max,
        model.sample_rate,
        model.hop_size,
    )


def _artifact_geometry(artifact: dict[str, Any]) -> tuple[int, ...]:
    fit = artifact["fit_dataset"]
    return tuple(
        int(fit[field])
        for field in (
            "feature_count",
            "outputs",
            "midi_min",
            "midi_max",
            "sample_rate",
            "hop_size",
        )
    )


def _append_aligned(payload: bytearray, absolute_base: int, value: bytes) -> tuple[int, int]:
    aligned = _align8(absolute_base + len(payload))
    payload.extend(bytes(aligned - (absolute_base + len(payload))))
    offset = absolute_base + len(payload)
    payload.extend(value)
    return offset, len(value)


def _validate_head_models(
    artifact: dict[str, Any],
    specs: Sequence[ModelSpec],
    expected_head: int,
) -> list[ParsedModel]:
    name = "activity" if expected_head == HEAD_ACTIVITY else "onset"
    expected_ids = [str(member["id"]) for member in artifact["members"]]
    actual_ids = [spec.identifier for spec in specs]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{name} model order/identity {actual_ids} does not match {expected_ids}"
        )
    parsed = [_parse_model(spec.path) for spec in specs]
    for spec, model, member in zip(specs, parsed, artifact["members"], strict=True):
        if model.head != expected_head:
            raise ValueError(f"{name} model {spec.identifier!r} has the wrong head")
        scale = np.float32(member["robust_scale"])
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(
                f"{name} model {spec.identifier!r} scale is invalid after float32 rounding"
            )
    return parsed


def _metadata_feature_fingerprint(path: Path) -> bytes:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "feature source must be a native dataset metadata JSON sidecar"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError("feature metadata must contain a JSON object")
    semantics = metadata.get("feature_semantics")
    if not isinstance(semantics, dict):
        raise ValueError("feature metadata has no feature_semantics descriptor")
    declared = semantics.get("fingerprint_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("feature semantics fingerprint is missing or malformed")
    try:
        fingerprint = bytes.fromhex(declared)
    except ValueError as error:
        raise ValueError("feature semantics fingerprint is not hexadecimal") from error
    if not any(fingerprint):
        raise ValueError("feature semantics fingerprint is zero")
    basis = dict(semantics)
    basis.pop("fingerprint_sha256", None)
    canonical = json.dumps(
        basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if hashlib.sha256(canonical).digest() != fingerprint:
        raise ValueError("feature semantics descriptor fingerprint is inconsistent")
    header = metadata.get("header")
    if isinstance(header, dict):
        header_fingerprint = header.get("feature_fingerprint_sha256")
        if header_fingerprint is not None and header_fingerprint != declared:
            raise ValueError(
                "feature metadata header and semantics fingerprints disagree"
            )
    return fingerprint


def export_ensemble_bundle(
    activity_ensemble_path_value: str | Path,
    activity_models: Sequence[ModelSpec],
    onset_ensemble_path_value: str | Path,
    onset_models: Sequence[ModelSpec],
    feature_source_value: str | Path,
    output_path_value: str | Path,
    *,
    allow_legacy_feature_contract: bool = False,
) -> dict[str, Any]:
    activity_ensemble_path = Path(activity_ensemble_path_value)
    onset_ensemble_path = Path(onset_ensemble_path_value)
    feature_source = Path(feature_source_value)
    output_path = Path(output_path_value)
    activity_artifact = _load_ensemble_artifact(
        activity_ensemble_path, "activity"
    )
    onset_artifact = _load_ensemble_artifact(onset_ensemble_path, "onset")
    parsed_activity = _validate_head_models(
        activity_artifact, activity_models, HEAD_ACTIVITY
    )
    parsed_onset = _validate_head_models(
        onset_artifact, onset_models, HEAD_ONSET
    )
    activity_ids = [spec.identifier for spec in activity_models]
    onset_ids = [spec.identifier for spec in onset_models]
    if len(set([*activity_ids, *onset_ids])) != len(activity_ids) + len(onset_ids):
        raise ValueError("activity and onset model IDs must be globally unique")

    expected_geometry = _artifact_geometry(onset_artifact)
    if _artifact_geometry(activity_artifact) != expected_geometry:
        raise ValueError("activity and onset ensemble geometries differ")
    all_models = [*parsed_activity, *parsed_onset]
    for model in all_models:
        if _same_geometry(model) != expected_geometry:
            raise ValueError(
                f"model geometry {_same_geometry(model)} does not match "
                f"ensemble {expected_geometry}: {model.path}"
            )

    semantic_fingerprints = {
        model.feature_fingerprint_sha256 for model in all_models
    }
    legacy_models = [
        model for model in all_models if not any(model.feature_fingerprint_sha256)
    ]
    if legacy_models:
        if len(legacy_models) != len(all_models):
            raise ValueError(
                "legacy and semantic TMGMMOD models cannot be mixed in one bundle"
            )
        if not allow_legacy_feature_contract:
            raise ValueError(
                "legacy TMGMMOD models have no feature-semantics fingerprint; "
                "pass an explicit legacy opt-in only for audit export"
            )
        bundle_format_version = BUNDLE_LEGACY_FORMAT_VERSION
        feature_fingerprint = hashlib.sha256(feature_source.read_bytes()).digest()
    else:
        if len(semantic_fingerprints) != 1:
            raise ValueError(
                "ensemble TMGMMOD feature-semantics fingerprints differ"
            )
        bundle_format_version = BUNDLE_FORMAT_VERSION
        feature_fingerprint = _metadata_feature_fingerprint(feature_source)
        model_fingerprint = next(iter(semantic_fingerprints))
        if feature_fingerprint != model_fingerprint:
            raise ValueError(
                "feature metadata fingerprint differs from ensemble models"
            )

    descriptors_offset = BUNDLE_HEADER_BYTES
    descriptor_bytes = len(all_models) * MEMBER_DESCRIPTOR_BYTES
    payload_offset = _align8(descriptors_offset + descriptor_bytes)
    payload = bytearray()
    descriptors: list[bytearray] = []
    specs = [*activity_models, *onset_models]
    artifact_members = [
        *activity_artifact["members"],
        *onset_artifact["members"],
    ]
    for spec, model, member_artifact in zip(
        specs, all_models, artifact_members, strict=True
    ):
        scale = np.float32(member_artifact["robust_scale"])
        offsets_offset, offsets_bytes = _append_aligned(
            payload, payload_offset, model.clause_offsets.tobytes(order="C")
        )
        literals_offset, literals_bytes = _append_aligned(
            payload, payload_offset, model.literal_ids.tobytes(order="C")
        )
        weights_offset, weights_bytes = _append_aligned(
            payload, payload_offset, model.weights.tobytes(order="C")
        )
        descriptor = bytearray(MEMBER_DESCRIPTOR_BYTES)
        identifier_bytes = spec.identifier.encode("ascii")
        descriptor[: len(identifier_bytes)] = identifier_bytes
        struct.pack_into(
            "<IIifIIIIII",
            descriptor,
            64,
            model.head,
            0,
            int(member_artifact["threshold"]),
            float(scale),
            model.feature_count,
            model.output_count,
            model.clause_count,
            model.literal_count,
            int(model.literal_ids.size),
            WEIGHT_BITS,
        )
        descriptor[104:136] = model.source_sha256
        struct.pack_into(
            "<QQQQQQQ",
            descriptor,
            136,
            offsets_offset,
            offsets_bytes,
            literals_offset,
            literals_bytes,
            weights_offset,
            weights_bytes,
            0,
        )
        descriptors.append(descriptor)

    file_bytes = payload_offset + len(payload)
    header = bytearray(BUNDLE_HEADER_BYTES)
    header[:8] = BUNDLE_MAGIC
    activity_order_fingerprint = hashlib.sha256(
        "\0".join(activity_ids).encode("utf-8")
    ).digest()
    onset_order_fingerprint = hashlib.sha256(
        "\0".join(onset_ids).encode("utf-8")
    ).digest()
    struct.pack_into(
        "<IIIIIIIIIiiIII",
        header,
        8,
        bundle_format_version,
        BUNDLE_HEADER_BYTES,
        MEMBER_DESCRIPTOR_BYTES,
        0,
        len(all_models),
        len(parsed_activity),
        len(parsed_onset),
        expected_geometry[0],
        expected_geometry[1],
        expected_geometry[2],
        expected_geometry[3],
        expected_geometry[4],
        expected_geometry[5],
        FUSION_MEAN,
    )
    struct.pack_into(
        "<IiIIiI",
        header,
        64,
        int(activity_artifact["quantization"]),
        int(activity_artifact["ensemble_threshold"]),
        FUSION_MEAN,
        int(onset_artifact["quantization"]),
        int(onset_artifact["ensemble_threshold"]),
        0,
    )
    struct.pack_into(
        "<QQQQQ",
        header,
        88,
        descriptors_offset,
        descriptor_bytes,
        payload_offset,
        len(payload),
        file_bytes,
    )
    header[128:160] = activity_order_fingerprint
    header[160:192] = onset_order_fingerprint
    header[192:224] = feature_fingerprint

    raw = bytearray(header)
    for descriptor in descriptors:
        raw.extend(descriptor)
    raw.extend(bytes(payload_offset - len(raw)))
    raw.extend(payload)
    if len(raw) != file_bytes:
        raise AssertionError("internal bundle layout error")
    digest = hashlib.sha256(raw).digest()
    raw[
        BUNDLE_CHECKSUM_OFFSET : BUNDLE_CHECKSUM_OFFSET + BUNDLE_CHECKSUM_BYTES
    ] = digest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(output_path)
    return {
        "format": (
            "TMGM_NATIVE_ENSEMBLE_BUNDLE_V2"
            if bundle_format_version == BUNDLE_FORMAT_VERSION
            else "TMGM_NATIVE_ENSEMBLE_BUNDLE_V1_LEGACY"
        ),
        "format_version": bundle_format_version,
        "legacy_feature_contract": (
            bundle_format_version == BUNDLE_LEGACY_FORMAT_VERSION
        ),
        "path": str(output_path.resolve()),
        "bytes": len(raw),
        "checksum_sha256": digest.hex(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "feature_fingerprint_sha256": feature_fingerprint.hex(),
        "activity_member_order_sha256": activity_order_fingerprint.hex(),
        "onset_member_order_sha256": onset_order_fingerprint.hex(),
        "activity_members": activity_ids,
        "onset_members": onset_ids,
        "feature_count": expected_geometry[0],
        "output_count": expected_geometry[1],
        "midi_min": expected_geometry[2],
        "midi_max": expected_geometry[3],
        "sample_rate": expected_geometry[4],
        "hop_size": expected_geometry[5],
        "activity": {
            "fusion": "mean",
            "quantization": int(activity_artifact["quantization"]),
            "ensemble_threshold": int(activity_artifact["ensemble_threshold"]),
        },
        "onset": {
            "fusion": "mean",
            "quantization": int(onset_artifact["quantization"]),
            "ensemble_threshold": int(onset_artifact["ensemble_threshold"]),
        },
    }


def _model_specs(values: Iterable[str]) -> list[ModelSpec]:
    return [parse_model_spec(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export validated TMGMMOD activity/onset models as one deterministic "
            "sparse CPU inference bundle."
        )
    )
    parser.add_argument("--activity-ensemble", type=Path, required=True)
    parser.add_argument(
        "--activity-model",
        action="append",
        required=True,
        metavar="ID=TMGMMOD",
        help="repeat in the exact order stored by the activity ensemble",
    )
    parser.add_argument("--onset-ensemble", type=Path, required=True)
    parser.add_argument(
        "--onset-model",
        action="append",
        required=True,
        metavar="ID=TMGMMOD",
        help="repeat in the exact order stored by the onset ensemble",
    )
    parser.add_argument(
        "--feature-source",
        type=Path,
        required=True,
        help=(
            "native dataset metadata JSON containing the canonical "
            "feature_semantics descriptor"
        ),
    )
    parser.add_argument(
        "--allow-legacy-feature-contract",
        action="store_true",
        help="audit-only export for legacy models without semantic fingerprints",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    activity = _model_specs(arguments.activity_model)
    onset = _model_specs(arguments.onset_model)
    result = export_ensemble_bundle(
        arguments.activity_ensemble,
        activity,
        arguments.onset_ensemble,
        onset,
        arguments.feature_source,
        arguments.output,
        allow_legacy_feature_contract=arguments.allow_legacy_feature_contract,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
