from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import ContextConfig, FrontendConfig
from .feature_semantics import (
    binary_feature_semantics,
    canonical_frontend_mapping,
    validate_feature_semantics,
)
from .midi import (
    NoteStateConfig,
    stabilize_frame_predictions,
    write_frame_predictions,
    write_teacher_events,
)
from .native_dataset import NativeDatasetHeader, read_native_dataset_header
from .native_score_ensemble import MemberSpec, apply_score_ensemble, load_score_file


MANIFEST_SCHEMA = "tmgm-offline-listening-manifest-v1"
RUN_SCHEMA = "tmgm-offline-listening-run-v1"
CACHE_SCHEMA = "tmgm-offline-listening-score-cache-v1"
ENSEMBLE_SIDECAR_SCHEMA = "tmgm-offline-listening-ensemble-output-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FileRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class PredictorSpec:
    executable: FileRef
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class BankProvenance:
    reference_metadata: FileRef
    binarizer_sha256: str


@dataclass(frozen=True)
class FeatureBankSpec:
    identifier: str
    fingerprint_sha256: str
    allow_legacy_zero_fingerprint: bool
    provenance: BankProvenance


@dataclass(frozen=True)
class TrackFeatureSpec:
    dataset: FileRef
    metadata: FileRef


@dataclass(frozen=True)
class TrackSpec:
    identifier: str
    source_wav: FileRef
    feature_banks: Mapping[str, TrackFeatureSpec]
    neuralnote_midi: FileRef | None = None
    teacher_events: FileRef | None = None


@dataclass(frozen=True)
class ModelProvenance:
    training_metadata: FileRef
    experiment_config: FileRef | None = None


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    head: str
    feature_bank: str
    feature_fingerprint_sha256: str
    model: FileRef
    allow_legacy_zero_fingerprint: bool
    provenance: ModelProvenance


@dataclass(frozen=True)
class HeadSpec:
    member: str | None
    artifact: FileRef | None
    members: tuple[str, ...]
    anchor_feature_bank: str | None


@dataclass(frozen=True)
class RenderSpec:
    identifier: str
    tracks: tuple[str, ...]
    activity: HeadSpec
    onset: HeadSpec
    stabilization: NoteStateConfig


@dataclass(frozen=True)
class ListeningManifest:
    path: Path
    predictor: PredictorSpec
    cache_root: Path
    output_root: Path
    feature_banks: Mapping[str, FeatureBankSpec]
    tracks: Mapping[str, TrackSpec]
    models: Mapping[str, ModelSpec]
    renders: tuple[RenderSpec, ...]


@dataclass(frozen=True)
class DatasetContract:
    format_version: int
    feature_count: int
    outputs: int
    midi_min: int
    midi_max: int
    sample_rate: int
    hop_size: int
    fingerprint_sha256: str
    legacy: bool
    checksum_sha256: str


@dataclass(frozen=True)
class ModelContract:
    format_version: int
    head: str
    feature_count: int
    outputs: int
    midi_min: int
    midi_max: int
    sample_rate: int
    hop_size: int
    fingerprint_sha256: str
    legacy: bool
    checksum_sha256: str


@dataclass(frozen=True)
class ValidatedFeature:
    track_id: str
    bank_id: str
    dataset: FileRef
    metadata: FileRef
    source_wav_sha256: str
    fingerprint_sha256: str
    header: NativeDatasetHeader
    contract: DatasetContract


@dataclass(frozen=True)
class PredictionPlan:
    track_id: str
    model_id: str
    head: str
    feature_bank: str
    dataset: FileRef
    model: FileRef
    source_wav_sha256: str
    feature_fingerprint_sha256: str
    cache_key: str
    score_path: Path
    sidecar_path: Path
    cache_hit: bool
    command: tuple[str, ...]


@dataclass(frozen=True)
class ListeningPlan:
    manifest: ListeningManifest
    features: Mapping[tuple[str, str], ValidatedFeature]
    predictions: tuple[PredictionPlan, ...]


ProcessRunner = Callable[[Sequence[str]], None]


def sha256_file(path_value: str | Path) -> str:
    path = Path(path_value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    name: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} must contain 1-96 ASCII letters, digits, '.', '_' or '-'"
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _resolve_path(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _file_ref(value: Any, base: Path, name: str) -> FileRef:
    raw = _require_object(value, name)
    _strict_keys(raw, {"path", "sha256"}, set(), name)
    return FileRef(
        path=_resolve_path(raw["path"], base, f"{name}.path"),
        sha256=_sha256(raw["sha256"], f"{name}.sha256"),
    )


def _optional_file_ref(value: Any, base: Path, name: str) -> FileRef | None:
    return None if value is None else _file_ref(value, base, name)


def _verify_file(reference: FileRef, name: str) -> None:
    if not reference.path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {reference.path}")
    actual = sha256_file(reference.path)
    if actual != reference.sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {reference.sha256}, "
            f"got {actual}: {reference.path}"
        )


def _read_json(reference: FileRef, name: str) -> dict[str, Any]:
    _verify_file(reference, name)
    try:
        value = json.loads(reference.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {reference.path}") from error
    return _require_object(value, name)


def feature_fingerprint_from_metadata(
    metadata_value: Mapping[str, Any], *, source: str = "metadata"
) -> tuple[str, dict[str, Any]]:
    """Rebuild the semantic fingerprint, including for pre-v2 sidecars.

    Width is deliberately not an identity. The ordered frontend/context
    semantics and the exact fitted thermometer both participate in this hash.
    """
    try:
        frontend_raw = _require_object(metadata_value["frontend"], f"{source}.frontend")
        context_raw = _require_object(metadata_value["context"], f"{source}.context")
        binarizer = _require_object(
            metadata_value["binarizer"], f"{source}.binarizer"
        )
    except KeyError as error:
        raise ValueError(f"{source} lacks feature provenance field {error.args[0]!r}") from error

    frontend = FrontendConfig(**canonical_frontend_mapping(frontend_raw))
    delays = context_raw.get("delays")
    if not isinstance(delays, list) or any(type(delay) is not int for delay in delays):
        raise ValueError(f"{source}.context.delays must be an integer array")
    context = ContextConfig(delays=tuple(delays))
    binarizer_sha = _sha256(binarizer.get("sha256"), f"{source}.binarizer.sha256")
    signature = binarizer.get("signature")
    if not isinstance(signature, str) or not signature:
        raise ValueError(f"{source}.binarizer.signature must be non-empty")

    continuous = metadata_value.get(
        "continuous_feature_count", binarizer.get("continuous_feature_count")
    )
    binary = metadata_value.get(
        "kept_binary_features", binarizer.get("kept_binary_features")
    )
    header = metadata_value.get("header")
    if binary is None and isinstance(header, dict):
        binary = header.get("feature_count")
    if type(continuous) is not int or continuous <= 0:
        raise ValueError(f"{source} has no positive continuous feature count")
    if type(binary) is not int or binary <= 0:
        raise ValueError(f"{source} has no positive binary feature count")

    semantics = binary_feature_semantics(
        frontend,
        context,
        binarizer_sha256=binarizer_sha,
        binarizer_signature=signature,
        continuous_feature_count=continuous,
        binary_feature_count=binary,
    )
    declared = metadata_value.get("feature_semantics")
    if declared is not None:
        if not isinstance(declared, dict):
            raise ValueError(f"{source}.feature_semantics must be an object")
        validate_feature_semantics(
            declared,
            frontend,
            context,
            binarizer_sha256=binarizer_sha,
            binarizer_signature=signature,
            continuous_feature_count=continuous,
            binary_feature_count=binary,
        )
    return str(semantics["fingerprint_sha256"]), semantics


def _dataset_contract(path: Path, allow_legacy: bool) -> DatasetContract:
    # feature_contract is owned by the native-format audit. Keeping this
    # adapter here prevents the listening pipeline from duplicating offsets.
    from .feature_contract import inspect_dataset_contract

    value = inspect_dataset_contract(path, allow_legacy=allow_legacy)
    raw = asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)
    return DatasetContract(
        format_version=int(raw["format_version"]),
        feature_count=int(raw["feature_count"]),
        outputs=int(raw["outputs"]),
        midi_min=int(raw["midi_min"]),
        midi_max=int(raw["midi_max"]),
        sample_rate=int(raw["sample_rate"]),
        hop_size=int(raw["hop_size"]),
        fingerprint_sha256=(
            str(raw["feature_fingerprint_sha256"])
            if raw.get("feature_fingerprint_sha256") is not None
            else "0" * 64
        ),
        legacy=bool(raw["legacy"]),
        checksum_sha256=str(raw["checksum_sha256"]),
    )


def _model_contract(path: Path, allow_legacy: bool) -> ModelContract:
    from .feature_contract import inspect_model_contract

    value = inspect_model_contract(path, allow_legacy=allow_legacy)
    raw = asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)
    return ModelContract(
        format_version=int(raw["format_version"]),
        head=str(raw["head"]),
        feature_count=int(raw["feature_count"]),
        outputs=int(raw["outputs"]),
        midi_min=int(raw["midi_min"]),
        midi_max=int(raw["midi_max"]),
        sample_rate=int(raw["sample_rate"]),
        hop_size=int(raw["hop_size"]),
        fingerprint_sha256=(
            str(raw["feature_fingerprint_sha256"])
            if raw.get("feature_fingerprint_sha256") is not None
            else "0" * 64
        ),
        legacy=bool(raw["legacy"]),
        checksum_sha256=str(raw["checksum_sha256"]),
    )


def _parse_head(value: Any, base: Path, name: str) -> HeadSpec:
    raw = _require_object(value, name)
    has_member = "member" in raw
    has_artifact = "artifact" in raw
    if has_member == has_artifact:
        raise ValueError(f"{name} must select exactly one of member or artifact")
    if has_member:
        _strict_keys(raw, {"member"}, set(), name)
        return HeadSpec(
            member=_identifier(raw["member"], f"{name}.member"),
            artifact=None,
            members=(),
            anchor_feature_bank=None,
        )
    _strict_keys(
        raw,
        {"artifact", "members", "anchor_feature_bank"},
        set(),
        name,
    )
    members = raw["members"]
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError(f"{name}.members must contain at least two IDs")
    return HeadSpec(
        member=None,
        artifact=_file_ref(raw["artifact"], base, f"{name}.artifact"),
        members=tuple(
            _identifier(member, f"{name}.members[{index}]")
            for index, member in enumerate(members)
        ),
        anchor_feature_bank=_identifier(
            raw["anchor_feature_bank"], f"{name}.anchor_feature_bank"
        ),
    )


def load_listening_manifest(path_value: str | Path) -> ListeningManifest:
    path = Path(path_value).resolve()
    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid listening manifest JSON: {path}") from error
    raw = _require_object(raw_value, "manifest")
    _strict_keys(
        raw,
        {
            "schema",
            "predictor",
            "cache_root",
            "output_root",
            "feature_banks",
            "tracks",
            "models",
            "renders",
        },
        set(),
        "manifest",
    )
    if raw["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported listening manifest schema: {raw['schema']!r}")
    base = path.parent

    predictor_raw = _require_object(raw["predictor"], "predictor")
    _strict_keys(predictor_raw, {"executable"}, {"arguments"}, "predictor")
    arguments = predictor_raw.get("arguments", [])
    if not isinstance(arguments, list) or any(
        not isinstance(value, str) for value in arguments
    ):
        raise ValueError("predictor.arguments must be a string array")
    predictor = PredictorSpec(
        executable=_file_ref(predictor_raw["executable"], base, "predictor.executable"),
        arguments=tuple(arguments),
    )

    banks_raw = _require_object(raw["feature_banks"], "feature_banks")
    if not banks_raw:
        raise ValueError("feature_banks must not be empty")
    banks: dict[str, FeatureBankSpec] = {}
    for key, value in banks_raw.items():
        identifier = _identifier(key, f"feature_banks key {key!r}")
        item = _require_object(value, f"feature_banks.{identifier}")
        _strict_keys(
            item,
            {"fingerprint_sha256", "allow_legacy_zero_fingerprint", "provenance"},
            set(),
            f"feature_banks.{identifier}",
        )
        if type(item["allow_legacy_zero_fingerprint"]) is not bool:
            raise ValueError(
                f"feature_banks.{identifier}.allow_legacy_zero_fingerprint "
                "must be a boolean"
            )
        provenance = _require_object(
            item["provenance"], f"feature_banks.{identifier}.provenance"
        )
        _strict_keys(
            provenance,
            {"reference_metadata", "binarizer_sha256"},
            set(),
            f"feature_banks.{identifier}.provenance",
        )
        banks[identifier] = FeatureBankSpec(
            identifier=identifier,
            fingerprint_sha256=_sha256(
                item["fingerprint_sha256"],
                f"feature_banks.{identifier}.fingerprint_sha256",
            ),
            allow_legacy_zero_fingerprint=item["allow_legacy_zero_fingerprint"],
            provenance=BankProvenance(
                reference_metadata=_file_ref(
                    provenance["reference_metadata"],
                    base,
                    f"feature_banks.{identifier}.provenance.reference_metadata",
                ),
                binarizer_sha256=_sha256(
                    provenance["binarizer_sha256"],
                    f"feature_banks.{identifier}.provenance.binarizer_sha256",
                ),
            ),
        )

    tracks_value = raw["tracks"]
    if not isinstance(tracks_value, list) or not tracks_value:
        raise ValueError("tracks must be a non-empty array")
    tracks: dict[str, TrackSpec] = {}
    for index, value in enumerate(tracks_value):
        name = f"tracks[{index}]"
        item = _require_object(value, name)
        _strict_keys(
            item,
            {"id", "source_wav", "feature_banks"},
            {"neuralnote_midi", "teacher_events"},
            name,
        )
        identifier = _identifier(item["id"], f"{name}.id")
        if identifier in tracks:
            raise ValueError(f"duplicate track ID {identifier!r}")
        features_raw = _require_object(item["feature_banks"], f"{name}.feature_banks")
        features: dict[str, TrackFeatureSpec] = {}
        for bank_key, feature_value in features_raw.items():
            bank_id = _identifier(bank_key, f"{name}.feature_banks key")
            if bank_id not in banks:
                raise ValueError(f"{name} references unknown feature bank {bank_id!r}")
            feature = _require_object(
                feature_value, f"{name}.feature_banks.{bank_id}"
            )
            _strict_keys(
                feature,
                {"dataset", "metadata"},
                set(),
                f"{name}.feature_banks.{bank_id}",
            )
            features[bank_id] = TrackFeatureSpec(
                dataset=_file_ref(
                    feature["dataset"], base, f"{name}.feature_banks.{bank_id}.dataset"
                ),
                metadata=_file_ref(
                    feature["metadata"], base, f"{name}.feature_banks.{bank_id}.metadata"
                ),
            )
        tracks[identifier] = TrackSpec(
            identifier=identifier,
            source_wav=_file_ref(item["source_wav"], base, f"{name}.source_wav"),
            feature_banks=features,
            neuralnote_midi=_optional_file_ref(
                item.get("neuralnote_midi"), base, f"{name}.neuralnote_midi"
            ),
            teacher_events=_optional_file_ref(
                item.get("teacher_events"), base, f"{name}.teacher_events"
            ),
        )

    models_value = raw["models"]
    if not isinstance(models_value, list) or not models_value:
        raise ValueError("models must be a non-empty array")
    models: dict[str, ModelSpec] = {}
    for index, value in enumerate(models_value):
        name = f"models[{index}]"
        item = _require_object(value, name)
        _strict_keys(
            item,
            {
                "id",
                "head",
                "feature_bank",
                "feature_fingerprint_sha256",
                "model",
                "allow_legacy_zero_fingerprint",
                "provenance",
            },
            set(),
            name,
        )
        identifier = _identifier(item["id"], f"{name}.id")
        if identifier in models:
            raise ValueError(f"duplicate model ID {identifier!r}")
        head = item["head"]
        if head not in {"activity", "onset"}:
            raise ValueError(f"{name}.head must be 'activity' or 'onset'")
        bank_id = _identifier(item["feature_bank"], f"{name}.feature_bank")
        if bank_id not in banks:
            raise ValueError(f"{name} references unknown feature bank {bank_id!r}")
        legacy = item["allow_legacy_zero_fingerprint"]
        if type(legacy) is not bool:
            raise ValueError(f"{name}.allow_legacy_zero_fingerprint must be a boolean")
        provenance = _require_object(item["provenance"], f"{name}.provenance")
        _strict_keys(
            provenance,
            {"training_metadata"},
            {"experiment_config"},
            f"{name}.provenance",
        )
        models[identifier] = ModelSpec(
            identifier=identifier,
            head=head,
            feature_bank=bank_id,
            feature_fingerprint_sha256=_sha256(
                item["feature_fingerprint_sha256"],
                f"{name}.feature_fingerprint_sha256",
            ),
            model=_file_ref(item["model"], base, f"{name}.model"),
            allow_legacy_zero_fingerprint=legacy,
            provenance=ModelProvenance(
                training_metadata=_file_ref(
                    provenance["training_metadata"],
                    base,
                    f"{name}.provenance.training_metadata",
                ),
                experiment_config=_optional_file_ref(
                    provenance.get("experiment_config"),
                    base,
                    f"{name}.provenance.experiment_config",
                ),
            ),
        )

    renders_value = raw["renders"]
    if not isinstance(renders_value, list) or not renders_value:
        raise ValueError("renders must be a non-empty array")
    renders: list[RenderSpec] = []
    render_ids: set[str] = set()
    for index, value in enumerate(renders_value):
        name = f"renders[{index}]"
        item = _require_object(value, name)
        _strict_keys(
            item,
            {"id", "tracks", "activity", "onset"},
            {"stabilization"},
            name,
        )
        identifier = _identifier(item["id"], f"{name}.id")
        if identifier in render_ids:
            raise ValueError(f"duplicate render ID {identifier!r}")
        render_ids.add(identifier)
        render_tracks = item["tracks"]
        if not isinstance(render_tracks, list) or not render_tracks:
            raise ValueError(f"{name}.tracks must be a non-empty array")
        track_ids = tuple(
            _identifier(track, f"{name}.tracks[{track_index}]")
            for track_index, track in enumerate(render_tracks)
        )
        if len(set(track_ids)) != len(track_ids):
            raise ValueError(f"{name}.tracks contains duplicates")
        unknown_tracks = [track for track in track_ids if track not in tracks]
        if unknown_tracks:
            raise ValueError(f"{name} references unknown tracks {unknown_tracks}")
        stabilization_raw = item.get("stabilization", {})
        if not isinstance(stabilization_raw, dict):
            raise ValueError(f"{name}.stabilization must be an object")
        _strict_keys(
            stabilization_raw,
            set(),
            {"attack_frames", "release_frames", "retrigger_refractory_frames"},
            f"{name}.stabilization",
        )
        renders.append(
            RenderSpec(
                identifier=identifier,
                tracks=track_ids,
                activity=_parse_head(item["activity"], base, f"{name}.activity"),
                onset=_parse_head(item["onset"], base, f"{name}.onset"),
                stabilization=NoteStateConfig(**stabilization_raw),
            )
        )

    manifest = ListeningManifest(
        path=path,
        predictor=predictor,
        cache_root=_resolve_path(raw["cache_root"], base, "cache_root"),
        output_root=_resolve_path(raw["output_root"], base, "output_root"),
        feature_banks=banks,
        tracks=tracks,
        models=models,
        renders=tuple(renders),
    )
    _validate_manifest_references(manifest)
    return manifest


def _head_model_ids(head: HeadSpec) -> tuple[str, ...]:
    return (head.member,) if head.member is not None else head.members


def _validate_manifest_references(manifest: ListeningManifest) -> None:
    for render in manifest.renders:
        for expected_head, head in (("activity", render.activity), ("onset", render.onset)):
            identifiers = _head_model_ids(head)
            for identifier in identifiers:
                if identifier not in manifest.models:
                    raise ValueError(
                        f"render {render.identifier!r} references unknown model "
                        f"{identifier!r}"
                    )
                if manifest.models[identifier].head != expected_head:
                    raise ValueError(
                        f"render {render.identifier!r} uses {identifier!r} as "
                        f"{expected_head}, but the model is {manifest.models[identifier].head}"
                    )
            if head.anchor_feature_bank is not None:
                if head.anchor_feature_bank not in manifest.feature_banks:
                    raise ValueError(
                        f"render {render.identifier!r} has unknown anchor bank "
                        f"{head.anchor_feature_bank!r}"
                    )
                if not any(
                    manifest.models[identifier].feature_bank == head.anchor_feature_bank
                    for identifier in identifiers
                ):
                    raise ValueError(
                        f"render {render.identifier!r} anchor bank "
                        f"{head.anchor_feature_bank!r} has no ensemble member"
                    )
        required_models = set(_head_model_ids(render.activity)) | set(
            _head_model_ids(render.onset)
        )
        for track_id in render.tracks:
            available = manifest.tracks[track_id].feature_banks
            for model_id in required_models:
                bank = manifest.models[model_id].feature_bank
                if bank not in available:
                    raise ValueError(
                        f"track {track_id!r} lacks bank {bank!r} required by "
                        f"model {model_id!r}"
                    )


def _metadata_header_check(
    metadata: Mapping[str, Any], header: NativeDatasetHeader, path: Path
) -> None:
    raw = metadata.get("header")
    if not isinstance(raw, dict):
        raise ValueError(f"feature metadata has no header object: {path}")
    expected = {
        "frame_count": header.frame_count,
        "feature_count": header.feature_count,
        "note_count": header.note_count,
        "midi_min": header.midi_min,
        "midi_max": header.midi_max,
        "sample_rate": header.sample_rate,
        "hop_size": header.hop_size,
        "payload_sha256": header.payload_sha256.hex(),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(
                f"feature metadata header.{key}={raw.get(key)!r} does not "
                f"match dataset {value!r}: {path}"
            )


def _validate_global_banks(manifest: ListeningManifest) -> None:
    for bank in manifest.feature_banks.values():
        metadata = _read_json(
            bank.provenance.reference_metadata,
            f"feature bank {bank.identifier!r} reference metadata",
        )
        fingerprint, _ = feature_fingerprint_from_metadata(
            metadata, source=f"feature bank {bank.identifier!r} reference metadata"
        )
        if fingerprint != bank.fingerprint_sha256:
            raise ValueError(
                f"feature bank {bank.identifier!r} fingerprint {bank.fingerprint_sha256} "
                f"does not match reference provenance {fingerprint}"
            )
        binarizer = _require_object(
            metadata.get("binarizer"),
            f"feature bank {bank.identifier!r} reference metadata.binarizer",
        )
        if binarizer.get("sha256") != bank.provenance.binarizer_sha256:
            raise ValueError(
                f"feature bank {bank.identifier!r} binarizer provenance mismatch"
            )


def _validate_track_feature(
    track: TrackSpec,
    bank: FeatureBankSpec,
    feature: TrackFeatureSpec,
) -> ValidatedFeature:
    _verify_file(track.source_wav, f"track {track.identifier!r} source WAV")
    _verify_file(feature.dataset, f"track {track.identifier!r} bank {bank.identifier!r} dataset")
    metadata = _read_json(
        feature.metadata,
        f"track {track.identifier!r} bank {bank.identifier!r} metadata",
    )
    header = read_native_dataset_header(feature.dataset.path)
    contract = _dataset_contract(
        feature.dataset.path, allow_legacy=bank.allow_legacy_zero_fingerprint
    )
    _metadata_header_check(metadata, header, feature.metadata.path)
    fingerprint, _ = feature_fingerprint_from_metadata(
        metadata,
        source=f"track {track.identifier!r} bank {bank.identifier!r} metadata",
    )
    if fingerprint != bank.fingerprint_sha256:
        raise ValueError(
            f"track {track.identifier!r} bank {bank.identifier!r} semantic "
            f"fingerprint {fingerprint} does not match manifest "
            f"{bank.fingerprint_sha256}"
        )
    if contract.fingerprint_sha256 not in {fingerprint, "0" * 64}:
        raise ValueError(
            f"track {track.identifier!r} bank {bank.identifier!r} embedded "
            "fingerprint disagrees with its sidecar"
        )
    if contract.legacy and not bank.allow_legacy_zero_fingerprint:
        raise ValueError(
            f"track {track.identifier!r} bank {bank.identifier!r} is legacy, "
            "but the manifest did not explicitly allow legacy provenance"
        )
    input_identity = metadata.get("input")
    if not isinstance(input_identity, dict) or input_identity.get("sha256") != track.source_wav.sha256:
        raise ValueError(
            f"track {track.identifier!r} bank {bank.identifier!r} was not "
            "exported from the declared source WAV"
        )
    output_identity = metadata.get("output")
    if isinstance(output_identity, dict) and output_identity.get("sha256") not in {
        None,
        feature.dataset.sha256,
    }:
        raise ValueError(
            f"track {track.identifier!r} bank {bank.identifier!r} dataset SHA "
            "does not match sidecar output provenance"
        )
    return ValidatedFeature(
        track_id=track.identifier,
        bank_id=bank.identifier,
        dataset=feature.dataset,
        metadata=feature.metadata,
        source_wav_sha256=track.source_wav.sha256,
        fingerprint_sha256=fingerprint,
        header=header,
        contract=contract,
    )


def _validate_model(
    manifest: ListeningManifest, model: ModelSpec
) -> ModelContract:
    _verify_file(model.model, f"model {model.identifier!r}")
    bank = manifest.feature_banks[model.feature_bank]
    if model.feature_fingerprint_sha256 != bank.fingerprint_sha256:
        raise ValueError(
            f"model {model.identifier!r} declares fingerprint "
            f"{model.feature_fingerprint_sha256}, but bank {bank.identifier!r} "
            f"is {bank.fingerprint_sha256}"
        )
    training_metadata = _read_json(
        model.provenance.training_metadata,
        f"model {model.identifier!r} training metadata",
    )
    provenance_fingerprint, _ = feature_fingerprint_from_metadata(
        training_metadata,
        source=f"model {model.identifier!r} training metadata",
    )
    if provenance_fingerprint != model.feature_fingerprint_sha256:
        raise ValueError(
            f"model {model.identifier!r} training provenance fingerprint "
            f"{provenance_fingerprint} does not match declared "
            f"{model.feature_fingerprint_sha256}"
        )
    if model.provenance.experiment_config is not None:
        _verify_file(
            model.provenance.experiment_config,
            f"model {model.identifier!r} experiment config",
        )
    contract = _model_contract(
        model.model.path, allow_legacy=model.allow_legacy_zero_fingerprint
    )
    if contract.head != model.head:
        raise ValueError(
            f"model {model.identifier!r} file head {contract.head!r} does not "
            f"match manifest {model.head!r}"
        )
    if contract.fingerprint_sha256 not in {
        model.feature_fingerprint_sha256,
        "0" * 64,
    }:
        raise ValueError(
            f"model {model.identifier!r} embedded fingerprint does not match manifest"
        )
    if contract.legacy and not model.allow_legacy_zero_fingerprint:
        raise ValueError(
            f"model {model.identifier!r} is legacy, but the manifest did not "
            "explicitly allow legacy provenance"
        )
    return contract


def _geometry(header: NativeDatasetHeader) -> tuple[int, int, int, int, int]:
    return (
        header.frame_count,
        header.note_count,
        header.midi_min,
        header.sample_rate,
        header.hop_size,
    )


def _validate_model_feature_geometry(
    model: ModelSpec,
    contract: ModelContract,
    feature: ValidatedFeature,
) -> None:
    expected = (
        feature.header.feature_count,
        feature.header.note_count,
        feature.header.midi_min,
        feature.header.midi_max,
        feature.header.sample_rate,
        feature.header.hop_size,
    )
    actual = (
        contract.feature_count,
        contract.outputs,
        contract.midi_min,
        contract.midi_max,
        contract.sample_rate,
        contract.hop_size,
    )
    if actual != expected:
        raise ValueError(
            f"model {model.identifier!r} geometry/timebase {actual} does not "
            f"match track {feature.track_id!r} bank {feature.bank_id!r} {expected}"
        )
    if feature.fingerprint_sha256 != model.feature_fingerprint_sha256:
        raise ValueError(
            f"model {model.identifier!r} cannot run on same-width wrong bank "
            f"{feature.bank_id!r}: semantic fingerprints differ"
        )


def prediction_cache_key(
    *,
    predictor_sha256: str,
    predictor_arguments: Sequence[str],
    model_sha256: str,
    dataset_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema": CACHE_SCHEMA,
            "predictor_sha256": predictor_sha256,
            "predictor_arguments": list(predictor_arguments),
            "model_sha256": model_sha256,
            "dataset_sha256": dataset_sha256,
        }
    )


def _cache_paths(root: Path, key: str) -> tuple[Path, Path]:
    score = root / "scores" / key[:2] / f"{key}.tsv"
    return score, score.with_suffix(score.suffix + ".json")


def _cache_metadata(plan: PredictionPlan) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "cache_key": plan.cache_key,
        "track_id": plan.track_id,
        "model_id": plan.model_id,
        "head": plan.head,
        "feature_bank": plan.feature_bank,
        "feature_fingerprint_sha256": plan.feature_fingerprint_sha256,
        "source_wav_sha256": plan.source_wav_sha256,
        "model": {"path": str(plan.model.path), "sha256": plan.model.sha256},
        "dataset": {"path": str(plan.dataset.path), "sha256": plan.dataset.sha256},
    }


def _cache_hit(plan: PredictionPlan) -> bool:
    if not plan.score_path.is_file() or not plan.sidecar_path.is_file():
        return False
    try:
        value = json.loads(plan.sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = _cache_metadata(plan)
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            return False
    score_sha = value.get("score_sha256")
    if not isinstance(score_sha, str) or sha256_file(plan.score_path) != score_sha:
        return False
    try:
        loaded = load_score_file(plan.score_path)
    except (OSError, ValueError):
        return False
    return loaded.metadata.head == plan.head


def build_listening_plan(manifest_value: str | Path | ListeningManifest) -> ListeningPlan:
    manifest = (
        manifest_value
        if isinstance(manifest_value, ListeningManifest)
        else load_listening_manifest(manifest_value)
    )
    _verify_file(manifest.predictor.executable, "predictor executable")
    _validate_global_banks(manifest)
    for track in manifest.tracks.values():
        if track.neuralnote_midi is not None:
            _verify_file(track.neuralnote_midi, f"track {track.identifier!r} NeuralNote MIDI")
        if track.teacher_events is not None:
            _verify_file(track.teacher_events, f"track {track.identifier!r} teacher events")

    needed_pairs: set[tuple[str, str]] = set()
    needed_models: set[str] = set()
    ordered_requests: list[tuple[str, str]] = []
    for render in manifest.renders:
        render_models = (*_head_model_ids(render.activity), *_head_model_ids(render.onset))
        for track_id in render.tracks:
            for model_id in render_models:
                pair = (track_id, model_id)
                if pair not in needed_pairs:
                    needed_pairs.add(pair)
                    ordered_requests.append(pair)
                needed_models.add(model_id)

    features: dict[tuple[str, str], ValidatedFeature] = {}
    for track_id, model_id in ordered_requests:
        model = manifest.models[model_id]
        key = (track_id, model.feature_bank)
        if key not in features:
            track = manifest.tracks[track_id]
            features[key] = _validate_track_feature(
                track,
                manifest.feature_banks[model.feature_bank],
                track.feature_banks[model.feature_bank],
            )

    model_contracts = {
        model_id: _validate_model(manifest, manifest.models[model_id])
        for model_id in sorted(needed_models)
    }
    # Validate every cross-bank fusion before planning even one predictor call.
    # This keeps --dry-run a complete preflight rather than a syntax check.
    for render in manifest.renders:
        for head_name, head in (("activity", render.activity), ("onset", render.onset)):
            artifact = _validate_artifact(head_name, head) if head.artifact else None
            for track_id in render.tracks:
                member_features = [
                    features[
                        (track_id, manifest.models[member_id].feature_bank)
                    ]
                    for member_id in _head_model_ids(head)
                ]
                geometries = {_geometry(feature.header) for feature in member_features}
                sources = {
                    feature.source_wav_sha256 for feature in member_features
                }
                if len(geometries) != 1:
                    raise ValueError(
                        f"render {render.identifier!r}/{head_name} cannot fuse "
                        f"track {track_id!r}: feature-bank frame/timebases differ"
                    )
                if sources != {manifest.tracks[track_id].source_wav.sha256}:
                    raise ValueError(
                        f"render {render.identifier!r}/{head_name} cannot fuse "
                        f"track {track_id!r}: source WAV identities differ"
                    )
                if artifact is not None:
                    assert head.anchor_feature_bank is not None
                    anchor = features[(track_id, head.anchor_feature_bank)]
                    fit = artifact.get("fit_dataset")
                    if (
                        not isinstance(fit, dict)
                        or fit.get("feature_count") != anchor.header.feature_count
                        or fit.get("outputs") != anchor.header.note_count
                        or fit.get("midi_min") != anchor.header.midi_min
                        or fit.get("sample_rate") != anchor.header.sample_rate
                        or fit.get("hop_size") != anchor.header.hop_size
                    ):
                        raise ValueError(
                            f"render {render.identifier!r}/{head_name} artifact "
                            f"does not match anchor bank {head.anchor_feature_bank!r}"
                        )
    predictions: list[PredictionPlan] = []
    for track_id, model_id in ordered_requests:
        model = manifest.models[model_id]
        feature = features[(track_id, model.feature_bank)]
        _validate_model_feature_geometry(model, model_contracts[model_id], feature)
        key = prediction_cache_key(
            predictor_sha256=manifest.predictor.executable.sha256,
            predictor_arguments=manifest.predictor.arguments,
            model_sha256=model.model.sha256,
            dataset_sha256=feature.dataset.sha256,
        )
        score_path, sidecar_path = _cache_paths(manifest.cache_root, key)
        command = (
            str(manifest.predictor.executable.path),
            *manifest.predictor.arguments,
            str(feature.dataset.path),
            str(model.model.path),
            "--output",
            str(score_path),
        )
        provisional = PredictionPlan(
            track_id=track_id,
            model_id=model_id,
            head=model.head,
            feature_bank=model.feature_bank,
            dataset=feature.dataset,
            model=model.model,
            source_wav_sha256=feature.source_wav_sha256,
            feature_fingerprint_sha256=feature.fingerprint_sha256,
            cache_key=key,
            score_path=score_path,
            sidecar_path=sidecar_path,
            cache_hit=False,
            command=command,
        )
        predictions.append(replace(provisional, cache_hit=_cache_hit(provisional)))
    return ListeningPlan(
        manifest=manifest,
        features=features,
        predictions=tuple(predictions),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _default_process_runner(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _execute_prediction(plan: PredictionPlan, runner: ProcessRunner) -> Path:
    if _cache_hit(plan):
        return plan.score_path
    plan.score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.score_path.with_suffix(
        plan.score_path.suffix + f".tmp.{os.getpid()}"
    )
    command = [*plan.command[:-1], str(temporary)]
    try:
        runner(command)
        loaded = load_score_file(temporary)
        if loaded.metadata.head != plan.head:
            raise ValueError(
                f"predictor returned {loaded.metadata.head!r} for model "
                f"{plan.model_id!r} ({plan.head!r})"
            )
        temporary.replace(plan.score_path)
        sidecar = {
            **_cache_metadata(plan),
            "score_sha256": sha256_file(plan.score_path),
            "score_metadata": dict(loaded.metadata.raw),
        }
        _atomic_json(plan.sidecar_path, sidecar)
    finally:
        if temporary.exists():
            temporary.unlink()
    return plan.score_path


def _prediction_lookup(plan: ListeningPlan) -> dict[tuple[str, str], PredictionPlan]:
    return {(item.track_id, item.model_id): item for item in plan.predictions}


def _validate_aligned_predictions(
    plan: ListeningPlan,
    track_id: str,
    member_ids: Sequence[str],
) -> None:
    lookup = _prediction_lookup(plan)
    geometries = set()
    source_hashes = set()
    for member_id in member_ids:
        prediction = lookup[(track_id, member_id)]
        sidecar = json.loads(prediction.sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("cache_key") != prediction.cache_key:
            raise ValueError(f"cache provenance mismatch for {track_id}/{member_id}")
        metadata = load_score_file(prediction.score_path).metadata
        geometries.add(
            (
                metadata.frames,
                metadata.outputs,
                metadata.midi_min,
                metadata.sample_rate,
                metadata.hop_size,
            )
        )
        source_hashes.add(sidecar.get("source_wav_sha256"))
    if len(geometries) != 1:
        raise ValueError(
            f"cannot fuse {track_id!r}: member frame/timebase geometries differ"
        )
    if source_hashes != {plan.manifest.tracks[track_id].source_wav.sha256}:
        raise ValueError(
            f"cannot fuse {track_id!r}: members were scored from different WAVs"
        )


def _validate_artifact(head: str, spec: HeadSpec) -> dict[str, Any]:
    assert spec.artifact is not None
    value = _read_json(spec.artifact, f"{head} ensemble artifact")
    if value.get("format") != "TMGM_NATIVE_SCORE_ENSEMBLE_V1":
        raise ValueError(f"unsupported {head} score ensemble artifact")
    if value.get("head") != head:
        raise ValueError(f"{head} ensemble artifact has head {value.get('head')!r}")
    members = value.get("members")
    artifact_ids = (
        [member.get("id") for member in members]
        if isinstance(members, list) and all(isinstance(member, dict) for member in members)
        else None
    )
    if artifact_ids != list(spec.members):
        raise ValueError(
            f"{head} ensemble member order {spec.members} does not match artifact "
            f"{artifact_ids}"
        )
    return value


def _materialize_head(
    plan: ListeningPlan,
    render: RenderSpec,
    track_id: str,
    head_name: str,
    head: HeadSpec,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    lookup = _prediction_lookup(plan)
    member_ids = _head_model_ids(head)
    _validate_aligned_predictions(plan, track_id, member_ids)
    if head.member is not None:
        prediction = lookup[(track_id, head.member)]
        return prediction.score_path, {
            "mode": "single",
            "member": head.member,
            "score_sha256": sha256_file(prediction.score_path),
        }

    assert head.artifact is not None and head.anchor_feature_bank is not None
    artifact = _validate_artifact(head_name, head)
    anchor = plan.features[(track_id, head.anchor_feature_bank)]
    fit = artifact.get("fit_dataset")
    if not isinstance(fit, dict) or fit.get("feature_count") != anchor.header.feature_count:
        raise ValueError(
            f"{head_name} ensemble artifact geometry does not match anchor bank "
            f"{head.anchor_feature_bank!r}"
        )
    members = [
        MemberSpec(identifier, lookup[(track_id, identifier)].score_path)
        for identifier in member_ids
    ]
    output = output_dir / f"{head_name}.ensemble.tsv"
    result = apply_score_ensemble(
        head.artifact.path,
        anchor.dataset.path,
        members,
        output,
    )
    sidecar = {
        "schema": ENSEMBLE_SIDECAR_SCHEMA,
        "track_id": track_id,
        "render_id": render.identifier,
        "head": head_name,
        "source_wav_sha256": plan.manifest.tracks[track_id].source_wav.sha256,
        "anchor_feature_bank": head.anchor_feature_bank,
        "artifact": {
            "path": str(head.artifact.path),
            "sha256": head.artifact.sha256,
        },
        "members": [
            {
                "id": identifier,
                "feature_bank": plan.manifest.models[identifier].feature_bank,
                "feature_fingerprint_sha256": plan.manifest.models[
                    identifier
                ].feature_fingerprint_sha256,
                "score_sha256": sha256_file(
                    lookup[(track_id, identifier)].score_path
                ),
            }
            for identifier in member_ids
        ],
        "output_sha256": sha256_file(output),
        "apply_result": result,
    }
    _atomic_json(output.with_suffix(output.suffix + ".json"), sidecar)
    return output, sidecar


def _atomic_midi_writer(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.mid")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_track(
    plan: ListeningPlan,
    render: RenderSpec,
    track_id: str,
) -> dict[str, Any]:
    track = plan.manifest.tracks[track_id]
    output_dir = plan.manifest.output_root / render.identifier / track_id
    output_dir.mkdir(parents=True, exist_ok=True)
    activity_path, activity_provenance = _materialize_head(
        plan, render, track_id, "activity", render.activity, output_dir
    )
    onset_path, onset_provenance = _materialize_head(
        plan, render, track_id, "onset", render.onset, output_dir
    )
    activity = load_score_file(activity_path)
    onset = load_score_file(onset_path)
    activity_prediction = activity.scores >= activity.metadata.threshold
    onset_prediction = onset.scores >= onset.metadata.threshold
    if activity_prediction.shape != onset_prediction.shape:
        raise ValueError(f"activity/onset dimensions differ for track {track_id!r}")
    raw = np.concatenate((activity_prediction, onset_prediction), axis=1).astype(
        np.uint32
    )
    stable = stabilize_frame_predictions(
        raw, activity.metadata.outputs, render.stabilization
    )
    frame_seconds = activity.metadata.hop_size / activity.metadata.sample_rate
    raw_midi = output_dir / "tm-raw.mid"
    stable_midi = output_dir / "tm-stable.mid"
    _atomic_midi_writer(
        raw_midi,
        lambda path: write_frame_predictions(
            path,
            raw,
            activity.metadata.midi_min,
            activity.metadata.outputs,
            frame_seconds,
        ),
    )
    _atomic_midi_writer(
        stable_midi,
        lambda path: write_frame_predictions(
            path,
            stable,
            activity.metadata.midi_min,
            activity.metadata.outputs,
            frame_seconds,
        ),
    )

    neuralnote_output: Path | None = None
    if track.neuralnote_midi is not None:
        neuralnote_output = output_dir / "neuralnote-reference.mid"
        temporary = neuralnote_output.with_suffix(
            neuralnote_output.suffix + f".tmp.{os.getpid()}"
        )
        shutil.copy2(track.neuralnote_midi.path, temporary)
        temporary.replace(neuralnote_output)
    elif track.teacher_events is not None:
        neuralnote_output = output_dir / "neuralnote-reference.mid"
        _atomic_midi_writer(
            neuralnote_output,
            lambda path: write_teacher_events(
                path,
                track.teacher_events.path,
                activity.metadata.midi_min,
                activity.metadata.midi_min + activity.metadata.outputs - 1,
            ),
        )

    (output_dir / "source-wav.txt").write_text(
        str(track.source_wav.path) + "\n", encoding="utf-8"
    )
    result = {
        "track_id": track_id,
        "source_wav": {
            "path": str(track.source_wav.path),
            "sha256": track.source_wav.sha256,
        },
        "frames": activity.metadata.frames,
        "sample_rate": activity.metadata.sample_rate,
        "hop_size": activity.metadata.hop_size,
        "activity": activity_provenance,
        "onset": onset_provenance,
        "raw_midi": {"path": str(raw_midi), "sha256": sha256_file(raw_midi)},
        "stable_midi": {
            "path": str(stable_midi),
            "sha256": sha256_file(stable_midi),
        },
        "neuralnote_midi": (
            {
                "path": str(neuralnote_output),
                "sha256": sha256_file(neuralnote_output),
            }
            if neuralnote_output is not None
            else None
        ),
    }
    _atomic_json(output_dir / "provenance.json", result)
    return result


def _plan_json(plan: ListeningPlan) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "dry_run": True,
        "manifest": {
            "path": str(plan.manifest.path),
            "sha256": sha256_file(plan.manifest.path),
        },
        "predictor": {
            "path": str(plan.manifest.predictor.executable.path),
            "sha256": plan.manifest.predictor.executable.sha256,
        },
        "predictions": [
            {
                "track_id": item.track_id,
                "model_id": item.model_id,
                "head": item.head,
                "feature_bank": item.feature_bank,
                "feature_fingerprint_sha256": item.feature_fingerprint_sha256,
                "dataset_sha256": item.dataset.sha256,
                "model_sha256": item.model.sha256,
                "cache_key": item.cache_key,
                "cache_hit": item.cache_hit,
                "score_path": str(item.score_path),
                "command": list(item.command),
            }
            for item in plan.predictions
        ],
        "renders": [
            {
                "id": render.identifier,
                "tracks": list(render.tracks),
                "activity_members": list(_head_model_ids(render.activity)),
                "onset_members": list(_head_model_ids(render.onset)),
                "output_root": str(plan.manifest.output_root / render.identifier),
            }
            for render in plan.manifest.renders
        ],
    }


def run_listening_manifest(
    manifest_value: str | Path | ListeningManifest,
    *,
    dry_run: bool = False,
    process_runner: ProcessRunner | None = None,
) -> dict[str, Any]:
    plan = build_listening_plan(manifest_value)
    if dry_run:
        return _plan_json(plan)
    runner = process_runner or _default_process_runner
    cache_hits = 0
    for prediction in plan.predictions:
        if _cache_hit(prediction):
            cache_hits += 1
        _execute_prediction(prediction, runner)
    rendered: dict[str, Any] = {}
    for render in plan.manifest.renders:
        rendered[render.identifier] = {
            track_id: _render_track(plan, render, track_id)
            for track_id in render.tracks
        }
    result = {
        "schema": RUN_SCHEMA,
        "dry_run": False,
        "manifest": {
            "path": str(plan.manifest.path),
            "sha256": sha256_file(plan.manifest.path),
        },
        "prediction_count": len(plan.predictions),
        "cache_hits": cache_hits,
        "cache_misses": len(plan.predictions) - cache_hits,
        "renders": rendered,
    }
    _atomic_json(plan.manifest.output_root / "run.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential, provenance-checked native TM inference, cross-bank "
            "score ensemble application and raw/stable MIDI rendering."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every identity/contract and print commands without executing them",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_listening_manifest(args.manifest, dry_run=args.dry_run)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        _atomic_json(args.output_json, result)
    print(rendered)
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
