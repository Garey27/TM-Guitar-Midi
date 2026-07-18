from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO, Iterator

import numpy as np

from .config import ContextConfig, FrontendConfig
from .feature_semantics import (
    binary_feature_semantics,
    canonical_frontend_mapping,
    feature_fingerprint_bytes,
    frontend_schema_descriptor,
    validate_feature_semantics,
)
from .native_dataset import (
    HEADER_BYTES,
    NativeDatasetHeader,
    _pack_header,
    read_native_dataset_header,
)


REPACK_SCHEMA = "tmgm-native-feature-label-repack-v1"
COPY_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class RepackResult:
    path: Path
    header: NativeDatasetHeader
    file_sha256: str
    metadata_path: Path
    feature_section_sha256: str
    activity_section_sha256: str
    onset_section_sha256: str
    onset_indices_sha256: str


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, sha256: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256,
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _header_metadata(header: NativeDatasetHeader) -> dict[str, Any]:
    return {
        **asdict(header),
        "payload_sha256": header.payload_sha256.hex(),
        "feature_fingerprint_sha256": (
            header.feature_fingerprint_sha256.hex()
        ),
    }


def _validate_metadata_header(
    metadata: dict[str, Any],
    header: NativeDatasetHeader,
    dataset_path: Path,
    name: str,
) -> None:
    declared = metadata.get("header")
    if not isinstance(declared, dict):
        raise ValueError(f"{name} metadata is missing its header object")
    expected = _header_metadata(header)
    if (
        not any(header.feature_fingerprint_sha256)
        and "feature_fingerprint_sha256" not in declared
    ):
        expected.pop("feature_fingerprint_sha256")
    missing = sorted(expected.keys() - declared.keys())
    if missing:
        raise ValueError(f"{name} metadata header is missing: {', '.join(missing)}")
    for field, value in expected.items():
        if declared[field] != value:
            raise ValueError(f"{name} metadata header.{field} is stale or invalid")
    if metadata.get("file_bytes") != dataset_path.stat().st_size:
        raise ValueError(f"{name} metadata file_bytes is stale or invalid")
    if metadata.get("rows") != header.frame_count:
        raise ValueError(f"{name} metadata rows disagrees with its header")
    if metadata.get("kept_binary_features") != header.feature_count:
        raise ValueError(
            f"{name} metadata kept_binary_features disagrees with its header"
        )


def _validate_payload_checksum(
    path: Path, header: NativeDatasetHeader, name: str
) -> str:
    payload_digest = hashlib.sha256()
    file_digest = hashlib.sha256()
    with path.open("rb") as stream:
        raw_header = stream.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            raise ValueError(f"{name} has a truncated header")
        file_digest.update(raw_header)
        remaining = path.stat().st_size - HEADER_BYTES
        while remaining:
            chunk = stream.read(min(remaining, COPY_CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"{name} has a truncated payload")
            remaining -= len(chunk)
            payload_digest.update(chunk)
            file_digest.update(chunk)
    if payload_digest.digest() != header.payload_sha256:
        raise ValueError(f"{name} payload checksum mismatch")
    return file_digest.hexdigest()


def _region_chunks(
    path: Path, offset: int, size: int
) -> Iterator[bytes]:
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, COPY_CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"truncated native dataset section: {path}")
            remaining -= len(chunk)
            yield chunk


def _compare_regions(
    first_path: Path,
    first_offset: int,
    second_path: Path,
    second_offset: int,
    size: int,
    name: str,
) -> str:
    digest = hashlib.sha256()
    first_chunks = _region_chunks(first_path, first_offset, size)
    second_chunks = _region_chunks(second_path, second_offset, size)
    for first, second in zip(first_chunks, second_chunks, strict=True):
        if first != second:
            raise ValueError(f"{name} sections are not byte-identical")
        digest.update(first)
    return digest.hexdigest()


def _copy_region(
    source_path: Path,
    offset: int,
    size: int,
    destination: BinaryIO,
    payload_digest: Any,
) -> str:
    section_digest = hashlib.sha256()
    for chunk in _region_chunks(source_path, offset, size):
        destination.write(chunk)
        payload_digest.update(chunk)
        section_digest.update(chunk)
    return section_digest.hexdigest()


def _require_object(metadata: dict[str, Any], key: str, name: str) -> dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{name} metadata.{key} must be an object")
    return value


def _require_tracks(metadata: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value = metadata.get("selected_tracks")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} metadata.selected_tracks must be a non-empty array")
    tracks: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name} selected_tracks[{index}] must be an object")
        for key in ("source", "id", "group"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(
                    f"{name} selected_tracks[{index}].{key} must be a string"
                )
        if type(item.get("rows")) is not int or item["rows"] <= 0:
            raise ValueError(
                f"{name} selected_tracks[{index}].rows must be a positive integer"
            )
        _track_cache_signature(item, name, index)
        tracks.append(item)
    return tracks


def _track_cache_signature(
    track: dict[str, Any], name: str, index: int
) -> str:
    value = track.get("cache_signature")
    if isinstance(value, str) and value:
        return value
    feature = track.get("feature_cache_signature")
    label = track.get("label_cache_signature")
    if (
        isinstance(feature, str)
        and feature
        and isinstance(label, str)
        and label
    ):
        return _canonical_hash(
            {
                "schema": REPACK_SCHEMA,
                "legacy_repack_feature_cache": feature,
                "legacy_repack_label_cache": label,
            }
        )
    raise ValueError(
        f"{name} selected_tracks[{index}] has no usable cache provenance"
    )


def _track_cache_lineage(
    track: dict[str, Any], name: str, index: int
) -> list[str]:
    value = track.get("cache_lineage")
    if value is not None:
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(
                f"{name} selected_tracks[{index}].cache_lineage is invalid"
            )
        return list(value)
    lineage: list[str] = []
    for key in (
        "cache_signature",
        "feature_cache_signature",
        "label_cache_signature",
    ):
        item = track.get(key)
        if isinstance(item, str) and item and item not in lineage:
            lineage.append(item)
    if not lineage:
        lineage.append(_track_cache_signature(track, name, index))
    return lineage


def _frontend_from_metadata(value: Any, name: str) -> FrontendConfig:
    if not isinstance(value, dict):
        raise ValueError(f"{name} frontend must be an object")
    return FrontendConfig(**canonical_frontend_mapping(value))


def _context_from_metadata(value: Any, name: str) -> ContextConfig:
    if not isinstance(value, dict) or set(value) != {"delays"}:
        raise ValueError(f"{name} context must contain only delays")
    delays = value["delays"]
    if not isinstance(delays, list) or any(type(item) is not int for item in delays):
        raise ValueError(f"{name} context.delays must be an integer array")
    return ContextConfig(delays=tuple(delays))


def _track_identity(track: dict[str, Any]) -> tuple[str, str, str, int]:
    return track["source"], track["id"], track["group"], track["rows"]


def _validate_metadata_pair(
    feature_metadata: dict[str, Any],
    label_metadata: dict[str, Any],
    feature_header: NativeDatasetHeader,
    split: str,
) -> tuple[list[dict[str, Any]], FrontendConfig, ContextConfig]:
    for name, metadata in (
        ("feature source", feature_metadata),
        ("label source", label_metadata),
    ):
        if metadata.get("format") != "TMGMDAT" or metadata.get("version") not in {
            1,
            2,
        }:
            raise ValueError(f"{name} metadata is not TMGMDAT v1/v2")
        if metadata.get("split") != split:
            raise ValueError(f"{name} metadata split disagrees with {split}")
        _require_object(metadata, "frontend", name)
        _require_object(metadata, "context", name)
        _require_object(metadata, "targets", name)
        _require_object(metadata, "binarizer", name)

    exact_shared = (
        "context",
        "sampling",
        "source_track_counts",
        "source_row_counts",
        "category_counts",
        "track_count",
        "rows",
    )
    for field in exact_shared:
        if feature_metadata.get(field) != label_metadata.get(field):
            raise ValueError(f"feature/label metadata {field} is not identical")

    feature_frontend_config = _frontend_from_metadata(
        feature_metadata["frontend"], "feature source"
    )
    label_frontend_config = _frontend_from_metadata(
        label_metadata["frontend"], "label source"
    )
    feature_frontend = asdict(feature_frontend_config)
    label_frontend = asdict(label_frontend_config)
    allowed_frontend_differences = {
        "harmonic_local_contrast",
        "contrast_offset_semitones",
        "expose_harmonic_local_profile",
        "contrast_attack_features",
    }
    for field in feature_frontend:
        if (
            field not in allowed_frontend_differences
            and feature_frontend[field] != label_frontend[field]
        ):
            raise ValueError(f"feature/label frontend.{field} is not identical")

    feature_context = _context_from_metadata(
        feature_metadata["context"], "feature source"
    )
    label_context = _context_from_metadata(
        label_metadata["context"], "label source"
    )
    if feature_context != label_context:
        raise ValueError("feature/label context is not identical")

    feature_targets = feature_metadata["targets"]
    label_targets = label_metadata["targets"]
    if feature_targets.keys() != label_targets.keys():
        raise ValueError("feature/label target fields are not identical")
    for field in feature_targets:
        if field not in {"onset_delay_frames", "onset_width_frames"} and (
            feature_targets[field] != label_targets[field]
        ):
            raise ValueError(f"feature/label targets.{field} is not identical")
    target_timing_differs = any(
        feature_targets[field] != label_targets[field]
        for field in ("onset_delay_frames", "onset_width_frames")
    )
    sampling = feature_metadata["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("feature/label sampling metadata must be an object")
    if target_timing_differs and sampling.get("frame_sampling_policy") != "natural":
        raise ValueError(
            "onset delay/width relabeling is supported only for natural sampling"
        )

    feature_tracks = _require_tracks(feature_metadata, "feature source")
    label_tracks = _require_tracks(label_metadata, "label source")
    if len(feature_tracks) != len(label_tracks):
        raise ValueError("feature/label selected track counts disagree")
    if [_track_identity(track) for track in feature_tracks] != [
        _track_identity(track) for track in label_tracks
    ]:
        raise ValueError("feature/label ordered selected_tracks disagree")
    if sum(track["rows"] for track in feature_tracks) != feature_header.frame_count:
        raise ValueError("selected track rows do not sum to the dataset frame count")
    if feature_metadata["track_count"] != len(feature_tracks):
        raise ValueError("metadata track_count disagrees with selected_tracks")
    track_counts = dict(
        sorted(Counter(track["source"] for track in feature_tracks).items())
    )
    row_counts: Counter[str] = Counter()
    for track in feature_tracks:
        row_counts[track["source"]] += track["rows"]
    if feature_metadata["source_track_counts"] != track_counts:
        raise ValueError("metadata source_track_counts disagrees with selected_tracks")
    if feature_metadata["source_row_counts"] != dict(sorted(row_counts.items())):
        raise ValueError("metadata source_row_counts disagrees with selected_tracks")
    onset_sampling = feature_metadata["sampling"].get("onset_training_indices")
    if not isinstance(onset_sampling, dict) or (
        onset_sampling.get("rows") != feature_header.onset_index_count
    ):
        raise ValueError(
            "metadata sampling onset row count disagrees with the dataset header"
        )

    combined: list[dict[str, Any]] = []
    for index, (feature_track, label_track) in enumerate(
        zip(feature_tracks, label_tracks, strict=True)
    ):
        feature_signature = _track_cache_signature(
            feature_track, "feature source", index
        )
        label_signature = _track_cache_signature(
            label_track, "label source", index
        )
        lineage: list[str] = []
        for item in [
            *_track_cache_lineage(feature_track, "feature source", index),
            *_track_cache_lineage(label_track, "label source", index),
        ]:
            if item not in lineage:
                lineage.append(item)
        combined_signature = _canonical_hash(
            {
                "schema": REPACK_SCHEMA,
                "feature_cache_signature": feature_signature,
                "label_cache_signature": label_signature,
                "lineage": lineage,
            }
        )
        combined.append(
            {
                "source": feature_track["source"],
                "id": feature_track["id"],
                "group": feature_track["group"],
                "rows": feature_track["rows"],
                "cache_signature": combined_signature,
                "feature_cache_signature": feature_signature,
                "label_cache_signature": label_signature,
                "cache_lineage": lineage,
            }
        )
    return combined, feature_frontend_config, feature_context


def _validate_compatible_headers(
    feature: NativeDatasetHeader, label: NativeDatasetHeader
) -> None:
    fields = (
        "frame_count",
        "note_count",
        "label_words_per_row",
        "midi_min",
        "midi_max",
        "sample_rate",
        "hop_size",
        "onset_index_count",
        "seed",
    )
    for field in fields:
        if getattr(feature, field) != getattr(label, field):
            raise ValueError(f"feature/label headers disagree on {field}")
    if feature.activity_bytes != label.activity_bytes:
        raise ValueError("feature/label activity section sizes disagree")
    if label.onset_indices_bytes != label.onset_index_count * np.dtype("<u4").itemsize:
        raise ValueError("label onset index dimensions are invalid")


def _validate_onset_indices(path: Path, header: NativeDatasetHeader) -> None:
    if header.onset_index_count == 0:
        return
    mapped = np.memmap(
        path,
        mode="r",
        dtype="<u4",
        offset=header.onset_indices_offset,
        shape=(header.onset_index_count,),
    )
    try:
        if int(mapped.max()) >= header.frame_count:
            raise ValueError("label onset training index is outside the frame matrix")
    finally:
        del mapped


def repack_native_split(
    feature_dataset: str | Path,
    label_dataset: str | Path,
    output_path: str | Path,
    *,
    split: str | None = None,
) -> RepackResult:
    """Combine one verified feature section with another export's labels."""
    feature_path = Path(feature_dataset).resolve()
    label_path = Path(label_dataset).resolve()
    destination = Path(output_path).resolve()
    if feature_path == label_path:
        raise ValueError("feature and label datasets must be different files")
    if destination in {feature_path, label_path}:
        raise ValueError("output must not overwrite an input dataset")
    destination_sidecar = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or destination_sidecar.exists():
        raise FileExistsError(f"repack output already exists: {destination}")

    feature_header = read_native_dataset_header(feature_path)
    label_header = read_native_dataset_header(label_path)
    _validate_compatible_headers(feature_header, label_header)
    split = split or feature_path.stem

    feature_metadata_path = feature_path.with_suffix(feature_path.suffix + ".json")
    label_metadata_path = label_path.with_suffix(label_path.suffix + ".json")
    feature_metadata = _read_json_object(
        feature_metadata_path, "feature source metadata"
    )
    label_metadata = _read_json_object(label_metadata_path, "label source metadata")
    _validate_metadata_header(
        feature_metadata, feature_header, feature_path, "feature source"
    )
    _validate_metadata_header(
        label_metadata, label_header, label_path, "label source"
    )
    selected_tracks, feature_frontend, feature_context = _validate_metadata_pair(
        feature_metadata, label_metadata, feature_header, split
    )
    feature_binarizer = feature_metadata["binarizer"]
    binarizer_sha256 = feature_binarizer.get("sha256")
    binarizer_signature = feature_binarizer.get("signature")
    if (
        not isinstance(binarizer_sha256, str)
        or len(binarizer_sha256) != 64
        or not isinstance(binarizer_signature, str)
        or not binarizer_signature
    ):
        raise ValueError("feature source has no complete binarizer identity")
    feature_semantics = binary_feature_semantics(
        feature_frontend,
        feature_context,
        binarizer_sha256=binarizer_sha256,
        binarizer_signature=binarizer_signature,
        continuous_feature_count=feature_metadata["continuous_feature_count"],
        binary_feature_count=feature_header.feature_count,
    )
    declared_feature_semantics = feature_metadata.get("feature_semantics")
    legacy_feature_contract_upgraded = declared_feature_semantics is None
    if declared_feature_semantics is not None:
        if not isinstance(declared_feature_semantics, dict):
            raise ValueError("feature source feature_semantics must be an object")
        validate_feature_semantics(
            declared_feature_semantics,
            feature_frontend,
            feature_context,
            binarizer_sha256=binarizer_sha256,
            binarizer_signature=binarizer_signature,
            continuous_feature_count=feature_metadata[
                "continuous_feature_count"
            ],
            binary_feature_count=feature_header.feature_count,
        )
    feature_fingerprint = feature_fingerprint_bytes(feature_semantics)
    if any(feature_header.feature_fingerprint_sha256) and (
        feature_header.feature_fingerprint_sha256 != feature_fingerprint
    ):
        raise ValueError(
            "feature source header fingerprint disagrees with feature semantics"
        )

    feature_file_sha256 = _validate_payload_checksum(
        feature_path, feature_header, "feature source"
    )
    label_file_sha256 = _validate_payload_checksum(
        label_path, label_header, "label source"
    )
    activity_sha256 = _compare_regions(
        feature_path,
        feature_header.activity_offset,
        label_path,
        label_header.activity_offset,
        feature_header.activity_bytes,
        "feature/label activity",
    )
    if feature_header.onset_indices_bytes != label_header.onset_indices_bytes:
        raise ValueError("feature/label onset index section sizes disagree")
    sampling_indices_sha256 = _compare_regions(
        feature_path,
        feature_header.onset_indices_offset,
        label_path,
        label_header.onset_indices_offset,
        label_header.onset_indices_bytes,
        "feature/label onset index",
    )
    _validate_onset_indices(label_path, label_header)

    section_sources = (
        (feature_path, feature_header.features_offset, feature_header.features_bytes),
        (label_path, label_header.activity_offset, label_header.activity_bytes),
        (label_path, label_header.onset_offset, label_header.onset_bytes),
        (
            label_path,
            label_header.onset_indices_offset,
            label_header.onset_indices_bytes,
        ),
    )
    sizes = tuple(source[2] for source in section_sources)
    offsets: list[int] = []
    next_offset = HEADER_BYTES
    for size in sizes:
        offsets.append(next_offset)
        next_offset += size

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    payload_digest = hashlib.sha256()
    try:
        with temporary.open("w+b") as output:
            output.write(bytes(HEADER_BYTES))
            section_sha256 = tuple(
                _copy_region(path, offset, size, output, payload_digest)
                for path, offset, size in section_sources
            )
            header = NativeDatasetHeader(
                frame_count=feature_header.frame_count,
                feature_count=feature_header.feature_count,
                feature_words_per_row=feature_header.feature_words_per_row,
                note_count=label_header.note_count,
                label_words_per_row=label_header.label_words_per_row,
                midi_min=label_header.midi_min,
                midi_max=label_header.midi_max,
                sample_rate=label_header.sample_rate,
                hop_size=label_header.hop_size,
                onset_index_count=label_header.onset_index_count,
                features_offset=offsets[0],
                features_bytes=sizes[0],
                activity_offset=offsets[1],
                activity_bytes=sizes[1],
                onset_offset=offsets[2],
                onset_bytes=sizes[2],
                onset_indices_offset=offsets[3],
                onset_indices_bytes=sizes[3],
                seed=label_header.seed,
                payload_sha256=payload_digest.digest(),
                feature_fingerprint_sha256=feature_fingerprint,
            )
            output.seek(0)
            output.write(_pack_header(header))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    written_header = read_native_dataset_header(destination)
    if written_header != header:
        raise AssertionError("repacked native header changed after writing")
    output_file_sha256 = _validate_payload_checksum(
        destination, header, "repacked output"
    )
    if section_sha256[1] != activity_sha256:
        raise AssertionError("repacked activity section changed while copying")
    if section_sha256[3] != sampling_indices_sha256:
        raise AssertionError("repacked onset index section changed while copying")

    feature_metadata_sha256 = _sha256_file(feature_metadata_path)
    label_metadata_sha256 = _sha256_file(label_metadata_path)
    signature_basis = {
        "schema": REPACK_SCHEMA,
        "split": split,
        "feature_source_sha256": feature_file_sha256,
        "feature_metadata_sha256": feature_metadata_sha256,
        "label_source_sha256": label_file_sha256,
        "label_metadata_sha256": label_metadata_sha256,
        "section_sha256": {
            "features": section_sha256[0],
            "activity": section_sha256[1],
            "onset": section_sha256[2],
            "onset_indices": section_sha256[3],
        },
        "frontend": asdict(feature_frontend),
        "context": feature_metadata["context"],
        "targets": label_metadata["targets"],
        "binarizer": feature_metadata["binarizer"],
        "feature_semantics": feature_semantics,
    }
    metadata = {
        "format": "TMGMDAT",
        "version": 2,
        "export_schema": feature_metadata.get("export_schema"),
        "export_signature": _canonical_hash(signature_basis),
        "split": split,
        "sampling": label_metadata["sampling"],
        "source_track_counts": label_metadata["source_track_counts"],
        "source_row_counts": label_metadata["source_row_counts"],
        "category_counts": label_metadata["category_counts"],
        "track_count": label_metadata["track_count"],
        "rows": header.frame_count,
        "continuous_feature_count": feature_metadata["continuous_feature_count"],
        "kept_binary_features": header.feature_count,
        "binarizer": feature_metadata["binarizer"],
        "frontend_schema": frontend_schema_descriptor(feature_frontend),
        "feature_semantics": feature_semantics,
        "frontend": asdict(feature_frontend),
        "context": feature_metadata["context"],
        "targets": label_metadata["targets"],
        "selected_tracks": selected_tracks,
        "header": _header_metadata(header),
        "file_bytes": destination.stat().st_size,
        "file_sha256": output_file_sha256,
        "repack": {
            "schema": REPACK_SCHEMA,
            "legacy_feature_contract_upgraded": (
                legacy_feature_contract_upgraded
                or not any(feature_header.feature_fingerprint_sha256)
            ),
            "feature_source": {
                **_file_identity(feature_path, feature_file_sha256),
                "metadata": _file_identity(
                    feature_metadata_path, feature_metadata_sha256
                ),
                "export_signature": feature_metadata.get("export_signature"),
                "targets": feature_metadata["targets"],
            },
            "label_source": {
                **_file_identity(label_path, label_file_sha256),
                "metadata": _file_identity(label_metadata_path, label_metadata_sha256),
                "export_signature": label_metadata.get("export_signature"),
                "frontend": label_metadata["frontend"],
            },
            "section_sources": {
                "features": "feature_source",
                "activity": "label_source",
                "onset": "label_source",
                "onset_indices": "label_source",
            },
            "section_sha256": {
                "features": section_sha256[0],
                "activity": section_sha256[1],
                "onset": section_sha256[2],
                "onset_indices": section_sha256[3],
            },
            "verification": {
                "input_payload_checksums": True,
                "header_timebase_rows_midi": True,
                "ordered_selected_tracks": True,
                "selected_track_rows_sum_to_frames": True,
                "activity_byte_identical": True,
                "sampling_identity": True,
                "onset_indices_byte_identical": True,
                "onset_indices_in_range": True,
                "output_payload_checksum": True,
            },
        },
    }
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    _write_json_atomic(metadata_path, metadata)
    return RepackResult(
        path=destination,
        header=header,
        file_sha256=output_file_sha256,
        metadata_path=metadata_path,
        feature_section_sha256=section_sha256[0],
        activity_section_sha256=section_sha256[1],
        onset_section_sha256=section_sha256[2],
        onset_indices_sha256=section_sha256[3],
    )


def _copy_verified_binarizer(
    feature_root: Path,
    destination_root: Path,
    feature_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    source = feature_root / "global-quantile-thermometer.npz"
    source_sidecar = source.with_suffix(source.suffix + ".json")
    metadata = _read_json_object(source_sidecar, "feature binarizer metadata")
    actual_sha256 = _sha256_file(source)
    if metadata.get("sha256") != actual_sha256:
        raise ValueError("feature binarizer checksum disagrees with its sidecar")
    split_semantics: list[dict[str, Any]] = []
    for split_metadata in feature_metadata:
        declared = split_metadata["binarizer"]
        for key in ("sha256", "signature", "quantiles", "kept_binary_features"):
            if declared.get(key) != metadata.get(key):
                raise ValueError(
                    f"feature split binarizer.{key} disagrees with the binarizer sidecar"
                )
        if (
            split_metadata.get("continuous_feature_count")
            != metadata.get("continuous_feature_count")
        ):
            raise ValueError(
                "feature split continuous feature count disagrees with the binarizer"
            )
        frontend = _frontend_from_metadata(
            split_metadata["frontend"], "feature split"
        )
        context = _context_from_metadata(
            split_metadata["context"], "feature split"
        )
        semantics = binary_feature_semantics(
            frontend,
            context,
            binarizer_sha256=actual_sha256,
            binarizer_signature=str(metadata["signature"]),
            continuous_feature_count=int(metadata["continuous_feature_count"]),
            binary_feature_count=int(metadata["kept_binary_features"]),
        )
        declared_semantics = split_metadata.get("feature_semantics")
        if declared_semantics is not None and declared_semantics != semantics:
            raise ValueError("feature split feature semantics is stale or invalid")
        split_semantics.append(semantics)
    if any(value != split_semantics[0] for value in split_semantics[1:]):
        raise ValueError("feature split semantic fingerprints disagree")

    destination = destination_root / source.name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=COPY_CHUNK_BYTES)
    temporary.replace(destination)
    if _sha256_file(destination) != actual_sha256:
        raise AssertionError("copied binarizer checksum changed")
    rebuilt = {
        key: metadata[key]
        for key in (
            "schema",
            "signature",
            "sha256",
            "train_rows",
            "continuous_feature_count",
            "quantiles",
            "raw_thermometer_literals",
            "kept_binary_features",
            "file_bytes",
        )
    }
    rebuilt["schema"] = 2
    rebuilt["frontend_schema"] = split_semantics[0]["frontend_schema"]
    rebuilt["frontend"] = split_semantics[0]["frontend"]
    rebuilt["context"] = split_semantics[0]["context"]
    rebuilt["feature_semantics"] = split_semantics[0]
    rebuilt["provenance"] = {
        "schema": REPACK_SCHEMA,
        "source": _file_identity(source, actual_sha256),
        "source_metadata": _file_identity(
            source_sidecar, _sha256_file(source_sidecar)
        ),
    }
    _write_json_atomic(destination.with_suffix(destination.suffix + ".json"), rebuilt)
    return rebuilt


def repack_native_corpus(
    feature_root: str | Path,
    label_root: str | Path,
    output_root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
    force: bool = False,
) -> list[RepackResult]:
    """Atomically build a corpus from verified feature and label exports."""
    features = Path(feature_root).resolve()
    labels = Path(label_root).resolve()
    destination = Path(output_root).resolve()
    if features == labels:
        raise ValueError("feature and label roots must be distinct")
    if any(
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
        for source in (features, labels)
    ):
        raise ValueError("feature, label and output roots must be distinct")
    if destination.exists() and not force:
        raise FileExistsError(f"output root already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.repack-", dir=destination.parent)
    )
    try:
        results = [
            repack_native_split(
                features / f"{split}.tmgd",
                labels / f"{split}.tmgd",
                staging / f"{split}.tmgd",
                split=split,
            )
            for split in splits
        ]
        feature_metadata = [
            _read_json_object(
                features / f"{split}.tmgd.json", f"{split} feature metadata"
            )
            for split in splits
        ]
        _copy_verified_binarizer(features, staging, feature_metadata)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return [
            RepackResult(
                path=destination / result.path.name,
                header=result.header,
                file_sha256=result.file_sha256,
                metadata_path=destination / result.metadata_path.name,
                feature_section_sha256=result.feature_section_sha256,
                activity_section_sha256=result.activity_section_sha256,
                onset_section_sha256=result.onset_section_sha256,
                onset_indices_sha256=result.onset_indices_sha256,
            )
            for result in results
        ]
    finally:
        if staging.exists():
            shutil.rmtree(staging)
