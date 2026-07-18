from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .binarize import QuantileThermometer
from .config import ContextConfig, FrontendConfig, TargetConfig
from .dataset import CorpusEntry, build_track_examples
from .feature_semantics import (
    binary_feature_semantics,
    feature_fingerprint_bytes,
    frontend_schema_descriptor,
)
from .native_dataset import (
    NativeDatasetHeader,
    onset_training_indices,
    read_native_dataset_header,
    write_native_dataset_batches,
)


TRACK_CACHE_SCHEMA = 2
EXPORT_SCHEMA = 2


@dataclass(frozen=True)
class CachedTrack:
    entry: CorpusEntry
    path: Path
    signature: str
    rows: int
    continuous_feature_count: int
    note_count: int
    category_counts: dict[str, int]
    sampling_policy: str


TrackBuilder = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_signature_value(targets: TargetConfig) -> dict[str, Any]:
    """Keep delay=0 compatible with caches created before causal-delay ablations."""
    value = asdict(targets)
    if value.get("onset_delay_frames") == 0:
        value.pop("onset_delay_frames")
    return value


def _frontend_signature_value(frontend: FrontendConfig) -> dict[str, Any]:
    """Canonical config: defaults stay explicit so signatures never alias."""
    return asdict(frontend)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _track_signature(
    entry: CorpusEntry,
    teacher_root: Path,
    split: str,
    frames_per_track: int,
    seed: int,
    frontend: FrontendConfig,
    context: ContextConfig,
    targets: TargetConfig,
    train_sampling: str = "balanced",
) -> str:
    teacher_base = teacher_root / entry.output_relative
    balanced_sampling = split == "train" and train_sampling == "balanced"
    return _canonical_hash(
        {
            "schema": TRACK_CACHE_SCHEMA,
            "split": split,
            "source": entry.source,
            "id": entry.identifier,
            "group": entry.group,
            "frames_per_track": frames_per_track,
            "seed": seed,
            # Keep the original key so the default balanced export continues
            # to hit caches written before train_sampling was configurable.
            "balanced_sampling": balanced_sampling,
            "frontend": _frontend_signature_value(frontend),
            "frontend_schema": frontend_schema_descriptor(frontend),
            "context": asdict(context),
            "targets": _target_signature_value(targets),
            "input": _file_identity(entry.input_path),
            "teacher_nnpg": _file_identity(teacher_base.with_suffix(".nnpg")),
            "teacher_events": _file_identity(
                teacher_base.with_suffix(".events.tsv")
            ),
        }
    )


def _cached_track_from_metadata(
    entry: CorpusEntry, path: Path, signature: str, metadata: dict[str, Any]
) -> CachedTrack | None:
    if (
        metadata.get("schema") != TRACK_CACHE_SCHEMA
        or metadata.get("signature") != signature
        or not path.is_file()
        or path.stat().st_size != metadata.get("file_bytes")
    ):
        return None
    rows = int(metadata.get("rows", 0))
    feature_count = int(metadata.get("continuous_feature_count", 0))
    note_count = int(metadata.get("note_count", 0))
    if rows <= 0 or feature_count <= 0 or note_count <= 0:
        return None
    return CachedTrack(
        entry=entry,
        path=path,
        signature=signature,
        rows=rows,
        continuous_feature_count=feature_count,
        note_count=note_count,
        category_counts={
            str(name): int(count)
            for name, count in metadata.get("category_counts", {}).items()
        },
        sampling_policy=str(
            metadata.get(
                "sampling_policy",
                "balanced" if entry.split == "train" else "natural",
            )
        ),
    )


def cache_split_tracks(
    entries: list[CorpusEntry],
    teacher_root: str | Path,
    cache_root: str | Path,
    split: str,
    frames_per_track: int,
    seed: int,
    frontend: FrontendConfig,
    context: ContextConfig,
    targets: TargetConfig,
    *,
    train_sampling: str = "balanced",
    force: bool = False,
    builder: TrackBuilder = build_track_examples,
) -> list[CachedTrack]:
    """Extract sampled tracks once and reuse them after interrupted exports."""
    if split not in {"train", "validation"}:
        raise ValueError("native corpus export supports train and validation splits")
    if frames_per_track <= 0:
        raise ValueError("frames per track must be positive")
    if train_sampling not in {"balanced", "natural"}:
        raise ValueError("train sampling must be 'balanced' or 'natural'")
    sampling_policy = train_sampling if split == "train" else "natural"
    balanced_sampling = split == "train" and sampling_policy == "balanced"
    teacher_root = Path(teacher_root)
    split_root = Path(cache_root) / "tracks" / split
    split_root.mkdir(parents=True, exist_ok=True)
    cached: list[CachedTrack] = []

    for index, entry in enumerate(entries):
        signature = _track_signature(
            entry,
            teacher_root,
            split,
            frames_per_track,
            seed,
            frontend,
            context,
            targets,
            train_sampling,
        )
        safe_source = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in entry.source
        )
        path = split_root / f"{index:04d}-{safe_source}-{signature[:16]}.npz"
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not force and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                hit = _cached_track_from_metadata(entry, path, signature, metadata)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                hit = None
            if hit is not None:
                print(
                    f"[{split} {index + 1}/{len(entries)}] cache "
                    f"{entry.source}/{entry.identifier} rows={hit.rows}"
                )
                cached.append(hit)
                continue

        print(
            f"[{split} {index + 1}/{len(entries)}] extract "
            f"{entry.source}/{entry.identifier}"
        )
        continuous, truth, _, _, category_counts = builder(
            entry,
            teacher_root,
            frontend,
            context,
            targets,
            frames_per_track,
            seed,
            balanced_sampling,
        )
        continuous = np.ascontiguousarray(continuous, dtype=np.float32)
        truth = np.asarray(truth)
        note_count = frontend.note_count
        if continuous.ndim != 2 or continuous.shape[0] == 0:
            raise ValueError(f"empty feature cache for {entry.identifier}")
        if truth.ndim != 2 or truth.shape != (continuous.shape[0], 2 * note_count):
            raise ValueError(f"unexpected target shape for {entry.identifier}")
        activity = np.ascontiguousarray(truth[:, :note_count], dtype=np.uint8)
        onset = np.ascontiguousarray(truth[:, note_count:], dtype=np.uint8)

        temporary = path.with_suffix(path.suffix + ".tmp.npz")
        np.savez(
            temporary,
            features=continuous,
            activity=activity,
            onset=onset,
        )
        temporary.replace(path)
        metadata = {
            "schema": TRACK_CACHE_SCHEMA,
            "signature": signature,
            "split": split,
            "source": entry.source,
            "id": entry.identifier,
            "group": entry.group,
            "rows": int(continuous.shape[0]),
            "continuous_feature_count": int(continuous.shape[1]),
            "note_count": note_count,
            "frontend_schema": frontend_schema_descriptor(frontend),
            "category_counts": {
                str(name): int(count) for name, count in category_counts.items()
            },
            "sampling_policy": sampling_policy,
            "balanced_sampling": balanced_sampling,
            "file_bytes": path.stat().st_size,
        }
        _write_json_atomic(metadata_path, metadata)
        cached_track = _cached_track_from_metadata(
            entry, path, signature, metadata
        )
        if cached_track is None:
            raise AssertionError("newly written track cache did not validate")
        cached.append(cached_track)
    return cached


def _validate_track_dimensions(tracks: list[CachedTrack]) -> tuple[int, int, int]:
    if not tracks:
        raise ValueError("cannot export an empty track list")
    feature_counts = {track.continuous_feature_count for track in tracks}
    note_counts = {track.note_count for track in tracks}
    if len(feature_counts) != 1 or len(note_counts) != 1:
        raise ValueError("cached track dimensions disagree")
    return (
        sum(track.rows for track in tracks),
        next(iter(feature_counts)),
        next(iter(note_counts)),
    )


def _load_track_features(track: CachedTrack) -> np.ndarray:
    with np.load(track.path, allow_pickle=False) as cached:
        features = np.asarray(cached["features"], dtype=np.float32)
    if features.shape != (track.rows, track.continuous_feature_count):
        raise ValueError(f"cached feature shape changed: {track.path}")
    return features


def _load_track_labels(track: CachedTrack) -> tuple[np.ndarray, np.ndarray]:
    with np.load(track.path, allow_pickle=False) as cached:
        activity = np.asarray(cached["activity"], dtype=np.uint8)
        onset = np.asarray(cached["onset"], dtype=np.uint8)
    expected_labels = (track.rows, track.note_count)
    if activity.shape != expected_labels or onset.shape != expected_labels:
        raise ValueError(f"cached label shape changed: {track.path}")
    return activity, onset


def _load_track_arrays(track: CachedTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _load_track_features(track)
    activity, onset = _load_track_labels(track)
    return features, activity, onset


def _continuous_memmap(
    tracks: list[CachedTrack], cache_root: Path, signature: str
) -> np.ndarray:
    rows, feature_count, _ = _validate_track_dimensions(tracks)
    path = cache_root / f"train-continuous-{signature[:20]}.npy"
    if path.is_file():
        try:
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.shape == (rows, feature_count) and existing.dtype == np.float32:
                print(f"[binarizer] reuse continuous memmap {path}")
                return existing
        except (OSError, ValueError):
            pass
        path.unlink(missing_ok=True)

    print(
        f"[binarizer] build continuous memmap rows={rows} "
        f"features={feature_count}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    mapped = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(rows, feature_count)
    )
    offset = 0
    for index, track in enumerate(tracks):
        features = _load_track_features(track)
        mapped[offset : offset + track.rows] = features
        offset += track.rows
        print(f"[binarizer {index + 1}/{len(tracks)}] cached rows={offset}/{rows}")
    mapped.flush()
    del mapped
    temporary.replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def save_quantile_thermometer(
    path: str | Path,
    binarizer: QuantileThermometer,
    *,
    signature: str,
    train_rows: int,
    continuous_feature_count: int,
    frontend: FrontendConfig | None = None,
    context: ContextConfig | None = None,
) -> str:
    if binarizer.thresholds is None or binarizer.keep_columns is None:
        raise RuntimeError("cannot save an unfitted quantile thermometer")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez(
        temporary,
        quantiles=np.asarray(binarizer.quantiles, dtype=np.float64),
        thresholds=np.asarray(binarizer.thresholds, dtype=np.float32),
        keep_columns=np.asarray(binarizer.keep_columns, dtype=np.bool_),
    )
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata = {
        "schema": EXPORT_SCHEMA,
        "signature": signature,
        "sha256": digest,
        "train_rows": train_rows,
        "continuous_feature_count": continuous_feature_count,
        "quantiles": list(binarizer.quantiles),
        "raw_thermometer_literals": int(binarizer.keep_columns.size),
        "kept_binary_features": int(binarizer.keep_columns.sum()),
        "file_bytes": destination.stat().st_size,
    }
    if (frontend is None) != (context is None):
        raise ValueError("frontend and context must be provided together")
    if frontend is not None and context is not None:
        semantics = binary_feature_semantics(
            frontend,
            context,
            binarizer_sha256=digest,
            binarizer_signature=signature,
            continuous_feature_count=continuous_feature_count,
            binary_feature_count=int(binarizer.keep_columns.sum()),
        )
        metadata["frontend_schema"] = frontend_schema_descriptor(frontend)
        metadata["frontend"] = asdict(frontend)
        metadata["context"] = asdict(context)
        metadata["feature_semantics"] = semantics
    _write_json_atomic(destination.with_suffix(destination.suffix + ".json"), metadata)
    return digest


def load_quantile_thermometer(path: str | Path) -> QuantileThermometer:
    source = Path(path)
    with np.load(source, allow_pickle=False) as stored:
        quantiles = tuple(float(value) for value in stored["quantiles"])
        thresholds = np.ascontiguousarray(stored["thresholds"], dtype=np.float32)
        keep_columns = np.ascontiguousarray(stored["keep_columns"], dtype=np.bool_)
    if thresholds.ndim != 2 or thresholds.shape[1] != len(quantiles):
        raise ValueError("saved quantile threshold dimensions are invalid")
    if keep_columns.shape != (thresholds.shape[0] * len(quantiles),):
        raise ValueError("saved quantile keep-column dimensions are invalid")
    return QuantileThermometer(
        quantiles=quantiles,
        thresholds=thresholds,
        keep_columns=keep_columns,
    )


def fit_or_load_global_binarizer(
    tracks: list[CachedTrack],
    cache_root: str | Path,
    output_path: str | Path,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95),
    constant_scan_batch_rows: int = 512,
    frontend: FrontendConfig | None = None,
    context: ContextConfig | None = None,
    force: bool = False,
) -> tuple[QuantileThermometer, str, str]:
    if (frontend is None) != (context is None):
        raise ValueError("frontend and context must be provided together")
    rows, feature_count, _ = _validate_track_dimensions(tracks)
    signature = _canonical_hash(
        {
            "schema": EXPORT_SCHEMA,
            "tracks": [track.signature for track in tracks],
            "quantiles": quantiles,
            "fit": "numpy.quantile-all-sampled-train-rows",
            "frontend_schema": (
                frontend_schema_descriptor(frontend)
                if frontend is not None
                else None
            ),
        }
    )
    destination = Path(output_path)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    if not force and destination.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("signature") == signature
                and destination.stat().st_size == metadata.get("file_bytes")
                and hashlib.sha256(destination.read_bytes()).hexdigest()
                == metadata.get("sha256")
            ):
                binarizer = load_quantile_thermometer(destination)
                if frontend is not None and context is not None:
                    expected_semantics = binary_feature_semantics(
                        frontend,
                        context,
                        binarizer_sha256=str(metadata["sha256"]),
                        binarizer_signature=signature,
                        continuous_feature_count=feature_count,
                        binary_feature_count=int(binarizer.keep_columns.sum()),
                    )
                    if metadata.get("feature_semantics") != expected_semantics:
                        raise ValueError("cached binarizer feature semantics changed")
                print(f"[binarizer] reuse {destination}")
                return binarizer, signature, str(metadata["sha256"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    continuous = _continuous_memmap(tracks, Path(cache_root), signature)
    binarizer = QuantileThermometer(quantiles).fit(
        continuous, constant_scan_batch_rows=constant_scan_batch_rows
    )
    del continuous
    digest = save_quantile_thermometer(
        destination,
        binarizer,
        signature=signature,
        train_rows=rows,
        continuous_feature_count=feature_count,
        frontend=frontend,
        context=context,
    )
    print(
        f"[binarizer] saved {destination} "
        f"kept={int(binarizer.keep_columns.sum())}/{binarizer.keep_columns.size}"
    )
    return binarizer, signature, digest


def _all_targets(tracks: list[CachedTrack]) -> np.ndarray:
    rows, _, note_count = _validate_track_dimensions(tracks)
    targets = np.empty((rows, 2 * note_count), dtype=np.uint8)
    offset = 0
    for track in tracks:
        activity, onset = _load_track_labels(track)
        targets[offset : offset + track.rows, :note_count] = activity
        targets[offset : offset + track.rows, note_count:] = onset
        offset += track.rows
    return targets


def iter_encoded_track_batches(
    tracks: list[CachedTrack],
    binarizer: QuantileThermometer,
    batch_rows: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if batch_rows <= 0:
        raise ValueError("encoding batch size must be positive")
    for index, track in enumerate(tracks):
        features, activity, onset = _load_track_arrays(track)
        for first in range(0, track.rows, batch_rows):
            last = min(first + batch_rows, track.rows)
            yield (
                binarizer.transform(features[first:last]),
                activity[first:last],
                onset[first:last],
            )
        print(f"[pack {index + 1}/{len(tracks)}] {track.entry.source}/{track.entry.identifier}")


def _header_metadata(header: NativeDatasetHeader) -> dict[str, Any]:
    return {
        **asdict(header),
        "payload_sha256": header.payload_sha256.hex(),
        "feature_fingerprint_sha256": (
            header.feature_fingerprint_sha256.hex()
        ),
    }


def export_cached_split(
    output_path: str | Path,
    tracks: list[CachedTrack],
    split: str,
    binarizer: QuantileThermometer,
    binarizer_signature: str,
    binarizer_sha256: str,
    frontend: FrontendConfig,
    context: ContextConfig,
    targets: TargetConfig,
    *,
    seed: int,
    batch_rows: int = 256,
    onset_row_multiplier: float = 2.0,
    force: bool = False,
) -> NativeDatasetHeader:
    rows, continuous_feature_count, note_count = _validate_track_dimensions(tracks)
    if binarizer.keep_columns is None:
        raise RuntimeError("cannot export with an unfitted binarizer")
    if split not in {"train", "validation"}:
        raise ValueError("native corpus export supports train and validation splits")
    if onset_row_multiplier <= 0:
        raise ValueError("onset row multiplier must be positive")
    sampling_policies = {track.sampling_policy for track in tracks}
    if len(sampling_policies) != 1:
        raise ValueError("cached tracks use different frame sampling policies")
    frame_sampling_policy = next(iter(sampling_policies))
    if frame_sampling_policy not in {"balanced", "natural"}:
        raise ValueError("cached track frame sampling policy is invalid")
    if split == "validation" and frame_sampling_policy != "natural":
        raise ValueError("validation tracks must use natural frame sampling")
    feature_count = int(binarizer.keep_columns.sum())
    feature_semantics = binary_feature_semantics(
        frontend,
        context,
        binarizer_sha256=binarizer_sha256,
        binarizer_signature=binarizer_signature,
        continuous_feature_count=continuous_feature_count,
        binary_feature_count=feature_count,
    )
    export_signature = _canonical_hash(
        {
            "schema": EXPORT_SCHEMA,
            "split": split,
            "tracks": [track.signature for track in tracks],
            "frame_sampling_policy": frame_sampling_policy,
            "binarizer": binarizer_signature,
            "seed": seed,
            "onset_policy": (
                {"balanced_multiplier": onset_row_multiplier}
                if split == "train" and frame_sampling_policy == "balanced"
                else {"natural_order": True}
            ),
            "frontend": _frontend_signature_value(frontend),
            "frontend_schema": frontend_schema_descriptor(frontend),
            "context": asdict(context),
            "feature_semantics": feature_semantics,
            "targets": _target_signature_value(targets),
        }
    )
    destination = Path(output_path)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    if not force and destination.is_file() and sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            header = read_native_dataset_header(destination)
            if (
                metadata.get("export_signature") == export_signature
                and metadata.get("file_bytes") == destination.stat().st_size
                and metadata.get("header", {}).get("payload_sha256")
                == header.payload_sha256.hex()
            ):
                print(f"[{split}] reuse {destination}")
                return header
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    if split == "train" and frame_sampling_policy == "balanced":
        all_targets = _all_targets(tracks)
        onset_rows = max(1, int(round(rows * onset_row_multiplier)))
        indices = onset_training_indices(all_targets, note_count, onset_rows, seed)
        del all_targets
        onset_policy: dict[str, Any] = {
            "name": "55pct_onset_35pct_active_10pct_silence",
            "rows": int(indices.size),
            "multiplier": onset_row_multiplier,
        }
    else:
        indices = np.arange(rows, dtype=np.uint32)
        onset_policy = {"name": "natural_order", "rows": int(indices.size)}

    header = write_native_dataset_batches(
        destination,
        iter_encoded_track_batches(tracks, binarizer, batch_rows),
        indices,
        feature_count=feature_count,
        note_count=note_count,
        midi_min=frontend.midi_min,
        sample_rate=frontend.sample_rate,
        hop_size=frontend.hop_size,
        seed=seed,
        feature_fingerprint_sha256=feature_fingerprint_bytes(feature_semantics),
    )

    source_track_counts = Counter(track.entry.source for track in tracks)
    source_row_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for track in tracks:
        source_row_counts[track.entry.source] += track.rows
        category_counts.update(track.category_counts)
    metadata = {
        "format": "TMGMDAT",
        "version": 2,
        "export_schema": EXPORT_SCHEMA,
        "export_signature": export_signature,
        "split": split,
        "sampling": {
            "frame_sampling_policy": frame_sampling_policy,
            "train_balanced_categories": (
                split == "train" and frame_sampling_policy == "balanced"
            ),
            "train_natural_uniform": (
                split == "train" and frame_sampling_policy == "natural"
            ),
            "validation_natural_uniform": split == "validation",
            "onset_training_indices": onset_policy,
        },
        "source_track_counts": dict(sorted(source_track_counts.items())),
        "source_row_counts": dict(sorted(source_row_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "track_count": len(tracks),
        "rows": rows,
        "continuous_feature_count": continuous_feature_count,
        "kept_binary_features": feature_count,
        "binarizer": {
            "signature": binarizer_signature,
            "sha256": binarizer_sha256,
            "quantiles": list(binarizer.quantiles),
            "thresholds_shape": list(binarizer.thresholds.shape)
            if binarizer.thresholds is not None
            else None,
            "raw_thermometer_literals": int(binarizer.keep_columns.size),
            "kept_binary_features": int(binarizer.keep_columns.sum()),
        },
        "frontend_schema": frontend_schema_descriptor(frontend),
        "feature_semantics": feature_semantics,
        "frontend": asdict(frontend),
        "context": asdict(context),
        "targets": asdict(targets),
        "selected_tracks": [
            {
                "source": track.entry.source,
                "id": track.entry.identifier,
                "group": track.entry.group,
                "rows": track.rows,
                "cache_signature": track.signature,
            }
            for track in tracks
        ],
        "header": _header_metadata(header),
        "file_bytes": destination.stat().st_size,
    }
    _write_json_atomic(sidecar, metadata)
    print(f"[{split}] saved {destination} ({destination.stat().st_size} bytes)")
    return header
