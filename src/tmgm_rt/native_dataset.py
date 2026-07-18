from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Iterable

import numpy as np


MAGIC = b"TMGMDAT\0"
LEGACY_VERSION = 1
VERSION = 2
HEADER_BYTES = 256
WORD_BITS = 64
FEATURE_FINGERPRINT_OFFSET = 176
FEATURE_FINGERPRINT_BYTES = 32

# All integers in the file are little-endian. Binary matrix column c is stored
# in bit (c % 64) of uint64 word (c // 64), so native code can mmap the payload
# and feed it directly to a word-oriented CPU/CUDA clause evaluator.
_HEADER_PREFIX = struct.Struct(
    "<8sIIIIQIIIIiiIIQ"  # identity, dimensions and audio grid
    "QQQQQQQQ"  # four payload offset/size pairs
    "Q32s"  # sampling seed and SHA-256 of the complete payload
)


@dataclass(frozen=True)
class NativeDatasetHeader:
    frame_count: int
    feature_count: int
    feature_words_per_row: int
    note_count: int
    label_words_per_row: int
    midi_min: int
    midi_max: int
    sample_rate: int
    hop_size: int
    onset_index_count: int
    features_offset: int
    features_bytes: int
    activity_offset: int
    activity_bytes: int
    onset_offset: int
    onset_bytes: int
    onset_indices_offset: int
    onset_indices_bytes: int
    seed: int
    payload_sha256: bytes
    feature_fingerprint_sha256: bytes = bytes(FEATURE_FINGERPRINT_BYTES)


@dataclass(frozen=True)
class NativeDataset:
    header: NativeDatasetHeader
    feature_words: np.ndarray
    activity_words: np.ndarray
    onset_words: np.ndarray
    onset_indices: np.ndarray


def _word_count(column_count: int) -> int:
    if column_count <= 0:
        raise ValueError("binary matrices need at least one column")
    return (column_count + WORD_BITS - 1) // WORD_BITS


def pack_binary_rows(values: np.ndarray) -> np.ndarray:
    """Pack a [rows, columns] binary matrix into little-endian uint64 words."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("binary matrix must have shape [rows, columns]")
    if array.shape[1] == 0:
        raise ValueError("binary matrix needs at least one column")
    if not np.logical_or(array == 0, array == 1).all():
        raise ValueError("binary matrix contains a value other than zero or one")

    word_count = _word_count(array.shape[1])
    byte_count = word_count * (WORD_BITS // 8)
    packed_bytes = np.packbits(array.astype(np.uint8, copy=False), axis=1, bitorder="little")
    if packed_bytes.shape[1] != byte_count:
        padded = np.zeros((array.shape[0], byte_count), dtype=np.uint8)
        padded[:, : packed_bytes.shape[1]] = packed_bytes
        packed_bytes = padded
    return np.ascontiguousarray(packed_bytes).view("<u8").reshape(
        array.shape[0], word_count
    )


def unpack_binary_rows(words: np.ndarray, column_count: int) -> np.ndarray:
    """Inverse of pack_binary_rows, primarily for validation and tests."""
    array = np.ascontiguousarray(words, dtype="<u8")
    if array.ndim != 2 or array.shape[1] != _word_count(column_count):
        raise ValueError("packed matrix dimensions disagree with column count")
    byte_view = array.view(np.uint8).reshape(array.shape[0], -1)
    return np.unpackbits(byte_view, axis=1, bitorder="little")[:, :column_count]


def onset_training_indices(
    targets: np.ndarray, note_count: int, rows: int, seed: int
) -> np.ndarray:
    """Reproduce the proven one-track onset sampling policy exactly."""
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] < 2 * note_count:
        raise ValueError("targets do not contain activity and onset heads")
    if rows <= 0:
        raise ValueError("onset training row count must be positive")

    activity = targets[:, :note_count].sum(axis=1)
    onset = targets[:, note_count : 2 * note_count].sum(axis=1)
    groups = (
        (np.flatnonzero(onset > 0), 0.55),
        (np.flatnonzero((onset == 0) & (activity > 0)), 0.35),
        (np.flatnonzero(activity == 0), 0.10),
    )
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    remaining = rows
    for index, (candidates, fraction) in enumerate(groups):
        if candidates.size == 0:
            continue
        count = remaining if index == len(groups) - 1 else int(round(rows * fraction))
        count = min(count, remaining)
        selected.append(rng.choice(candidates, size=count, replace=True))
        remaining -= count
    if remaining:
        selected.append(rng.choice(targets.shape[0], size=remaining, replace=True))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return np.ascontiguousarray(result, dtype="<u4")


def _pack_header(header: NativeDatasetHeader) -> bytes:
    fingerprint = _normalize_feature_fingerprint(
        header.feature_fingerprint_sha256
    )
    version = VERSION if any(fingerprint) else LEGACY_VERSION
    prefix = _HEADER_PREFIX.pack(
        MAGIC,
        version,
        HEADER_BYTES,
        WORD_BITS,
        0,
        header.frame_count,
        header.feature_count,
        header.feature_words_per_row,
        header.note_count,
        header.label_words_per_row,
        header.midi_min,
        header.midi_max,
        header.sample_rate,
        header.hop_size,
        header.onset_index_count,
        header.features_offset,
        header.features_bytes,
        header.activity_offset,
        header.activity_bytes,
        header.onset_offset,
        header.onset_bytes,
        header.onset_indices_offset,
        header.onset_indices_bytes,
        header.seed,
        header.payload_sha256,
    )
    if len(prefix) > HEADER_BYTES:
        raise AssertionError("native dataset header exceeded its fixed size")
    packed = bytearray(prefix.ljust(HEADER_BYTES, b"\0"))
    if version == VERSION:
        packed[
            FEATURE_FINGERPRINT_OFFSET :
            FEATURE_FINGERPRINT_OFFSET + FEATURE_FINGERPRINT_BYTES
        ] = fingerprint
    return bytes(packed)


def _normalize_feature_fingerprint(value: bytes | str | None) -> bytes:
    if value is None:
        return bytes(FEATURE_FINGERPRINT_BYTES)
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("feature fingerprint is not hexadecimal") from error
    result = bytes(value)
    if len(result) != FEATURE_FINGERPRINT_BYTES:
        raise ValueError("feature fingerprint must contain exactly 32 bytes")
    return result


def _unpack_header(raw: bytes) -> NativeDatasetHeader:
    if len(raw) != HEADER_BYTES:
        raise ValueError("truncated native dataset header")
    values = _HEADER_PREFIX.unpack_from(raw)
    if values[0] != MAGIC:
        raise ValueError("bad native dataset magic")
    version = values[1]
    if version not in {LEGACY_VERSION, VERSION}:
        raise ValueError(f"unsupported native dataset version: {values[1]}")
    if values[2] != HEADER_BYTES or values[3] != WORD_BITS:
        raise ValueError("unsupported native dataset header or word size")
    if values[4] != 0:
        raise ValueError("unsupported native dataset flags")
    fingerprint = bytes(
        raw[
            FEATURE_FINGERPRINT_OFFSET :
            FEATURE_FINGERPRINT_OFFSET + FEATURE_FINGERPRINT_BYTES
        ]
    )
    if version == LEGACY_VERSION:
        if any(raw[FEATURE_FINGERPRINT_OFFSET:]):
            raise ValueError("legacy native dataset reserved header bytes are non-zero")
        fingerprint = bytes(FEATURE_FINGERPRINT_BYTES)
    else:
        if not any(fingerprint):
            raise ValueError("native dataset v2 feature fingerprint is zero")
        if any(raw[FEATURE_FINGERPRINT_OFFSET + FEATURE_FINGERPRINT_BYTES :]):
            raise ValueError("native dataset v2 reserved header bytes are non-zero")
    return NativeDatasetHeader(
        frame_count=values[5],
        feature_count=values[6],
        feature_words_per_row=values[7],
        note_count=values[8],
        label_words_per_row=values[9],
        midi_min=values[10],
        midi_max=values[11],
        sample_rate=values[12],
        hop_size=values[13],
        onset_index_count=values[14],
        features_offset=values[15],
        features_bytes=values[16],
        activity_offset=values[17],
        activity_bytes=values[18],
        onset_offset=values[19],
        onset_bytes=values[20],
        onset_indices_offset=values[21],
        onset_indices_bytes=values[22],
        seed=values[23],
        payload_sha256=values[24],
        feature_fingerprint_sha256=fingerprint,
    )


def write_native_dataset(
    path: str | Path,
    binary_features: np.ndarray,
    activity: np.ndarray,
    onset: np.ndarray,
    onset_indices: np.ndarray,
    *,
    midi_min: int,
    sample_rate: int,
    hop_size: int,
    seed: int,
    feature_fingerprint_sha256: bytes | str | None = None,
) -> NativeDatasetHeader:
    """Atomically write a deterministic, mmap-friendly native training file."""
    binary_features = np.asarray(binary_features)
    activity = np.asarray(activity)
    onset = np.asarray(onset)
    if binary_features.ndim != 2:
        raise ValueError("features must have shape [frames, features]")
    if activity.ndim != 2 or onset.shape != activity.shape:
        raise ValueError("activity/onset matrices must have equal 2D shapes")
    if activity.shape[0] != binary_features.shape[0]:
        raise ValueError("feature and label frame counts disagree")

    indices = np.ascontiguousarray(onset_indices, dtype="<u4").reshape(-1)
    if indices.size and int(indices.max()) >= binary_features.shape[0]:
        raise ValueError("onset training index is outside the feature matrix")
    feature_words = pack_binary_rows(binary_features)
    activity_words = pack_binary_rows(activity)
    onset_words = pack_binary_rows(onset)

    payloads = (feature_words, activity_words, onset_words, indices)
    offsets: list[int] = []
    sizes: list[int] = []
    offset = HEADER_BYTES
    for payload in payloads:
        offsets.append(offset)
        size = int(payload.nbytes)
        sizes.append(size)
        offset += size

    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(memoryview(payload).cast("B"))
    note_count = int(activity.shape[1])
    header = NativeDatasetHeader(
        frame_count=int(binary_features.shape[0]),
        feature_count=int(binary_features.shape[1]),
        feature_words_per_row=int(feature_words.shape[1]),
        note_count=note_count,
        label_words_per_row=int(activity_words.shape[1]),
        midi_min=int(midi_min),
        midi_max=int(midi_min + note_count - 1),
        sample_rate=int(sample_rate),
        hop_size=int(hop_size),
        onset_index_count=int(indices.size),
        features_offset=offsets[0],
        features_bytes=sizes[0],
        activity_offset=offsets[1],
        activity_bytes=sizes[1],
        onset_offset=offsets[2],
        onset_bytes=sizes[2],
        onset_indices_offset=offsets[3],
        onset_indices_bytes=sizes[3],
        seed=int(seed),
        payload_sha256=digest.digest(),
        feature_fingerprint_sha256=_normalize_feature_fingerprint(
            feature_fingerprint_sha256
        ),
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_pack_header(header))
        for payload in payloads:
            stream.write(memoryview(payload).cast("B"))
    temporary.replace(destination)
    return header


def write_native_dataset_batches(
    path: str | Path,
    batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    onset_indices: np.ndarray,
    *,
    feature_count: int,
    note_count: int,
    midi_min: int,
    sample_rate: int,
    hop_size: int,
    seed: int,
    feature_fingerprint_sha256: bytes | str | None = None,
) -> NativeDatasetHeader:
    """Stream binary rows into a TMGMDAT v1 file.

    Each input batch is ``(binary_features, activity, onset)``. Only one batch
    and its packed representation are resident at a time; the complete
    ``rows x feature_count`` uint32 matrix is never constructed. The temporary
    section files are joined atomically because TMGMDAT stores its three
    matrices in contiguous sections rather than interleaved by batch.
    """
    if feature_count <= 0 or note_count <= 0:
        raise ValueError("native dataset dimensions must be positive")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    indices = np.ascontiguousarray(onset_indices, dtype="<u4").reshape(-1)
    feature_words_per_row = _word_count(feature_count)
    label_words_per_row = _word_count(note_count)
    frame_count = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.parts-", dir=destination.parent
        ) as parts_directory:
            parts_root = Path(parts_directory)
            section_paths = tuple(
                parts_root / name
                for name in ("features.bin", "activity.bin", "onset.bin")
            )
            streams = tuple(section.open("wb") for section in section_paths)
            try:
                for batch_index, (binary_features, activity, onset) in enumerate(
                    batches
                ):
                    binary_features = np.asarray(binary_features)
                    activity = np.asarray(activity)
                    onset = np.asarray(onset)
                    if binary_features.ndim != 2:
                        raise ValueError(
                            f"feature batch {batch_index} must have shape [rows, features]"
                        )
                    if binary_features.shape[1] != feature_count:
                        raise ValueError(
                            f"feature batch {batch_index} width disagrees with feature_count"
                        )
                    if activity.ndim != 2 or activity.shape[1] != note_count:
                        raise ValueError(
                            f"activity batch {batch_index} width disagrees with note_count"
                        )
                    if onset.shape != activity.shape:
                        raise ValueError(
                            f"onset batch {batch_index} shape disagrees with activity"
                        )
                    if activity.shape[0] != binary_features.shape[0]:
                        raise ValueError(
                            f"feature/label row counts disagree in batch {batch_index}"
                        )
                    if binary_features.shape[0] == 0:
                        continue

                    packed = (
                        pack_binary_rows(binary_features),
                        pack_binary_rows(activity),
                        pack_binary_rows(onset),
                    )
                    for stream, payload in zip(streams, packed, strict=True):
                        stream.write(memoryview(payload).cast("B"))
                    frame_count += int(binary_features.shape[0])
            finally:
                for stream in streams:
                    stream.close()

            if frame_count == 0:
                raise ValueError("cannot write an empty native dataset")
            if indices.size and int(indices.max()) >= frame_count:
                raise ValueError("onset training index is outside the feature matrix")

            feature_bytes = frame_count * feature_words_per_row * np.dtype("<u8").itemsize
            label_bytes = frame_count * label_words_per_row * np.dtype("<u8").itemsize
            expected_sizes = (feature_bytes, label_bytes, label_bytes)
            actual_sizes = tuple(section.stat().st_size for section in section_paths)
            if actual_sizes != expected_sizes:
                raise AssertionError("streamed native section has an unexpected size")

            index_path = parts_root / "onset-indices.bin"
            with index_path.open("wb") as stream:
                stream.write(memoryview(indices).cast("B"))
            payload_paths = (*section_paths, index_path)
            payload_sizes = (*actual_sizes, int(indices.nbytes))

            offsets: list[int] = []
            offset = HEADER_BYTES
            for size in payload_sizes:
                offsets.append(offset)
                offset += size

            digest = hashlib.sha256()
            for payload_path in payload_paths:
                with payload_path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)

            header = NativeDatasetHeader(
                frame_count=frame_count,
                feature_count=int(feature_count),
                feature_words_per_row=feature_words_per_row,
                note_count=int(note_count),
                label_words_per_row=label_words_per_row,
                midi_min=int(midi_min),
                midi_max=int(midi_min + note_count - 1),
                sample_rate=int(sample_rate),
                hop_size=int(hop_size),
                onset_index_count=int(indices.size),
                features_offset=offsets[0],
                features_bytes=payload_sizes[0],
                activity_offset=offsets[1],
                activity_bytes=payload_sizes[1],
                onset_offset=offsets[2],
                onset_bytes=payload_sizes[2],
                onset_indices_offset=offsets[3],
                onset_indices_bytes=payload_sizes[3],
                seed=int(seed),
                payload_sha256=digest.digest(),
                feature_fingerprint_sha256=_normalize_feature_fingerprint(
                    feature_fingerprint_sha256
                ),
            )

            with temporary.open("wb") as destination_stream:
                destination_stream.write(_pack_header(header))
                for payload_path in payload_paths:
                    with payload_path.open("rb") as source_stream:
                        shutil.copyfileobj(
                            source_stream, destination_stream, length=8 * 1024 * 1024
                        )
            temporary.replace(destination)
            return header
    finally:
        temporary.unlink(missing_ok=True)


def read_native_dataset(
    path: str | Path, *, verify_checksum: bool = True
) -> NativeDataset:
    """Read and validate an exported native training file."""
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        header = _unpack_header(stream.read(HEADER_BYTES))
        sections = (
            (header.features_offset, header.features_bytes),
            (header.activity_offset, header.activity_bytes),
            (header.onset_offset, header.onset_bytes),
            (header.onset_indices_offset, header.onset_indices_bytes),
        )
        expected_offset = HEADER_BYTES
        for offset, size in sections:
            if offset != expected_offset or size < 0 or offset + size > file_size:
                raise ValueError("invalid native dataset payload layout")
            expected_offset += size
        if expected_offset != file_size:
            raise ValueError("native dataset has trailing or missing payload bytes")
        payload = stream.read()

    if verify_checksum and hashlib.sha256(payload).digest() != header.payload_sha256:
        raise ValueError("native dataset payload checksum mismatch")
    expected_feature_bytes = (
        header.frame_count * header.feature_words_per_row * np.dtype("<u8").itemsize
    )
    expected_label_bytes = (
        header.frame_count * header.label_words_per_row * np.dtype("<u8").itemsize
    )
    if header.features_bytes != expected_feature_bytes:
        raise ValueError("feature payload size disagrees with header dimensions")
    if header.activity_bytes != expected_label_bytes or header.onset_bytes != expected_label_bytes:
        raise ValueError("label payload size disagrees with header dimensions")
    if header.onset_indices_bytes != header.onset_index_count * np.dtype("<u4").itemsize:
        raise ValueError("onset index payload size disagrees with header dimensions")

    base = HEADER_BYTES
    feature_words = np.frombuffer(
        payload,
        dtype="<u8",
        count=header.frame_count * header.feature_words_per_row,
        offset=header.features_offset - base,
    ).reshape(header.frame_count, header.feature_words_per_row)
    activity_words = np.frombuffer(
        payload,
        dtype="<u8",
        count=header.frame_count * header.label_words_per_row,
        offset=header.activity_offset - base,
    ).reshape(header.frame_count, header.label_words_per_row)
    onset_words = np.frombuffer(
        payload,
        dtype="<u8",
        count=header.frame_count * header.label_words_per_row,
        offset=header.onset_offset - base,
    ).reshape(header.frame_count, header.label_words_per_row)
    onset_indices = np.frombuffer(
        payload,
        dtype="<u4",
        count=header.onset_index_count,
        offset=header.onset_indices_offset - base,
    )
    if onset_indices.size and int(onset_indices.max()) >= header.frame_count:
        raise ValueError("onset training index is outside the feature matrix")
    return NativeDataset(
        header=header,
        feature_words=feature_words,
        activity_words=activity_words,
        onset_words=onset_words,
        onset_indices=onset_indices,
    )


def read_native_dataset_header(path: str | Path) -> NativeDatasetHeader:
    """Validate the v1 layout and return its header without loading payloads."""
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        header = _unpack_header(stream.read(HEADER_BYTES))
    sections = (
        (header.features_offset, header.features_bytes),
        (header.activity_offset, header.activity_bytes),
        (header.onset_offset, header.onset_bytes),
        (header.onset_indices_offset, header.onset_indices_bytes),
    )
    expected_offset = HEADER_BYTES
    for offset, size in sections:
        if offset != expected_offset or size < 0 or offset + size > file_size:
            raise ValueError("invalid native dataset payload layout")
        expected_offset += size
    if expected_offset != file_size:
        raise ValueError("native dataset has trailing or missing payload bytes")
    expected_feature_bytes = (
        header.frame_count * header.feature_words_per_row * np.dtype("<u8").itemsize
    )
    expected_label_bytes = (
        header.frame_count * header.label_words_per_row * np.dtype("<u8").itemsize
    )
    if header.features_bytes != expected_feature_bytes:
        raise ValueError("feature payload size disagrees with header dimensions")
    if (
        header.activity_bytes != expected_label_bytes
        or header.onset_bytes != expected_label_bytes
    ):
        raise ValueError("label payload size disagrees with header dimensions")
    if header.onset_indices_bytes != header.onset_index_count * np.dtype("<u4").itemsize:
        raise ValueError("onset index payload size disagrees with header dimensions")
    return header
