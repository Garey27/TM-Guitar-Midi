from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from .native_dataset import NativeDatasetHeader, read_native_dataset_header
from .native_score_eval import SCORE_MAGIC, ScoreMetadata


ARTIFACT_FORMAT = "TMGM_NATIVE_SCORE_ENSEMBLE_V1"
DEFAULT_QUANTIZATION = 1024
_MEMBER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_I32 = np.iinfo(np.int32)


@dataclass(frozen=True)
class MemberSpec:
    identifier: str
    path: Path


@dataclass(frozen=True)
class LoadedScores:
    metadata: ScoreMetadata
    scores: np.ndarray


def parse_member_spec(value: str) -> MemberSpec:
    """Parse an explicit, stable ``ID=path`` member assignment."""
    if "=" not in value:
        raise ValueError(f"member must use ID=path syntax: {value!r}")
    identifier, path_text = value.split("=", 1)
    if not _MEMBER_ID.fullmatch(identifier):
        raise ValueError(
            "member ID must be 1-64 ASCII letters, digits, '.', '_' or '-'"
        )
    if not path_text:
        raise ValueError(f"member {identifier!r} has an empty path")
    return MemberSpec(identifier=identifier, path=Path(path_text))


def _parse_integer(value: str, name: str, path: Path) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise ValueError(f"invalid integer {name}={value!r}: {path}") from error
    if str(parsed) != value and str(parsed) != value.lstrip("+"):
        raise ValueError(f"non-canonical integer {name}={value!r}: {path}")
    return parsed


def _metadata(values: dict[str, str], path: Path) -> ScoreMetadata:
    required = (
        "head",
        "frames",
        "outputs",
        "midi_min",
        "sample_rate",
        "hop_size",
        "threshold",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"score metadata is missing {missing}: {path}")
    head = values["head"]
    if head not in {"activity", "onset"}:
        raise ValueError(f"unsupported score head {head!r}: {path}")
    result = ScoreMetadata(
        head=head,
        frames=_parse_integer(values["frames"], "frames", path),
        outputs=_parse_integer(values["outputs"], "outputs", path),
        midi_min=_parse_integer(values["midi_min"], "midi_min", path),
        sample_rate=_parse_integer(values["sample_rate"], "sample_rate", path),
        hop_size=_parse_integer(values["hop_size"], "hop_size", path),
        threshold=_parse_integer(values["threshold"], "threshold", path),
        raw=dict(values),
    )
    if result.frames <= 0 or result.outputs <= 0:
        raise ValueError(f"score dimensions must be positive: {path}")
    if result.sample_rate <= 0 or result.hop_size <= 0:
        raise ValueError(f"score timebase must be positive: {path}")
    if result.threshold < _I32.min or result.threshold > _I32.max:
        raise ValueError(f"score threshold is outside int32: {path}")
    return result


def load_score_file(path_value: str | Path) -> LoadedScores:
    """Load raw integer scores while strictly validating the V1 TSV contract."""
    path = Path(path_value)
    metadata_values: dict[str, str] = {}
    metadata: ScoreMetadata | None = None
    scores: np.ndarray | None = None
    magic_seen = False
    header_seen = False
    expected_frame = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"blank line at {line_number}: {path}")
            if line.startswith("#"):
                if header_seen:
                    raise ValueError(
                        f"metadata after score header at line {line_number}: {path}"
                    )
                if line == SCORE_MAGIC:
                    if magic_seen:
                        raise ValueError(f"duplicate {SCORE_MAGIC} marker: {path}")
                    magic_seen = True
                    continue
                text = line[1:]
                if "=" not in text:
                    raise ValueError(f"malformed metadata at line {line_number}: {path}")
                key, value = text.split("=", 1)
                if not key or key in metadata_values:
                    raise ValueError(
                        f"duplicate or empty metadata key at line {line_number}: {path}"
                    )
                metadata_values[key] = value
                continue

            if not header_seen:
                if not magic_seen:
                    raise ValueError(f"missing {SCORE_MAGIC} marker: {path}")
                metadata = _metadata(metadata_values, path)
                expected_scores = [
                    f"score_{note}"
                    for note in range(
                        metadata.midi_min, metadata.midi_min + metadata.outputs
                    )
                ]
                expected_predictions = [
                    f"pred_{note}"
                    for note in range(
                        metadata.midi_min, metadata.midi_min + metadata.outputs
                    )
                ]
                if line.split("\t") != [
                    "frame",
                    *expected_scores,
                    *expected_predictions,
                ]:
                    raise ValueError(f"unexpected score TSV columns: {path}")
                scores = np.empty(
                    (metadata.frames, metadata.outputs), dtype=np.int32
                )
                header_seen = True
                continue

            assert metadata is not None and scores is not None
            if expected_frame >= metadata.frames:
                raise ValueError(f"score file has more than {metadata.frames} rows: {path}")
            values = np.fromstring(line, sep="\t", dtype=np.int64)
            expected_columns = 1 + 2 * metadata.outputs
            if values.size != expected_columns:
                raise ValueError(
                    f"score row {line_number} has {values.size} columns, "
                    f"expected {expected_columns}: {path}"
                )
            if int(values[0]) != expected_frame:
                raise ValueError(
                    f"score frame index {int(values[0])} is not expected "
                    f"{expected_frame}: {path}"
                )
            score_row = values[1 : 1 + metadata.outputs]
            if np.any(score_row < _I32.min) or np.any(score_row > _I32.max):
                raise ValueError(f"raw score outside int32 at frame {expected_frame}: {path}")
            predictions = values[1 + metadata.outputs :]
            if not np.logical_or(predictions == 0, predictions == 1).all():
                raise ValueError(
                    f"non-binary prediction at frame {expected_frame}: {path}"
                )
            scores[expected_frame] = score_row
            expected_frame += 1

    if not header_seen or metadata is None or scores is None:
        raise ValueError(f"empty native score file: {path}")
    if expected_frame != metadata.frames:
        raise ValueError(
            f"score row count {expected_frame} disagrees with "
            f"frames={metadata.frames}: {path}"
        )
    return LoadedScores(metadata=metadata, scores=scores)


def _validate_metadata(
    metadata: ScoreMetadata,
    header: NativeDatasetHeader,
    path: Path,
    *,
    expected_head: str | None = None,
) -> None:
    actual = (
        metadata.frames,
        metadata.outputs,
        metadata.midi_min,
        metadata.sample_rate,
        metadata.hop_size,
    )
    expected = (
        header.frame_count,
        header.note_count,
        header.midi_min,
        header.sample_rate,
        header.hop_size,
    )
    if actual != expected:
        raise ValueError(
            f"score metadata does not match dataset; got {actual}, "
            f"expected {expected}: {path}"
        )
    if expected_head is not None and metadata.head != expected_head:
        raise ValueError(
            f"score head {metadata.head!r} does not match expected "
            f"{expected_head!r}: {path}"
        )


def _validate_members(members: Sequence[MemberSpec]) -> None:
    if len(members) < 2:
        raise ValueError("an ensemble needs at least two members")
    identifiers = [member.identifier for member in members]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ensemble member IDs must be unique")
    for identifier in identifiers:
        if not _MEMBER_ID.fullmatch(identifier):
            raise ValueError(f"invalid member ID: {identifier!r}")


def estimate_robust_scale(centered_scores: np.ndarray) -> float:
    """Estimate dispersion using MAD, with deterministic robust fallbacks."""
    values = np.asarray(centered_scores)
    if values.size == 0:
        raise ValueError("cannot estimate scale from no scores")
    flat = values.reshape(-1)
    maximum_samples = 1_000_000
    stride = max(1, math.ceil(flat.size / maximum_samples))
    sample = flat[::stride][:maximum_samples].astype(np.float64, copy=False)
    if not np.isfinite(sample).all():
        raise ValueError("score matrix contains a non-finite value")
    median = float(np.median(sample))
    scale = 1.4826 * float(np.median(np.abs(sample - median)))
    if not math.isfinite(scale) or scale < 1.0:
        q25, q75 = np.percentile(sample, (25.0, 75.0))
        scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1.0:
        scale = float(np.median(np.abs(sample)))
    if not math.isfinite(scale) or scale < 1.0:
        scale = 1.0
    return scale


def _normalized_member(scores: np.ndarray, threshold: int, scale: float) -> np.ndarray:
    return (scores.astype(np.float32) - np.float32(threshold)) / np.float32(scale)


def _fuse(normalized: np.ndarray, rule: str) -> np.ndarray:
    if normalized.ndim != 3 or normalized.shape[0] < 2:
        raise ValueError("normalized member stack must have shape [members, frames, notes]")
    if rule == "mean":
        return np.mean(normalized, axis=0, dtype=np.float32)
    if rule == "max":
        return np.max(normalized, axis=0)
    if rule == "top2_mean":
        top_two = np.partition(normalized, normalized.shape[0] - 2, axis=0)[-2:]
        return np.mean(top_two, axis=0, dtype=np.float32)
    raise ValueError(f"unsupported fusion rule: {rule!r}")


def _quantize(scores: np.ndarray, quantization: int) -> np.ndarray:
    if (
        not isinstance(quantization, int)
        or isinstance(quantization, bool)
        or quantization <= 0
        or quantization > 1_000_000
    ):
        raise ValueError("quantization must be in [1, 1000000]")
    scaled = np.rint(scores.astype(np.float64) * quantization)
    # Reserve INT32_MAX for the valid predict-nothing threshold max(score)+1.
    np.clip(scaled, _I32.min, _I32.max - 1, out=scaled)
    return scaled.astype(np.int32)


def _labels(dataset_path: Path, header: NativeDatasetHeader, head: str) -> np.ndarray:
    offset = header.activity_offset if head == "activity" else header.onset_offset
    words = np.memmap(
        dataset_path,
        mode="r",
        dtype="<u8",
        offset=offset,
        shape=(header.frame_count, header.label_words_per_row),
    )
    byte_view = np.ascontiguousarray(words).view(np.uint8).reshape(header.frame_count, -1)
    return np.unpackbits(byte_view, axis=1, bitorder="little")[
        :, : header.note_count
    ].astype(bool, copy=False)


def calibrate_common_threshold(
    scores: np.ndarray,
    truth: np.ndarray,
    maximum_polyphony_ratio: float,
) -> dict[str, int | float]:
    """Find the highest-threshold best F1 point allowed by the polyphony cap."""
    score_array = np.asarray(scores)
    target = np.asarray(truth, dtype=bool)
    if score_array.shape != target.shape or score_array.ndim != 2:
        raise ValueError("score/truth matrices must have equal 2D shapes")
    if not math.isfinite(maximum_polyphony_ratio) or maximum_polyphony_ratio < 0.5:
        raise ValueError("maximum_polyphony_ratio must be finite and at least 0.5")
    flat_scores = score_array.reshape(-1)
    flat_truth = target.reshape(-1)
    target_count = int(flat_truth.sum())
    frame_count = score_array.shape[0]
    maximum_score = int(flat_scores.max())
    empty_threshold = maximum_score + 1
    best: dict[str, int | float] = {
        "threshold": empty_threshold,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": target_count,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "predicted_mean_polyphony": 0.0,
        "target_mean_polyphony": target_count / frame_count,
    }
    if target_count == 0:
        return best

    order = np.argsort(flat_scores, kind="stable")[::-1]
    ordered_scores = flat_scores[order]
    ordered_truth = flat_truth[order].astype(np.int64, copy=False)
    cumulative_true = np.cumsum(ordered_truth, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    )
    maximum_predictions = maximum_polyphony_ratio * target_count
    epsilon = 1.0e-15
    for group_end in group_ends:
        predicted = int(group_end) + 1
        if predicted > maximum_predictions + epsilon:
            break
        true_positive = int(cumulative_true[group_end])
        threshold = int(ordered_scores[group_end])
        precision = true_positive / predicted
        recall = true_positive / target_count
        f1 = 2.0 * true_positive / (predicted + target_count)
        if f1 > float(best["f1"]) + epsilon or (
            abs(f1 - float(best["f1"])) <= epsilon
            and threshold > int(best["threshold"])
        ):
            best = {
                "threshold": threshold,
                "true_positives": true_positive,
                "false_positives": predicted - true_positive,
                "false_negatives": target_count - true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_mean_polyphony": predicted / frame_count,
                "target_mean_polyphony": target_count / frame_count,
            }
    return best


def _dataset_metadata(header: NativeDatasetHeader) -> dict[str, Any]:
    return {
        "frames": header.frame_count,
        "feature_count": header.feature_count,
        "outputs": header.note_count,
        "midi_min": header.midi_min,
        "midi_max": header.midi_max,
        "sample_rate": header.sample_rate,
        "hop_size": header.hop_size,
        "payload_sha256": header.payload_sha256.hex(),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fit_score_ensemble(
    dataset_path_value: str | Path,
    members: Sequence[MemberSpec],
    output_path_value: str | Path,
    *,
    maximum_polyphony_ratio: float | None = None,
    quantization: int = DEFAULT_QUANTIZATION,
) -> dict[str, Any]:
    _validate_members(members)
    dataset_path = Path(dataset_path_value)
    output_path = Path(output_path_value)
    header = read_native_dataset_header(dataset_path)
    loaded: list[LoadedScores] = []
    head: str | None = None
    for member in members:
        current = load_score_file(member.path)
        if head is None:
            head = current.metadata.head
        _validate_metadata(current.metadata, header, member.path, expected_head=head)
        metadata_member = current.metadata.raw.get("member_id")
        if metadata_member is not None and metadata_member != member.identifier:
            raise ValueError(
                f"score member_id={metadata_member!r} does not match explicit "
                f"ID {member.identifier!r}: {member.path}"
            )
        loaded.append(current)
    assert head is not None
    ratio = maximum_polyphony_ratio
    if ratio is None:
        ratio = 1.5 if head == "activity" else 4.0
    if not math.isfinite(ratio) or ratio < 0.5:
        raise ValueError("maximum_polyphony_ratio must be finite and at least 0.5")

    normalized_members: list[np.ndarray] = []
    member_artifacts: list[dict[str, Any]] = []
    for spec, current in zip(members, loaded, strict=True):
        centered = current.scores.astype(np.int64) - current.metadata.threshold
        scale = estimate_robust_scale(centered)
        normalized_members.append(
            _normalized_member(current.scores, current.metadata.threshold, scale)
        )
        member_artifacts.append(
            {
                "id": spec.identifier,
                "threshold": current.metadata.threshold,
                "robust_scale": scale,
                "fit_score_file": str(spec.path.resolve()),
            }
        )
    stack = np.stack(normalized_members)
    truth = _labels(dataset_path, header, head)
    rules = ["mean"] if head == "activity" else ["mean", "top2_mean", "max"]
    candidates: dict[str, dict[str, int | float]] = {}
    best_rule = rules[0]
    best_f1 = -1.0
    for rule in rules:
        fused = _quantize(_fuse(stack, rule), quantization)
        calibration = calibrate_common_threshold(fused, truth, ratio)
        candidates[rule] = calibration
        candidate_f1 = float(calibration["f1"])
        # Stable rule order intentionally prefers mean, then top-2, then max.
        if candidate_f1 > best_f1 + 1.0e-15:
            best_rule = rule
            best_f1 = candidate_f1

    chosen = candidates[best_rule]
    member_order = [member.identifier for member in members]
    artifact: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "head": head,
        "fusion": best_rule,
        "quantization": quantization,
        "ensemble_threshold": int(chosen["threshold"]),
        "maximum_polyphony_ratio": ratio,
        "members": member_artifacts,
        "member_order_sha256": hashlib.sha256(
            "\0".join(member_order).encode("utf-8")
        ).hexdigest(),
        "fit_dataset": {
            "path": str(dataset_path.resolve()),
            **_dataset_metadata(header),
        },
        "calibration": {
            "selection_metric": "exact_binary_f1_with_max_polyphony_guard",
            "chosen": chosen,
            "candidates": candidates,
        },
    }
    _atomic_json(output_path, artifact)
    return artifact


def _artifact(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"unsupported score ensemble artifact: {path}")
    if value.get("head") not in {"activity", "onset"}:
        raise ValueError(f"artifact has invalid head: {path}")
    if value.get("fusion") not in {"mean", "top2_mean", "max"}:
        raise ValueError(f"artifact has invalid fusion rule: {path}")
    if value["head"] == "activity" and value["fusion"] != "mean":
        raise ValueError(f"activity artifact must use mean fusion: {path}")
    members = value.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError(f"artifact needs at least two members: {path}")
    identifiers: list[str] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError(f"artifact member {index} is not an object: {path}")
        identifier = member.get("id")
        threshold = member.get("threshold")
        scale = member.get("robust_scale")
        if not isinstance(identifier, str) or not _MEMBER_ID.fullmatch(identifier):
            raise ValueError(f"artifact member {index} has invalid ID: {path}")
        if not isinstance(threshold, int) or isinstance(threshold, bool):
            raise ValueError(f"artifact member {index} has invalid threshold: {path}")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise ValueError(f"artifact member {index} has invalid robust scale: {path}")
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(f"artifact member {index} has invalid robust scale: {path}")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"artifact has duplicate member IDs: {path}")
    expected_digest = hashlib.sha256("\0".join(identifiers).encode("utf-8")).hexdigest()
    if value.get("member_order_sha256") != expected_digest:
        raise ValueError(f"artifact member order checksum mismatch: {path}")
    quantization = value.get("quantization")
    threshold = value.get("ensemble_threshold")
    if (
        not isinstance(quantization, int)
        or isinstance(quantization, bool)
        or quantization <= 0
        or quantization > 1_000_000
    ):
        raise ValueError(f"artifact has invalid quantization: {path}")
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold < _I32.min
        or threshold > _I32.max
    ):
        raise ValueError(f"artifact has invalid ensemble threshold: {path}")
    fit_dataset = value.get("fit_dataset")
    required_geometry = (
        "feature_count",
        "outputs",
        "midi_min",
        "midi_max",
        "sample_rate",
        "hop_size",
    )
    if not isinstance(fit_dataset, dict) or any(
        not isinstance(fit_dataset.get(name), int)
        or isinstance(fit_dataset.get(name), bool)
        for name in required_geometry
    ):
        raise ValueError(f"artifact has invalid fit dataset geometry: {path}")
    return value


def _validate_apply_geometry(
    artifact: dict[str, Any], header: NativeDatasetHeader, artifact_path: Path
) -> None:
    fit = artifact["fit_dataset"]
    actual = (
        header.feature_count,
        header.note_count,
        header.midi_min,
        header.midi_max,
        header.sample_rate,
        header.hop_size,
    )
    expected = tuple(
        fit[name]
        for name in (
            "feature_count",
            "outputs",
            "midi_min",
            "midi_max",
            "sample_rate",
            "hop_size",
        )
    )
    if actual != expected:
        raise ValueError(
            f"dataset geometry/timebase {actual} does not match artifact "
            f"{expected}: {artifact_path}"
        )


def _write_score_file(
    path: Path,
    header: NativeDatasetHeader,
    head: str,
    scores: np.ndarray,
    threshold: int,
    artifact_path: Path,
    fusion: str,
    member_ids: Sequence[str],
) -> None:
    predictions = scores >= threshold
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{SCORE_MAGIC}\n")
        stream.write(f"#head={head}\n")
        stream.write(f"#frames={header.frame_count}\n")
        stream.write(f"#outputs={header.note_count}\n")
        stream.write(f"#midi_min={header.midi_min}\n")
        stream.write(f"#sample_rate={header.sample_rate}\n")
        stream.write(f"#hop_size={header.hop_size}\n")
        stream.write(f"#threshold={threshold}\n")
        stream.write(f"#ensemble_format={ARTIFACT_FORMAT}\n")
        stream.write(f"#fusion={fusion}\n")
        stream.write(f"#member_ids={','.join(member_ids)}\n")
        stream.write(f"#ensemble_artifact={artifact_path.resolve()}\n")
        columns = ["frame"]
        columns.extend(
            f"score_{note}"
            for note in range(header.midi_min, header.midi_max + 1)
        )
        columns.extend(
            f"pred_{note}"
            for note in range(header.midi_min, header.midi_max + 1)
        )
        stream.write("\t".join(columns) + "\n")
        for frame in range(header.frame_count):
            score_values = "\t".join(str(int(value)) for value in scores[frame])
            prediction_values = "\t".join(
                str(int(value)) for value in predictions[frame]
            )
            stream.write(f"{frame}\t{score_values}\t{prediction_values}\n")
    temporary.replace(path)


def apply_score_ensemble(
    artifact_path_value: str | Path,
    dataset_path_value: str | Path,
    members: Sequence[MemberSpec],
    output_path_value: str | Path,
) -> dict[str, Any]:
    _validate_members(members)
    artifact_path = Path(artifact_path_value)
    dataset_path = Path(dataset_path_value)
    output_path = Path(output_path_value)
    artifact = _artifact(artifact_path)
    header = read_native_dataset_header(dataset_path)
    _validate_apply_geometry(artifact, header, artifact_path)

    expected_members = artifact["members"]
    expected_ids = [member["id"] for member in expected_members]
    actual_ids = [member.identifier for member in members]
    if actual_ids != expected_ids:
        raise ValueError(
            f"member order/identity {actual_ids} does not match artifact {expected_ids}"
        )

    normalized_members: list[np.ndarray] = []
    head = str(artifact["head"])
    for spec, member_artifact in zip(members, expected_members, strict=True):
        current = load_score_file(spec.path)
        _validate_metadata(current.metadata, header, spec.path, expected_head=head)
        expected_threshold = int(member_artifact["threshold"])
        if current.metadata.threshold != expected_threshold:
            raise ValueError(
                f"member {spec.identifier!r} threshold "
                f"{current.metadata.threshold} does not match artifact "
                f"{expected_threshold}: {spec.path}"
            )
        metadata_member = current.metadata.raw.get("member_id")
        if metadata_member is not None and metadata_member != spec.identifier:
            raise ValueError(
                f"score member_id={metadata_member!r} does not match explicit "
                f"ID {spec.identifier!r}: {spec.path}"
            )
        normalized_members.append(
            _normalized_member(
                current.scores,
                expected_threshold,
                float(member_artifact["robust_scale"]),
            )
        )
    stack = np.stack(normalized_members)
    fused = _quantize(
        _fuse(stack, str(artifact["fusion"])), int(artifact["quantization"])
    )
    threshold = int(artifact["ensemble_threshold"])
    _write_score_file(
        output_path,
        header,
        head,
        fused,
        threshold,
        artifact_path,
        str(artifact["fusion"]),
        actual_ids,
    )
    return {
        "format": SCORE_MAGIC[1:],
        "output": str(output_path.resolve()),
        "head": head,
        "frames": header.frame_count,
        "outputs": header.note_count,
        "fusion": artifact["fusion"],
        "threshold": threshold,
        "predicted_mean_polyphony": float((fused >= threshold).sum())
        / header.frame_count,
    }


def _members(values: Iterable[str]) -> list[MemberSpec]:
    return [parse_member_spec(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit/apply an offline ensemble over native TM score TSVs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument(
        "--member",
        action="append",
        required=True,
        metavar="ID=TSV",
        help="repeat in stable model order; at least two are required",
    )
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--maximum-polyphony-ratio", type=float)
    fit.add_argument("--quantization", type=int, default=DEFAULT_QUANTIZATION)

    apply = commands.add_parser("apply")
    apply.add_argument("--artifact", type=Path, required=True)
    apply.add_argument("--dataset", type=Path, required=True)
    apply.add_argument(
        "--member",
        action="append",
        required=True,
        metavar="ID=TSV",
        help="repeat in exactly the order stored by fit",
    )
    apply.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    members = _members(args.member)
    if args.command == "fit":
        result = fit_score_ensemble(
            args.dataset,
            members,
            args.output,
            maximum_polyphony_ratio=args.maximum_polyphony_ratio,
            quantization=args.quantization,
        )
        summary = {
            "artifact": str(args.output.resolve()),
            "head": result["head"],
            "fusion": result["fusion"],
            "threshold": result["ensemble_threshold"],
            "calibration": result["calibration"]["chosen"],
        }
    else:
        summary = apply_score_ensemble(
            args.artifact, args.dataset, members, args.output
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
