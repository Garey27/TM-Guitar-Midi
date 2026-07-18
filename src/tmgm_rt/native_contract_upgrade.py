from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ContextConfig, FrontendConfig
from .feature_contract import inspect_dataset_contract
from .feature_semantics import (
    binary_feature_semantics,
    canonical_frontend_mapping,
    feature_fingerprint_bytes,
    frontend_schema_descriptor,
)
from .native_dataset import _pack_header, read_native_dataset_header


UPGRADE_SCHEMA = "tmgm-native-feature-contract-upgrade-v1"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".upgrade.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _header_metadata(header: Any) -> dict[str, Any]:
    return {
        **asdict(header),
        "payload_sha256": header.payload_sha256.hex(),
        "feature_fingerprint_sha256": (
            header.feature_fingerprint_sha256.hex()
        ),
    }


def _verify_binarizer(
    binarizer_path: Path,
    metadata: dict[str, Any],
) -> tuple[str, str, int, int]:
    digest = hashlib.sha256(binarizer_path.read_bytes()).hexdigest()
    if metadata.get("sha256") != digest:
        raise ValueError("binarizer SHA-256 disagrees with its sidecar")
    signature = metadata.get("signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("binarizer sidecar has no valid signature")
    with np.load(binarizer_path, allow_pickle=False) as stored:
        thresholds = np.asarray(stored["thresholds"])
        keep_columns = np.asarray(stored["keep_columns"])
        quantiles = np.asarray(stored["quantiles"])
    if thresholds.ndim != 2 or thresholds.shape[1] != quantiles.size:
        raise ValueError("binarizer threshold dimensions are invalid")
    if keep_columns.shape != (thresholds.size,):
        raise ValueError("binarizer keep-column dimensions are invalid")
    continuous_count = int(thresholds.shape[0])
    binary_count = int(np.count_nonzero(keep_columns))
    if metadata.get("continuous_feature_count") != continuous_count:
        raise ValueError("binarizer continuous feature count disagrees")
    if metadata.get("kept_binary_features") != binary_count:
        raise ValueError("binarizer binary feature count disagrees")
    return digest, signature, continuous_count, binary_count


def upgrade_native_feature_contract(
    corpus_directory_value: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upgrade a verified legacy full-native corpus without rewriting payloads.

    This is intentionally explicit and idempotent. It authenticates every
    dataset payload and the binarizer before deriving one canonical semantics
    fingerprint, writes upgraded sidecars first, and patches only the fixed
    256-byte TMGD headers last. Interrupted runs can safely be repeated.
    """

    root = Path(corpus_directory_value)
    binarizer_path = root / "global-quantile-thermometer.npz"
    binarizer_metadata_path = binarizer_path.with_suffix(
        binarizer_path.suffix + ".json"
    )
    split_paths = [root / "train.tmgd", root / "validation.tmgd"]
    required = [binarizer_path, binarizer_metadata_path]
    required.extend(split_paths)
    required.extend(path.with_suffix(path.suffix + ".json") for path in split_paths)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"native corpus is incomplete: {missing}")
    if list(root.glob("*.tmp")) or list(root.glob("*.upgrade.tmp")):
        raise ValueError("native corpus contains an unfinished temporary file")

    binarizer_metadata = _load_object(binarizer_metadata_path)
    binarizer_sha, binarizer_signature, continuous_count, binary_count = (
        _verify_binarizer(binarizer_path, binarizer_metadata)
    )

    split_metadata: list[tuple[Path, dict[str, Any]]] = []
    canonical_frontend = None
    canonical_context = None
    semantics = None
    fingerprint = None
    headers = []
    for dataset_path in split_paths:
        metadata_path = dataset_path.with_suffix(dataset_path.suffix + ".json")
        metadata = _load_object(metadata_path)
        if metadata.get("format") != "TMGMDAT":
            raise ValueError(f"unexpected dataset metadata format: {metadata_path}")
        frontend_value = metadata.get("frontend")
        context_value = metadata.get("context")
        if not isinstance(frontend_value, dict) or not isinstance(context_value, dict):
            raise ValueError(f"dataset frontend/context metadata is missing: {metadata_path}")
        frontend = FrontendConfig(**canonical_frontend_mapping(frontend_value))
        delays = context_value.get("delays")
        if not isinstance(delays, (list, tuple)):
            raise ValueError(f"dataset context delays are invalid: {metadata_path}")
        context = ContextConfig(delays=tuple(int(value) for value in delays))
        candidate_semantics = binary_feature_semantics(
            frontend,
            context,
            binarizer_sha256=binarizer_sha,
            binarizer_signature=binarizer_signature,
            continuous_feature_count=continuous_count,
            binary_feature_count=binary_count,
        )
        candidate_fingerprint = feature_fingerprint_bytes(candidate_semantics)
        if canonical_frontend is None:
            canonical_frontend = frontend
            canonical_context = context
            semantics = candidate_semantics
            fingerprint = candidate_fingerprint
        elif (
            frontend != canonical_frontend
            or context != canonical_context
            or candidate_fingerprint != fingerprint
        ):
            raise ValueError("train/validation feature semantics differ")

        contract = inspect_dataset_contract(dataset_path, allow_legacy=True)
        header = read_native_dataset_header(dataset_path)
        if header.feature_count != binary_count:
            raise ValueError(f"dataset/binarizer feature count differs: {dataset_path}")
        declared_header = metadata.get("header")
        if not isinstance(declared_header, dict):
            raise ValueError(f"dataset header metadata is missing: {metadata_path}")
        if declared_header.get("payload_sha256") != contract.checksum_sha256:
            raise ValueError(f"dataset payload checksum metadata differs: {dataset_path}")
        if any(header.feature_fingerprint_sha256) and (
            header.feature_fingerprint_sha256 != candidate_fingerprint
        ):
            raise ValueError(f"dataset already has a different fingerprint: {dataset_path}")
        headers.append(header)
        split_metadata.append((metadata_path, metadata))

    assert canonical_frontend is not None
    assert canonical_context is not None
    assert semantics is not None
    assert fingerprint is not None
    frontend_schema = frontend_schema_descriptor(canonical_frontend)
    source_exports = []
    for path, metadata in split_metadata:
        existing_upgrade = metadata.get("feature_contract_upgrade")
        if isinstance(existing_upgrade, dict):
            source_schema = existing_upgrade.get("source_export_schema")
            source_signature = existing_upgrade.get("source_export_signature")
        else:
            source_schema = metadata.get("export_schema")
            source_signature = metadata.get("export_signature")
        source_exports.append(
            {
                "path": path.name,
                "export_schema": source_schema,
                "export_signature": source_signature,
            }
        )
    upgrade_basis = {
        "schema": UPGRADE_SCHEMA,
        "binarizer_sha256": binarizer_sha,
        "binarizer_signature": binarizer_signature,
        "feature_semantics": semantics,
        "source_exports": source_exports,
    }
    upgrade_id = _canonical_hash(upgrade_basis)

    upgraded_binarizer_metadata = dict(binarizer_metadata)
    upgraded_binarizer_metadata.update(
        {
            "schema": 2,
            "frontend_schema": frontend_schema,
            "frontend": asdict(canonical_frontend),
            "context": asdict(canonical_context),
            "feature_semantics": semantics,
            "feature_contract_upgrade": {
                "schema": UPGRADE_SCHEMA,
                "id": upgrade_id,
                "payload_rewritten": False,
            },
        }
    )

    upgraded_splits: list[tuple[Path, dict[str, Any], Any]] = []
    for (metadata_path, metadata), header in zip(
        split_metadata, headers, strict=True
    ):
        existing_upgrade = metadata.get("feature_contract_upgrade")
        if isinstance(existing_upgrade, dict):
            source_export_schema = existing_upgrade.get("source_export_schema")
            source_export_signature = existing_upgrade.get(
                "source_export_signature"
            )
        else:
            source_export_schema = metadata.get("export_schema")
            source_export_signature = metadata.get("export_signature")
        upgraded_header = replace(
            header, feature_fingerprint_sha256=fingerprint
        )
        upgraded = dict(metadata)
        upgraded.update(
            {
                "version": 2,
                "export_schema": 2,
                "export_signature": _canonical_hash(
                    {
                        "schema": UPGRADE_SCHEMA,
                        "upgrade_id": upgrade_id,
                        "source_export_signature": source_export_signature,
                        "split": metadata.get("split"),
                    }
                ),
                "frontend_schema": frontend_schema,
                "feature_semantics": semantics,
                "frontend": asdict(canonical_frontend),
                "context": asdict(canonical_context),
                "header": _header_metadata(upgraded_header),
                "feature_contract_upgrade": {
                    "schema": UPGRADE_SCHEMA,
                    "id": upgrade_id,
                    "payload_rewritten": False,
                    "source_export_schema": source_export_schema,
                    "source_export_signature": source_export_signature,
                },
            }
        )
        binarizer = dict(upgraded.get("binarizer", {}))
        binarizer["sha256"] = binarizer_sha
        binarizer["signature"] = binarizer_signature
        upgraded["binarizer"] = binarizer
        upgraded_splits.append((metadata_path, upgraded, upgraded_header))

    if not dry_run:
        _write_json_atomic(binarizer_metadata_path, upgraded_binarizer_metadata)
        for metadata_path, metadata, _ in upgraded_splits:
            _write_json_atomic(metadata_path, metadata)
        for dataset_path, (_, _, upgraded_header) in zip(
            split_paths, upgraded_splits, strict=True
        ):
            packed = _pack_header(upgraded_header)
            with dataset_path.open("r+b") as stream:
                stream.seek(0)
                stream.write(packed)
                stream.flush()

    return {
        "schema": UPGRADE_SCHEMA,
        "corpus_directory": str(root.resolve()),
        "dry_run": dry_run,
        "upgrade_id": upgrade_id,
        "feature_fingerprint_sha256": fingerprint.hex(),
        "datasets": [path.name for path in split_paths],
        "payload_rewritten": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly upgrade a completed legacy native corpus to the "
            "authenticated feature-semantics contract."
        )
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = upgrade_native_feature_contract(
        arguments.corpus_dir, dry_run=arguments.dry_run
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
