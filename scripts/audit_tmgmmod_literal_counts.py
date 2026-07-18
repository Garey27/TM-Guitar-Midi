from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import mmap
from pathlib import Path
import statistics
import struct
from typing import Any

from tmgm_rt.feature_contract import inspect_model_contract


HEADER_BYTES = 256


def parse_model_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("model must use LABEL=PATH")
    return label, Path(path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_model(label: str, path: Path) -> dict[str, Any]:
    contract = inspect_model_contract(path, allow_legacy=True)
    with path.open("rb") as stream:
        header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise ValueError(f"truncated TMGMMOD header: {path}")
        state_bits, feature_count, output_count, clause_count, literal_count, words = (
            struct.unpack_from("<IIIIII", header, 28)
        )
        cap = struct.unpack_from("<I", header, 60)[0]
        ta_offset, ta_bytes = struct.unpack_from("<QQ", header, 112)
        expected_ta_bytes = clause_count * state_bits * words * 4
        if ta_bytes != expected_ta_bytes:
            raise ValueError(f"TA payload size mismatch: {path}")

        tail_bits = literal_count % 32
        tail_mask = (1 << tail_bits) - 1 if tail_bits else 0xFFFFFFFF
        counts: list[int] = []
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for clause in range(clause_count):
                action_word = (clause * state_bits + state_bits - 1) * words
                count = 0
                for word_index in range(words):
                    value = struct.unpack_from(
                        "<I", mapped, ta_offset + (action_word + word_index) * 4
                    )[0]
                    if word_index + 1 == words:
                        value &= tail_mask
                    count += value.bit_count()
                counts.append(count)

    violations = [index for index, count in enumerate(counts) if count > cap]
    histogram = {str(key): value for key, value in sorted(Counter(counts).items())}
    return {
        "label": label,
        "path": str(path),
        "file_sha256": file_sha256(path),
        "embedded_model_checksum_sha256": contract.checksum_sha256,
        "format_version": contract.format_version,
        "head": contract.head,
        "feature_count": feature_count,
        "literal_count": literal_count,
        "output_count": output_count,
        "clause_count": clause_count,
        "state_bits": state_bits,
        "header_max_included_literals": cap,
        "shared_clause_bank_across_outputs": True,
        "per_output_literal_counts_identical": True,
        "included_literals_per_clause": counts,
        "summary": {
            "minimum": min(counts),
            "maximum": max(counts),
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "histogram": histogram,
            "violating_clause_count": len(violations),
            "violating_clause_indices": violations,
            "strict_cap_satisfied": not violations,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TMGMMOD included-literal audit",
        "",
        "The TA action/MSB bit-plane was read directly from each authenticated model. "
        "Padding bits in the final literal word were masked before popcount.",
        "",
        "The native architecture has one clause/TA bank shared by every output. "
        "Consequently, a clause has the same included-literal count for all 49 MIDI outputs; "
        "output-specific weights do not change clause literals.",
        "",
        "| Model | Header cap | Min | Mean | Median | Max | Violations | SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in report["models"]:
        summary = model["summary"]
        lines.append(
            "| {label} | {cap} | {minimum} | {mean:.3f} | {median:.1f} | "
            "{maximum} | {violations} | `{sha}` |".format(
                label=model["label"],
                cap=model["header_max_included_literals"],
                minimum=summary["minimum"],
                mean=summary["mean"],
                median=summary["median"],
                maximum=summary["maximum"],
                violations=summary["violating_clause_count"],
                sha=model["file_sha256"],
            )
        )
    lines.extend(
        [
            "",
            f"Overall strict-cap result: **{'PASS' if report['all_models_pass'] else 'FAIL'}**.",
            "",
            "Exact per-clause counts and embedded model checksums are in "
            "`literal-count-audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Authenticate TMGMMOD files and audit included literals per shared clause."
    )
    parser.add_argument(
        "--model", action="append", required=True, type=parse_model_argument
    )
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    labels = [label for label, _ in args.model]
    if len(labels) != len(set(labels)):
        parser.error("model labels must be unique")
    models = [audit_model(label, path) for label, path in args.model]
    report = {
        "schema_version": 1,
        "method": {
            "model_authentication": "TMGMMOD embedded canonical SHA-256 verified",
            "literal_count": "popcount of clause action/MSB bit-plane with tail padding masked",
            "output_semantics": "one shared TA clause bank; counts are identical for every output",
        },
        "models": models,
        "all_models_pass": all(
            model["summary"]["strict_cap_satisfied"] for model in models
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    return 0 if report["all_models_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
