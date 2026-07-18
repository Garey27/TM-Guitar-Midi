from __future__ import annotations

"""Prepare the frozen cattack-v2 cross-bank CPU listening package.

No fitting, score inspection, or threshold search is performed here.  The
polyphony-matched -486 and balanced -583 operating points come verbatim from
the grouped calibration-only cattack audit.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from prepare_multibank_cpu_listening import (
    CPU_PREDICTOR,
    DUMMIES,
    MODELS,
    file_ref,
    load_json,
    member_map,
    minimal_artifact,
    sha256_file,
    write_json,
)
from tmgm_rt.feature_contract import inspect_dataset_contract, inspect_model_contract
from tmgm_rt.native_ensemble_bundle import ModelSpec, export_ensemble_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/native-listening-comparison-20260718-cattack-v2"
V1 = ROOT / "artifacts/native-listening-comparison-20260718-hprofile-v1"
CATTACK_ROOT = ROOT / "artifacts/native-full-natural-d3w3-hcontrast15-cattack"
AUDIT_ROOT = CATTACK_ROOT / "threshold-audit-cross-bank"
SOURCE_ACTIVITY = (
    AUDIT_ROOT
    / "heldout/ensembles/activity-best_six_plus_cattack/ensemble.json"
)
SOURCE_ONSET_BALANCED = (
    AUDIT_ROOT / "heldout/ensembles/onset-all9_hprofile_cattack/ensemble.json"
)
CATTACK_MODELS = {
    "activity_cattack_c256": CATTACK_ROOT
    / "ablations/c256-natural-d3w3-hcontrast15-cattack-q4/activity/model.tmgmmod",
    "onset_cattack_c256": CATTACK_ROOT
    / "ablations/c256-natural-d3w3-hcontrast15-cattack-q4/onset/model.tmgmmod",
}
TRACKS = {
    "2222": {
        "wav": ROOT / "datasets/private-listening/2222.wav",
        "neuralnote": ROOT / "datasets/private-listening/2222-neuralnote.mid",
    },
    "2232": {
        "wav": ROOT / "datasets/private-listening/2232.wav",
        "neuralnote": ROOT / "datasets/private-listening/2232-neuralnote.mid",
    },
}


def prepare_references(output: Path) -> dict[str, dict[str, str]]:
    old_banks = load_json(V1 / "feature-banks.json")
    references = output / "references"
    references.mkdir(parents=True, exist_ok=True)
    banks: dict[str, dict[str, str]] = {}
    for bank, value in old_banks.items():
        destination = references / f"{bank}.reference.json"
        shutil.copy2(value["reference"], destination)
        banks[bank] = {
            "binarizer": value["binarizer"],
            "reference": str(destination.resolve()),
            "fingerprint_sha256": value["fingerprint_sha256"],
        }
    cattack_reference = references / "cattack-d3.reference.json"
    shutil.copy2(CATTACK_ROOT / "validation.tmgd.json", cattack_reference)
    cattack_metadata = load_json(cattack_reference)
    banks["cattack-d3"] = {
        "binarizer": str(
            (CATTACK_ROOT / "global-quantile-thermometer.npz").resolve()
        ),
        "reference": str(cattack_reference.resolve()),
        "fingerprint_sha256": cattack_metadata["feature_semantics"][
            "fingerprint_sha256"
        ],
    }
    write_json(output / "feature-banks.json", banks)
    return banks


def export_features(output: Path, banks: dict[str, dict[str, str]]) -> None:
    exporter = ROOT / "scripts/export_wav_native.py"
    for bank in sorted(banks):
        directory = output / "features" / bank
        directory.mkdir(parents=True, exist_ok=True)
        for track, sources in TRACKS.items():
            command = [
                sys.executable,
                str(exporter),
                "--wav",
                str(sources["wav"]),
                "--binarizer",
                banks[bank]["binarizer"],
                "--reference-metadata",
                banks[bank]["reference"],
                "--output",
                str(directory / f"{track}.tmgd"),
                "--batch-frames",
                "2048",
            ]
            subprocess.run(command, cwd=ROOT, check=True)


def freeze_ensembles(output: Path) -> tuple[Path, Path, Path, Path]:
    ensemble_dir = output / "ensembles"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    activity = ensemble_dir / "activity-prior5-hprofile-cattack.json"
    balanced = ensemble_dir / "onset-all9-cattack-balanced.json"
    primary = ensemble_dir / "onset-all9-cattack-polyphony-matched.json"
    audit_copy = ensemble_dir / "cattack-audit.json"
    shutil.copy2(SOURCE_ACTIVITY, activity)
    shutil.copy2(SOURCE_ONSET_BALANCED, balanced)
    shutil.copy2(AUDIT_ROOT / "cattack-audit.json", audit_copy)

    audit = load_json(audit_copy)
    operating_point = audit["onset_operating_points"]["all9_hprofile_cattack"][
        "polyphony_matched_f1"
    ]
    if operating_point["threshold"] != -486:
        raise ValueError("audited cattack polyphony-matched threshold is not -486")
    primary_value = deepcopy(load_json(balanced))
    primary_value["ensemble_threshold"] = -486
    primary_value["calibration"] = {
        "selection_metric": operating_point["selection"],
        "chosen": operating_point["calibration"],
        "candidates": {},
    }
    primary_value["operating_point_provenance"] = {
        "policy": "polyphony_matched_f1",
        "source_audit": file_ref(audit_copy),
        "source_ensemble_artifact": file_ref(SOURCE_ONSET_BALANCED),
        "selection_rows": "grouped calibration only",
        "disjoint_evaluation": operating_point["evaluation"],
        "test_or_control_wavs_used": False,
        "threshold": -486,
    }
    write_json(primary, primary_value)
    if load_json(balanced)["ensemble_threshold"] != -583:
        raise ValueError("audited cattack balanced threshold is not -583")
    return activity, primary, balanced, audit_copy


def model_path(key: str) -> Path:
    if key in CATTACK_MODELS:
        return CATTACK_MODELS[key]
    return ROOT / MODELS[key][1]


def build_bundles(
    output: Path,
    banks: dict[str, dict[str, str]],
    activity_path: Path,
    onset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    activity = load_json(activity_path)
    onset = load_json(onset_path)
    activity_calibration = member_map(activity)
    onset_calibration = member_map(onset)
    specs: dict[str, dict[str, list[tuple[str, str]]]] = {
        bank: {"activity": [], "onset": []} for bank in banks
    }
    specs["plain"]["activity"] = [
        (name, name) for name in ("plain_c256", "plain_c512", "plain_c1024")
    ]
    specs["hcontrast-d2"]["activity"] = [
        (name, name) for name in ("hc_c256", "hc_c512")
    ]
    specs["hprofile-d3"]["activity"] = [
        ("activity_hprofile_c256", "hprofile_c256")
    ]
    specs["cattack-d3"]["activity"] = [
        ("activity_cattack_c256", "cattack_c256")
    ]
    specs["hcontrast-d3"]["onset"] = [
        (name, name)
        for name in (
            "c256_q1",
            "c256_q2",
            "c256_q4",
            "c256_q8",
            "c256_q4_seed19",
            "c512_q4",
            "c1024_q4",
        )
    ]
    specs["hprofile-d3"]["onset"] = [
        ("onset_hprofile_c256", "hprofile_c256")
    ]
    specs["cattack-d3"]["onset"] = [
        ("onset_cattack_c256", "cattack_c256")
    ]

    bundle_records: dict[str, Any] = {}
    model_records: dict[str, Any] = {}
    bundle_dir = output / "bundles"
    for bank, heads in specs.items():
        contract = inspect_dataset_contract(output / "features" / bank / "2222.tmgd")
        if contract.feature_fingerprint_sha256 != banks[bank]["fingerprint_sha256"]:
            raise ValueError(f"feature fingerprint differs for {bank}")
        packed: dict[str, list[tuple[str, Path, dict[str, Any] | None]]] = {
            "activity": [],
            "onset": [],
        }
        for internal, final in heads["activity"]:
            packed["activity"].append(
                (internal, model_path(internal), activity_calibration[final])
            )
        for internal, final in heads["onset"]:
            packed["onset"].append(
                (internal, model_path(internal), onset_calibration[final])
            )
        if not packed["activity"]:
            dummy = ROOT / DUMMIES[bank]["activity"]
            packed["activity"].append((f"dummy_{bank}_activity", dummy, None))
        if not packed["onset"]:
            dummy = ROOT / DUMMIES[bank]["onset"]
            packed["onset"].append((f"dummy_{bank}_onset", dummy, None))

        activity_packaging = bundle_dir / f"{bank}.activity-packaging.json"
        onset_packaging = bundle_dir / f"{bank}.onset-packaging.json"
        write_json(
            activity_packaging,
            minimal_artifact(
                "activity", packed["activity"], contract.feature_count
            ),
        )
        write_json(
            onset_packaging,
            minimal_artifact("onset", packed["onset"], contract.feature_count),
        )
        bundle_path = bundle_dir / f"{bank}.tmgmbundle"
        result = export_ensemble_bundle(
            activity_packaging,
            [ModelSpec(identifier=i, path=p) for i, p, _ in packed["activity"]],
            onset_packaging,
            [ModelSpec(identifier=i, path=p) for i, p, _ in packed["onset"]],
            Path(banks[bank]["reference"]),
            bundle_path,
            allow_legacy_feature_contract=True,
        )
        sidecar = bundle_path.with_suffix(bundle_path.suffix + ".json")
        write_json(
            sidecar,
            {
                **result,
                "semantic_feature_fingerprint_sha256": banks[bank][
                    "fingerprint_sha256"
                ],
                "legacy_audit_opt_in_required": bool(
                    result["legacy_feature_contract"]
                ),
                "activity_packaging_artifact": file_ref(activity_packaging),
                "onset_packaging_artifact": file_ref(onset_packaging),
            },
        )
        aliases = {
            internal: final
            for head in ("activity", "onset")
            for internal, final in heads[head]
            if internal != final
        }
        bundle_records[bank] = {
            "bundle": file_ref(bundle_path),
            "sidecar": file_ref(sidecar),
            "activity_members": [item[0] for item in packed["activity"]],
            "onset_members": [item[0] for item in packed["onset"]],
            "artifact_member_aliases": aliases,
            "semantic_feature_fingerprint_sha256": banks[bank][
                "fingerprint_sha256"
            ],
            "legacy_audit_opt_in_required": bool(
                result["legacy_feature_contract"]
            ),
        }
        for identifier, path, calibrated in [
            *packed["activity"],
            *packed["onset"],
        ]:
            model_records[f"{bank}:{identifier}"] = {
                **file_ref(path),
                "bank": bank,
                "selected": calibrated is not None,
                "contract": inspect_model_contract(
                    path, allow_legacy=True
                ).to_dict(),
            }
    return bundle_records, model_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output == V1.resolve():
        raise ValueError("refusing to modify the frozen hprofile-v1 package")
    output.mkdir(parents=True, exist_ok=True)
    banks = prepare_references(output)
    export_features(output, banks)
    activity, primary, balanced, audit = freeze_ensembles(output)
    bundles, models = build_bundles(output, banks, activity, primary)

    tracks: dict[str, Any] = {}
    for track, sources in TRACKS.items():
        tracks[track] = {
            "source_wav": file_ref(sources["wav"]),
            "neuralnote": file_ref(sources["neuralnote"]),
            "features": {
                bank: {
                    "dataset": file_ref(
                        output / "features" / bank / f"{track}.tmgd"
                    ),
                    "metadata": file_ref(
                        output / "features" / bank / f"{track}.tmgd.json"
                    ),
                }
                for bank in banks
            },
        }
    manifest = {
        "schema": "tmgm-multibank-cpu-listening-v2",
        "version": "cattack-v2",
        "policy": {
            "inference_device": "CPU sequential",
            "control_tracks_used_for_tuning": False,
            "threshold_fitting_in_this_run": False,
            "legacy_feature_contract_scope": "four pre-cattack authenticated TMGMMOD banks only",
            "semantic_v3_bank": "cattack-d3",
        },
        "predictor": file_ref(CPU_PREDICTOR),
        "activity_ensemble": file_ref(activity),
        "onset_primary_ensemble": file_ref(primary),
        "onset_balanced_ensemble": file_ref(balanced),
        "calibration_audit": file_ref(audit),
        "feature_banks": {
            bank: {
                "semantic_feature_fingerprint_sha256": value[
                    "fingerprint_sha256"
                ],
                "binarizer": file_ref(Path(value["binarizer"])),
                "reference": file_ref(Path(value["reference"])),
            }
            for bank, value in banks.items()
        },
        "bundles": bundles,
        "models": models,
        "tracks": tracks,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"manifest": str(manifest_path), "sha256": sha256_file(manifest_path)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
