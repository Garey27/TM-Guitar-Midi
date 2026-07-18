from __future__ import annotations

"""Prepare the frozen strict-cap16-v3 cross-bank CPU listening package.

The candidate, member thresholds, and ensemble operating points are copied
from the grouped calibration audit and its frozen contiguous-confirmation
plan.  This script never reads prediction scores from the listening WAVs.
"""

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from prepare_cattack_v2_listening import (
    CATTACK_MODELS,
    TRACKS,
    build_bundles as build_cattack_bundles,
    export_features,
    model_path,
    prepare_references,
)
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
DEFAULT_OUTPUT = (
    ROOT / "artifacts/native-listening-comparison-20260718-strict-cap16-v3"
)
STRICT_ROOT = ROOT / "artifacts/native-full-natural-d3w3-hcontrast15-strict-cap"
AUDIT_ROOT = STRICT_ROOT / "threshold-audit"
CONFIRMATION_ROOT = AUDIT_ROOT / "contiguous-confirmation"
STRICT_MODEL = (
    STRICT_ROOT
    / "ablations/c512-onset-strict-cap16/onset/model.tmgmmod"
)
SOURCE_ACTIVITY = CONFIRMATION_ROOT / "frozen/activity-current-cattack-minus169.ensemble.json"
SOURCE_PRIMARY = (
    CONFIRMATION_ROOT
    / "frozen/onset-add_cap16-polyphony_matched_f1.ensemble.json"
)
SOURCE_BALANCED = (
    AUDIT_ROOT / "heldout/ensembles/onset-strict-cap-add_cap16/ensemble.json"
)
SOURCE_PLAN = CONFIRMATION_ROOT / "frozen-plan.json"
SOURCE_CONFIRMATION = CONFIRMATION_ROOT / "contiguous-confirmation.json"
EXPECTED_MEMBERS = [
    "c256_q1",
    "c256_q2",
    "c256_q4",
    "c256_q8",
    "c256_q4_seed19",
    "c512_q4",
    "hprofile_c256",
    "c1024_q4",
    "cattack_c256",
    "strict_cap16",
]
EXPECTED_THRESHOLDS = {
    "c256_q1": 151,
    "c256_q2": 88,
    "c256_q4": 80,
    "c256_q8": 38,
    "c256_q4_seed19": 60,
    "c512_q4": 153,
    "hprofile_c256": 60,
    "c1024_q4": 283,
    "cattack_c256": 87,
    "strict_cap16": 144,
}


def _member_thresholds(value: dict[str, Any]) -> dict[str, int]:
    return {member["id"]: int(member["threshold"]) for member in value["members"]}


def freeze_artifacts(output: Path) -> tuple[Path, Path, Path, Path, Path]:
    ensemble_dir = output / "ensembles"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    activity = ensemble_dir / "activity-prior5-hprofile-cattack.json"
    primary = ensemble_dir / "onset-all10-strict-cap16-polyphony-matched.json"
    balanced = ensemble_dir / "onset-all10-strict-cap16-balanced.json"
    audit = ensemble_dir / "strict-cap-audit.json"
    plan = ensemble_dir / "strict-cap16-frozen-plan.json"
    confirmation = ensemble_dir / "strict-cap16-contiguous-confirmation.json"
    for source, destination in (
        (SOURCE_ACTIVITY, activity),
        (SOURCE_PRIMARY, primary),
        (SOURCE_BALANCED, balanced),
        (AUDIT_ROOT / "strict-cap-audit.json", audit),
        (SOURCE_PLAN, plan),
        (SOURCE_CONFIRMATION, confirmation),
    ):
        shutil.copy2(source, destination)

    activity_value = load_json(activity)
    primary_value = load_json(primary)
    balanced_value = load_json(balanced)
    plan_value = load_json(plan)
    if activity_value["ensemble_threshold"] != -169 or len(activity_value["members"]) != 7:
        raise ValueError("activity ensemble is not the frozen current7 -169 artifact")
    if primary_value["ensemble_threshold"] != -492:
        raise ValueError("primary onset threshold differs from frozen -492")
    if balanced_value["ensemble_threshold"] != -648:
        raise ValueError("balanced onset threshold differs from calibrated -648")
    for label, value in (("primary", primary_value), ("balanced", balanced_value)):
        if [member["id"] for member in value["members"]] != EXPECTED_MEMBERS:
            raise ValueError(f"{label} onset member order differs from frozen plan")
        if _member_thresholds(value) != EXPECTED_THRESHOLDS:
            raise ValueError(f"{label} onset member thresholds differ from frozen plan")
    selection = plan_value["selection"]
    if (
        selection["candidate"] != "add_cap16"
        or selection["ensemble_threshold"] != -492
        or selection["member_calibration_thresholds"] != EXPECTED_THRESHOLDS
        or selection["members"] != EXPECTED_MEMBERS
    ):
        raise ValueError("contiguous-confirmation plan no longer matches strict-cap16")
    return activity, primary, balanced, audit, plan


def build_bundles(
    output: Path,
    banks: dict[str, dict[str, str]],
    activity_path: Path,
    onset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_records, model_records = build_cattack_bundles(
        output, banks, activity_path, onset_path
    )

    bank = "hcontrast-d3"
    bundle_dir = output / "bundles"
    contract = inspect_dataset_contract(output / "features" / bank / "2222.tmgd")
    onset_calibration = member_map(load_json(onset_path))
    activity_model = ROOT / DUMMIES[bank]["activity"]
    onset_models: list[tuple[str, Path, dict[str, Any]]] = [
        (name, ROOT / MODELS[name][1], onset_calibration[name])
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
    onset_models.append(("strict_cap16", STRICT_MODEL, onset_calibration["strict_cap16"]))
    activity_packaging = bundle_dir / f"{bank}.activity-packaging.json"
    onset_packaging = bundle_dir / f"{bank}.onset-packaging.json"
    activity_models = [(f"dummy_{bank}_activity", activity_model, None)]
    write_json(
        activity_packaging,
        minimal_artifact("activity", activity_models, contract.feature_count),
    )
    write_json(
        onset_packaging,
        minimal_artifact("onset", onset_models, contract.feature_count),
    )
    bundle_path = bundle_dir / f"{bank}.tmgmbundle"
    result = export_ensemble_bundle(
        activity_packaging,
        [ModelSpec(identifier=i, path=p) for i, p, _ in activity_models],
        onset_packaging,
        [ModelSpec(identifier=i, path=p) for i, p, _ in onset_models],
        Path(banks[bank]["reference"]),
        bundle_path,
        allow_legacy_feature_contract=True,
    )
    sidecar = bundle_path.with_suffix(bundle_path.suffix + ".json")
    write_json(
        sidecar,
        {
            **result,
            "semantic_feature_fingerprint_sha256": banks[bank]["fingerprint_sha256"],
            "legacy_audit_opt_in_required": bool(result["legacy_feature_contract"]),
            "activity_packaging_artifact": file_ref(activity_packaging),
            "onset_packaging_artifact": file_ref(onset_packaging),
        },
    )
    bundle_records[bank] = {
        "bundle": file_ref(bundle_path),
        "sidecar": file_ref(sidecar),
        "activity_members": [activity_models[0][0]],
        "onset_members": [item[0] for item in onset_models],
        "artifact_member_aliases": {},
        "semantic_feature_fingerprint_sha256": banks[bank]["fingerprint_sha256"],
        "legacy_audit_opt_in_required": bool(result["legacy_feature_contract"]),
    }
    model_records[f"{bank}:strict_cap16"] = {
        **file_ref(STRICT_MODEL),
        "bank": bank,
        "selected": True,
        "contract": inspect_model_contract(STRICT_MODEL, allow_legacy=True).to_dict(),
        "strict_max_literals": 16,
    }
    return bundle_records, model_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    banks = prepare_references(output)
    export_features(output, banks)
    activity, primary, balanced, audit, plan = freeze_artifacts(output)
    bundles, models = build_bundles(output, banks, activity, primary)

    tracks: dict[str, Any] = {}
    for track, sources in TRACKS.items():
        tracks[track] = {
            "source_wav": file_ref(sources["wav"]),
            "neuralnote": file_ref(sources["neuralnote"]),
            "features": {
                bank: {
                    "dataset": file_ref(output / "features" / bank / f"{track}.tmgd"),
                    "metadata": file_ref(
                        output / "features" / bank / f"{track}.tmgd.json"
                    ),
                }
                for bank in banks
            },
        }
    manifest = {
        "schema": "tmgm-multibank-cpu-listening-v3",
        "version": "strict-cap16-v3",
        "policy": {
            "inference_device": "CPU sequential",
            "control_tracks_used_for_tuning": False,
            "threshold_fitting_in_this_run": False,
            "candidate_selection": "grouped calibration only; frozen before contiguous/control inference",
            "production_modified": False,
            "legacy_feature_contract_scope": (
                "authenticated pre-cattack banks plus audited strict-cap16 legacy model"
            ),
            "semantic_v3_bank": "cattack-d3",
        },
        "predictor": file_ref(CPU_PREDICTOR),
        "activity_ensemble": file_ref(activity),
        "onset_primary_ensemble": file_ref(primary),
        "onset_balanced_ensemble": file_ref(balanced),
        "calibration_audit": file_ref(audit),
        "frozen_selection_plan": file_ref(plan),
        "contiguous_confirmation": file_ref(
            output / "ensembles/strict-cap16-contiguous-confirmation.json"
        ),
        "feature_banks": {
            bank: {
                "semantic_feature_fingerprint_sha256": value["fingerprint_sha256"],
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
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
