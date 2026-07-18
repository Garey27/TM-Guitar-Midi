from __future__ import annotations

"""Freeze the audited cross-bank TM listening experiment.

This script deliberately does no fitting or threshold search.  It packages
the already calibrated activity/onset members into one legacy-audit CPU
bundle per feature bank and writes a strict, hashed run manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from tmgm_rt.feature_contract import inspect_dataset_contract
from tmgm_rt.native_ensemble_bundle import (
    ModelSpec,
    _parse_model,
    export_ensemble_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "native-listening-comparison-20260718-hprofile-v1"
ACTIVITY_ENSEMBLE = (
    ROOT
    / "artifacts/native-full-natural-d3w3-hcontrast15-hprofile/threshold-audit-cross-bank"
    / "heldout/ensembles/activity-cross_bank_five_plus_hprofile/ensemble.json"
)
ONSET_ENSEMBLE = (
    ROOT
    / "artifacts/contiguous-test-v1/scores/night-20260718-hprofile-additive"
    / "onset-all8-hprofile-polyphony-matched.ensemble.json"
)
ONSET_RECALL_ENSEMBLE = (
    ROOT
    / "artifacts/native-full-natural-d3w3-hcontrast15-hprofile/threshold-audit-cross-bank"
    / "heldout/ensembles/onset-all_seven_plus_hprofile/ensemble.json"
)
CPU_PREDICTOR = ROOT / "native/build-ensemble-cpu/Release/tmgm_ensemble_predict.exe"


MODELS: dict[str, tuple[str, str]] = {
    "plain_c256": (
        "plain",
        "artifacts/native-full-natural-d2w3/ablations/c256-natural-d2w3-q05/activity/model.tmgmmod",
    ),
    "plain_c512": (
        "plain",
        "artifacts/native-full-natural-d2w3/ablations/c512-t256-natural-d2w3-q8/activity/model.tmgmmod",
    ),
    "plain_c1024": (
        "plain",
        "artifacts/native-full-natural-d2w3/ablations/c1024-t512-natural-d2w3-q8/activity/model.tmgmmod",
    ),
    "hc_c256": (
        "hcontrast-d2",
        "artifacts/native-full-natural-d2w3-hcontrast15/ablations/c256-natural-d2w3-hcontrast15-q4/activity/model.tmgmmod",
    ),
    "hc_c512": (
        "hcontrast-d2",
        "artifacts/native-full-natural-d2w3-hcontrast15/ablations/c512-t256-natural-d2w3-hcontrast15-q4/activity/model.tmgmmod",
    ),
    "c256_q1": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/c256-natural-d3w3-hcontrast15-q1/onset/model.tmgmmod",
    ),
    "c256_q2": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/c256-natural-d3w3-hcontrast15-q2/onset/model.tmgmmod",
    ),
    "c256_q4": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/c256-natural-d3w3-hcontrast15-q4/onset/model.tmgmmod",
    ),
    "c256_q8": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/c256-natural-d3w3-hcontrast15-q8/onset/model.tmgmmod",
    ),
    "c256_q4_seed19": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/c256-natural-d3w3-hcontrast15-q4-seed19/onset/model.tmgmmod",
    ),
    "c512_q4": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/activity-c256-onset-c512-t256-natural-d3w3-hcontrast15-q4/onset/model.tmgmmod",
    ),
    "c1024_q4": (
        "hcontrast-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15/ablations/activity-c256-onset-c1024-t512-natural-d3w3-hcontrast15-q4/onset/model.tmgmmod",
    ),
    "activity_hprofile_c256": (
        "hprofile-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15-hprofile/ablations/c256-natural-d3w3-hcontrast15-hprofile-q4/activity/model.tmgmmod",
    ),
    "onset_hprofile_c256": (
        "hprofile-d3",
        "artifacts/native-full-natural-d3w3-hcontrast15-hprofile/ablations/c256-natural-d3w3-hcontrast15-hprofile-q4/onset/model.tmgmmod",
    ),
}

DUMMIES = {
    "plain": {
        "onset": "artifacts/native-full-natural-d2w3/ablations/c256-natural-d2w3-q05/onset/model.tmgmmod"
    },
    "hcontrast-d2": {
        "onset": "artifacts/native-full-natural-d2w3-hcontrast15/ablations/c256-natural-d2w3-hcontrast15-q4/onset/model.tmgmmod"
    },
    "hcontrast-d3": {
        "activity": "artifacts/native-full-natural-d3w3-hcontrast15/ablations/c256-natural-d3w3-hcontrast15-q4/activity/model.tmgmmod"
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def member_map(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(member["id"]): dict(member) for member in artifact["members"]}


def minimal_artifact(
    head: str,
    members: list[tuple[str, Path, dict[str, Any] | None]],
    feature_count: int,
) -> dict[str, Any]:
    packed: list[dict[str, Any]] = []
    for identifier, model_path, calibrated in members:
        model = _parse_model(model_path)
        if calibrated is None:
            packed.append(
                {
                    "id": identifier,
                    "threshold": int(model.score_threshold),
                    "robust_scale": 1.0,
                    "audit_role": "opposite-head bundle placeholder",
                }
            )
        else:
            packed.append(
                {
                    "id": identifier,
                    "threshold": int(calibrated["threshold"]),
                    "robust_scale": float(calibrated["robust_scale"]),
                    "source_fit_score_file": calibrated.get("fit_score_file"),
                }
            )
    order = [member["id"] for member in packed]
    return {
        "format": "TMGM_NATIVE_SCORE_ENSEMBLE_V1",
        "head": head,
        "fusion": "mean",
        "quantization": 1024,
        "ensemble_threshold": 0,
        "members": packed,
        "member_order_sha256": hashlib.sha256("\0".join(order).encode()).hexdigest(),
        "fit_dataset": {
            "feature_count": feature_count,
            "outputs": 49,
            "midi_min": 40,
            "midi_max": 88,
            "sample_rate": 22050,
            "hop_size": 256,
        },
        "calibration": {"selection_metric": "none; packaging only"},
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    banks = load_json(output / "feature-banks.json")
    activity = load_json(ACTIVITY_ENSEMBLE)
    onset = load_json(ONSET_ENSEMBLE)
    activity_members = member_map(activity)
    onset_members = member_map(onset)

    selected_by_bank: dict[str, dict[str, list[str]]] = {
        bank: {"activity": [], "onset": []} for bank in banks
    }
    for identifier in [member["id"] for member in activity["members"]]:
        key = "activity_hprofile_c256" if identifier == "hprofile_c256" else identifier
        selected_by_bank[MODELS[key][0]]["activity"].append(key)
    for identifier in [member["id"] for member in onset["members"]]:
        key = "onset_hprofile_c256" if identifier == "hprofile_c256" else identifier
        selected_by_bank[MODELS[key][0]]["onset"].append(key)

    bundle_records: dict[str, Any] = {}
    model_records: dict[str, Any] = {}
    bundle_dir = output / "bundles"
    for bank, heads in selected_by_bank.items():
        dataset = output / "features" / bank / "2222.tmgd"
        contract = inspect_dataset_contract(dataset)
        if contract.feature_fingerprint_sha256 != banks[bank]["fingerprint_sha256"]:
            raise ValueError(f"feature contract mismatch for {bank}")

        activity_specs: list[tuple[str, Path, dict[str, Any] | None]] = []
        onset_specs: list[tuple[str, Path, dict[str, Any] | None]] = []
        for key in heads["activity"]:
            artifact_id = "hprofile_c256" if key == "activity_hprofile_c256" else key
            path = ROOT / MODELS[key][1]
            activity_specs.append((key, path, activity_members[artifact_id]))
        for key in heads["onset"]:
            artifact_id = "hprofile_c256" if key == "onset_hprofile_c256" else key
            path = ROOT / MODELS[key][1]
            onset_specs.append((key, path, onset_members[artifact_id]))
        if not activity_specs:
            path = ROOT / DUMMIES[bank]["activity"]
            activity_specs.append((f"dummy_{bank}_activity", path, None))
        if not onset_specs:
            path = ROOT / DUMMIES[bank]["onset"]
            onset_specs.append((f"dummy_{bank}_onset", path, None))

        activity_path = bundle_dir / f"{bank}.activity-packaging.json"
        onset_path = bundle_dir / f"{bank}.onset-packaging.json"
        write_json(
            activity_path,
            minimal_artifact("activity", activity_specs, contract.feature_count),
        )
        write_json(onset_path, minimal_artifact("onset", onset_specs, contract.feature_count))
        bundle_path = bundle_dir / f"{bank}.tmgmbundle"
        bundle = export_ensemble_bundle(
            activity_path,
            [ModelSpec(identifier=i, path=p) for i, p, _ in activity_specs],
            onset_path,
            [ModelSpec(identifier=i, path=p) for i, p, _ in onset_specs],
            Path(banks[bank]["reference"]),
            bundle_path,
            allow_legacy_feature_contract=True,
        )
        for identifier, path, calibrated in [*activity_specs, *onset_specs]:
            model_records[f"{bank}:{identifier}"] = {
                **file_ref(path),
                "bank": bank,
                "selected": calibrated is not None,
                "embedded_contract": _parse_model(path).format_version,
            }
        sidecar = bundle_path.with_suffix(bundle_path.suffix + ".json")
        write_json(
            sidecar,
            {
                **bundle,
                "semantic_feature_fingerprint_sha256": banks[bank]["fingerprint_sha256"],
                "legacy_audit_opt_in_required": True,
                "activity_packaging_artifact": file_ref(activity_path),
                "onset_packaging_artifact": file_ref(onset_path),
            },
        )
        bundle_records[bank] = {
            "bundle": file_ref(bundle_path),
            "sidecar": file_ref(sidecar),
            "activity_members": [item[0] for item in activity_specs],
            "onset_members": [item[0] for item in onset_specs],
            "artifact_member_aliases": {
                "activity_hprofile_c256": "hprofile_c256",
                "onset_hprofile_c256": "hprofile_c256",
            }
            if bank == "hprofile-d3"
            else {},
            "semantic_feature_fingerprint_sha256": banks[bank]["fingerprint_sha256"],
        }

    sources = {
        "2222": {
            "wav": ROOT / "datasets/private-listening/2222.wav",
            "neuralnote": ROOT / "datasets/private-listening/2222-neuralnote.mid",
        },
        "2232": {
            "wav": ROOT / "datasets/private-listening/2232.wav",
            "neuralnote": ROOT / "datasets/private-listening/2232-neuralnote.mid",
        },
    }
    tracks: dict[str, Any] = {}
    for track, source in sources.items():
        tracks[track] = {
            "source_wav": file_ref(source["wav"]),
            "neuralnote": file_ref(source["neuralnote"]),
            "features": {
                bank: {
                    "dataset": file_ref(output / "features" / bank / f"{track}.tmgd"),
                    "metadata": file_ref(output / "features" / bank / f"{track}.tmgd.json"),
                }
                for bank in banks
            },
        }

    manifest = {
        "schema": "tmgm-multibank-cpu-listening-v1",
        "policy": {
            "inference_device": "CPU sequential",
            "allow_legacy_feature_contract": True,
            "legacy_scope": "authenticated legacy TMGMMOD bundle payloads only",
            "control_tracks_used_for_tuning": False,
            "threshold_fitting_in_this_run": False,
        },
        "predictor": file_ref(CPU_PREDICTOR),
        "activity_ensemble": file_ref(ACTIVITY_ENSEMBLE),
        "onset_ensemble": file_ref(ONSET_ENSEMBLE),
        "onset_recall_ensemble": file_ref(ONSET_RECALL_ENSEMBLE),
        "feature_banks": {
            bank: {
                "semantic_feature_fingerprint_sha256": value["fingerprint_sha256"],
                "binarizer": file_ref(Path(value["binarizer"])),
                "reference": file_ref(Path(value["reference"])),
            }
            for bank, value in banks.items()
        },
        "bundles": bundle_records,
        "models": model_records,
        "tracks": tracks,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
