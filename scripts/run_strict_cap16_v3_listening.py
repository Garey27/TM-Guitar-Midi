from __future__ import annotations

"""Execute and audit the frozen strict-cap16-v3 listening package."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from prepare_strict_cap16_v3_listening import EXPECTED_MEMBERS, EXPECTED_THRESHOLDS
from run_cattack_v2_listening import ensemble, midi_event_audit
from run_multibank_cpu_listening import (
    artifact_member_ids,
    locate_member,
    normalized_member_copy,
    render_pair,
    sha256_file,
    verify_file_ref,
    write_json,
)
from tmgm_rt.feature_contract import inspect_dataset_contract
from tmgm_rt.native_score_ensemble import MemberSpec, apply_score_ensemble


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "artifacts/native-listening-comparison-20260718-strict-cap16-v3/manifest.json"
)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "tmgm-multibank-cpu-listening-v3":
        raise ValueError("unsupported strict-cap16 listening manifest")
    if manifest.get("version") != "strict-cap16-v3":
        raise ValueError("manifest is not strict-cap16-v3")
    policy = manifest["policy"]
    if (
        policy.get("inference_device") != "CPU sequential"
        or policy.get("control_tracks_used_for_tuning") is not False
        or policy.get("threshold_fitting_in_this_run") is not False
        or policy.get("production_modified") is not False
    ):
        raise ValueError("manifest violates frozen no-tuning/no-production policy")
    verify_file_ref(manifest["predictor"], "CPU predictor")
    activity_path = verify_file_ref(manifest["activity_ensemble"], "activity")
    primary_path = verify_file_ref(manifest["onset_primary_ensemble"], "primary")
    balanced_path = verify_file_ref(manifest["onset_balanced_ensemble"], "balanced")
    verify_file_ref(manifest["calibration_audit"], "strict-cap audit")
    plan_path = verify_file_ref(manifest["frozen_selection_plan"], "frozen plan")
    verify_file_ref(manifest["contiguous_confirmation"], "contiguous confirmation")

    activity = ensemble(activity_path)
    primary = ensemble(primary_path)
    balanced = ensemble(balanced_path)
    if activity["ensemble_threshold"] != -169 or len(activity["members"]) != 7:
        raise ValueError("activity artifact differs from frozen current7")
    if primary["ensemble_threshold"] != -492:
        raise ValueError("primary onset operating point is not -492")
    if balanced["ensemble_threshold"] != -648:
        raise ValueError("balanced onset operating point is not -648")
    for label, value in (("primary", primary), ("balanced", balanced)):
        members = [member["id"] for member in value["members"]]
        thresholds = {member["id"]: int(member["threshold"]) for member in value["members"]}
        if members != EXPECTED_MEMBERS or thresholds != EXPECTED_THRESHOLDS:
            raise ValueError(f"{label} member contract differs from frozen plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selection = plan["selection"]
    if (
        selection["candidate"] != "add_cap16"
        or selection["ensemble_threshold"] != -492
        or selection["member_calibration_thresholds"] != EXPECTED_THRESHOLDS
        or selection["members"] != EXPECTED_MEMBERS
        or plan["guard"]["test_threshold_tuning"] is not False
    ):
        raise ValueError("frozen selection plan differs from requested candidate")

    for bank, value in manifest["feature_banks"].items():
        verify_file_ref(value["binarizer"], f"{bank} binarizer")
        verify_file_ref(value["reference"], f"{bank} reference")
    for bank, value in manifest["bundles"].items():
        verify_file_ref(value["bundle"], f"{bank} bundle")
        verify_file_ref(value["sidecar"], f"{bank} bundle sidecar")
        expected = manifest["feature_banks"][bank]["semantic_feature_fingerprint_sha256"]
        if value["semantic_feature_fingerprint_sha256"] != expected:
            raise ValueError(f"{bank} bundle fingerprint differs")
    if "strict_cap16" not in manifest["bundles"]["hcontrast-d3"]["onset_members"]:
        raise ValueError("strict-cap16 model is absent from hcontrast-d3 bundle")
    for key, value in manifest["models"].items():
        verify_file_ref(value, f"model {key}")
    for track, value in manifest["tracks"].items():
        wav = verify_file_ref(value["source_wav"], f"{track} source WAV")
        verify_file_ref(value["neuralnote"], f"{track} NeuralNote")
        for bank, feature in value["features"].items():
            dataset = verify_file_ref(feature["dataset"], f"{track}/{bank} dataset")
            metadata_path = verify_file_ref(feature["metadata"], f"{track}/{bank} metadata")
            contract = inspect_dataset_contract(dataset)
            fingerprint = manifest["feature_banks"][bank]["semantic_feature_fingerprint_sha256"]
            if contract.feature_fingerprint_sha256 != fingerprint:
                raise ValueError(f"{track}/{bank} embedded fingerprint differs")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["feature_semantics"]["fingerprint_sha256"] != fingerprint:
                raise ValueError(f"{track}/{bank} sidecar fingerprint differs")
            if metadata["input"]["sha256"] != value["source_wav"]["sha256"]:
                raise ValueError(f"{track}/{bank} source WAV SHA differs")
            if Path(metadata["input"]["path"]).resolve() != wav.resolve():
                raise ValueError(f"{track}/{bank} source WAV path differs")


def audited_midi(path: Path) -> dict[str, Any]:
    result = midi_event_audit(path)
    if result["note_ons"] != result["note_offs"]:
        raise ValueError(f"unbalanced NoteOn/NoteOff count: {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)

    output = manifest_path.parent
    predictor = Path(manifest["predictor"]["path"])
    activity_path = Path(manifest["activity_ensemble"]["path"])
    primary_path = Path(manifest["onset_primary_ensemble"]["path"])
    balanced_path = Path(manifest["onset_balanced_ensemble"]["path"])
    activity_ids = artifact_member_ids(activity_path)
    onset_ids = artifact_member_ids(primary_path)
    run_tracks: dict[str, Any] = {}

    for track, track_value in manifest["tracks"].items():
        score_root = output / "scores" / track
        for bank, bundle in manifest["bundles"].items():
            bank_root = score_root / bank
            member_root = bank_root / "members"
            member_root.mkdir(parents=True, exist_ok=True)
            command = [
                str(predictor),
                track_value["features"][bank]["dataset"]["path"],
                bundle["bundle"]["path"],
                "--activity-output",
                str(bank_root / "activity-packaging.tsv"),
                "--onset-output",
                str(bank_root / "onset-packaging.tsv"),
                "--member-output-dir",
                str(member_root),
            ]
            if bundle["legacy_audit_opt_in_required"]:
                command.append("--allow-legacy-feature-contract")
            subprocess.run(command, cwd=ROOT, check=True)

        selected: dict[str, list[MemberSpec]] = {"activity": [], "onset": []}
        for head, identifiers in (("activity", activity_ids), ("onset", onset_ids)):
            for final_id in identifiers:
                bank, internal_id = locate_member(manifest, head, final_id)
                source = score_root / bank / "members" / f"{internal_id}.tsv"
                destination = score_root / "selected" / head / f"{final_id}.tsv"
                normalized_member_copy(source, destination, final_id)
                selected[head].append(MemberSpec(final_id, destination))

        anchor = Path(track_value["features"]["plain"]["dataset"]["path"])
        activity_scores = score_root / "activity-final.tsv"
        primary_scores = score_root / "onset-primary.tsv"
        balanced_scores = score_root / "onset-balanced.tsv"
        apply_score_ensemble(activity_path, anchor, selected["activity"], activity_scores)
        apply_score_ensemble(primary_path, anchor, selected["onset"], primary_scores)
        apply_score_ensemble(balanced_path, anchor, selected["onset"], balanced_scores)

        listening = output / "listening" / track
        listening.mkdir(parents=True, exist_ok=True)
        neuralnote = listening / "neuralnote.mid"
        shutil.copy2(track_value["neuralnote"]["path"], neuralnote)
        primary_stats = render_pair(
            activity_scores,
            primary_scores,
            anchor,
            listening / "tm-raw.mid",
            listening / "tm-stable.mid",
        )
        balanced_stats = render_pair(
            activity_scores,
            balanced_scores,
            anchor,
            listening / "tm-balanced-raw.mid",
            listening / "tm-balanced-stable.mid",
        )
        midi_paths = [
            listening / "tm-raw.mid",
            listening / "tm-stable.mid",
            listening / "tm-balanced-raw.mid",
            listening / "tm-balanced-stable.mid",
            neuralnote,
        ]
        run_tracks[track] = {
            "primary_polyphony_matched": primary_stats,
            "balanced_recall": balanced_stats,
            "midi": [audited_midi(path) for path in midi_paths],
        }

    run = {
        "schema": "tmgm-multibank-cpu-listening-run-v3",
        "version": "strict-cap16-v3",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "device": "CPU sequential",
        "thresholds_fitted_on_control_tracks": False,
        "production_modified": False,
        "operating_points": {
            "activity": -169,
            "onset_primary_polyphony_matched": -492,
            "onset_balanced_recall": -648,
        },
        "tracks": run_tracks,
    }
    run_path = output / "run.json"
    write_json(run_path, run)
    readme = output / "README.md"
    readme.write_text(
        "# Frozen strict-cap16-v3 cross-bank TM listening comparison\n\n"
        "Activity is unchanged from cattack-v2: prior5 + hprofile + cattack, "
        "seven members at threshold -169. Primary onset is the frozen current9 "
        "+ strict-cap16 candidate, ten members at the grouped-calibration-only "
        "polyphony-matched threshold -492. Optional balanced/recall files use "
        "the same ten members at the pre-existing calibration threshold -648.\n\n"
        "The control WAVs 2222 and 2232 were used only for inference/listening. "
        "No fitting, candidate selection, threshold calibration, or score-driven "
        "choice was performed on them. Inference was sequential and CPU-only. "
        "Production and VST artifacts were not modified.\n\n"
        "For each track, `tm-raw.mid` and `tm-stable.mid` are primary. "
        "`tm-balanced-raw.mid` and `tm-balanced-stable.mid` are the optional "
        "higher-recall comparison. `neuralnote.mid` is the teacher reference. "
        "Every MIDI file is nonempty and has equal NoteOn/NoteOff counts.\n",
        encoding="utf-8",
    )
    checksum_paths = [manifest_path, run_path, readme]
    checksum_paths.extend(sorted((output / "ensembles").glob("*.json")))
    checksum_paths.extend(sorted((output / "listening").rglob("*.mid")))
    rows = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in checksum_paths
    ]
    (output / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="ascii")
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
