from __future__ import annotations

"""Execute and audit the frozen cattack-v2 listening package."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import mido

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
    / "artifacts/native-listening-comparison-20260718-cattack-v2/manifest.json"
)
V1_MANIFEST = (
    ROOT
    / "artifacts/native-listening-comparison-20260718-hprofile-v1/manifest.json"
)
V1_MANIFEST_SHA256 = "d3ae59d971b9b6904da53d1856de8ee8c91ccfe29ff46d7ee5eb1d300bec8c5a"


def ensemble(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "TMGM_NATIVE_SCORE_ENSEMBLE_V1":
        raise ValueError(f"unsupported ensemble: {path}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "tmgm-multibank-cpu-listening-v2":
        raise ValueError("unsupported cattack listening manifest")
    if manifest.get("version") != "cattack-v2":
        raise ValueError("manifest is not cattack-v2")
    policy = manifest["policy"]
    if (
        policy.get("inference_device") != "CPU sequential"
        or policy.get("control_tracks_used_for_tuning") is not False
        or policy.get("threshold_fitting_in_this_run") is not False
        or policy.get("semantic_v3_bank") != "cattack-d3"
    ):
        raise ValueError("manifest violates the frozen no-control-tuning CPU policy")
    verify_file_ref(manifest["predictor"], "CPU predictor")
    activity_path = verify_file_ref(manifest["activity_ensemble"], "activity")
    primary_path = verify_file_ref(manifest["onset_primary_ensemble"], "onset primary")
    balanced_path = verify_file_ref(
        manifest["onset_balanced_ensemble"], "onset balanced"
    )
    audit_path = verify_file_ref(manifest["calibration_audit"], "cattack audit")
    activity = ensemble(activity_path)
    primary = ensemble(primary_path)
    balanced = ensemble(balanced_path)
    if activity["ensemble_threshold"] != -169 or len(activity["members"]) != 7:
        raise ValueError("activity artifact is not prior5+hprofile+cattack")
    if primary["ensemble_threshold"] != -486 or len(primary["members"]) != 9:
        raise ValueError("primary onset artifact is not all9 cattack at -486")
    if balanced["ensemble_threshold"] != -583 or len(balanced["members"]) != 9:
        raise ValueError("balanced onset artifact is not all9 cattack at -583")
    if [m["id"] for m in primary["members"]] != [
        m["id"] for m in balanced["members"]
    ]:
        raise ValueError("primary and balanced onset members differ")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    guard = audit["guard"]
    if (
        guard.get("test_or_control_wavs") != "not used"
        or guard.get("thresholds") != "calibration rows only"
    ):
        raise ValueError("cattack audit leakage guard failed")
    audited = audit["onset_operating_points"]["all9_hprofile_cattack"]
    if (
        audited["polyphony_matched_f1"]["threshold"] != -486
        or audited["balanced_f1"]["threshold"] != -583
    ):
        raise ValueError("manifest operating points differ from cattack audit")

    for bank, value in manifest["feature_banks"].items():
        verify_file_ref(value["binarizer"], f"{bank} binarizer")
        verify_file_ref(value["reference"], f"{bank} reference")
    for bank, value in manifest["bundles"].items():
        verify_file_ref(value["bundle"], f"{bank} bundle")
        verify_file_ref(value["sidecar"], f"{bank} bundle sidecar")
        expected = manifest["feature_banks"][bank][
            "semantic_feature_fingerprint_sha256"
        ]
        if value["semantic_feature_fingerprint_sha256"] != expected:
            raise ValueError(f"{bank} bundle semantic fingerprint declaration differs")
        if bank == "cattack-d3" and value["legacy_audit_opt_in_required"]:
            raise ValueError("cattack bank unexpectedly has a legacy feature contract")
    for key, value in manifest["models"].items():
        verify_file_ref(value, f"model {key}")
    for track, value in manifest["tracks"].items():
        wav = verify_file_ref(value["source_wav"], f"{track} source WAV")
        verify_file_ref(value["neuralnote"], f"{track} NeuralNote")
        for bank, feature in value["features"].items():
            dataset = verify_file_ref(feature["dataset"], f"{track}/{bank} dataset")
            metadata_path = verify_file_ref(
                feature["metadata"], f"{track}/{bank} metadata"
            )
            contract = inspect_dataset_contract(dataset)
            fingerprint = manifest["feature_banks"][bank][
                "semantic_feature_fingerprint_sha256"
            ]
            if contract.feature_fingerprint_sha256 != fingerprint:
                raise ValueError(f"{track}/{bank} embedded feature fingerprint differs")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["feature_semantics"]["fingerprint_sha256"] != fingerprint:
                raise ValueError(f"{track}/{bank} sidecar feature fingerprint differs")
            if metadata["input"]["sha256"] != value["source_wav"]["sha256"]:
                raise ValueError(f"{track}/{bank} source WAV SHA differs")
            if Path(metadata["input"]["path"]).resolve() != wav.resolve():
                raise ValueError(f"{track}/{bank} source WAV path differs")


def midi_event_audit(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    note_ons = 0
    note_offs = 0
    pitches: set[int] = set()
    maximum_polyphony = 0
    for track in midi.tracks:
        active: set[int] = set()
        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                note_ons += 1
                pitches.add(message.note)
                active.add(message.note)
                maximum_polyphony = max(maximum_polyphony, len(active))
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                note_offs += 1
                active.discard(message.note)
    if note_ons <= 0 or note_offs <= 0 or midi.length <= 0.0:
        raise ValueError(f"MIDI is empty or incomplete: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "type": midi.type,
        "tracks": len(midi.tracks),
        "note_ons": note_ons,
        "note_offs": note_offs,
        "distinct_pitches": len(pitches),
        "maximum_polyphony": maximum_polyphony,
        "duration_seconds": midi.length,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if sha256_file(V1_MANIFEST) != V1_MANIFEST_SHA256:
        raise ValueError("frozen hprofile-v1 manifest changed")

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
            "midi": [midi_event_audit(path) for path in midi_paths],
        }

    run = {
        "schema": "tmgm-multibank-cpu-listening-run-v2",
        "version": "cattack-v2",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "frozen_v1_manifest_unchanged": {
            "path": str(V1_MANIFEST.resolve()),
            "sha256": V1_MANIFEST_SHA256,
        },
        "device": "CPU sequential",
        "thresholds_fitted_on_control_tracks": False,
        "operating_points": {
            "activity": -169,
            "onset_primary_polyphony_matched": -486,
            "onset_balanced_recall": -583,
        },
        "tracks": run_tracks,
    }
    run_path = output / "run.json"
    write_json(run_path, run)
    readme = output / "README.md"
    readme.write_text(
        "# Frozen cattack-v2 cross-bank TM listening comparison\n\n"
        "This package uses prior5 + hprofile + cattack for activity and all8 + "
        "hprofile + cattack for onset. The primary onset operating point is the "
        "grouped-calibration-only polyphony-matched threshold -486; the optional "
        "balanced/recall files use -583. Activity uses -169.\n\n"
        "The two control WAVs were used only for inference/listening. No fitting, "
        "threshold calibration, or score-driven selection was performed on them. "
        "Inference was sequential and CPU-only. Legacy opt-in is restricted to the "
        "four authenticated pre-cattack banks; cattack uses a schema-2 dataset and "
        "v3 models with semantic fingerprint 59d6e0de...\n\n"
        "For each track, `tm-raw.mid` and `tm-stable.mid` are primary. "
        "`tm-balanced-raw.mid` and `tm-balanced-stable.mid` are the optional "
        "higher-recall comparison. `neuralnote.mid` is the teacher reference.\n",
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
