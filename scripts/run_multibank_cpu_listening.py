from __future__ import annotations

"""Run the frozen cross-bank TM comparison and render listening MIDI files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import mido
import numpy as np

from tmgm_rt.feature_contract import inspect_dataset_contract
from tmgm_rt.midi import NoteStateConfig, stabilize_frame_predictions, write_frame_predictions
from tmgm_rt.native_score_ensemble import (
    MemberSpec,
    apply_score_ensemble,
    load_score_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "artifacts/native-listening-comparison-20260718-hprofile-v1/manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_ref(value: dict[str, Any], label: str) -> Path:
    path = Path(value["path"])
    expected = str(value["sha256"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: {actual} != {expected}")
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "tmgm-multibank-cpu-listening-v1":
        raise ValueError("unsupported manifest schema")
    policy = manifest["policy"]
    if policy != {
        "allow_legacy_feature_contract": True,
        "control_tracks_used_for_tuning": False,
        "inference_device": "CPU sequential",
        "legacy_scope": "authenticated legacy TMGMMOD bundle payloads only",
        "threshold_fitting_in_this_run": False,
    }:
        raise ValueError("run policy is not the frozen CPU/no-tuning legacy-audit policy")
    verify_file_ref(manifest["predictor"], "CPU predictor")
    verify_file_ref(manifest["activity_ensemble"], "activity ensemble")
    verify_file_ref(manifest["onset_ensemble"], "onset ensemble")
    verify_file_ref(manifest["onset_recall_ensemble"], "recall onset ensemble")
    for bank, value in manifest["feature_banks"].items():
        verify_file_ref(value["binarizer"], f"{bank} binarizer")
        verify_file_ref(value["reference"], f"{bank} reference")
    for bank, value in manifest["bundles"].items():
        verify_file_ref(value["bundle"], f"{bank} bundle")
        verify_file_ref(value["sidecar"], f"{bank} bundle sidecar")
        declared = value["semantic_feature_fingerprint_sha256"]
        expected = manifest["feature_banks"][bank][
            "semantic_feature_fingerprint_sha256"
        ]
        if declared != expected:
            raise ValueError(f"{bank} bundle semantic bank declaration differs")
    for key, value in manifest["models"].items():
        verify_file_ref(value, f"model {key}")
    for track, value in manifest["tracks"].items():
        wav = verify_file_ref(value["source_wav"], f"{track} WAV")
        verify_file_ref(value["neuralnote"], f"{track} NeuralNote")
        for bank, features in value["features"].items():
            dataset = verify_file_ref(features["dataset"], f"{track}/{bank} dataset")
            metadata_path = verify_file_ref(
                features["metadata"], f"{track}/{bank} metadata"
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
                raise ValueError(f"{track}/{bank} feature source WAV differs")
            if Path(metadata["input"]["path"]).resolve() != wav.resolve():
                raise ValueError(f"{track}/{bank} feature source path differs")


def normalized_member_copy(source: Path, destination: Path, final_id: str) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("#member_id="):
            lines[index] = f"#member_id={final_id}"
            replaced = True
            break
    if not replaced:
        raise ValueError(f"member score has no member_id: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def locate_member(
    manifest: dict[str, Any], head: str, final_id: str
) -> tuple[str, str]:
    matches: list[tuple[str, str]] = []
    key = f"{head}_members"
    for bank, bundle in manifest["bundles"].items():
        aliases = bundle.get("artifact_member_aliases", {})
        for internal_id in bundle[key]:
            if internal_id.startswith("dummy_"):
                continue
            if aliases.get(internal_id, internal_id) == final_id:
                matches.append((bank, internal_id))
    if len(matches) != 1:
        raise ValueError(f"expected one {head} source for {final_id}, got {matches}")
    return matches[0]


def artifact_member_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return [str(member["id"]) for member in value["members"]]


def render_pair(
    activity_path: Path,
    onset_path: Path,
    dataset_path: Path,
    raw_path: Path,
    stable_path: Path,
) -> dict[str, Any]:
    contract = inspect_dataset_contract(dataset_path)
    activity = load_score_file(activity_path)
    onset = load_score_file(onset_path)
    if activity.metadata.frames != onset.metadata.frames:
        raise ValueError("activity/onset frame counts differ")
    raw = np.concatenate(
        (
            activity.scores >= activity.metadata.threshold,
            onset.scores >= onset.metadata.threshold,
        ),
        axis=1,
    ).astype(np.uint32)
    stable = stabilize_frame_predictions(
        raw,
        contract.outputs,
        NoteStateConfig(
            attack_frames=2,
            release_frames=4,
            retrigger_refractory_frames=6,
        ),
    )
    frame_seconds = contract.hop_size / contract.sample_rate
    write_frame_predictions(
        raw_path, raw, contract.midi_min, contract.outputs, frame_seconds, velocity=80
    )
    write_frame_predictions(
        stable_path,
        stable,
        contract.midi_min,
        contract.outputs,
        frame_seconds,
        velocity=80,
    )
    return {
        "frames": int(raw.shape[0]),
        "raw_activity_positive": int(raw[:, : contract.outputs].sum()),
        "raw_onset_positive": int(raw[:, contract.outputs :].sum()),
    }


def midi_audit(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    note_ons = sum(
        1
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    )
    if note_ons <= 0 or midi.length <= 0.0:
        raise ValueError(f"MIDI is empty: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "type": midi.type,
        "tracks": len(midi.tracks),
        "note_ons": note_ons,
        "duration_seconds": midi.length,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    output = manifest_path.parent
    predictor = Path(manifest["predictor"]["path"])
    activity_artifact = Path(manifest["activity_ensemble"]["path"])
    onset_artifact = Path(manifest["onset_ensemble"]["path"])
    recall_artifact = Path(manifest["onset_recall_ensemble"]["path"])
    activity_ids = artifact_member_ids(activity_artifact)
    onset_ids = artifact_member_ids(onset_artifact)
    run_tracks: dict[str, Any] = {}

    for track, track_value in manifest["tracks"].items():
        score_root = output / "scores" / track
        for bank, bundle_value in manifest["bundles"].items():
            bank_root = score_root / bank
            member_root = bank_root / "members"
            member_root.mkdir(parents=True, exist_ok=True)
            command = [
                str(predictor),
                track_value["features"][bank]["dataset"]["path"],
                bundle_value["bundle"]["path"],
                "--activity-output",
                str(bank_root / "activity-packaging.tsv"),
                "--onset-output",
                str(bank_root / "onset-packaging.tsv"),
                "--member-output-dir",
                str(member_root),
                "--allow-legacy-feature-contract",
            ]
            subprocess.run(command, cwd=ROOT, check=True)

        selected: dict[str, list[MemberSpec]] = {"activity": [], "onset": []}
        for head, member_ids in (("activity", activity_ids), ("onset", onset_ids)):
            for final_id in member_ids:
                bank, internal_id = locate_member(manifest, head, final_id)
                source = score_root / bank / "members" / f"{internal_id}.tsv"
                destination = score_root / "selected" / head / f"{final_id}.tsv"
                normalized_member_copy(source, destination, final_id)
                selected[head].append(MemberSpec(final_id, destination))

        anchor = Path(track_value["features"]["plain"]["dataset"]["path"])
        activity_scores = score_root / "activity-final.tsv"
        onset_scores = score_root / "onset-final.tsv"
        recall_scores = score_root / "onset-recall-heavy.tsv"
        apply_score_ensemble(activity_artifact, anchor, selected["activity"], activity_scores)
        apply_score_ensemble(onset_artifact, anchor, selected["onset"], onset_scores)
        apply_score_ensemble(recall_artifact, anchor, selected["onset"], recall_scores)

        listening_root = output / "listening" / track
        listening_root.mkdir(parents=True, exist_ok=True)
        neuralnote = listening_root / "neuralnote.mid"
        shutil.copy2(track_value["neuralnote"]["path"], neuralnote)
        primary_stats = render_pair(
            activity_scores,
            onset_scores,
            anchor,
            listening_root / "tm-raw.mid",
            listening_root / "tm-stable.mid",
        )
        recall_stats = render_pair(
            activity_scores,
            recall_scores,
            anchor,
            listening_root / "tm-recall-heavy-raw.mid",
            listening_root / "tm-recall-heavy-stable.mid",
        )
        midi_files = [
            listening_root / "tm-raw.mid",
            listening_root / "tm-stable.mid",
            listening_root / "tm-recall-heavy-raw.mid",
            listening_root / "tm-recall-heavy-stable.mid",
            neuralnote,
        ]
        run_tracks[track] = {
            "primary": primary_stats,
            "recall_heavy": recall_stats,
            "midi": [midi_audit(path) for path in midi_files],
        }

    run = {
        "schema": "tmgm-multibank-cpu-listening-run-v1",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "device": "CPU sequential",
        "explicit_legacy_audit_flag": True,
        "thresholds_fitted_on_control_tracks": False,
        "tracks": run_tracks,
    }
    run_path = output / "run.json"
    write_json(run_path, run)
    checksum_rows: list[str] = []
    for path in sorted((output / "listening").rglob("*.mid")):
        checksum_rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_rows) + "\n", encoding="ascii"
    )
    readme = output / "README.md"
    readme.write_text(
        "# Frozen cross-bank TM listening comparison\n\n"
        "This is a sequential CPU-only inference run. The control recordings were "
        "used only for listening inference: no model fitting, calibration or threshold "
        "selection used either recording. Legacy TMGMMOD files were admitted only with "
        "the explicit audit flag; every source/model/bundle/dataset is pinned by SHA-256, "
        "and every schema-2 dataset carries its semantic feature fingerprint.\n\n"
        "Primary onset operating point: calibration-only polyphony-matched threshold "
        "-465. Recall-heavy files reuse the same frozen raw scores at threshold -643.\n\n"
        "For each track, listen to `tm-raw.mid`, `tm-stable.mid`, and `neuralnote.mid`. "
        "The `tm-recall-heavy-*` pair is an optional higher-recall comparison. See "
        "`manifest.json`, `run.json`, and `checksums.sha256` for provenance.\n",
        encoding="utf-8",
    )
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
