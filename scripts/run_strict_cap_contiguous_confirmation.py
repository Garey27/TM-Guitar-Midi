from __future__ import annotations

"""Frozen contiguous confirmation for a calibration-selected strict-cap model.

This runner is intentionally fail-closed and has no implicit execution mode.
`--plan` performs read-only validation. `--execute` is required before it will
write artifacts or launch inference. The selected candidate and every score
threshold come exclusively from the grouped calibration report; contiguous
test tracks are evaluated only after that selection is frozen.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRICT_ROOT = (
    PROJECT_ROOT
    / "artifacts/native-full-natural-d3w3-hcontrast15-strict-cap"
)
CONTIGUOUS_ROOT = PROJECT_ROOT / "artifacts/contiguous-test-v1"
HPROFILE_ADDITIVE = (
    CONTIGUOUS_ROOT / "scores/night-20260718-hprofile-additive"
)
CATTACK_ADDITIVE = (
    CONTIGUOUS_ROOT / "scores/night-20260718-cattack-additive"
)

REPORT_FORMAT = "TMGM_D3_C512_STRICT_CAP_FAIR_AUDIT_V1"
CONFIRMATION_FORMAT = "TMGM_STRICT_CAP_CONTIGUOUS_CONFIRMATION_V1"
POLICY = "polyphony_matched_f1"
FEATURE_SET = "hcontrast15_d2w3"
CAPS = (16, 24, 32, 48, 64)
CURRENT_ACTIVITY_IDS = (
    "plain_c256",
    "plain_c512",
    "plain_c1024",
    "hc_c256",
    "hc_c512",
    "hprofile_c256",
    "cattack_c256",
)
CURRENT_ONSET_IDS = (
    "c256_q1",
    "c256_q2",
    "c256_q4",
    "c256_q8",
    "c256_q4_seed19",
    "c512_q4",
    "hprofile_c256",
    "c1024_q4",
    "cattack_c256",
)
STRICT_NAME = re.compile(r"^(add|replace)_cap(16|24|32|48|64)$")


@dataclass(frozen=True)
class FrozenSelection:
    candidate: str
    mode: str
    cap: int
    strict_member: str
    members: tuple[str, ...]
    ensemble_threshold: int
    member_thresholds: dict[str, int]
    calibration_f1: float
    source_artifact: Path
    model: Path


@dataclass(frozen=True)
class RunnerPaths:
    report: Path
    literal_audit: Path
    manifest: Path
    anchor_features: Path
    hprofile_pinned: Path
    cattack_pinned: Path
    activity_artifact: Path
    current_onset_artifact: Path
    predict: Path
    python: Path
    rewrite: Path
    ensemble: Path
    evaluator: Path
    output_root: Path


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, str):
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value) is None:
            raise ValueError(f"{label} must be an integer")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if float(value) != float(result):
        raise ValueError(f"{label} must be an integer")
    return result


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{label} must be finite")
    return result


def _member_ids(artifact: Mapping[str, Any]) -> tuple[str, ...]:
    members = artifact.get("members")
    if not isinstance(members, list):
        raise ValueError("ensemble artifact has no member list")
    result: list[str] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict) or not isinstance(member.get("id"), str):
            raise ValueError(f"invalid ensemble member at index {index}")
        result.append(member["id"])
    if len(result) != len(set(result)):
        raise ValueError("ensemble artifact repeats a member ID")
    return tuple(result)


def select_frozen_candidate(
    report: Mapping[str, Any],
    literal_audit: Mapping[str, Any],
    *,
    report_path: Path,
    literal_path: Path,
) -> FrozenSelection:
    if report.get("format") != REPORT_FORMAT:
        raise ValueError(f"unexpected grouped report format: {report.get('format')!r}")
    split = report.get("split", {})
    if (
        _as_int(split.get("calibration_rows"), "split.calibration_rows") != 40_800
        or _as_int(split.get("evaluation_rows"), "split.evaluation_rows") != 40_000
    ):
        raise ValueError("grouped report split geometry changed")
    guard = report.get("guard", {})
    if (
        guard.get("production") != "not modified"
        or guard.get("test_or_control_wavs") != "not used"
        or guard.get("fusion") != "forced equal mean"
    ):
        raise ValueError("grouped report guard is not the frozen fair protocol")

    if literal_audit.get("all_models_pass") is not True:
        raise ValueError("literal-count audit has not passed all models")
    literal_models = literal_audit.get("models")
    if not isinstance(literal_models, list):
        raise ValueError("literal-count audit has no model list")
    literal_by_label = {
        str(item.get("label")): item
        for item in literal_models
        if isinstance(item, dict)
    }
    expected_labels = {f"cap{cap}" for cap in CAPS}
    if set(literal_by_label) != expected_labels:
        raise ValueError(
            "literal-count audit is not final for all strict caps: "
            f"{sorted(literal_by_label)}"
        )
    embedded_literal = report.get("literal_count_audit")
    if embedded_literal != literal_audit:
        raise ValueError("grouped report did not embed the exact final literal audit")
    provenance = report.get("provenance", {})
    literal_identity = provenance.get("literal_count_audit", {})
    if literal_identity.get("sha256") != sha256(literal_path):
        raise ValueError("grouped report literal-audit SHA-256 is stale")

    selection_root = report.get("calibration_only_selection", {})
    strict_root = selection_root.get("strict_ensemble_only", {})
    selection = strict_root.get(POLICY)
    if not isinstance(selection, dict):
        raise ValueError(f"missing calibration-only strict selection for {POLICY}")
    candidate = selection.get("selected")
    if not isinstance(candidate, str):
        raise ValueError("strict selection has no candidate name")
    match = STRICT_NAME.fullmatch(candidate)
    if match is None:
        raise ValueError(f"invalid strict candidate name: {candidate!r}")
    mode, cap_text = match.groups()
    cap = int(cap_text)
    strict_member = f"strict_cap{cap}"

    ensemble_members = report.get("ensemble_members")
    ensemble_points = report.get("ensemble_operating_points")
    if not isinstance(ensemble_members, dict) or not isinstance(ensemble_points, dict):
        raise ValueError("grouped report lacks ensemble candidates")
    strict_names = [name for name in ensemble_members if STRICT_NAME.fullmatch(name)]
    if set(strict_names) != {
        f"{mode_name}_cap{value}"
        for value in CAPS
        for mode_name in ("add", "replace")
    }:
        raise ValueError("grouped report strict candidate grid is incomplete")
    calibration_f1 = {
        name: _as_float(
            ensemble_points[name][POLICY]["calibration"]["f1"],
            f"{name}.{POLICY}.calibration.f1",
        )
        for name in strict_names
    }
    maximum_f1 = max(calibration_f1.values())
    if calibration_f1[candidate] != maximum_f1:
        raise ValueError(
            "reported strict selection is not the maximum calibration "
            f"{POLICY} F1"
        )

    raw_members = selection.get("members")
    if not isinstance(raw_members, list) or not all(
        isinstance(item, str) for item in raw_members
    ):
        raise ValueError("strict selection has no valid member order")
    members = tuple(raw_members)
    if list(members) != ensemble_members.get(candidate):
        raise ValueError("strict selection member order disagrees with report registry")
    expected_members = (
        (*CURRENT_ONSET_IDS, strict_member)
        if mode == "add"
        else tuple(
            strict_member if item == "c512_q4" else item
            for item in CURRENT_ONSET_IDS
        )
    )
    if members != expected_members:
        raise ValueError(
            f"{candidate} members are not the declared add/replace transformation"
        )

    point = selection.get("point")
    if not isinstance(point, dict):
        raise ValueError("strict selection has no operating point")
    canonical_point = ensemble_points[candidate][POLICY]
    if point != canonical_point:
        raise ValueError("strict selection point disagrees with ensemble report")
    ensemble_threshold = _as_int(point.get("threshold"), "ensemble threshold")
    raw_thresholds = selection.get("member_calibration_thresholds")
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(members):
        raise ValueError("strict member-threshold registry is incomplete")
    member_thresholds = {
        member: _as_int(raw_thresholds[member], f"threshold for {member}")
        for member in members
    }

    source_artifact_value = selection.get("fit_artifact")
    if not isinstance(source_artifact_value, str):
        raise ValueError("strict selection has no fit artifact path")
    source_artifact = Path(source_artifact_value).resolve()
    if not source_artifact.is_file():
        raise FileNotFoundError(source_artifact)
    allowed_artifact_root = (report_path.parent / "heldout/ensembles").resolve()
    if allowed_artifact_root not in source_artifact.parents:
        raise ValueError("fit artifact escapes strict-cap heldout ensemble root")

    model = (
        STRICT_ROOT
        / f"ablations/c512-onset-strict-cap{cap}/onset/model.tmgmmod"
    ).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    model_identity = provenance.get("strict_models", {}).get(f"cap{cap}", {})
    if model_identity.get("path") != str(model):
        raise ValueError("grouped report selected-model path is stale")
    model_sha = sha256(model)
    if model_identity.get("sha256") != model_sha:
        raise ValueError("grouped report selected-model SHA-256 is stale")
    literal = literal_by_label[f"cap{cap}"]
    summary = literal.get("summary", {})
    if (
        literal.get("path") != str(model)
        or literal.get("file_sha256") != model_sha
        or _as_int(literal.get("header_max_included_literals"), "literal cap") != cap
        or summary.get("strict_cap_satisfied") is not True
        or _as_int(summary.get("violating_clause_count"), "literal violations") != 0
        or _as_int(summary.get("maximum"), "literal maximum") > cap
    ):
        raise ValueError("selected model failed frozen literal-cap validation")

    return FrozenSelection(
        candidate=candidate,
        mode=mode,
        cap=cap,
        strict_member=strict_member,
        members=members,
        ensemble_threshold=ensemble_threshold,
        member_thresholds=member_thresholds,
        calibration_f1=calibration_f1[candidate],
        source_artifact=source_artifact,
        model=model,
    )


def freeze_selected_artifact(
    source: Mapping[str, Any],
    selection: FrozenSelection,
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(source))
    if artifact.get("format") != "TMGM_NATIVE_SCORE_ENSEMBLE_V1":
        raise ValueError("selected fit artifact has an unexpected format")
    if artifact.get("head") != "onset" or artifact.get("fusion") != "mean":
        raise ValueError("selected fit artifact is not a mean onset ensemble")
    if _member_ids(artifact) != selection.members:
        raise ValueError("selected fit artifact member order changed")
    for member in artifact["members"]:
        member["threshold"] = selection.member_thresholds[member["id"]]
    point = report["calibration_only_selection"]["strict_ensemble_only"][POLICY][
        "point"
    ]
    artifact["ensemble_threshold"] = selection.ensemble_threshold
    calibration = artifact.setdefault("calibration", {})
    calibration["chosen"] = dict(point["calibration"])
    calibration["chosen"]["threshold"] = selection.ensemble_threshold
    calibration["selection_metric"] = (
        "maximum grouped-calibration F1 with predicted polyphony <= target "
        "polyphony; candidate selected by grouped calibration only"
    )
    artifact["operating_point_provenance"] = {
        "policy": POLICY,
        "candidate": selection.candidate,
        "selection_rows": "40,800 grouped calibration rows only",
        "contiguous_test_tracks_used": False,
        "production_modified": False,
        "ensemble_threshold": selection.ensemble_threshold,
        "member_calibration_thresholds": selection.member_thresholds,
        "source_fit_artifact": str(selection.source_artifact),
        "source_fit_artifact_sha256": sha256(selection.source_artifact),
        "source_grouped_report": str(report_path.resolve()),
        "source_grouped_report_sha256": sha256(report_path),
        "source_calibration_point": point["calibration"],
        "source_disjoint_grouped_evaluation": point["evaluation"],
    }
    return artifact


def parse_score_header(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().rstrip("\r\n") != "#TMGM_SCORES_V1":
            raise ValueError(f"not a native score file: {path}")
        for line in stream:
            if not line.startswith("#"):
                break
            key, separator, value = line[1:].rstrip("\r\n").partition("=")
            if not separator or key in result:
                raise ValueError(f"malformed score metadata: {path}")
            result[key] = value
    return result


def pinned_score(paths: RunnerPaths, head: str, member: str, key: str) -> Path:
    root = paths.cattack_pinned if member == "cattack_c256" else paths.hprofile_pinned
    return root / head / member / f"{key}.{head}.tsv"


def validate_pinned_score(
    path: Path, *, head: str, member: str, threshold: int
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = parse_score_header(path)
    if (
        metadata.get("head") != head
        or metadata.get("member_id") != member
        or _as_int(metadata.get("threshold"), f"{path} threshold") != threshold
    ):
        raise ValueError(f"pinned member metadata disagrees with artifact: {path}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def run_command(
    command: Sequence[Path | str], stage: str, records: list[dict[str, Any]], root: Path
) -> None:
    argv = [str(value) for value in command]
    started = datetime.now(timezone.utc)
    tick = time.perf_counter()
    print(f"[{len(records) + 1:03d}] {stage}", flush=True)
    print(command_text(argv), flush=True)
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    records.append(
        {
            "sequence": len(records) + 1,
            "stage": stage,
            "command": argv,
            "command_line": command_text(argv),
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - tick,
            "return_code": completed.returncode,
            "output_tail": completed.stdout[-8000:] if completed.stdout else "",
        }
    )
    atomic_json(
        root / "execution-log.json",
        {
            "format": "TMGM_STRICT_SEQUENTIAL_EXECUTION_LOG_V1",
            "parallelism": 1,
            "test_threshold_tuning": False,
            "commands": records,
        },
    )
    (root / "commands.txt").write_text(
        "\n".join(item["command_line"] for item in records) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)


def ensemble_command(
    paths: RunnerPaths,
    *,
    artifact: Path,
    dataset: Path,
    head: str,
    members: Sequence[str],
    thresholds: Mapping[str, int],
    key: str,
    strict_member: str | None,
    strict_score: Path | None,
    output: Path,
) -> list[Path | str]:
    command: list[Path | str] = [
        paths.python,
        paths.ensemble,
        "apply",
        "--artifact",
        artifact,
        "--dataset",
        dataset,
    ]
    for member in members:
        if member == strict_member:
            if strict_score is None:
                raise ValueError("strict member score was not provided")
            score = strict_score
        else:
            score = pinned_score(paths, head, member, key)
            validate_pinned_score(
                score, head=head, member=member, threshold=thresholds[member]
            )
        command.extend(["--member", f"{member}={score}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    command.extend(["--output", output])
    return command


def evaluator_command(
    paths: RunnerPaths, score_root: Path, output: Path
) -> list[Path | str]:
    return [
        paths.python,
        paths.evaluator,
        "--manifest",
        paths.manifest,
        "--feature-set",
        FEATURE_SET,
        "--scores-root",
        score_root,
        "--training-onset-delay-frames",
        "3",
        "--onset-width-frames",
        "3",
        "--target-aligned-tolerances",
        "2",
        "3",
        "4",
        "--wall-clock-tolerances",
        "2",
        "3",
        "4",
        "6",
        "--retrigger-silence-frames",
        "3",
        "--chord-window-frames",
        "3",
        "--low-midi-max",
        "59",
        "--batch-rows",
        "4096",
        "--output",
        output,
    ]


def validate_static_inputs(paths: RunnerPaths) -> None:
    required_files = (
        paths.report,
        paths.literal_audit,
        paths.manifest,
        paths.activity_artifact,
        paths.current_onset_artifact,
        paths.predict,
        paths.python,
        paths.rewrite,
        paths.ensemble,
        paths.evaluator,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing confirmation input: " + ", ".join(missing))
    for directory in (
        paths.anchor_features,
        paths.hprofile_pinned,
        paths.cattack_pinned,
    ):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    protected = (
        paths.anchor_features.resolve(),
        paths.hprofile_pinned.resolve(),
        paths.cattack_pinned.resolve(),
        (STRICT_ROOT / "ablations").resolve(),
    )
    resolved_output = paths.output_root.resolve()
    if any(
        resolved_output == source or source in resolved_output.parents
        for source in protected
    ):
        raise ValueError("output root overlaps a frozen input/production artifact")


def artifact_thresholds(
    artifact: Mapping[str, Any], expected_ids: Sequence[str], expected_ensemble: int
) -> dict[str, int]:
    if _member_ids(artifact) != tuple(expected_ids):
        raise ValueError("frozen artifact member order changed")
    if _as_int(artifact.get("ensemble_threshold"), "artifact threshold") != expected_ensemble:
        raise ValueError("frozen artifact ensemble threshold changed")
    return {
        str(member["id"]): _as_int(
            member.get("threshold"), f"threshold for {member.get('id')}"
        )
        for member in artifact["members"]
    }


def build_plan(paths: RunnerPaths) -> tuple[dict[str, Any], FrozenSelection]:
    validate_static_inputs(paths)
    report = read_object(paths.report)
    literal = read_object(paths.literal_audit)
    selection = select_frozen_candidate(
        report,
        literal,
        report_path=paths.report,
        literal_path=paths.literal_audit,
    )
    activity = read_object(paths.activity_artifact)
    current_onset = read_object(paths.current_onset_artifact)
    if activity.get("head") != "activity" or activity.get("fusion") != "mean":
        raise ValueError("current activity artifact is not a mean activity ensemble")
    activity_thresholds = artifact_thresholds(activity, CURRENT_ACTIVITY_IDS, -169)
    if current_onset.get("head") != "onset" or current_onset.get("fusion") != "mean":
        raise ValueError("current onset artifact is not a mean onset ensemble")
    current_onset_thresholds = artifact_thresholds(
        current_onset, CURRENT_ONSET_IDS, -486
    )
    source_selected = read_object(selection.source_artifact)
    frozen_selected = freeze_selected_artifact(
        source_selected, selection, report, report_path=paths.report
    )

    manifest = read_object(paths.manifest)
    tracks = manifest.get("tracks")
    if manifest.get("schema") != "tmgm-contiguous-eval-manifest-v1":
        raise ValueError("unexpected contiguous manifest schema")
    if not isinstance(tracks, list) or len(tracks) != 6:
        raise ValueError("contiguous confirmation requires the frozen six tracks")
    track_rows: list[dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, dict) or not isinstance(track.get("key"), str):
            raise ValueError("invalid contiguous track entry")
        key = track["key"]
        dataset = paths.anchor_features / f"{key}.tmgd"
        metadata = dataset.with_suffix(dataset.suffix + ".json")
        if not dataset.is_file() or not metadata.is_file():
            raise FileNotFoundError(dataset)
        declared = track.get("feature_sets", {}).get(FEATURE_SET, {})
        declared_dataset = (paths.manifest.parent / declared.get("dataset", "")).resolve()
        if declared_dataset != dataset.resolve():
            raise ValueError(f"manifest anchor path changed for {key}")
        for member, threshold in activity_thresholds.items():
            validate_pinned_score(
                pinned_score(paths, "activity", member, key),
                head="activity",
                member=member,
                threshold=threshold,
            )
        for member, threshold in current_onset_thresholds.items():
            validate_pinned_score(
                pinned_score(paths, "onset", member, key),
                head="onset",
                member=member,
                threshold=threshold,
            )
        # Selected ensembles share every old member threshold with the pinned
        # current all9 bank. Only the strict member is inferred below.
        for member, threshold in selection.member_thresholds.items():
            if member != selection.strict_member:
                validate_pinned_score(
                    pinned_score(paths, "onset", member, key),
                    head="onset",
                    member=member,
                    threshold=threshold,
                )
        track_rows.append(
            {
                "key": key,
                "source": track.get("source"),
                "anchor_dataset": identity(dataset),
                "anchor_metadata": identity(metadata),
            }
        )

    plan = {
        "format": CONFIRMATION_FORMAT,
        "status": "frozen plan; no contiguous score consulted for selection",
        "selection": {
            "policy": POLICY,
            "candidate": selection.candidate,
            "mode": selection.mode,
            "cap": selection.cap,
            "strict_member": selection.strict_member,
            "members": list(selection.members),
            "ensemble_threshold": selection.ensemble_threshold,
            "member_calibration_thresholds": selection.member_thresholds,
            "calibration_f1": selection.calibration_f1,
            "source_fit_artifact": identity(selection.source_artifact),
            "strict_model": identity(selection.model),
        },
        "current_baseline": {
            "activity_members": list(CURRENT_ACTIVITY_IDS),
            "activity_threshold": -169,
            "activity_member_thresholds": activity_thresholds,
            "onset_members": list(CURRENT_ONSET_IDS),
            "onset_threshold": -486,
            "onset_member_thresholds": current_onset_thresholds,
        },
        "frozen_selected_artifact": frozen_selected,
        "tracks": track_rows,
        "protocol": {
            "feature_set": FEATURE_SET,
            "training_onset_delay_frames": 3,
            "onset_width_frames": 3,
            "target_aligned_tolerances": [2, 3, 4],
            "wall_clock_tolerances": [2, 3, 4, 6],
            "retrigger_silence_frames": 3,
            "chord_window_frames": 3,
            "low_midi_max": 59,
            "batch_rows": 4096,
        },
        "guard": {
            "candidate_selection": "grouped calibration report only",
            "test_threshold_tuning": False,
            "new_model_inference": [selection.strict_member],
            "other_member_scores": "reused frozen hprofile/cattack additive scores",
            "production_modified": False,
            "execution_parallelism": 1,
        },
        "provenance": {
            "grouped_report": identity(paths.report),
            "literal_audit": identity(paths.literal_audit),
            "contiguous_manifest": identity(paths.manifest),
            "activity_artifact": identity(paths.activity_artifact),
            "current_onset_artifact": identity(paths.current_onset_artifact),
            "runner": identity(Path(__file__)),
        },
    }
    return plan, selection


def execute(paths: RunnerPaths, plan: dict[str, Any], selection: FrozenSelection) -> None:
    if paths.output_root.exists() and any(paths.output_root.iterdir()):
        raise FileExistsError(
            f"confirmation output root is not empty: {paths.output_root}"
        )
    paths.output_root.mkdir(parents=True, exist_ok=False)
    frozen_root = paths.output_root / "frozen"
    activity_frozen = frozen_root / "activity-current-cattack-minus169.ensemble.json"
    current_onset_frozen = (
        frozen_root / "onset-current9-cattack-minus486.ensemble.json"
    )
    selected_onset_frozen = (
        frozen_root / f"onset-{selection.candidate}-{POLICY}.ensemble.json"
    )
    atomic_copy(paths.activity_artifact, activity_frozen)
    atomic_copy(paths.current_onset_artifact, current_onset_frozen)
    atomic_json(selected_onset_frozen, plan.pop("frozen_selected_artifact"))
    plan["frozen_artifacts"] = {
        "activity": identity(activity_frozen),
        "current_onset": identity(current_onset_frozen),
        "selected_onset": identity(selected_onset_frozen),
    }
    atomic_json(paths.output_root / "frozen-plan.json", plan)

    activity_thresholds = plan["current_baseline"]["activity_member_thresholds"]
    current_thresholds = plan["current_baseline"]["onset_member_thresholds"]
    tracks = read_object(paths.manifest)["tracks"]
    records: list[dict[str, Any]] = []
    strict_scores: dict[str, Path] = {}

    # The only new native-model inference in this confirmation.
    for track in tracks:
        key = track["key"]
        dataset = paths.anchor_features / f"{key}.tmgd"
        raw = (
            paths.output_root
            / "members/raw/onset"
            / selection.strict_member
            / f"{key}.onset.tsv"
        )
        raw.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                paths.predict,
                dataset,
                selection.model,
                "--output",
                raw,
                "--allow-legacy-feature-contract",
            ],
            f"infer:{selection.strict_member}:{key}",
            records,
            paths.output_root,
        )
        rewritten = (
            paths.output_root
            / "members/frozen-threshold/onset"
            / selection.strict_member
            / f"{key}.onset.tsv"
        )
        run_command(
            [
                paths.python,
                paths.rewrite,
                raw,
                rewritten,
                "--threshold",
                str(selection.member_thresholds[selection.strict_member]),
                "--member-id",
                selection.strict_member,
            ],
            f"rewrite:{selection.strict_member}:{key}",
            records,
            paths.output_root,
        )
        strict_scores[key] = rewritten

    candidates = (
        (
            "current9_hprofile_cattack",
            current_onset_frozen,
            CURRENT_ONSET_IDS,
            current_thresholds,
            None,
        ),
        (
            selection.candidate,
            selected_onset_frozen,
            selection.members,
            selection.member_thresholds,
            selection.strict_member,
        ),
    )
    evaluations: dict[str, Path] = {}
    for candidate, onset_artifact, onset_ids, onset_thresholds, strict_member in candidates:
        score_root = paths.output_root / "evaluations" / candidate / FEATURE_SET
        for track in tracks:
            key = track["key"]
            dataset = paths.anchor_features / f"{key}.tmgd"
            run_command(
                ensemble_command(
                    paths,
                    artifact=activity_frozen,
                    dataset=dataset,
                    head="activity",
                    members=CURRENT_ACTIVITY_IDS,
                    thresholds=activity_thresholds,
                    key=key,
                    strict_member=None,
                    strict_score=None,
                    output=score_root / f"{key}.activity.tsv",
                ),
                f"ensemble:{candidate}:activity:{key}",
                records,
                paths.output_root,
            )
            run_command(
                ensemble_command(
                    paths,
                    artifact=onset_artifact,
                    dataset=dataset,
                    head="onset",
                    members=onset_ids,
                    thresholds=onset_thresholds,
                    key=key,
                    strict_member=strict_member,
                    strict_score=(strict_scores[key] if strict_member else None),
                    output=score_root / f"{key}.onset.tsv",
                ),
                f"ensemble:{candidate}:onset:{key}",
                records,
                paths.output_root,
            )
        evaluation = score_root / "evaluation.json"
        run_command(
            evaluator_command(paths, score_root, evaluation),
            f"evaluate:{candidate}",
            records,
            paths.output_root,
        )
        evaluations[candidate] = evaluation

    final_report = {
        "format": CONFIRMATION_FORMAT,
        "status": "complete; frozen selection evaluated without test tuning",
        "selection": plan["selection"],
        "protocol": plan["protocol"],
        "guard": plan["guard"],
        "frozen_plan": identity(paths.output_root / "frozen-plan.json"),
        "evaluations": {
            name: {
                "identity": identity(path),
                "report": read_object(path),
            }
            for name, path in evaluations.items()
        },
        "execution_log": identity(paths.output_root / "execution-log.json"),
    }
    atomic_json(paths.output_root / "contiguous-confirmation.json", final_report)


def resolve_paths(args: argparse.Namespace) -> RunnerPaths:
    return RunnerPaths(
        report=args.grouped_report.resolve(),
        literal_audit=args.literal_audit.resolve(),
        manifest=args.manifest.resolve(),
        anchor_features=args.anchor_features.resolve(),
        hprofile_pinned=args.hprofile_pinned.resolve(),
        cattack_pinned=args.cattack_pinned.resolve(),
        activity_artifact=args.activity_artifact.resolve(),
        current_onset_artifact=args.current_onset_artifact.resolve(),
        predict=args.predict_exe.resolve(),
        python=args.python_exe.resolve(),
        rewrite=args.rewrite_script.resolve(),
        ensemble=args.ensemble_script.resolve(),
        evaluator=args.evaluator_script.resolve(),
        output_root=args.output_root.resolve(),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Freeze the calibration-selected strict-cap onset candidate, infer "
            "only that member on contiguous-test-v1, and compare it with current9."
        )
    )
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="read-only validation")
    mode.add_argument("--execute", action="store_true", help="run frozen confirmation")
    result.add_argument(
        "--grouped-report",
        type=Path,
        default=STRICT_ROOT / "threshold-audit/strict-cap-audit.json",
    )
    result.add_argument(
        "--literal-audit",
        type=Path,
        default=STRICT_ROOT / "threshold-audit/literal-count-audit.json",
    )
    result.add_argument(
        "--manifest", type=Path, default=CONTIGUOUS_ROOT / "manifest.json"
    )
    result.add_argument(
        "--anchor-features",
        type=Path,
        default=CONTIGUOUS_ROOT / "features/hcontrast15_d2w3",
    )
    result.add_argument(
        "--hprofile-pinned",
        type=Path,
        default=HPROFILE_ADDITIVE / "members-validation-threshold",
    )
    result.add_argument(
        "--cattack-pinned",
        type=Path,
        default=CATTACK_ADDITIVE / "members-validation-threshold",
    )
    result.add_argument(
        "--activity-artifact",
        type=Path,
        default=CATTACK_ADDITIVE / "activity-all7-cattack-frozen.ensemble.json",
    )
    result.add_argument(
        "--current-onset-artifact",
        type=Path,
        default=CATTACK_ADDITIVE / "onset-all9-cattack-polyphony-matched.ensemble.json",
    )
    result.add_argument(
        "--predict-exe",
        type=Path,
        default=PROJECT_ROOT / "native/build-contract-cuda/Release/tmgm_predict.exe",
    )
    result.add_argument(
        "--python-exe", type=Path, default=PROJECT_ROOT / ".venv/Scripts/python.exe"
    )
    result.add_argument(
        "--rewrite-script",
        type=Path,
        default=(
            CONTIGUOUS_ROOT
            / "scores/night-20260718/rewrite_member_thresholds.py"
        ),
    )
    result.add_argument(
        "--ensemble-script",
        type=Path,
        default=PROJECT_ROOT / "scripts/native_score_ensemble.py",
    )
    result.add_argument(
        "--evaluator-script",
        type=Path,
        default=PROJECT_ROOT / "scripts/evaluate_contiguous_tracks.py",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=STRICT_ROOT / "threshold-audit/contiguous-confirmation",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = resolve_paths(args)
    plan, selection = build_plan(paths)
    if args.plan:
        printable = dict(plan)
        printable.pop("frozen_selected_artifact", None)
        print(json.dumps(printable, indent=2, sort_keys=True))
        return 0
    execute(paths, plan, selection)
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate": selection.candidate,
                "output": str(paths.output_root / "contiguous-confirmation.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
