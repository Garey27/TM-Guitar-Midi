from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_strict_cap_contiguous_confirmation as confirmation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, Path, Path, dict]:
    strict_root = tmp_path / "strict"
    report_root = strict_root / "threshold-audit"
    report_root.mkdir(parents=True)
    monkeypatch.setattr(confirmation, "STRICT_ROOT", strict_root)

    literal_models = []
    model_identities = {}
    for cap in confirmation.CAPS:
        model = (
            strict_root
            / f"ablations/c512-onset-strict-cap{cap}/onset/model.tmgmmod"
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(f"strict-cap-{cap}".encode())
        digest = _sha(model)
        literal_models.append(
            {
                "label": f"cap{cap}",
                "path": str(model.resolve()),
                "file_sha256": digest,
                "header_max_included_literals": cap,
                "summary": {
                    "strict_cap_satisfied": True,
                    "violating_clause_count": 0,
                    "maximum": cap,
                },
            }
        )
        model_identities[f"cap{cap}"] = {
            "path": str(model.resolve()),
            "bytes": model.stat().st_size,
            "sha256": digest,
        }
    literal = {"models": literal_models, "all_models_pass": True}
    literal_path = report_root / "literal-count-audit.json"
    literal_path.write_text(json.dumps(literal), encoding="utf-8")

    candidates: dict[str, list[str]] = {}
    points: dict[str, dict] = {}
    for cap in confirmation.CAPS:
        strict = f"strict_cap{cap}"
        candidates[f"add_cap{cap}"] = [*confirmation.CURRENT_ONSET_IDS, strict]
        candidates[f"replace_cap{cap}"] = [
            strict if item == "c512_q4" else item
            for item in confirmation.CURRENT_ONSET_IDS
        ]
        for mode in ("add", "replace"):
            name = f"{mode}_cap{cap}"
            score = 0.40 + cap / 10_000
            if name == "replace_cap24":
                score = 0.75
            points[name] = {
                confirmation.POLICY: {
                    "threshold": -321,
                    "calibration": {"f1": score},
                    "evaluation": {"f1": 0.01},
                }
            }

    selected = "replace_cap24"
    selected_members = candidates[selected]
    source_artifact_path = (
        report_root / "heldout/ensembles/onset-strict-cap-replace_cap24/ensemble.json"
    )
    source_artifact_path.parent.mkdir(parents=True)
    source_artifact = {
        "format": "TMGM_NATIVE_SCORE_ENSEMBLE_V1",
        "head": "onset",
        "fusion": "mean",
        "ensemble_threshold": -999,
        "members": [
            {"id": member, "threshold": index, "robust_scale": 1.0}
            for index, member in enumerate(selected_members)
        ],
        "calibration": {},
    }
    source_artifact_path.write_text(json.dumps(source_artifact), encoding="utf-8")
    selected_point = points[selected][confirmation.POLICY]
    thresholds = {member: 100 + index for index, member in enumerate(selected_members)}
    report = {
        "format": confirmation.REPORT_FORMAT,
        "split": {"calibration_rows": 40_800, "evaluation_rows": 40_000},
        "guard": {
            "production": "not modified",
            "test_or_control_wavs": "not used",
            "fusion": "forced equal mean",
        },
        "literal_count_audit": literal,
        "ensemble_members": candidates,
        "ensemble_operating_points": points,
        "calibration_only_selection": {
            "strict_ensemble_only": {
                confirmation.POLICY: {
                    "selected": selected,
                    "point": selected_point,
                    "members": selected_members,
                    "fit_artifact": str(source_artifact_path.resolve()),
                    "member_calibration_thresholds": thresholds,
                }
            }
        },
        "provenance": {
            "literal_count_audit": {"sha256": _sha(literal_path)},
            "strict_models": model_identities,
        },
    }
    report_path = report_root / "strict-cap-audit.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report, literal, report_path, literal_path, source_artifact


def test_selection_is_calibration_only_and_freezes_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report, literal, report_path, literal_path, source = _fixture(
        tmp_path, monkeypatch
    )
    selected = confirmation.select_frozen_candidate(
        report,
        literal,
        report_path=report_path,
        literal_path=literal_path,
    )
    assert selected.candidate == "replace_cap24"
    assert selected.strict_member == "strict_cap24"
    assert selected.members[5] == "strict_cap24"
    assert "c512_q4" not in selected.members
    assert selected.calibration_f1 == 0.75

    frozen = confirmation.freeze_selected_artifact(
        source, selected, report, report_path=report_path
    )
    assert frozen["ensemble_threshold"] == -321
    assert {
        member["id"]: member["threshold"] for member in frozen["members"]
    } == selected.member_thresholds
    provenance = frozen["operating_point_provenance"]
    assert provenance["policy"] == confirmation.POLICY
    assert provenance["contiguous_test_tracks_used"] is False
    assert provenance["production_modified"] is False


def test_selection_rejects_reported_candidate_that_is_not_calibration_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    report, literal, report_path, literal_path, _ = _fixture(tmp_path, monkeypatch)
    selection = report["calibration_only_selection"]["strict_ensemble_only"][
        confirmation.POLICY
    ]
    selection["selected"] = "add_cap16"
    selection["point"] = report["ensemble_operating_points"]["add_cap16"][
        confirmation.POLICY
    ]
    selection["members"] = report["ensemble_members"]["add_cap16"]
    selection["member_calibration_thresholds"] = {
        member: index for index, member in enumerate(selection["members"])
    }
    with pytest.raises(ValueError, match="maximum calibration"):
        confirmation.select_frozen_candidate(
            report,
            literal,
            report_path=report_path,
            literal_path=literal_path,
        )


def test_cli_has_no_implicit_execution_mode():
    with pytest.raises(SystemExit):
        confirmation.parser().parse_args([])


def test_integer_parser_accepts_score_header_text_but_rejects_noncanonical_values():
    assert confirmation._as_int("73", "threshold") == 73
    assert confirmation._as_int("-492", "threshold") == -492
    for value in ("73.0", " 73", "+73", "01", True):
        with pytest.raises(ValueError, match="must be an integer"):
            confirmation._as_int(value, "threshold")
