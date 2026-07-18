from __future__ import annotations

"""Run native TM ablations serially and leave reproducible artifacts.

The runner deliberately treats the native trainer, predictor, and streaming
evaluator as separate processes.  Consequently only one GPU process exists at
a time, and the Python process never loads a TMGMDAT corpus into memory.
"""

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from tmgm_rt.feature_contract import inspect_dataset_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = PROJECT_ROOT / "native" / "build-vs2019-cuda129-earlystop" / "Release"
RUN_CONFIG_FORMAT = "TMGM_NATIVE_ABLATION_V1"
MODEL_MAGIC = b"TMGMMOD\0"
MODEL_HEADER_BYTES = 256
MODEL_LEGACY_VERSIONS = frozenset((1, 2))
MODEL_SUPPORTED_VERSIONS = frozenset((1, 2, 3))
MODEL_FEATURE_FINGERPRINT_OFFSET = 192
MODEL_FEATURE_FINGERPRINT_BYTES = 32
MODEL_CHECKSUM_OFFSET = 160
MODEL_CHECKSUM_BYTES = 32
MODEL_WORD_BITS = 32
MODEL_KNOWN_FLAGS = 0x0F
SCORE_MAGIC = "#TMGM_SCORES_V1"

PARAMETER_ORDER = (
    "epochs",
    "clauses",
    "threshold",
    "specificity",
    "negative_samples",
    "max_literals",
    "samples_per_launch",
    "validation_patience",
    "seed",
)
BOOLEAN_PARAMETERS = (
    "onset_sustain_hard_negatives",
    "onset_sustain_hard_negative_weight_only",
    "rotate_output_update_order",
)
OPTIONAL_NUMERIC_PARAMETERS = (
    "onset_sustain_hard_negative_probability",
)
PARAMETER_OPTIONS = {
    key: "--" + key.replace("_", "-") for key in PARAMETER_ORDER
}
BUILTIN_COMMON: dict[str, int | float] = {
    "epochs": 30,
    "clauses": 256,
    "threshold": 128,
    "max_literals": 64,
    "samples_per_launch": 128,
    "validation_patience": 6,
}
BUILTIN_HEADS: dict[str, dict[str, int | float | bool]] = {
    "activity": {
        "specificity": 5.0,
        "negative_samples": 8.0,
        "seed": 20260718,
        "onset_sustain_hard_negatives": False,
        "onset_sustain_hard_negative_probability": 1.0,
        "onset_sustain_hard_negative_weight_only": False,
        "rotate_output_update_order": False,
    },
    "onset": {
        "specificity": 4.0,
        "negative_samples": 4.0,
        "seed": 20270718,
        "onset_sustain_hard_negatives": False,
        "onset_sustain_hard_negative_probability": 1.0,
        "onset_sustain_hard_negative_weight_only": False,
        "rotate_output_update_order": False,
    },
}
HEADS = ("activity", "onset")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    heads: dict[str, dict[str, int | float | bool]]


@dataclass(frozen=True)
class RunnerPaths:
    train_dataset: Path
    validation_dataset: Path
    output_root: Path
    train_executable: Path
    predict_executable: Path
    python_executable: Path
    evaluator: Path
    feature_fingerprint_sha256: str | None = None
    allow_legacy_feature_contract: bool = False


class CommandError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _split_level(value: Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    allowed = {
        *PARAMETER_ORDER,
        *BOOLEAN_PARAMETERS,
        *OPTIONAL_NUMERIC_PARAMETERS,
        *HEADS,
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {label}: {unknown}")
    common = {
        key: value[key]
        for key in (
            *PARAMETER_ORDER,
            *BOOLEAN_PARAMETERS,
            *OPTIONAL_NUMERIC_PARAMETERS,
        )
        if key in value
    }
    per_head: dict[str, dict[str, Any]] = {}
    for head in HEADS:
        if head in value:
            head_values = _mapping(value[head], f"{label}.{head}")
            unknown_head = sorted(
                set(head_values) - set(PARAMETER_ORDER) - set(BOOLEAN_PARAMETERS)
                - set(OPTIONAL_NUMERIC_PARAMETERS)
            )
            if unknown_head:
                raise ValueError(
                    f"unknown keys in {label}.{head}: {unknown_head}"
                )
            per_head[head] = head_values
    return common, per_head


def _validated_parameters(
    values: Mapping[str, Any], label: str
) -> dict[str, int | float | bool]:
    missing = [key for key in PARAMETER_ORDER if key not in values]
    if missing:
        raise ValueError(f"missing parameters in {label}: {missing}")

    result: dict[str, int | float | bool] = {}
    integer_keys = {
        "epochs",
        "clauses",
        "threshold",
        "max_literals",
        "samples_per_launch",
        "validation_patience",
        "seed",
    }
    for key in PARAMETER_ORDER:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{key} must be numeric")
        if key in integer_keys:
            if not isinstance(value, int):
                raise ValueError(f"{label}.{key} must be an integer")
            result[key] = value
        else:
            result[key] = float(value)

    for key in ("epochs", "clauses", "threshold", "max_literals"):
        if int(result[key]) <= 0:
            raise ValueError(f"{label}.{key} must be positive")
    if not 1 <= int(result["samples_per_launch"]) <= 512:
        raise ValueError(f"{label}.samples_per_launch must be in [1, 512]")
    if int(result["validation_patience"]) <= 0:
        raise ValueError(f"{label}.validation_patience must be positive")
    if int(result["seed"]) < 0 or int(result["seed"]) > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{label}.seed must fit uint64")
    if float(result["specificity"]) <= 1.0:
        raise ValueError(f"{label}.specificity must be greater than 1")
    if float(result["negative_samples"]) < 0.0:
        raise ValueError(f"{label}.negative_samples must be non-negative")
    hard_negatives = values.get("onset_sustain_hard_negatives", False)
    if not isinstance(hard_negatives, bool):
        raise ValueError(
            f"{label}.onset_sustain_hard_negatives must be boolean"
        )
    result["onset_sustain_hard_negatives"] = hard_negatives
    weight_only = values.get(
        "onset_sustain_hard_negative_weight_only", False
    )
    if not isinstance(weight_only, bool):
        raise ValueError(
            f"{label}.onset_sustain_hard_negative_weight_only must be boolean"
        )
    if weight_only and not hard_negatives:
        raise ValueError(
            f"{label}.onset_sustain_hard_negative_weight_only requires hard negatives"
        )
    result["onset_sustain_hard_negative_weight_only"] = weight_only
    hard_probability = values.get(
        "onset_sustain_hard_negative_probability", 1.0
    )
    if (
        isinstance(hard_probability, bool)
        or not isinstance(hard_probability, (int, float))
        or not 0.0 < float(hard_probability) <= 1.0
    ):
        raise ValueError(
            f"{label}.onset_sustain_hard_negative_probability must be in (0, 1]"
        )
    result["onset_sustain_hard_negative_probability"] = float(
        hard_probability
    )
    rotate_order = values.get("rotate_output_update_order", False)
    if not isinstance(rotate_order, bool):
        raise ValueError(
            f"{label}.rotate_output_update_order must be boolean"
        )
    result["rotate_output_update_order"] = rotate_order
    return result


def load_experiments(path: str | Path) -> list[ExperimentSpec]:
    """Load and fully resolve a compact JSON ablation manifest."""
    manifest_path = Path(path)
    root = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "manifest"
    )
    unknown_root = sorted(set(root) - {"defaults", "experiments"})
    if unknown_root:
        raise ValueError(f"unknown manifest keys: {unknown_root}")

    defaults = _mapping(root.get("defaults", {}), "defaults")
    default_common, default_heads = _split_level(defaults, "defaults")
    raw_experiments = root.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("manifest.experiments must be a non-empty JSON array")

    experiments: list[ExperimentSpec] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_experiments):
        item = _mapping(raw, f"experiments[{index}]")
        name = item.pop("name", None)
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"experiments[{index}].name must match {NAME_PATTERN.pattern}"
            )
        if name in seen_names:
            raise ValueError(f"duplicate experiment name: {name}")
        seen_names.add(name)
        experiment_common, experiment_heads = _split_level(
            item, f"experiments[{index}]"
        )

        resolved_heads: dict[str, dict[str, int | float | bool]] = {}
        for head in HEADS:
            resolved: dict[str, Any] = {}
            resolved.update(BUILTIN_COMMON)
            resolved.update(BUILTIN_HEADS[head])
            resolved.update(default_common)
            resolved.update(default_heads.get(head, {}))
            resolved.update(experiment_common)
            resolved.update(experiment_heads.get(head, {}))
            resolved_heads[head] = _validated_parameters(
                resolved, f"experiment {name}.{head}"
            )
            if (
                head == "activity"
                and resolved_heads[head]["onset_sustain_hard_negatives"]
            ):
                raise ValueError(
                    f"experiment {name}.activity cannot enable onset sustain hard negatives"
                )
        experiments.append(ExperimentSpec(name=name, heads=resolved_heads))
    return experiments


def _number_text(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def build_train_command(
    executable: str | Path,
    train_dataset: str | Path,
    validation_dataset: str | Path,
    head: str,
    parameters: Mapping[str, int | float | bool],
    score_output: str | Path,
    model_output: str | Path,
    *,
    allow_legacy_feature_contract: bool = False,
) -> list[str]:
    if head not in HEADS:
        raise ValueError(f"unsupported head: {head}")
    command = [
        str(executable),
        str(train_dataset),
        "--validation",
        str(validation_dataset),
        "--head",
        head,
    ]
    for key in PARAMETER_ORDER:
        command.extend([PARAMETER_OPTIONS[key], _number_text(parameters[key])])
    if parameters.get("onset_sustain_hard_negatives", False):
        command.extend(
            [
                "--onset-sustain-hard-negative-probability",
                _number_text(
                    parameters["onset_sustain_hard_negative_probability"]
                ),
            ]
        )
        if parameters.get("onset_sustain_hard_negative_weight_only", False):
            command.append("--onset-sustain-hard-negative-weight-only")
    if parameters.get("rotate_output_update_order", False):
        command.append("--rotate-output-update-order")
    if allow_legacy_feature_contract:
        command.append("--allow-legacy-feature-contract")
    command.extend(["--output", str(score_output), "--model", str(model_output)])
    return command


def build_predict_command(
    executable: str | Path,
    validation_dataset: str | Path,
    model: str | Path,
    score_output: str | Path,
    *,
    allow_legacy_feature_contract: bool = False,
) -> list[str]:
    command = [
        str(executable),
        str(validation_dataset),
        str(model),
        "--output",
        str(score_output),
    ]
    if allow_legacy_feature_contract:
        command.append("--allow-legacy-feature-contract")
    return command


def build_evaluate_command(
    python_executable: str | Path,
    evaluator: str | Path,
    validation_dataset: str | Path,
    score_files: Iterable[str | Path],
    output: str | Path,
) -> list[str]:
    return [
        str(python_executable),
        str(evaluator),
        "--dataset",
        str(validation_dataset),
        "--scores",
        *(str(path) for path in score_files),
        "--output",
        str(output),
    ]


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _model_checksum_matches(path: Path, stored: bytes) -> bool:
    digest = hashlib.sha256()
    offset = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            mutable = bytearray(chunk)
            begin = max(offset, MODEL_CHECKSUM_OFFSET)
            end = min(
                offset + len(mutable),
                MODEL_CHECKSUM_OFFSET + MODEL_CHECKSUM_BYTES,
            )
            if begin < end:
                mutable[begin - offset : end - offset] = bytes(end - begin)
            digest.update(mutable)
            offset += len(mutable)
    return digest.digest() == stored


def _model_ready(
    path: Path,
    expected_feature_fingerprint: bytes | None = None,
    *,
    allow_legacy_feature_contract: bool = False,
) -> bool:
    try:
        file_size = path.stat().st_size
        if file_size < MODEL_HEADER_BYTES:
            return False
        with path.open("rb") as stream:
            header = stream.read(MODEL_HEADER_BYTES)
        if len(header) != MODEL_HEADER_BYTES or header[:8] != MODEL_MAGIC:
            return False

        u32 = lambda offset: struct.unpack_from("<I", header, offset)[0]
        u64 = lambda offset: struct.unpack_from("<Q", header, offset)[0]
        i32 = lambda offset: struct.unpack_from("<i", header, offset)[0]
        f32 = lambda offset: struct.unpack_from("<f", header, offset)[0]

        version = u32(8)
        flags = u32(20)
        head = u32(24)
        state_bits = u32(28)
        feature_count = u32(32)
        output_count = u32(36)
        clause_count = u32(40)
        literal_count = u32(44)
        literal_words = u32(48)
        training_threshold = i32(52)
        max_literals = u32(60)
        specificity = f32(64)
        negative_samples = f32(68)
        feedback_ratio = f32(72)
        midi_minimum = i32(88)
        midi_maximum = i32(92)
        hard_probability = f32(108)

        if (
            version not in MODEL_SUPPORTED_VERSIONS
            or u32(12) != MODEL_HEADER_BYTES
            or u32(16) != MODEL_WORD_BITS
            or flags & ~MODEL_KNOWN_FLAGS
            or head not in (1, 2)
            or not 2 <= state_bits <= 16
            or feature_count == 0
            or output_count == 0
            or clause_count == 0
            or feature_count > 0xFFFFFFFF // 2
            or literal_count != feature_count * 2
            or literal_words != (literal_count + 31) // 32
            or training_threshold <= 0
            or max_literals > literal_count
            or not math.isfinite(specificity)
            or specificity <= 0.0
            or not math.isfinite(negative_samples)
            or negative_samples < 0.0
            or not math.isfinite(feedback_ratio)
            or feedback_ratio <= 0.0
            or midi_minimum < 0
            or midi_maximum > 127
            or midi_maximum - midi_minimum + 1 != output_count
            or not 1 <= u32(96) <= 16
            or u32(100) == 0
            or u32(104) == 0
        ):
            return False

        if version in MODEL_LEGACY_VERSIONS:
            if any(header[MODEL_FEATURE_FINGERPRINT_OFFSET:]):
                return False
            if not allow_legacy_feature_contract:
                return False
        else:
            feature_fingerprint = bytes(
                header[
                    MODEL_FEATURE_FINGERPRINT_OFFSET :
                    MODEL_FEATURE_FINGERPRINT_OFFSET
                    + MODEL_FEATURE_FINGERPRINT_BYTES
                ]
            )
            if (
                not any(feature_fingerprint)
                or any(header[224:])
                or (
                    expected_feature_fingerprint is not None
                    and feature_fingerprint != expected_feature_fingerprint
                )
            ):
                return False
            stored_checksum = bytes(
                header[
                    MODEL_CHECKSUM_OFFSET :
                    MODEL_CHECKSUM_OFFSET + MODEL_CHECKSUM_BYTES
                ]
            )
            if not any(stored_checksum) or not _model_checksum_matches(
                path, stored_checksum
            ):
                return False

        hard_negatives = bool(flags & (1 << 2))
        weight_only = bool(flags & (1 << 3))
        if (
            (hard_negatives and head != 2)
            or (weight_only and not hard_negatives)
            or not math.isfinite(hard_probability)
            or hard_probability < 0.0
            or hard_probability > 1.0
            or (not hard_negatives and hard_probability != 0.0)
            or (
                version >= 2
                and hard_negatives
                and hard_probability <= 0.0
            )
        ):
            return False

        ta_bytes = clause_count * state_bits * literal_words * 4
        weight_bytes = output_count * clause_count * 4
        return (
            u64(112) == MODEL_HEADER_BYTES
            and u64(120) == ta_bytes
            and u64(128) == MODEL_HEADER_BYTES + ta_bytes
            and u64(136) == weight_bytes
            and u64(144) == ta_bytes + weight_bytes
            and u64(152) == file_size
            and file_size == MODEL_HEADER_BYTES + ta_bytes + weight_bytes
        )
    except (OSError, struct.error):
        return False


def _score_ready(path: Path) -> bool:
    try:
        if path.stat().st_size <= len(SCORE_MAGIC):
            return False
        with path.open("r", encoding="utf-8") as stream:
            return stream.readline().rstrip("\r\n") == SCORE_MAGIC
    except (OSError, UnicodeError):
        return False


def _json_ready(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(value, dict) and bool(value)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _fresh(output: Path, inputs: Iterable[Path]) -> bool:
    try:
        output_time = output.stat().st_mtime_ns
        return all(output_time >= path.stat().st_mtime_ns for path in inputs)
    except OSError:
        return False


class RunnerLock(AbstractContextManager["RunnerLock"]):
    """Cross-platform advisory lock released automatically after a crash."""

    def __init__(self, path: Path):
        self.path = path
        self.stream: Any | None = None

    def __enter__(self) -> "RunnerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError(
                f"another native ablation runner owns {self.path}"
            ) from error
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def _write_command(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_command(command) + "\n", encoding="utf-8")


def _run_command(
    command: Sequence[str], log_path: Path, *, cwd: Path = PROJECT_ROOT
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    print(f"RUN {format_command(command)}", flush=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write(f"# command: {format_command(command)}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if return_code != 0:
            raise CommandError(
                f"command exited with {return_code}; see {log_path}"
            )


def _resolved_run_config(spec: ExperimentSpec, paths: RunnerPaths) -> dict[str, Any]:
    result = {
        "format": RUN_CONFIG_FORMAT,
        "name": spec.name,
        "train_dataset": str(paths.train_dataset),
        "validation_dataset": str(paths.validation_dataset),
        "train_executable": str(paths.train_executable),
        "predict_executable": str(paths.predict_executable),
        "python_executable": str(paths.python_executable),
        "evaluator": str(paths.evaluator),
        "heads": spec.heads,
        "allow_legacy_feature_contract": paths.allow_legacy_feature_contract,
    }
    if paths.feature_fingerprint_sha256 is not None:
        result["feature_fingerprint_sha256"] = (
            paths.feature_fingerprint_sha256
        )
    return result


def _expected_feature_fingerprint(paths: RunnerPaths) -> bytes | None:
    if paths.feature_fingerprint_sha256 is None:
        return None
    return bytes.fromhex(paths.feature_fingerprint_sha256)


def _dataset_feature_fingerprint(
    train_dataset: Path,
    validation_dataset: Path,
    *,
    allow_legacy_feature_contract: bool,
) -> str | None:
    train = inspect_dataset_contract(train_dataset, allow_legacy=True)
    validation = inspect_dataset_contract(validation_dataset, allow_legacy=True)
    for field in (
        "feature_count",
        "outputs",
        "midi_min",
        "midi_max",
        "sample_rate",
        "hop_size",
    ):
        if getattr(train, field) != getattr(validation, field):
            raise ValueError(f"train/validation dataset {field} differs")
    if train.legacy or validation.legacy:
        if train.legacy != validation.legacy:
            raise ValueError(
                "train/validation cannot mix legacy and semantic feature contracts"
            )
        if not allow_legacy_feature_contract:
            raise ValueError(
                "legacy train/validation datasets require explicit "
                "--allow-legacy-feature-contract audit opt-in"
            )
        return None
    if (
        train.feature_fingerprint_sha256
        != validation.feature_fingerprint_sha256
    ):
        raise ValueError(
            "train/validation feature-semantics fingerprints differ"
        )
    return train.feature_fingerprint_sha256


def _check_or_write_config(
    path: Path, config: Mapping[str, Any], *, force: bool
) -> None:
    if path.exists() and not force:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read existing run config: {path}") from error
        if previous != config:
            raise RuntimeError(
                f"resolved config changed for {path.parent}; use --force or a new name"
            )
        return
    _atomic_json(path, config)


def _temporary(path: Path) -> Path:
    return path.with_name(path.name + f".partial-{os.getpid()}")


def _replace_checked(temporary: Path, final: Path, ready: Any) -> None:
    if not ready(temporary):
        raise CommandError(f"command succeeded but output is invalid: {temporary}")
    temporary.replace(final)


def run_experiment(
    spec: ExperimentSpec, paths: RunnerPaths, *, force: bool = False
) -> dict[str, Any]:
    experiment_dir = paths.output_root / spec.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _check_or_write_config(
        experiment_dir / "run-config.json",
        _resolved_run_config(spec, paths),
        force=force,
    )
    summary: dict[str, Any] = {
        "format": RUN_CONFIG_FORMAT,
        "name": spec.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "heads": {},
        "allow_legacy_feature_contract": paths.allow_legacy_feature_contract,
    }
    if paths.feature_fingerprint_sha256 is not None:
        summary["feature_fingerprint_sha256"] = (
            paths.feature_fingerprint_sha256
        )
    expected_fingerprint = _expected_feature_fingerprint(paths)
    validation_scores: list[Path] = []

    for head in HEADS:
        head_dir = experiment_dir / head
        head_dir.mkdir(parents=True, exist_ok=True)
        model = head_dir / "model.tmgmmod"
        train_scores = head_dir / "train.tsv"
        validation_scores_path = head_dir / "validation.tsv"
        validation_scores.append(validation_scores_path)
        head_summary: dict[str, str] = {}
        summary["heads"][head] = head_summary

        model_temporary = _temporary(model)
        train_temporary = _temporary(train_scores)
        train_command = build_train_command(
            paths.train_executable,
            paths.train_dataset,
            paths.validation_dataset,
            head,
            spec.heads[head],
            train_temporary,
            model_temporary,
            allow_legacy_feature_contract=paths.allow_legacy_feature_contract,
        )
        _write_command(head_dir / "train.command.txt", train_command)
        if not force and _model_ready(
            model,
            expected_fingerprint,
            allow_legacy_feature_contract=paths.allow_legacy_feature_contract,
        ):
            print(f"SKIP {spec.name}/{head} train: model is ready", flush=True)
            head_summary["train"] = "skipped"
        else:
            for partial in (model_temporary, train_temporary):
                partial.unlink(missing_ok=True)
            _run_command(train_command, head_dir / "train.log")
            _replace_checked(
                model_temporary,
                model,
                lambda candidate: _model_ready(
                    candidate,
                    expected_fingerprint,
                    allow_legacy_feature_contract=(
                        paths.allow_legacy_feature_contract
                    ),
                ),
            )
            _replace_checked(train_temporary, train_scores, _score_ready)
            head_summary["train"] = "completed"

        validation_temporary = _temporary(validation_scores_path)
        predict_command = build_predict_command(
            paths.predict_executable,
            paths.validation_dataset,
            model,
            validation_temporary,
            allow_legacy_feature_contract=paths.allow_legacy_feature_contract,
        )
        _write_command(head_dir / "predict.command.txt", predict_command)
        prediction_ready = (
            _score_ready(validation_scores_path)
            and _fresh(validation_scores_path, (model, paths.validation_dataset))
        )
        if not force and prediction_ready:
            print(f"SKIP {spec.name}/{head} predict: scores are ready", flush=True)
            head_summary["predict"] = "skipped"
        else:
            validation_temporary.unlink(missing_ok=True)
            _run_command(predict_command, head_dir / "predict.log")
            _replace_checked(validation_temporary, validation_scores_path, _score_ready)
            head_summary["predict"] = "completed"
        _atomic_json(experiment_dir / "run-summary.json", summary)

    metrics = experiment_dir / "validation-metrics.json"
    metrics_temporary = _temporary(metrics)
    evaluate_command = build_evaluate_command(
        paths.python_executable,
        paths.evaluator,
        paths.validation_dataset,
        validation_scores,
        metrics_temporary,
    )
    _write_command(experiment_dir / "evaluate.command.txt", evaluate_command)
    metrics_ready = _json_ready(metrics) and _fresh(metrics, validation_scores)
    if not force and metrics_ready:
        print(f"SKIP {spec.name} evaluate: metrics are ready", flush=True)
        summary["evaluate"] = "skipped"
    else:
        metrics_temporary.unlink(missing_ok=True)
        _run_command(evaluate_command, experiment_dir / "evaluate.log")
        _replace_checked(metrics_temporary, metrics, _json_ready)
        summary["evaluate"] = "completed"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(experiment_dir / "run-summary.json", summary)
    return summary


def _dry_run(specs: Sequence[ExperimentSpec], paths: RunnerPaths, *, force: bool) -> None:
    expected_fingerprint = _expected_feature_fingerprint(paths)
    for spec in specs:
        experiment_dir = paths.output_root / spec.name
        print(f"EXPERIMENT {spec.name}")
        print(json.dumps(_resolved_run_config(spec, paths), indent=2, sort_keys=True))
        score_files: list[Path] = []
        for head in HEADS:
            head_dir = experiment_dir / head
            model = head_dir / "model.tmgmmod"
            validation_score = head_dir / "validation.tsv"
            score_files.append(validation_score)
            train_status = "RUN" if force or not _model_ready(
                model,
                expected_fingerprint,
                allow_legacy_feature_contract=paths.allow_legacy_feature_contract,
            ) else "SKIP"
            print(
                f"{train_status} {format_command(build_train_command(paths.train_executable, paths.train_dataset, paths.validation_dataset, head, spec.heads[head], _temporary(head_dir / 'train.tsv'), _temporary(model), allow_legacy_feature_contract=paths.allow_legacy_feature_contract))}"
            )
            prediction_ready = _score_ready(validation_score) and _fresh(
                validation_score, (model, paths.validation_dataset)
            )
            predict_status = "RUN" if force or not prediction_ready else "SKIP"
            print(
                f"{predict_status} {format_command(build_predict_command(paths.predict_executable, paths.validation_dataset, model, _temporary(validation_score), allow_legacy_feature_contract=paths.allow_legacy_feature_contract))}"
            )
        metrics = experiment_dir / "validation-metrics.json"
        evaluate_ready = _json_ready(metrics) and _fresh(metrics, score_files)
        evaluate_status = "RUN" if force or not evaluate_ready else "SKIP"
        print(
            f"{evaluate_status} {format_command(build_evaluate_command(paths.python_executable, paths.evaluator, paths.validation_dataset, score_files, _temporary(metrics)))}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serial native activity/onset TM ablations with validation checkpoints, "
            "prediction, and streaming evaluation."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--train-exe", type=Path, default=DEFAULT_BUILD / "tmgm_train.exe"
    )
    parser.add_argument(
        "--allow-legacy-feature-contract",
        action="store_true",
        help=(
            "explicit audit-only opt-in for legacy datasets/models without "
            "authenticated feature semantics"
        ),
    )
    parser.add_argument(
        "--predict-exe", type=Path, default=DEFAULT_BUILD / "tmgm_predict.exe"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "evaluate_native_scores.py",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="run only named experiments"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun completed stages and accept a changed resolved config",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = load_experiments(args.config)
    if args.only:
        requested = set(args.only)
        known = {spec.name for spec in specs}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"--only names are not in manifest: {unknown}")
        specs = [spec for spec in specs if spec.name in requested]

    train_dataset = args.train_dataset.resolve()
    validation_dataset = args.validation_dataset.resolve()
    fingerprint: str | None = None
    if train_dataset.is_file() and validation_dataset.is_file():
        fingerprint = _dataset_feature_fingerprint(
            train_dataset,
            validation_dataset,
            allow_legacy_feature_contract=args.allow_legacy_feature_contract,
        )
    elif train_dataset.is_file() != validation_dataset.is_file():
        raise FileNotFoundError(
            "train and validation datasets must either both exist or both be absent"
        )

    paths = RunnerPaths(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        output_root=args.output_root.resolve(),
        train_executable=args.train_exe.resolve(),
        predict_executable=args.predict_exe.resolve(),
        python_executable=args.python.resolve(),
        evaluator=args.evaluator.resolve(),
        feature_fingerprint_sha256=fingerprint,
        allow_legacy_feature_contract=args.allow_legacy_feature_contract,
    )
    if args.dry_run:
        _dry_run(specs, paths, force=args.force)
        return 0

    required_files = {
        "training dataset": paths.train_dataset,
        "validation dataset": paths.validation_dataset,
        "trainer": paths.train_executable,
        "predictor": paths.predict_executable,
        "Python": paths.python_executable,
        "evaluator": paths.evaluator,
    }
    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))

    with RunnerLock(paths.output_root / ".native-ablation.lock"):
        for spec in specs:
            run_experiment(spec, paths, force=args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CommandError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
