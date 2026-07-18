from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .audio import load_audio_mono_channel_zero
from .binarize import QuantileThermometer
from .config import ContextConfig, FrontendConfig
from .context import StreamingContext
from .feature_semantics import (
    binary_feature_semantics,
    feature_fingerprint_bytes,
    frontend_schema_descriptor,
    validate_feature_semantics,
)
from .native_dataset import NativeDatasetHeader, write_native_dataset_batches
from .native_export import load_quantile_thermometer
from .stft_plus import CausalSTFTPlus


WAV_EXPORT_SCHEMA = "tmgm-native-wav-inference-v1"


@dataclass(frozen=True)
class ReferenceInferenceMetadata:
    """Frontend/context and encoder identity copied from a native export."""

    path: Path
    sha256: str
    frontend: FrontendConfig
    context: ContextConfig
    export_schema: int | None
    export_signature: str | None
    binarizer_sha256: str
    binarizer_signature: str
    continuous_feature_count: int | None
    kept_binary_features: int | None
    header: dict[str, Any]
    feature_semantics: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _header_metadata(header: NativeDatasetHeader) -> dict[str, Any]:
    return {
        **asdict(header),
        "payload_sha256": header.payload_sha256.hex(),
        "feature_fingerprint_sha256": (
            header.feature_fingerprint_sha256.hex()
        ),
    }


def _file_identity(path: Path, *, sha256: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256,
    }


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _strict_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _strict_int(value, name)


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _strict_optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_sha256(value: Any, name: str) -> str | None:
    result = _strict_optional_string(value, name)
    if result is None:
        return None
    normalized = result.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _parse_frontend_config(value: Any) -> FrontendConfig:
    raw = _require_object(value, "reference metadata frontend")
    core_fields = {
        "sample_rate",
        "hop_size",
        "fft_size",
        "midi_min",
        "midi_max",
        "harmonics",
        "ema_alpha",
    }
    contrast_fields = {
        "harmonic_local_contrast",
        "contrast_offset_semitones",
    }
    ablation_fields = {
        "expose_harmonic_local_profile",
        "contrast_attack_features",
    }
    keys = set(raw)
    known_schemas = (
        core_fields,
        core_fields | contrast_fields,
        core_fields | contrast_fields | ablation_fields,
    )
    if keys not in known_schemas:
        # Diagnose against the current schema. The two older exact schemas
        # above remain supported so existing production sidecars keep their
        # original semantic defaults without accepting arbitrary omissions.
        _strict_keys(
            raw,
            core_fields | contrast_fields | ablation_fields,
            "reference metadata frontend",
        )
        raise AssertionError("unreachable frontend metadata schema")

    harmonic_local_contrast = raw.get("harmonic_local_contrast", False)
    expose_harmonic_local_profile = raw.get(
        "expose_harmonic_local_profile", False
    )
    contrast_attack_features = raw.get("contrast_attack_features", False)
    for field, field_value in (
        ("harmonic_local_contrast", harmonic_local_contrast),
        ("expose_harmonic_local_profile", expose_harmonic_local_profile),
        ("contrast_attack_features", contrast_attack_features),
    ):
        if type(field_value) is not bool:
            raise ValueError(
                f"reference metadata frontend.{field} must be a boolean"
            )
    frontend = FrontendConfig(
        sample_rate=_strict_int(
            raw["sample_rate"], "reference metadata frontend.sample_rate"
        ),
        hop_size=_strict_int(
            raw["hop_size"], "reference metadata frontend.hop_size"
        ),
        fft_size=_strict_int(
            raw["fft_size"], "reference metadata frontend.fft_size"
        ),
        midi_min=_strict_int(
            raw["midi_min"], "reference metadata frontend.midi_min"
        ),
        midi_max=_strict_int(
            raw["midi_max"], "reference metadata frontend.midi_max"
        ),
        harmonics=_strict_int(
            raw["harmonics"], "reference metadata frontend.harmonics"
        ),
        ema_alpha=_strict_float(
            raw["ema_alpha"], "reference metadata frontend.ema_alpha"
        ),
        harmonic_local_contrast=harmonic_local_contrast,
        contrast_offset_semitones=_strict_float(
            raw.get("contrast_offset_semitones", 0.5),
            "reference metadata frontend.contrast_offset_semitones",
        ),
        expose_harmonic_local_profile=expose_harmonic_local_profile,
        contrast_attack_features=contrast_attack_features,
    )
    if frontend.sample_rate <= 0:
        raise ValueError("reference metadata frontend.sample_rate must be positive")
    if frontend.hop_size <= 0:
        raise ValueError("reference metadata frontend.hop_size must be positive")
    if frontend.fft_size <= frontend.hop_size:
        raise ValueError(
            "reference metadata frontend.fft_size must exceed hop_size"
        )
    if not 0 <= frontend.midi_min <= frontend.midi_max <= 127:
        raise ValueError(
            "reference metadata frontend MIDI range must be within 0..127"
        )
    if frontend.harmonics <= 0:
        raise ValueError("reference metadata frontend.harmonics must be positive")
    if not 0.0 < frontend.ema_alpha <= 1.0:
        raise ValueError(
            "reference metadata frontend.ema_alpha must be in the interval (0, 1]"
        )
    return frontend


def _parse_context_config(value: Any) -> ContextConfig:
    raw = _require_object(value, "reference metadata context")
    _strict_keys(raw, {"delays"}, "reference metadata context")
    delays_value = raw["delays"]
    if not isinstance(delays_value, list):
        raise ValueError("reference metadata context.delays must be an array")
    delays = tuple(
        _strict_int(delay, f"reference metadata context.delays[{index}]")
        for index, delay in enumerate(delays_value)
    )
    if any(delay < 0 for delay in delays):
        raise ValueError("reference metadata context delays cannot be negative")
    return ContextConfig(delays=delays)


def load_reference_inference_metadata(
    path: str | Path,
) -> ReferenceInferenceMetadata:
    """Strictly load inference geometry and binarizer identity from TMGD JSON."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference metadata does not exist: {source}")
    try:
        raw_value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"reference metadata is not valid JSON: {source}") from error
    raw = _require_object(raw_value, "reference metadata")
    if "frontend" not in raw or "context" not in raw:
        raise ValueError("reference metadata must contain frontend and context")
    frontend = _parse_frontend_config(raw["frontend"])
    context = _parse_context_config(raw["context"])

    binarizer_value = raw.get("binarizer", {})
    binarizer = _require_object(binarizer_value, "reference metadata binarizer")
    binarizer_sha256 = _strict_sha256(
        binarizer.get("sha256"), "reference metadata binarizer.sha256"
    )
    binarizer_signature = _strict_optional_string(
        binarizer.get("signature"), "reference metadata binarizer.signature"
    )
    if binarizer_sha256 is None or binarizer_signature is None:
        raise ValueError(
            "reference metadata requires binarizer.sha256 and binarizer.signature"
        )
    continuous_feature_count = _strict_optional_int(
        raw.get("continuous_feature_count"),
        "reference metadata continuous_feature_count",
    )
    kept_binary_features = _strict_optional_int(
        raw.get("kept_binary_features"),
        "reference metadata kept_binary_features",
    )
    if continuous_feature_count is not None and continuous_feature_count <= 0:
        raise ValueError(
            "reference metadata continuous_feature_count must be positive"
        )
    if kept_binary_features is not None and kept_binary_features <= 0:
        raise ValueError("reference metadata kept_binary_features must be positive")

    header = _require_object(raw.get("header", {}), "reference metadata header")
    for field in (
        "feature_count",
        "note_count",
        "midi_min",
        "midi_max",
        "sample_rate",
        "hop_size",
    ):
        if field in header:
            _strict_int(header[field], f"reference metadata header.{field}")
    export_schema = _strict_optional_int(
        raw.get("export_schema"), "reference metadata export_schema"
    )
    export_signature = _strict_optional_string(
        raw.get("export_signature"), "reference metadata export_signature"
    )
    feature_semantics_value = raw.get("feature_semantics")
    if not isinstance(feature_semantics_value, dict):
        raise ValueError("reference metadata requires feature_semantics")
    return ReferenceInferenceMetadata(
        path=source,
        sha256=_sha256_file(source),
        frontend=frontend,
        context=context,
        export_schema=export_schema,
        export_signature=export_signature,
        binarizer_sha256=binarizer_sha256,
        binarizer_signature=binarizer_signature,
        continuous_feature_count=continuous_feature_count,
        kept_binary_features=kept_binary_features,
        header=dict(header),
        feature_semantics=dict(feature_semantics_value),
    )


def _validate_binarizer(
    binarizer: QuantileThermometer,
    frontend: FrontendConfig,
    context: ContextConfig,
) -> tuple[int, int]:
    if binarizer.thresholds is None or binarizer.keep_columns is None:
        raise ValueError("the quantile thermometer has not been fitted")
    spectral_feature_count = CausalSTFTPlus(frontend).feature_count
    continuous_feature_count = spectral_feature_count * len(context.delays)
    if binarizer.thresholds.shape[0] != continuous_feature_count:
        raise ValueError(
            "binarizer expects "
            f"{binarizer.thresholds.shape[0]} continuous features, but the current "
            f"frontend/context produces {continuous_feature_count}"
        )
    binary_feature_count = int(np.count_nonzero(binarizer.keep_columns))
    if binary_feature_count <= 0:
        raise ValueError("binarizer keeps no binary features")
    return continuous_feature_count, binary_feature_count


def iter_causal_inference_batches(
    samples: np.ndarray,
    binarizer: QuantileThermometer,
    frontend: FrontendConfig,
    context: ContextConfig,
    *,
    batch_frames: int = 512,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield sequential binary STFT+/context rows with zero inference labels.

    The frontend and context objects retain state across batches. Therefore batch
    boundaries neither reset history nor add lookahead, and every emitted row is
    equivalent to processing the complete waveform in one causal pass.
    """
    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")
    samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("cannot export an empty waveform")

    continuous_feature_count, _ = _validate_binarizer(
        binarizer, frontend, context
    )
    extractor = CausalSTFTPlus(frontend)
    context_stack = StreamingContext(extractor.feature_count, context)
    samples_per_batch = batch_frames * frontend.hop_size

    for first in range(0, samples.size, samples_per_batch):
        spectral = extractor.push(samples[first : first + samples_per_batch])
        if spectral.shape[0] == 0:
            continue
        continuous = np.empty(
            (spectral.shape[0], continuous_feature_count), dtype=np.float32
        )
        for row, frame in enumerate(spectral):
            continuous[row] = context_stack.push(frame)
        binary = binarizer.transform(continuous)
        labels = np.zeros(
            (binary.shape[0], frontend.note_count), dtype=np.uint8
        )
        yield binary, labels, labels


def export_wav_native(
    wav_path: str | Path,
    binarizer_path: str | Path,
    output_path: str | Path,
    *,
    frontend: FrontendConfig | None = None,
    context: ContextConfig | None = None,
    reference_metadata: str | Path | None = None,
    batch_frames: int = 512,
) -> NativeDatasetHeader:
    """Export every causal frame of an arbitrary WAV to an unlabeled TMGMDAT."""
    source = Path(wav_path).resolve()
    encoder_path = Path(binarizer_path).resolve()
    destination = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input waveform does not exist: {source}")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"binarizer does not exist: {encoder_path}")
    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")

    if reference_metadata is not None and (frontend is not None or context is not None):
        raise ValueError(
            "reference_metadata cannot be combined with explicit frontend/context"
        )
    reference: ReferenceInferenceMetadata | None = None
    if reference_metadata is not None:
        reference = load_reference_inference_metadata(reference_metadata)
        frontend = reference.frontend
        context = reference.context
        configuration_mode = "reference-metadata"
    else:
        configuration_mode = (
            "explicit" if frontend is not None or context is not None else "default"
        )
        frontend = frontend or FrontendConfig()
        context = context or ContextConfig()

    binarizer_sha256 = _sha256_file(encoder_path)
    binarizer = load_quantile_thermometer(encoder_path)
    continuous_feature_count, binary_feature_count = _validate_binarizer(
        binarizer, frontend, context
    )

    encoder_metadata_path = encoder_path.with_suffix(encoder_path.suffix + ".json")
    encoder_metadata: dict[str, Any] = {}
    if encoder_metadata_path.is_file():
        try:
            encoder_metadata_value = json.loads(
                encoder_metadata_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "binarizer metadata sidecar is not valid JSON"
            ) from error
        encoder_metadata = _require_object(
            encoder_metadata_value, "binarizer metadata sidecar"
        )
        declared_sha256 = encoder_metadata.get("sha256")
        if declared_sha256 is not None and declared_sha256 != binarizer_sha256:
            raise ValueError("binarizer checksum disagrees with its metadata sidecar")
        declared_width = encoder_metadata.get("continuous_feature_count")
        if (
            declared_width is not None
            and int(declared_width) != continuous_feature_count
        ):
            raise ValueError(
                "binarizer metadata continuous feature count disagrees with its arrays"
            )

    binarizer_verification = {
        "sha256": "not-applicable",
        "signature": "not-applicable",
    }
    if reference is not None:
        if (
            reference.continuous_feature_count is not None
            and reference.continuous_feature_count != continuous_feature_count
        ):
            raise ValueError(
                "reference metadata continuous feature count disagrees with the "
                "selected frontend/context"
            )
        if (
            reference.kept_binary_features is not None
            and reference.kept_binary_features != binary_feature_count
        ):
            raise ValueError(
                "reference metadata kept binary feature count disagrees with the "
                "selected binarizer"
            )
        if reference.binarizer_sha256 != binarizer_sha256:
            raise ValueError(
                "reference metadata binarizer SHA-256 does not match the selected "
                "binarizer"
            )
        binarizer_verification["sha256"] = "matched"
        actual_binarizer_signature = encoder_metadata.get("signature")
        if actual_binarizer_signature is None:
            raise ValueError(
                "reference-mode requires a binarizer sidecar signature"
            )
        if actual_binarizer_signature != reference.binarizer_signature:
            raise ValueError(
                "reference metadata binarizer signature does not match the "
                "selected binarizer sidecar"
            )
        binarizer_verification["signature"] = "matched"

        expected_header = {
            "feature_count": binary_feature_count,
            "note_count": frontend.note_count,
            "midi_min": frontend.midi_min,
            "midi_max": frontend.midi_max,
            "sample_rate": frontend.sample_rate,
            "hop_size": frontend.hop_size,
        }
        for field, expected in expected_header.items():
            declared = reference.header.get(field)
            if declared is not None and declared != expected:
                raise ValueError(
                    f"reference metadata header.{field} disagrees with the selected "
                    "frontend/binarizer"
                )

    encoder_signature = encoder_metadata.get("signature")
    if not isinstance(encoder_signature, str) or not encoder_signature:
        raise ValueError("binarizer metadata sidecar requires a signature")
    feature_semantics = binary_feature_semantics(
        frontend,
        context,
        binarizer_sha256=binarizer_sha256,
        binarizer_signature=encoder_signature,
        continuous_feature_count=continuous_feature_count,
        binary_feature_count=binary_feature_count,
    )
    if reference is not None:
        validate_feature_semantics(
            reference.feature_semantics,
            frontend,
            context,
            binarizer_sha256=binarizer_sha256,
            binarizer_signature=encoder_signature,
            continuous_feature_count=continuous_feature_count,
            binary_feature_count=binary_feature_count,
        )

    input_sha256 = _sha256_file(source)
    input_identity = _file_identity(source, sha256=input_sha256)
    audio = load_audio_mono_channel_zero(source, frontend.sample_rate)
    if audio.size == 0:
        raise ValueError("cannot export an empty waveform")

    batches = iter_causal_inference_batches(
        audio,
        binarizer,
        frontend,
        context,
        batch_frames=batch_frames,
    )
    header = write_native_dataset_batches(
        destination,
        batches,
        np.empty(0, dtype=np.uint32),
        feature_count=binary_feature_count,
        note_count=frontend.note_count,
        midi_min=frontend.midi_min,
        sample_rate=frontend.sample_rate,
        hop_size=frontend.hop_size,
        seed=0,
        feature_fingerprint_sha256=feature_fingerprint_bytes(feature_semantics),
    )
    expected_frames = (int(audio.size) + frontend.hop_size - 1) // frontend.hop_size
    if header.frame_count != expected_frames:
        raise AssertionError(
            f"frontend emitted {header.frame_count} frames; expected {expected_frames}"
        )

    encoder_identity = _file_identity(encoder_path, sha256=binarizer_sha256)
    signature_basis = {
        "schema": WAV_EXPORT_SCHEMA,
        "input_sha256": input_sha256,
        "binarizer_sha256": binarizer_sha256,
        "frontend": asdict(frontend),
        "context": asdict(context),
    }
    metadata = {
        "schema": WAV_EXPORT_SCHEMA,
        "signature": _canonical_hash(signature_basis),
        "format": "TMGMDAT",
        "version": 2,
        "purpose": "causal-unlabeled-inference",
        "input": input_identity,
        "resampled_mono_samples": int(audio.size),
        "binarizer": {
            **encoder_identity,
            "signature": encoder_metadata.get("signature"),
            "quantiles": list(binarizer.quantiles),
            "continuous_feature_count": continuous_feature_count,
            "kept_binary_features": binary_feature_count,
        },
        "configuration_source": {
            "mode": configuration_mode,
            "reference_metadata": (
                {
                    **_file_identity(reference.path, sha256=reference.sha256),
                    "export_schema": reference.export_schema,
                    "export_signature": reference.export_signature,
                }
                if reference is not None
                else None
            ),
            "binarizer_verification": binarizer_verification,
        },
        "frontend_schema": frontend_schema_descriptor(frontend),
        "feature_semantics": feature_semantics,
        "frontend": asdict(frontend),
        "context": asdict(context),
        "causality": {
            "strictly_causal": True,
            "lookahead_frames": 0,
            "all_frames_in_source_order": True,
        },
        "labels": {
            "activity": "all-zero",
            "onset": "all-zero",
            "onset_training_indices": 0,
        },
        "header": _header_metadata(header),
        "output": {
            **_file_identity(destination, sha256=_sha256_file(destination)),
        },
    }
    _write_json_atomic(
        destination.with_suffix(destination.suffix + ".json"), metadata
    )
    return header
