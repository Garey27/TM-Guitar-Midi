from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Mapping

from .config import ContextConfig, FrontendConfig
from .stft_plus import CausalSTFTPlus


# Bump this ID whenever a formula, state update, interpolation rule, channel
# order, or normalization in CausalSTFTPlus changes. It deliberately lives
# outside cache/export schema versions so stale same-width features cannot hit.
FRONTEND_SCHEMA_ID = "tmgm-causal-stft-plus-v1"
FEATURE_SEMANTICS_SCHEMA_ID = "tmgm-binary-feature-semantics-v1"

_LEGACY_OPTIONAL_FRONTEND_DEFAULTS: dict[str, Any] = {
    "harmonic_local_contrast": False,
    "contrast_offset_semitones": 0.5,
    "expose_harmonic_local_profile": False,
    "contrast_attack_features": False,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def canonical_frontend_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize known legacy omissions without accepting unknown fields."""
    defaults = asdict(FrontendConfig())
    raw = dict(value)
    unknown = sorted(raw.keys() - defaults.keys())
    missing_required = sorted(
        defaults.keys() - raw.keys() - _LEGACY_OPTIONAL_FRONTEND_DEFAULTS.keys()
    )
    if unknown or missing_required:
        details: list[str] = []
        if missing_required:
            details.append(f"missing keys: {', '.join(missing_required)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"invalid frontend metadata ({'; '.join(details)})")
    for key, default in _LEGACY_OPTIONAL_FRONTEND_DEFAULTS.items():
        raw.setdefault(key, default)
    # FrontendConfig performs strict type/range and dependency validation.
    return asdict(FrontendConfig(**raw))


def frontend_schema_descriptor(frontend: FrontendConfig) -> dict[str, Any]:
    extractor = CausalSTFTPlus(frontend)
    names = extractor.feature_names()
    return {
        "id": FRONTEND_SCHEMA_ID,
        "feature_count": extractor.feature_count,
        "ordered_feature_names_sha256": _sha256_json(names),
        "feature_names_hash_encoding": "canonical-json-array-v1",
    }


def context_feature_names(
    frontend: FrontendConfig, context: ContextConfig
) -> list[str]:
    spectral_names = CausalSTFTPlus(frontend).feature_names()
    return [
        f"delay_{delay}:{name}"
        for delay in context.delays
        for name in spectral_names
    ]


def binary_feature_semantics(
    frontend: FrontendConfig,
    context: ContextConfig,
    *,
    binarizer_sha256: str,
    binarizer_signature: str,
    continuous_feature_count: int,
    binary_feature_count: int,
) -> dict[str, Any]:
    if (
        len(binarizer_sha256) != 64
        or any(character not in "0123456789abcdef" for character in binarizer_sha256)
    ):
        raise ValueError("binarizer_sha256 must be lowercase hexadecimal SHA-256")
    if not binarizer_signature:
        raise ValueError("binarizer_signature must be non-empty")
    names = context_feature_names(frontend, context)
    if len(names) != continuous_feature_count:
        raise ValueError(
            "ordered context feature names disagree with continuous feature count"
        )
    basis = {
        "schema": FEATURE_SEMANTICS_SCHEMA_ID,
        "frontend_schema": frontend_schema_descriptor(frontend),
        "frontend": asdict(frontend),
        "context": asdict(context),
        "ordered_feature_names_sha256": _sha256_json(names),
        "continuous_feature_count": continuous_feature_count,
        "binary_feature_count": binary_feature_count,
        "binarizer": {
            "sha256": binarizer_sha256,
            "signature": binarizer_signature,
        },
    }
    # Sidecars are JSON; normalize tuples (notably ContextConfig.delays) to
    # arrays before returning so an encode/decode round-trip stays identical.
    basis = json.loads(_canonical_json(basis))
    return {**basis, "fingerprint_sha256": _sha256_json(basis)}


def feature_fingerprint_bytes(semantics: Mapping[str, Any]) -> bytes:
    value = semantics.get("fingerprint_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("feature semantics has no valid fingerprint_sha256")
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            "feature semantics fingerprint_sha256 is not hexadecimal"
        ) from error
    if not any(result):
        raise ValueError("feature semantics fingerprint cannot be zero")
    return result


def validate_feature_semantics(
    semantics: Mapping[str, Any],
    frontend: FrontendConfig,
    context: ContextConfig,
    *,
    binarizer_sha256: str,
    binarizer_signature: str,
    continuous_feature_count: int,
    binary_feature_count: int,
) -> dict[str, Any]:
    expected = binary_feature_semantics(
        frontend,
        context,
        binarizer_sha256=binarizer_sha256,
        binarizer_signature=binarizer_signature,
        continuous_feature_count=continuous_feature_count,
        binary_feature_count=binary_feature_count,
    )
    if dict(semantics) != expected:
        raise ValueError("feature semantics descriptor does not match its artifacts")
    return expected
