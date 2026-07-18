from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tmgm_rt.binarize import QuantileThermometer
from tmgm_rt.config import ContextConfig, FrontendConfig
from tmgm_rt.context import stack_causal_context
from tmgm_rt.feature_semantics import binary_feature_semantics
from tmgm_rt.native_dataset import read_native_dataset, unpack_binary_rows
from tmgm_rt.native_export import save_quantile_thermometer
from tmgm_rt.native_wav_export import (
    WAV_EXPORT_SCHEMA,
    export_wav_native,
    load_reference_inference_metadata,
)
from tmgm_rt.stft_plus import extract_stft_plus


def _fixture(
    tmp_path: Path,
    *,
    frontend: FrontendConfig | None = None,
    context: ContextConfig | None = None,
    sample_count: int = 333,
):
    frontend = frontend or FrontendConfig(
        sample_rate=8_000,
        hop_size=16,
        fft_size=64,
        midi_min=40,
        midi_max=42,
        harmonics=2,
    )
    context = context or ContextConfig(delays=(0, 1, 3))
    time = np.arange(sample_count, dtype=np.float32) / frontend.sample_rate
    audio = (
        0.25 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.1 * np.sin(2.0 * np.pi * 164.81 * time)
    ).astype(np.float32)
    wav = tmp_path / "input.wav"
    sf.write(wav, audio, frontend.sample_rate, subtype="FLOAT")
    decoded = sf.read(wav, dtype="float32")[0]
    continuous = stack_causal_context(
        extract_stft_plus(decoded, frontend), context
    )
    binarizer = QuantileThermometer(quantiles=(0.4, 0.7)).fit(continuous)
    binarizer_path = tmp_path / "global-quantile-thermometer.npz"
    save_quantile_thermometer(
        binarizer_path,
        binarizer,
        signature="unit-test-binarizer",
        train_rows=continuous.shape[0],
        continuous_feature_count=continuous.shape[1],
    )
    return frontend, context, wav, continuous, binarizer, binarizer_path


def _reference_metadata(
    tmp_path: Path,
    frontend: FrontendConfig,
    context: ContextConfig,
    continuous: np.ndarray,
    binarizer: QuantileThermometer,
    binarizer_path: Path,
) -> Path:
    encoder_metadata = json.loads(
        binarizer_path.with_suffix(binarizer_path.suffix + ".json").read_text(
            encoding="utf-8"
        )
    )
    reference = {
        "export_schema": 1,
        "export_signature": "unit-test-reference-export",
        "frontend": asdict(frontend),
        "context": asdict(context),
        "continuous_feature_count": int(continuous.shape[1]),
        "kept_binary_features": int(np.count_nonzero(binarizer.keep_columns)),
        "binarizer": {
            "sha256": encoder_metadata["sha256"],
            "signature": encoder_metadata["signature"],
        },
        "feature_semantics": binary_feature_semantics(
            frontend,
            context,
            binarizer_sha256=encoder_metadata["sha256"],
            binarizer_signature=encoder_metadata["signature"],
            continuous_feature_count=int(continuous.shape[1]),
            binary_feature_count=int(np.count_nonzero(binarizer.keep_columns)),
        ),
        "header": {
            "feature_count": int(np.count_nonzero(binarizer.keep_columns)),
            "note_count": frontend.note_count,
            "midi_min": frontend.midi_min,
            "midi_max": frontend.midi_max,
            "sample_rate": frontend.sample_rate,
            "hop_size": frontend.hop_size,
        },
    }
    path = tmp_path / "validation.tmgd.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    return path


def test_wav_export_matches_one_pass_causal_frontend_and_has_zero_labels(
    tmp_path: Path,
):
    frontend, context, wav, continuous, binarizer, binarizer_path = _fixture(
        tmp_path
    )
    output = tmp_path / "inference.tmgd"
    header = export_wav_native(
        wav,
        binarizer_path,
        output,
        frontend=frontend,
        context=context,
        batch_frames=5,
    )
    dataset = read_native_dataset(output)

    expected = binarizer.transform(continuous)
    assert header.frame_count == int(np.ceil(333 / frontend.hop_size))
    np.testing.assert_array_equal(
        unpack_binary_rows(dataset.feature_words, header.feature_count), expected
    )
    assert not unpack_binary_rows(
        dataset.activity_words, header.note_count
    ).any()
    assert not unpack_binary_rows(dataset.onset_words, header.note_count).any()
    assert dataset.onset_indices.size == 0

    metadata = json.loads(
        output.with_suffix(".tmgd.json").read_text(encoding="utf-8")
    )
    assert metadata["schema"] == WAV_EXPORT_SCHEMA
    assert metadata["input"]["path"] == str(wav.resolve())
    assert metadata["input"]["sha256"] == hashlib.sha256(wav.read_bytes()).hexdigest()
    assert metadata["binarizer"]["signature"] == "unit-test-binarizer"
    assert metadata["header"]["payload_sha256"] == header.payload_sha256.hex()
    assert metadata["causality"] == {
        "all_frames_in_source_order": True,
        "lookahead_frames": 0,
        "strictly_causal": True,
    }
    assert metadata["configuration_source"]["mode"] == "explicit"


def test_wav_export_is_independent_of_batch_boundaries(tmp_path: Path):
    frontend, context, wav, _, _, binarizer_path = _fixture(tmp_path)
    first = tmp_path / "one-frame-batches.tmgd"
    second = tmp_path / "seventeen-frame-batches.tmgd"
    export_wav_native(
        wav,
        binarizer_path,
        first,
        frontend=frontend,
        context=context,
        batch_frames=1,
    )
    export_wav_native(
        wav,
        binarizer_path,
        second,
        frontend=frontend,
        context=context,
        batch_frames=17,
    )
    assert first.read_bytes() == second.read_bytes()


def test_wav_export_rejects_binarizer_for_another_frontend(tmp_path: Path):
    frontend, context, wav, _, _, binarizer_path = _fixture(tmp_path)
    incompatible = FrontendConfig(
        sample_rate=frontend.sample_rate,
        hop_size=frontend.hop_size,
        fft_size=frontend.fft_size,
        midi_min=40,
        midi_max=43,
        harmonics=frontend.harmonics,
    )
    with pytest.raises(ValueError, match="current frontend/context produces"):
        export_wav_native(
            wav,
            binarizer_path,
            tmp_path / "bad.tmgd",
            frontend=incompatible,
            context=context,
        )


def test_reference_metadata_selects_semantic_frontend_and_records_provenance(
    tmp_path: Path,
):
    corrected_frontend = FrontendConfig(
        sample_rate=8_000,
        hop_size=16,
        fft_size=64,
        midi_min=40,
        midi_max=42,
        harmonics=2,
        harmonic_local_contrast=True,
        contrast_offset_semitones=1.5,
    )
    (
        frontend,
        context,
        wav,
        continuous,
        binarizer,
        binarizer_path,
    ) = _fixture(tmp_path, frontend=corrected_frontend)
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )

    selected = tmp_path / "reference-selected.tmgd"
    export_wav_native(
        wav,
        binarizer_path,
        selected,
        reference_metadata=reference,
        batch_frames=5,
    )
    wrong_semantics = tmp_path / "legacy-width-compatible.tmgd"
    export_wav_native(
        wav,
        binarizer_path,
        wrong_semantics,
        frontend=replace(
            frontend,
            harmonic_local_contrast=False,
            contrast_offset_semitones=0.5,
        ),
        context=context,
        batch_frames=5,
    )

    selected_dataset = read_native_dataset(selected)
    wrong_dataset = read_native_dataset(wrong_semantics)
    assert selected_dataset.header.feature_count == wrong_dataset.header.feature_count
    assert not np.array_equal(
        selected_dataset.feature_words, wrong_dataset.feature_words
    )
    metadata = json.loads(
        selected.with_suffix(selected.suffix + ".json").read_text(encoding="utf-8")
    )
    assert metadata["frontend"]["harmonic_local_contrast"] is True
    assert metadata["frontend"]["contrast_offset_semitones"] == 1.5
    assert metadata["configuration_source"]["mode"] == "reference-metadata"
    assert metadata["configuration_source"]["reference_metadata"]["path"] == str(
        reference.resolve()
    )
    assert metadata["configuration_source"]["binarizer_verification"] == {
        "sha256": "matched",
        "signature": "matched",
    }


def test_reference_metadata_round_trips_optional_frontend_ablation_flags(
    tmp_path: Path,
):
    ablation_frontend = FrontendConfig(
        sample_rate=8_000,
        hop_size=16,
        fft_size=64,
        midi_min=40,
        midi_max=42,
        harmonics=2,
        harmonic_local_contrast=True,
        contrast_offset_semitones=1.5,
        expose_harmonic_local_profile=True,
        contrast_attack_features=True,
    )
    (
        frontend,
        context,
        wav,
        continuous,
        binarizer,
        binarizer_path,
    ) = _fixture(tmp_path, frontend=ablation_frontend)
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )

    loaded = load_reference_inference_metadata(reference)
    assert loaded.frontend == frontend
    assert loaded.context == context

    output = tmp_path / "ablation-reference-selected.tmgd"
    export_wav_native(
        wav,
        binarizer_path,
        output,
        reference_metadata=reference,
        batch_frames=3,
    )
    metadata = json.loads(
        output.with_suffix(output.suffix + ".json").read_text(encoding="utf-8")
    )
    assert metadata["frontend"] == asdict(frontend)
    assert metadata["binarizer"]["continuous_feature_count"] == (
        continuous.shape[1]
    )


def test_reference_metadata_accepts_only_complete_known_legacy_frontend_schemas(
    tmp_path: Path,
):
    frontend, context, _, continuous, binarizer, binarizer_path = _fixture(
        tmp_path
    )
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )
    raw = json.loads(reference.read_text(encoding="utf-8"))

    # Schema immediately preceding these ablations: corrected-contrast fields
    # are explicit, both new feature flags are absent and therefore false.
    contrast_schema = json.loads(json.dumps(raw))
    contrast_schema["frontend"].pop("expose_harmonic_local_profile")
    contrast_schema["frontend"].pop("contrast_attack_features")
    reference.write_text(json.dumps(contrast_schema), encoding="utf-8")
    assert load_reference_inference_metadata(reference).frontend == frontend

    # Original STFT+ schema: contrast semantics and ablation flags all take
    # their established defaults.
    original_schema = json.loads(json.dumps(contrast_schema))
    original_schema["frontend"].pop("harmonic_local_contrast")
    original_schema["frontend"].pop("contrast_offset_semitones")
    reference.write_text(json.dumps(original_schema), encoding="utf-8")
    assert load_reference_inference_metadata(reference).frontend == frontend

    partial_schema = json.loads(json.dumps(raw))
    partial_schema["frontend"].pop("contrast_attack_features")
    reference.write_text(json.dumps(partial_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys: contrast_attack_features"):
        load_reference_inference_metadata(reference)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sha256", "0" * 64, "SHA-256 does not match"),
        ("signature", "another-binarizer", "signature does not match"),
    ],
)
def test_reference_metadata_rejects_wrong_binarizer_identity(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
):
    frontend, context, wav, continuous, binarizer, binarizer_path = _fixture(
        tmp_path
    )
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["binarizer"][field] = replacement
    reference.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        export_wav_native(
            wav,
            binarizer_path,
            tmp_path / "identity-mismatch.tmgd",
            reference_metadata=reference,
        )


@pytest.mark.parametrize("missing", ["sha256", "signature"])
def test_reference_metadata_requires_complete_binarizer_identity(
    tmp_path: Path, missing: str
):
    frontend, context, wav, continuous, binarizer, binarizer_path = _fixture(
        tmp_path
    )
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["binarizer"].pop(missing)
    reference.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="requires binarizer.sha256"):
        export_wav_native(
            wav,
            binarizer_path,
            tmp_path / "missing-identity.tmgd",
            reference_metadata=reference,
        )


def test_reference_metadata_parses_frontend_and_context_strictly(tmp_path: Path):
    frontend, context, wav, continuous, binarizer, binarizer_path = _fixture(
        tmp_path
    )
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["frontend"]["sample_rate"] = True
    raw["context"]["future_field"] = 1
    reference.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_rate must be an integer"):
        export_wav_native(
            wav,
            binarizer_path,
            tmp_path / "invalid-config.tmgd",
            reference_metadata=reference,
        )


def test_reference_metadata_cannot_override_explicit_configs(tmp_path: Path):
    frontend, context, wav, continuous, binarizer, binarizer_path = _fixture(
        tmp_path
    )
    reference = _reference_metadata(
        tmp_path,
        frontend,
        context,
        continuous,
        binarizer,
        binarizer_path,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        export_wav_native(
            wav,
            binarizer_path,
            tmp_path / "ambiguous.tmgd",
            frontend=frontend,
            reference_metadata=reference,
        )


def test_wav_export_default_configuration_remains_supported(tmp_path: Path):
    default_frontend = FrontendConfig()
    default_context = ContextConfig()
    _, _, wav, _, _, binarizer_path = _fixture(
        tmp_path,
        frontend=default_frontend,
        context=default_context,
        sample_count=4_097,
    )
    output = tmp_path / "default.tmgd"
    export_wav_native(wav, binarizer_path, output, batch_frames=3)
    metadata = json.loads(
        output.with_suffix(output.suffix + ".json").read_text(encoding="utf-8")
    )
    assert metadata["configuration_source"]["mode"] == "default"
    assert metadata["frontend"] == asdict(default_frontend)
    assert metadata["context"] == {"delays": list(default_context.delays)}
