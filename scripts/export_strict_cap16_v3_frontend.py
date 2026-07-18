from __future__ import annotations

"""Export the frozen strict-cap16-v3 streaming frontend contract.

NPZ is convenient for training but unsuitable for a realtime plugin loader.
This script verifies the four frozen reference/binarizer pairs and writes one
canonical little-endian TMGMFRT artifact containing:

* exact NumPy float32 window/frequency constants;
* the exact float64 interpolation-bin positions used by STFT+;
* kept thermometer raw-column indices, float32 thresholds, and authenticated
  per-entry ULP equality policy;
* source-binarizer/reference identities and semantic fingerprints.

The optional 2222 float fixture is already resampled mono channel zero at
22050 Hz. Resampling remains deliberately outside the native frontend.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from tmgm_rt.audio import load_audio_mono_channel_zero
from tmgm_rt.native_export import load_quantile_thermometer
from tmgm_rt.native_wav_export import load_reference_inference_metadata
from tmgm_rt.stft_plus import CausalSTFTPlus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT / "artifacts/native-listening-comparison-20260718-strict-cap16-v3"
)
MAGIC = b"TMGMFRT\0"
FORMAT_VERSION = 2
HEADER_BYTES = 512
DESCRIPTOR_BYTES = 256
CHECKSUM_OFFSET = 376
CHECKSUM_BYTES = 32
THERMOMETER_EQUALITY_ULPS = 4
POCKETFFT_COMMIT = "33ae5dc94c9cdc7f1c78346504a85de87cadaa12"
VARIANTS = (
    ("plain", "plain", 1),
    ("hcontrast", "hcontrast-d2", 2),
    ("hprofile", "hprofile-d3", 3),
    ("cattack", "cattack-d3", 4),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def verify_file_ref(value: dict[str, Any], label: str) -> Path:
    path = Path(value["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256_file(path)
    if actual != str(value["sha256"]):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return path


def align8(value: int) -> int:
    return (value + 7) & ~7


def digest_bytes(value: str, label: str) -> bytes:
    if len(value) != 64:
        raise ValueError(f"{label} is not SHA-256 hex")
    return bytes.fromhex(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--without-2222-audio", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else package / "native-frontend/strict-cap16-v3.tmgmfront"
    )
    manifest_path = package / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("version") != "strict-cap16-v3":
        raise ValueError("package is not frozen strict-cap16-v3")

    banks: list[dict[str, Any]] = []
    extractors: dict[str, CausalSTFTPlus] = {}
    for native_id, package_id, variant in VARIANTS:
        feature_record = manifest["feature_banks"][package_id]
        reference_path = verify_file_ref(
            feature_record["reference"], f"{package_id} reference"
        )
        binarizer_path = verify_file_ref(
            feature_record["binarizer"], f"{package_id} binarizer"
        )
        reference = load_reference_inference_metadata(reference_path)
        binarizer = load_quantile_thermometer(binarizer_path)
        thresholds = np.ascontiguousarray(binarizer.thresholds, dtype="<f4")
        keep = np.ascontiguousarray(binarizer.keep_columns, dtype=np.bool_)
        if tuple(binarizer.quantiles) != (0.5, 0.7, 0.85, 0.95):
            raise ValueError(f"{package_id} quantile order differs")
        spectral_width = CausalSTFTPlus(reference.frontend).feature_count
        continuous_width = spectral_width * len(reference.context.delays)
        if thresholds.shape != (continuous_width, 4):
            raise ValueError(f"{package_id} threshold geometry differs")
        if keep.shape != (continuous_width * 4,):
            raise ValueError(f"{package_id} keep-column geometry differs")
        kept_indices = np.flatnonzero(keep).astype("<u4", copy=False)
        kept_thresholds = thresholds.reshape(-1)[keep].astype("<f4", copy=False)
        entries = np.empty(
            kept_indices.size,
            dtype=np.dtype(
                [
                    ("raw_column", "<u4"),
                    ("threshold", "<f4"),
                    ("equality_ulps", "<u4"),
                ]
            ),
        )
        entries["raw_column"] = kept_indices
        entries["threshold"] = kept_thresholds
        entries["equality_ulps"] = 0
        equality_records: list[dict[str, int | str]] = []
        if native_id == "cattack":
            equality_raw_column = (5 * spectral_width + 359) * 4
            matches = np.flatnonzero(kept_indices == equality_raw_column)
            if matches.tolist() != [8748]:
                raise ValueError("frozen cattack equality entry moved")
            entries["equality_ulps"][matches[0]] = THERMOMETER_EQUALITY_ULPS
            equality_records.append(
                {
                    "output_column": int(matches[0]),
                    "raw_column": equality_raw_column,
                    "context_slot": 5,
                    "context_delay": 16,
                    "spectral_feature": 359,
                    "semantic_feature": "contrast_fast_slow_attack:midi_56",
                    "quantile_index": 0,
                    "one_sided_float32_equality_ulps": THERMOMETER_EQUALITY_ULPS,
                }
            )
        semantic = str(feature_record["semantic_feature_fingerprint_sha256"])
        if reference.feature_semantics.get("fingerprint_sha256") != semantic:
            raise ValueError(f"{package_id} semantic fingerprint differs")
        if reference.binarizer_sha256 != sha256_file(binarizer_path):
            raise ValueError(f"{package_id} reference binarizer identity differs")
        sidecar = read_json(binarizer_path.with_suffix(binarizer_path.suffix + ".json"))
        if sidecar.get("signature") != reference.binarizer_signature:
            raise ValueError(f"{package_id} binarizer signature differs")
        extractor = CausalSTFTPlus(reference.frontend)
        extractors[native_id] = extractor
        flags = (
            (1 if reference.frontend.harmonic_local_contrast else 0)
            | (2 if reference.frontend.expose_harmonic_local_profile else 0)
            | (4 if reference.frontend.contrast_attack_features else 0)
        )
        banks.append(
            {
                "native_id": native_id,
                "package_id": package_id,
                "variant": variant,
                "flags": flags,
                "spectral_width": spectral_width,
                "continuous_width": continuous_width,
                "raw_width": continuous_width * 4,
                "binary_width": int(kept_indices.size),
                "contrast_offset": float(reference.frontend.contrast_offset_semitones),
                "entries": entries,
                "equality_records": equality_records,
                "binarizer_sha256": reference.binarizer_sha256,
                "binarizer_signature": reference.binarizer_signature,
                "semantic_sha256": semantic,
                "reference_sha256": str(feature_record["reference"]["sha256"]),
                "delays": tuple(reference.context.delays),
            }
        )

    plain = extractors["plain"]
    corrected = extractors["hcontrast"]
    for name, extractor in extractors.items():
        for field in (
            "window",
            "harmonic_bins",
            "harmonic_weights",
            "subharmonic_bins",
            "frequency_axis",
        ):
            if not np.array_equal(getattr(plain, field), getattr(extractor, field)):
                raise ValueError(f"common frontend constant {field} differs for {name}")
    if plain.side_low_bins.ndim != 1 or corrected.side_low_bins.shape != (49, 6):
        raise ValueError("plain/corrected side-bin geometry differs")
    if any(bank["delays"] != (0, 1, 2, 4, 8, 16, 32) for bank in banks):
        raise ValueError("frozen context delays differ")

    descriptors_offset = HEADER_BYTES
    descriptors_bytes = DESCRIPTOR_BYTES * len(banks)
    payload_offset = align8(descriptors_offset + descriptors_bytes)
    raw = bytearray(payload_offset)

    def append_payload(value: np.ndarray) -> tuple[int, int]:
        nonlocal raw
        offset = align8(len(raw))
        raw.extend(bytes(offset - len(raw)))
        payload = np.ascontiguousarray(value).tobytes(order="C")
        raw.extend(payload)
        return offset, len(payload)

    common_arrays = (
        np.asarray(plain.window, dtype="<f4"),
        np.asarray(plain.harmonic_bins, dtype="<f8"),
        np.asarray(plain.harmonic_weights, dtype="<f4"),
        np.asarray(plain.side_low_bins, dtype="<f8"),
        np.asarray(plain.side_high_bins, dtype="<f8"),
        np.asarray(corrected.side_low_bins, dtype="<f8"),
        np.asarray(corrected.side_high_bins, dtype="<f8"),
        np.asarray(plain.subharmonic_bins, dtype="<f8"),
        np.asarray(plain.frequency_axis, dtype="<f4"),
    )
    common_sections = [append_payload(value) for value in common_arrays]
    for bank in banks:
        bank["entry_section"] = append_payload(bank["entries"])

    file_bytes = len(raw)
    header = bytearray(HEADER_BYTES)
    header[:8] = MAGIC
    struct.pack_into(
        "<IIIIIIIIiiiIfIIII",
        header,
        8,
        FORMAT_VERSION,
        HEADER_BYTES,
        DESCRIPTOR_BYTES,
        len(banks),
        22_050,
        256,
        2_048,
        1,  # first frame after sample one
        40,
        88,
        6,
        4,  # quantiles
        np.float32(0.08),
        7,  # delays
        32,  # max delay
        THERMOMETER_EQUALITY_ULPS,
        0,
    )
    struct.pack_into(
        "<QQQQQ",
        header,
        80,
        descriptors_offset,
        descriptors_bytes,
        payload_offset,
        file_bytes - payload_offset,
        file_bytes,
    )
    common_cursor = 120
    for offset, size in common_sections:
        struct.pack_into("<QQ", header, common_cursor, offset, size)
        common_cursor += 16
    normalizer = np.float32(max(float(plain.window.sum()) * 0.5, 1.0))
    struct.pack_into("<f", header, 264, normalizer)
    struct.pack_into("<4f", header, 268, 0.5, 0.7, 0.85, 0.95)
    struct.pack_into("<7I", header, 284, 0, 1, 2, 4, 8, 16, 32)
    header[312:352] = POCKETFFT_COMMIT.encode("ascii")

    descriptors = bytearray(descriptors_bytes)
    for index, bank in enumerate(banks):
        descriptor = memoryview(descriptors)[
            index * DESCRIPTOR_BYTES : (index + 1) * DESCRIPTOR_BYTES
        ]
        identifier = bank["native_id"].encode("ascii")
        descriptor[: len(identifier)] = identifier
        entry_offset, entry_bytes = bank["entry_section"]
        struct.pack_into(
            "<IIIIIfIQQ",
            descriptor,
            32,
            bank["variant"],
            bank["spectral_width"],
            bank["continuous_width"],
            bank["raw_width"],
            bank["binary_width"],
            np.float32(bank["contrast_offset"]),
            bank["flags"],
            entry_offset,
            entry_bytes,
        )
        descriptor[80:112] = digest_bytes(
            bank["binarizer_sha256"], "binarizer SHA-256"
        )
        descriptor[112:144] = digest_bytes(
            bank["binarizer_signature"], "binarizer signature"
        )
        descriptor[144:176] = digest_bytes(
            bank["semantic_sha256"], "semantic fingerprint"
        )
        descriptor[176:208] = digest_bytes(
            bank["reference_sha256"], "reference SHA-256"
        )

    raw[:HEADER_BYTES] = header
    raw[descriptors_offset : descriptors_offset + descriptors_bytes] = descriptors
    canonical = bytearray(raw)
    canonical[CHECKSUM_OFFSET : CHECKSUM_OFFSET + CHECKSUM_BYTES] = bytes(
        CHECKSUM_BYTES
    )
    checksum = hashlib.sha256(canonical).digest()
    raw[CHECKSUM_OFFSET : CHECKSUM_OFFSET + CHECKSUM_BYTES] = checksum
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(output)

    metadata: dict[str, Any] = {
        "format": "TMGMFRT",
        "format_version": FORMAT_VERSION,
        "path": str(output),
        "bytes": len(raw),
        "checksum_sha256": checksum.hex(),
        "file_sha256": sha256_file(output),
        "pocketfft": {
            "upstream": "mreineck/pocketfft",
            "numpy_submodule_commit": POCKETFFT_COMMIT,
        },
        "timebase": {
            "sample_rate": 22_050,
            "hop_size": 256,
            "fft_size": 2_048,
            "first_frame_sample": 1,
        },
        "thermometer_comparison": {
            "operator": ">=",
            "maximum_one_sided_float32_equality_ulps": THERMOMETER_EQUALITY_ULPS,
            "policy": "per-entry authenticated equality_ulps; zero means exact >=",
            "records": [
                record
                for bank in banks
                for record in bank["equality_records"]
            ],
            "purpose": "portable equality across NumPy-wheel and native PocketFFT builds",
        },
        "banks": [
            {
                key: value
                for key, value in bank.items()
                if key
                not in {
                    "entries",
                    "entry_section",
                    "delays",
                }
            }
            for bank in banks
        ],
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
    }

    if not args.without_2222_audio:
        wav = verify_file_ref(manifest["tracks"]["2222"]["source_wav"], "2222 WAV")
        audio = load_audio_mono_channel_zero(wav, 22_050)
        audio_path = output.parent / "2222-mono-22050.f32le"
        audio_bytes = np.asarray(audio, dtype="<f4").tobytes(order="C")
        audio_temporary = audio_path.with_suffix(audio_path.suffix + ".tmp")
        audio_temporary.write_bytes(audio_bytes)
        audio_temporary.replace(audio_path)
        metadata["parity_audio"] = {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "samples": int(audio.size),
            "dtype": "little-endian-float32",
            "source_wav": manifest["tracks"]["2222"]["source_wav"],
        }

    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
