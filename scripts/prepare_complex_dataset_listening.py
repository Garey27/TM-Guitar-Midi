from __future__ import annotations

"""Prepare arbitrary dataset tracks for the frozen strict-cap16-v3 runner.

This module only exports inference features and time-normalized reference MIDI.
It never fits a model, searches a threshold, or modifies the frozen package.
"""

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

import mido

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_multibank_cpu_listening import file_ref, sha256_file, write_json
from tmgm_rt.feature_contract import inspect_dataset_contract
from tmgm_rt.midi import write_teacher_events


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = (
    ROOT / "artifacts/native-listening-comparison-20260718-strict-cap16-v3"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/complex-dataset-listening-20260718-strict-cap16-v3"
)
EXPECTED_SOURCES = {"goat", "guitarset", "guitar-techs"}
MIDI_MIN = 40
MIDI_MAX = 88
TICKS_PER_BEAT = 480
TEMPO = 500_000
TICKS_PER_SECOND = 960.0


def safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError("empty track package_id")
    return result


def _resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


def load_selection(path: Path) -> list[dict[str, Any]]:
    selection_path = path.resolve()
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    if value.get("schema") != "tmgm-complex-dataset-selection-v1":
        raise ValueError("unsupported complex-dataset selection schema")
    tracks = value.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != 6:
        raise ValueError("selection must contain exactly six tracks")
    counts = Counter(str(track.get("source")) for track in tracks)
    if counts != Counter({source: 2 for source in EXPECTED_SOURCES}):
        raise ValueError("selection must contain two tracks from each dataset")

    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in tracks:
        track = copy.deepcopy(raw)
        required = {"package_id", "source", "split", "dataset_id", "wav", "teacher_events"}
        missing = sorted(required - set(track))
        if missing:
            raise ValueError(f"selection track lacks fields: {missing}")
        identifier = safe_id(str(track["package_id"]))
        if identifier in identifiers:
            raise ValueError(f"duplicate package_id: {identifier}")
        identifiers.add(identifier)
        if track["source"] not in EXPECTED_SOURCES:
            raise ValueError(f"unsupported source: {track['source']}")
        track["package_id"] = identifier
        for key in ("wav", "teacher_events"):
            track[key] = str(_resolve(str(track[key]), selection_path.parent))
        reference = track.get("dataset_reference")
        if reference is not None:
            if not isinstance(reference, dict) or not {"kind", "path"} <= set(reference):
                raise ValueError(f"{identifier}: invalid dataset_reference")
            if reference["kind"] not in {"absolute-midi", "labels-tsv", "events-tsv"}:
                raise ValueError(f"{identifier}: unsupported reference kind")
            if track["source"] == "guitar-techs" and reference["kind"] == "absolute-midi":
                raise ValueError("Guitar-TECHS raw MIDI is forbidden because its tempo map is unsafe")
            reference["path"] = str(_resolve(str(reference["path"]), selection_path.parent))
        result.append(track)
    return result


def _write_absolute_events(
    destination: Path,
    events: Iterable[tuple[float, int, int, int, int]],
) -> None:
    packed: list[tuple[int, int, int, int, int]] = []
    final_tick = 0
    for seconds, is_on, channel, pitch, velocity in events:
        if pitch < MIDI_MIN or pitch > MIDI_MAX:
            continue
        tick = max(0, round(float(seconds) * TICKS_PER_SECOND))
        packed.append((tick, int(is_on), int(channel), int(pitch), int(velocity)))
        final_tick = max(final_tick, tick)
    # At a shared timestamp, release old voices before starting new ones.
    packed.sort(key=lambda event: (event[0], event[1], event[2], event[3]))
    midi = mido.MidiFile(type=0, ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    previous = 0
    for tick, is_on, channel, pitch, velocity in packed:
        delta = tick - previous
        previous = tick
        if is_on:
            track.append(
                mido.Message(
                    "note_on", channel=channel, note=pitch,
                    velocity=max(1, min(127, velocity)), time=delta,
                )
            )
        else:
            track.append(
                mido.Message(
                    "note_off", channel=channel, note=pitch, velocity=0, time=delta
                )
            )
    track.append(mido.MetaMessage("end_of_track", time=max(0, final_tick - previous)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.save(destination)


def normalize_midi_absolute_time(source: Path, destination: Path) -> None:
    """Remove the source tempo map while preserving every note's wall-clock time."""
    midi = mido.MidiFile(source)
    tempo = TEMPO
    seconds = 0.0
    events: list[tuple[float, int, int, int, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "note_on" and message.velocity > 0:
            events.append((seconds, 1, message.channel, message.note, message.velocity))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            events.append((seconds, 0, message.channel, message.note, 0))
    _write_absolute_events(destination, events)


def write_labels_reference(source: Path, destination: Path) -> None:
    events: list[tuple[float, int, int, int, int]] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            start = float(row["start"])
            end = float(row["end"])
            pitch = int(row["midi"])
            if end <= start:
                continue
            velocity = int(round(float(row.get("velocity", 80))))
            string = int(row.get("string", 0))
            channel = string if 0 <= string <= 5 else 0
            events.append((start, 1, channel, pitch, velocity))
            events.append((end, 0, channel, pitch, 0))
    _write_absolute_events(destination, events)


def make_reference(reference: dict[str, str], destination: Path) -> None:
    source = Path(reference["path"])
    kind = reference["kind"]
    if kind == "absolute-midi":
        normalize_midi_absolute_time(source, destination)
    elif kind == "labels-tsv":
        write_labels_reference(source, destination)
    elif kind == "events-tsv":
        write_teacher_events(destination, source, MIDI_MIN, MIDI_MAX)
    else:  # guarded by load_selection
        raise ValueError(f"unsupported reference kind: {kind}")


def feature_is_reusable(
    dataset: Path,
    wav: Path,
    expected_fingerprint: str,
) -> bool:
    metadata_path = dataset.with_suffix(dataset.suffix + ".json")
    if not dataset.exists() and not metadata_path.exists():
        return False
    if not dataset.is_file() or not metadata_path.is_file():
        raise ValueError(f"partial cached feature pair: {dataset}")
    contract = inspect_dataset_contract(dataset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embedded = contract.feature_fingerprint_sha256
    sidecar = metadata.get("feature_semantics", {}).get("fingerprint_sha256")
    input_value = metadata.get("input", {})
    actual_wav_sha = sha256_file(wav)
    actual_wav_path = wav.resolve()
    cached_path = Path(str(input_value.get("path", ""))).resolve()
    problems: list[str] = []
    if embedded != expected_fingerprint or sidecar != expected_fingerprint:
        problems.append("semantic feature fingerprint")
    if input_value.get("sha256") != actual_wav_sha:
        problems.append("source WAV SHA-256")
    if cached_path != actual_wav_path:
        problems.append("source WAV absolute path")
    if problems:
        raise ValueError(f"cached feature contract differs for {dataset}: {', '.join(problems)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frozen-package", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--batch-frames", type=int, default=2048)
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="explicitly replace existing feature pairs instead of fail-closed reuse",
    )
    args = parser.parse_args()
    if args.batch_frames <= 0:
        parser.error("--batch-frames must be positive")

    output = args.output_root.resolve()
    frozen = args.frozen_package.resolve()
    if output == frozen:
        raise ValueError("refusing to overwrite the frozen strict-cap16-v3 package")
    base_manifest_path = frozen / "manifest.json"
    manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "strict-cap16-v3":
        raise ValueError("frozen package is not strict-cap16-v3")
    tracks = load_selection(args.selection)
    output.mkdir(parents=True, exist_ok=True)

    banks_path = frozen / "feature-banks.json"
    banks = json.loads(banks_path.read_text(encoding="utf-8"))
    if set(banks) != {"plain", "hcontrast-d2", "hcontrast-d3", "hprofile-d3", "cattack-d3"}:
        raise ValueError("frozen five-bank contract changed")
    exporter = ROOT / "scripts/export_wav_native.py"
    manifest_tracks: dict[str, Any] = {}
    frozen_selection: list[dict[str, Any]] = []

    for track in tracks:
        identifier = track["package_id"]
        wav = Path(track["wav"])
        teacher_events = Path(track["teacher_events"])
        if not wav.is_file() or not teacher_events.is_file():
            raise FileNotFoundError(f"missing WAV or NeuralNote events for {identifier}")
        teacher_midi = output / "references/neuralnote" / f"{identifier}.mid"
        write_teacher_events(teacher_midi, teacher_events, MIDI_MIN, MIDI_MAX)

        dataset_reference_ref: dict[str, Any] | None = None
        reference = track.get("dataset_reference")
        if reference is not None:
            reference_source = Path(reference["path"])
            if not reference_source.is_file():
                raise FileNotFoundError(reference_source)
            reference_midi = output / "references/dataset" / f"{identifier}.mid"
            make_reference(reference, reference_midi)
            dataset_reference_ref = {
                **file_ref(reference_midi),
                "kind": reference["kind"],
                "source": file_ref(reference_source),
                "note": reference.get("note"),
            }

        features: dict[str, Any] = {}
        for bank in sorted(banks):
            directory = output / "features" / bank
            directory.mkdir(parents=True, exist_ok=True)
            dataset = directory / f"{identifier}.tmgd"
            reusable = False
            if not args.force_export:
                reusable = feature_is_reusable(
                    dataset, wav, str(banks[bank]["fingerprint_sha256"])
                )
            if not reusable:
                subprocess.run(
                    [
                        sys.executable,
                        str(exporter),
                        "--wav", str(wav),
                        "--binarizer", banks[bank]["binarizer"],
                        "--reference-metadata", banks[bank]["reference"],
                        "--output", str(dataset),
                        "--batch-frames", str(args.batch_frames),
                    ],
                    cwd=ROOT,
                    check=True,
                )
            features[bank] = {
                "dataset": file_ref(dataset),
                "metadata": file_ref(dataset.with_suffix(dataset.suffix + ".json")),
            }

        manifest_tracks[identifier] = {
            "source": track["source"],
            "split": track["split"],
            "dataset_id": track["dataset_id"],
            "complexity": track.get("complexity"),
            "source_wav": file_ref(wav),
            "teacher_events": file_ref(teacher_events),
            "neuralnote": file_ref(teacher_midi),
            "dataset_reference": dataset_reference_ref,
            "dataset_reference_unavailable_reason": track.get(
                "dataset_reference_unavailable_reason"
            ),
            "features": features,
        }
        frozen_selection.append(
            {
                "package_id": identifier,
                "source": track["source"],
                "split": track["split"],
                "dataset_id": track["dataset_id"],
                "complexity": track.get("complexity"),
                "source_wav": file_ref(wav),
                "teacher_events": file_ref(teacher_events),
                "dataset_reference": dataset_reference_ref,
            }
        )

    selection_copy = output / "selected-tracks.frozen.json"
    write_json(
        selection_copy,
        {
            "schema": "tmgm-complex-dataset-selection-frozen-v1",
            "source_selection": file_ref(args.selection.resolve()),
            "tracks": frozen_selection,
        },
    )
    manifest["tracks"] = manifest_tracks
    manifest["complex_dataset_selection"] = {
        "schema": "tmgm-complex-dataset-listening-v1",
        "selection": file_ref(selection_copy),
        "frozen_base_manifest": file_ref(base_manifest_path),
        "threshold_or_model_fitting": False,
        "production_modified": False,
        "source_wavs_copied": False,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
