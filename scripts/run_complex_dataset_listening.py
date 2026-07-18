from __future__ import annotations

"""Run and strongly audit a prepared frozen complex-dataset package."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import mido

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_complex_dataset_listening import DEFAULT_OUTPUT, MIDI_MAX, MIDI_MIN
from prepare_multibank_cpu_listening import sha256_file, write_json
from run_strict_cap16_v3_listening import validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def audit_midi_integrity(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    active: Counter[tuple[int, int]] = Counter()
    note_ons = 0
    note_offs = 0
    maximum_polyphony = 0
    pitches: set[int] = set()
    for message in mido.merge_tracks(midi.tracks):
        if message.type == "note_on" and message.velocity > 0:
            if not MIDI_MIN <= message.note <= MIDI_MAX:
                raise ValueError(f"MIDI pitch outside {MIDI_MIN}..{MIDI_MAX}: {path}")
            key = (message.channel, message.note)
            active[key] += 1
            note_ons += 1
            pitches.add(message.note)
            maximum_polyphony = max(maximum_polyphony, sum(active.values()))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            key = (message.channel, message.note)
            if active[key] <= 0:
                raise ValueError(f"orphan NoteOff for channel/pitch {key}: {path}")
            active[key] -= 1
            note_offs += 1
    dangling = {str(key): count for key, count in active.items() if count}
    if dangling:
        raise ValueError(f"dangling NoteOn voices in {path}: {dangling}")
    if note_ons <= 0 or note_ons != note_offs or midi.length <= 0.0:
        raise ValueError(f"empty or unbalanced MIDI: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "type": midi.type,
        "tracks": len(midi.tracks),
        "note_ons": note_ons,
        "note_offs": note_offs,
        "distinct_pitches": len(pitches),
        "maximum_voice_polyphony": maximum_polyphony,
        "duration_seconds": midi.length,
        "per_channel_pitch_balanced": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "manifest.json")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    selection = manifest.get("complex_dataset_selection")
    if not isinstance(selection, dict) or selection.get("threshold_or_model_fitting") is not False:
        raise ValueError("manifest lacks frozen complex-dataset policy")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_strict_cap16_v3_listening.py"),
            "--manifest", str(manifest_path),
        ],
        cwd=ROOT,
        check=True,
    )

    output = manifest_path.parent
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    readme_rows: list[str] = []
    for identifier, track in manifest["tracks"].items():
        listening = output / "listening" / identifier
        reference = track.get("dataset_reference")
        if reference is not None:
            shutil.copy2(reference["path"], listening / "dataset-reference.mid")
        midi_paths = sorted(listening.glob("*.mid"))
        run["tracks"][identifier]["midi"] = [
            audit_midi_integrity(path) for path in midi_paths
        ]
        run["tracks"][identifier]["source"] = track["source"]
        run["tracks"][identifier]["split"] = track["split"]
        run["tracks"][identifier]["dataset_id"] = track["dataset_id"]
        run["tracks"][identifier]["complexity"] = track.get("complexity")
        run["tracks"][identifier]["source_wav"] = track["source_wav"]
        run["tracks"][identifier]["dataset_reference_available"] = reference is not None
        complexity = track.get("complexity") or {}
        complexity_text = (
            f"duration {float(complexity['duration_seconds']):.1f}s; "
            f"teacher events {int(complexity['teacher_events'])}; "
            f"mean/max polyphony {float(complexity['mean_polyphony']):.2f}/"
            f"{int(complexity['max_polyphony'])}; "
            if complexity else ""
        )
        readme_rows.append(
            f"- `{identifier}` — {track['source']} / {track['split']}; "
            f"{complexity_text}"
            f"WAV `{track['source_wav']['path']}`; SHA-256 `{track['source_wav']['sha256']}`; "
            f"dataset reference: {'yes' if reference is not None else 'unavailable'}."
        )
    run["schema"] = "tmgm-complex-dataset-listening-run-v1"
    run["selection"] = selection
    run["thresholds_fitted_on_selected_tracks"] = False
    run["production_modified"] = False
    write_json(run_path, run)

    readme = output / "README.md"
    readme.write_text(
        "# Frozen strict-cap16-v3: six complex dataset tracks\n\n"
        "The five feature banks, all TM members, member thresholds, and final "
        "activity/onset thresholds are reused verbatim from strict-cap16-v3. "
        "These tracks are inference/listening only: no fitting or threshold search "
        "was performed, and production/VST artifacts were not modified.\n\n"
        "Each folder contains `tm-raw.mid`, `tm-stable.mid`, "
        "`tm-balanced-raw.mid`, `tm-balanced-stable.mid`, and `neuralnote.mid`. "
        "`dataset-reference.mid` is included only where an independently usable "
        "absolute-time annotation was available. Source WAVs are not copied.\n\n"
        "## Tracks\n\n" + "\n".join(readme_rows) + "\n\n"
        "All MIDI files passed a per-channel/per-pitch voice-counter audit: no "
        "orphan NoteOff, no dangling NoteOn, nonempty duration, and pitch range "
        "40..88. Duplicate same-pitch voices are counted rather than collapsed.\n",
        encoding="utf-8",
    )
    checksum_paths = [manifest_path, run_path, readme, output / "selected-tracks.frozen.json"]
    checksum_paths.extend(sorted((output / "references").rglob("*.mid")))
    checksum_paths.extend(sorted((output / "listening").rglob("*.mid")))
    (output / "checksums.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="ascii",
    )
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
