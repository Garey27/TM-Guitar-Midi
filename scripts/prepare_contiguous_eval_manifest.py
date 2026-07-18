from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf

from tmgm_rt.contiguous_eval import MANIFEST_SCHEMA, sha256_file


# Fixed after inspecting duration and teacher-event availability. Keeping the
# exact IDs here makes future manifests independent of filesystem enumeration.
SELECTION = (
    ("goat", "test", "item_39", "goat-item_39"),
    ("goat", "test", "item_29", "goat-item_29"),
    (
        "guitarset",
        "test",
        "05_Jazz1-200-B_comp_mix",
        "guitarset-05_Jazz1-200-B_comp_mix",
    ),
    (
        "guitarset",
        "test",
        "05_BN1-147-Gb_solo_mix",
        "guitarset-05_BN1-147-Gb_solo_mix",
    ),
    (
        "guitar-techs",
        "validation",
        "P3_music_audio_directinput_directinput_11.",
        "guitar-techs-directinput_11",
    ),
    (
        "guitar-techs",
        "validation",
        "P3_music_audio_directinput_directinput_08.",
        "guitar-techs-directinput_08",
    ),
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the deterministic six-track contiguous evaluation manifest."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    corpus = args.corpus.resolve()
    teacher_root = args.teacher_root.resolve()
    output_root = args.output_root.resolve()
    with corpus.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    lookup = {(row["source"], row["split"], row["id"]): row for row in rows}

    tracks = []
    for source, split, track_id, key in SELECTION:
        try:
            row = lookup[(source, split, track_id)]
        except KeyError as error:
            raise ValueError(
                f"selected corpus entry is missing: {(source, split, track_id)}"
            ) from error
        wav = Path(row["input"]).resolve()
        events = (teacher_root / f"{row['output_rel']}.events.tsv").resolve()
        if not wav.is_file():
            raise FileNotFoundError(f"selected WAV does not exist: {wav}")
        if not events.is_file():
            raise FileNotFoundError(f"selected teacher events do not exist: {events}")
        info = sf.info(wav)
        feature_sets = {}
        for feature_name in ("plain_d2w3", "hcontrast15_d2w3"):
            relative = Path("features") / feature_name / f"{key}.tmgd"
            feature_sets[feature_name] = {
                "dataset": relative.as_posix(),
                "metadata": f"{relative.as_posix()}.json",
            }
        tracks.append(
            {
                "key": key,
                "source": source,
                "corpus_split": split,
                "evaluation_role": "test",
                "id": track_id,
                "group": row["group"],
                "wav": str(wav),
                "events": str(events),
                "wav_sha256": sha256_file(wav),
                "events_sha256": sha256_file(events),
                "duration_seconds": info.duration,
                "source_sample_rate": info.samplerate,
                "source_channels": info.channels,
                "feature_sets": feature_sets,
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "corpus": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "teacher_root": str(teacher_root),
        "selection_policy": {
            "count_per_source": 2,
            "goat": "two short official-test tracks, fixed IDs",
            "guitarset": (
                "short official-test comp and solo tracks from different pieces, "
                "fixed IDs"
            ),
            "guitar-techs": (
                "the corpus has no official test rows; two shortest distinct "
                "direct-input validation performances are held out as evaluation-role=test"
            ),
            "sparse_sampled_validation_used": False,
            "track_order": "the explicit order in this manifest",
        },
        "timing_views": {
            "target_aligned": (
                "teacher start_frame plus the requested training label delay"
            ),
            "wall_clock": "unshifted teacher start_frame (delay zero)",
        },
        "tracks": tracks,
    }
    manifest_path = output_root / "manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    columns = (
        "key",
        "source",
        "corpus_split",
        "evaluation_role",
        "id",
        "group",
        "duration_seconds",
        "wav",
        "events",
        "wav_sha256",
        "events_sha256",
    )
    lines = ["\t".join(columns)]
    for track in tracks:
        lines.append("\t".join(str(track[column]) for column in columns))
    _atomic_text(output_root / "manifest.tsv", "\n".join(lines) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "tracks": len(tracks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
