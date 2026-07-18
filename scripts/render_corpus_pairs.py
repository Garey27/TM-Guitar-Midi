from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np

from tmgm_rt.audio import load_audio_mono_channel_zero
from tmgm_rt.context import stack_causal_context
from tmgm_rt.dataset import read_corpus
from tmgm_rt.midi import (
    NoteStateConfig,
    stabilize_frame_predictions,
    write_frame_predictions,
    write_teacher_events,
)
from tmgm_rt.model import load_bundle
from tmgm_rt.stft_plus import extract_stft_plus


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tracks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    entries = read_corpus(args.corpus, args.split, args.tracks, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    decoder = NoteStateConfig()
    manifest: list[dict[str, str]] = []
    for index, entry in enumerate(entries, start=1):
        stem = f"{index:02d}_{safe_name(entry.source)}_{safe_name(entry.identifier)}"
        print(f"[{index}/{len(entries)}] {entry.source}/{entry.identifier}")
        audio = load_audio_mono_channel_zero(
            entry.input_path, bundle.frontend.sample_rate
        )
        features = stack_causal_context(
            extract_stft_plus(audio, bundle.frontend), bundle.context
        )
        scores = bundle.predict_scores(features)
        raw = (scores >= bundle.output_thresholds[None, :]).astype(np.uint32)
        stable = stabilize_frame_predictions(
            raw, bundle.frontend.note_count, decoder
        )
        raw_path = args.output / f"{stem}.tm-raw.mid"
        stable_path = args.output / f"{stem}.tm-stable.mid"
        teacher_path = args.output / f"{stem}.neuralnote.mid"
        write_frame_predictions(
            raw_path,
            raw,
            bundle.frontend.midi_min,
            bundle.frontend.note_count,
            bundle.frontend.frame_seconds,
        )
        write_frame_predictions(
            stable_path,
            stable,
            bundle.frontend.midi_min,
            bundle.frontend.note_count,
            bundle.frontend.frame_seconds,
        )
        teacher_base = args.teacher_root / entry.output_relative
        write_teacher_events(
            teacher_path,
            teacher_base.with_suffix(".events.tsv"),
            bundle.frontend.midi_min,
            bundle.frontend.midi_max,
        )
        manifest.append(
            {
                "index": str(index),
                "source": entry.source,
                "id": entry.identifier,
                "group": entry.group,
                "wav": str(entry.input_path),
                "tm_raw": str(raw_path),
                "tm_stable": str(stable_path),
                "neuralnote": str(teacher_path),
            }
        )
    with (args.output / "manifest.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
