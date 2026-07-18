from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tmgm_rt.binarize import QuantileThermometer
from tmgm_rt.config import ContextConfig, FrontendConfig, TargetConfig
from tmgm_rt.dataset import build_track_examples, read_corpus
from tmgm_rt.native_dataset import onset_training_indices, write_native_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the proven one-track overfit data for a native trainer."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--source", default="guitarset")
    parser.add_argument("--id", default="00_Jazz1-130-D_comp_mix")
    parser.add_argument("--onset-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entry = next(
        candidate
        for candidate in read_corpus(args.corpus, "train")
        if candidate.source == args.source and candidate.identifier == args.id
    )
    frontend = FrontendConfig()
    context = ContextConfig()
    targets = TargetConfig()
    continuous, truth, _, _, categories = build_track_examples(
        entry,
        args.teacher_root,
        frontend,
        context,
        targets,
        frames_per_track=10_000_000,
        seed=args.seed,
        balanced_sampling=False,
    )
    binarizer = QuantileThermometer()
    binary = binarizer.fit_transform(continuous)
    note_count = frontend.note_count
    activity = truth[:, :note_count]
    onset = truth[:, note_count : 2 * note_count]
    indices = onset_training_indices(truth, note_count, args.onset_rows, args.seed)
    header = write_native_dataset(
        args.output,
        binary,
        activity,
        onset,
        indices,
        midi_min=frontend.midi_min,
        sample_rate=frontend.sample_rate,
        hop_size=frontend.hop_size,
        seed=args.seed,
    )

    metadata = {
        "format": "TMGMDAT",
        "version": 1,
        "binary_word": "little-endian uint64; column c is bit c%64 of word c//64",
        "source": entry.source,
        "id": entry.identifier,
        "input": str(entry.input_path),
        "teacher_base": str(args.teacher_root / entry.output_relative),
        "frontend": asdict(frontend),
        "context": asdict(context),
        "targets": asdict(targets),
        "quantiles": list(binarizer.quantiles),
        "continuous_features": int(continuous.shape[1]),
        "raw_thermometer_literals": int(binarizer.keep_columns.size),
        "kept_binary_features": int(binary.shape[1]),
        "category_counts": categories,
        "header": {
            **asdict(header),
            "payload_sha256": header.payload_sha256.hex(),
            "feature_fingerprint_sha256": (
                header.feature_fingerprint_sha256.hex()
            ),
        },
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(sidecar)
    print(json.dumps(metadata["header"], indent=2, sort_keys=True))
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
