from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tmgm_rt.contiguous_eval import evaluate_manifest_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full contiguous teacher timelines instead of sparse sampled "
            "validation rows. Score files are <track-key>.activity.tsv and "
            "<track-key>.onset.tsv under --scores-root."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-onset-delay-frames", type=int, default=2)
    parser.add_argument("--onset-width-frames", type=int, default=3)
    parser.add_argument(
        "--target-aligned-tolerances", type=int, nargs="+", default=(2, 3, 4)
    )
    parser.add_argument(
        "--wall-clock-tolerances", type=int, nargs="+", default=(2, 3, 4, 6)
    )
    parser.add_argument("--retrigger-silence-frames", type=int, default=3)
    parser.add_argument("--chord-window-frames", type=int, default=3)
    parser.add_argument("--low-midi-max", type=int, default=59)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args(argv)

    result = evaluate_manifest_scores(
        args.manifest,
        args.feature_set,
        args.scores_root,
        training_onset_delay_frames=args.training_onset_delay_frames,
        onset_width_frames=args.onset_width_frames,
        target_aligned_tolerances=args.target_aligned_tolerances,
        wall_clock_tolerances=args.wall_clock_tolerances,
        retrigger_silence_frames=args.retrigger_silence_frames,
        chord_window_frames=args.chord_window_frames,
        low_midi_max=args.low_midi_max,
        batch_rows=args.batch_rows,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
