from __future__ import annotations

import argparse
from pathlib import Path

from tmgm_rt.config import ContextConfig, FrontendConfig, TargetConfig
from tmgm_rt.dataset import read_corpus
from tmgm_rt.native_export import (
    cache_split_tracks,
    export_cached_split,
    fit_or_load_global_binarizer,
)


def _track_limit(value: str) -> int | None:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("track count must be an integer") from error
    if count < 0:
        raise argparse.ArgumentTypeError("track count cannot be negative")
    return None if count == 0 else count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache, globally binarize and stream-pack the sampled train/validation "
            "corpus for the native C++/CUDA trainer. A zero track limit means all "
            "available tracks."
        )
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="persistent extraction cache (default: OUTPUT_DIR/cache)",
    )
    parser.add_argument("--train-tracks", type=_track_limit, default=None)
    parser.add_argument("--validation-tracks", type=_track_limit, default=None)
    parser.add_argument("--frames-per-track", type=int, default=800)
    parser.add_argument(
        "--train-sampling",
        choices=("balanced", "natural"),
        default="balanced",
        help=(
            "train frame selection: category-balanced sampling with replacement "
            "(default), or uniform temporal natural sampling"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--harmonic-local-contrast",
        action="store_true",
        help="measure side energy around every harmonic instead of only the fundamental",
    )
    parser.add_argument(
        "--contrast-offset-semitones",
        type=float,
        default=0.5,
        help="side-reference distance used by local contrast",
    )
    parser.add_argument(
        "--expose-harmonic-local-profile",
        action="store_true",
        help=(
            "append one locally whitened channel per pitch/harmonic; requires "
            "--harmonic-local-contrast"
        ),
    )
    parser.add_argument(
        "--contrast-attack-features",
        action="store_true",
        help=(
            "append corrected-contrast positive-flux and fast/slow channels; "
            "requires --harmonic-local-contrast"
        ),
    )
    parser.add_argument("--pack-batch-rows", type=int, default=256)
    parser.add_argument("--quantile-scan-batch-rows", type=int, default=512)
    parser.add_argument("--onset-row-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--onset-width-frames",
        type=int,
        default=2,
        help="number of consecutive positive onset target frames",
    )
    parser.add_argument(
        "--onset-delay-frames",
        type=int,
        default=0,
        help=(
            "causal target delay in frontend frames; 2 means about 23 ms at "
            "the default 22050/256 timebase"
        ),
    )
    parser.add_argument("--rebuild-track-cache", action="store_true")
    parser.add_argument("--refit-binarizer", action="store_true")
    parser.add_argument("--repack", action="store_true")
    args = parser.parse_args()

    if args.frames_per_track <= 0:
        parser.error("--frames-per-track must be positive")
    if args.pack_batch_rows <= 0 or args.quantile_scan_batch_rows <= 0:
        parser.error("batch sizes must be positive")
    if args.onset_row_multiplier <= 0:
        parser.error("--onset-row-multiplier must be positive")
    if args.onset_width_frames <= 0:
        parser.error("--onset-width-frames must be positive")
    if args.onset_delay_frames < 0:
        parser.error("--onset-delay-frames cannot be negative")
    if args.contrast_offset_semitones <= 0.0:
        parser.error("--contrast-offset-semitones must be positive")
    if (
        args.expose_harmonic_local_profile or args.contrast_attack_features
    ) and not args.harmonic_local_contrast:
        parser.error(
            "harmonic profile/contrast attack features require "
            "--harmonic-local-contrast"
        )

    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir or output_dir / "cache").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frontend = FrontendConfig(
        harmonic_local_contrast=args.harmonic_local_contrast,
        contrast_offset_semitones=args.contrast_offset_semitones,
        expose_harmonic_local_profile=args.expose_harmonic_local_profile,
        contrast_attack_features=args.contrast_attack_features,
    )
    context = ContextConfig()
    targets = TargetConfig(
        onset_width_frames=args.onset_width_frames,
        onset_delay_frames=args.onset_delay_frames,
    )

    train_entries = read_corpus(
        args.corpus, "train", args.train_tracks, args.seed
    )
    validation_entries = read_corpus(
        args.corpus, "validation", args.validation_tracks, args.seed
    )
    print(
        f"selected train={len(train_entries)} validation={len(validation_entries)} "
        f"frames_per_track={args.frames_per_track} "
        f"train_sampling={args.train_sampling} "
        f"onset_delay_frames={args.onset_delay_frames} "
        f"onset_width_frames={args.onset_width_frames} "
        f"harmonic_local_contrast={args.harmonic_local_contrast} "
        f"contrast_offset_semitones={args.contrast_offset_semitones} "
        f"expose_harmonic_local_profile="
        f"{args.expose_harmonic_local_profile} "
        f"contrast_attack_features={args.contrast_attack_features}"
    )
    train_tracks = cache_split_tracks(
        train_entries,
        args.teacher_root,
        cache_dir,
        "train",
        args.frames_per_track,
        args.seed,
        frontend,
        context,
        targets,
        train_sampling=args.train_sampling,
        force=args.rebuild_track_cache,
    )
    validation_tracks = cache_split_tracks(
        validation_entries,
        args.teacher_root,
        cache_dir,
        "validation",
        args.frames_per_track,
        args.seed,
        frontend,
        context,
        targets,
        train_sampling=args.train_sampling,
        force=args.rebuild_track_cache,
    )

    binarizer, binarizer_signature, binarizer_sha256 = (
        fit_or_load_global_binarizer(
            train_tracks,
            cache_dir,
            output_dir / "global-quantile-thermometer.npz",
            constant_scan_batch_rows=args.quantile_scan_batch_rows,
            frontend=frontend,
            context=context,
            force=args.refit_binarizer,
        )
    )
    export_cached_split(
        output_dir / "train.tmgd",
        train_tracks,
        "train",
        binarizer,
        binarizer_signature,
        binarizer_sha256,
        frontend,
        context,
        targets,
        seed=args.seed,
        batch_rows=args.pack_batch_rows,
        onset_row_multiplier=args.onset_row_multiplier,
        force=args.repack,
    )
    export_cached_split(
        output_dir / "validation.tmgd",
        validation_tracks,
        "validation",
        binarizer,
        binarizer_signature,
        binarizer_sha256,
        frontend,
        context,
        targets,
        seed=args.seed,
        batch_rows=args.pack_batch_rows,
        onset_row_multiplier=args.onset_row_multiplier,
        force=args.repack,
    )
    print(f"native corpus export complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
