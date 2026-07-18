from __future__ import annotations

import argparse
import json
from pathlib import Path

from tmgm_rt.native_wav_export import export_wav_native


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export every causal STFT+/context frame of a WAV through a fitted "
            "quantile thermometer into an unlabeled native TMGMDAT file."
        )
    )
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--binarizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-metadata",
        type=Path,
        help=(
            "TMGD metadata JSON whose frontend/context and binarizer identity "
            "must match this inference export"
        ),
    )
    parser.add_argument("--batch-frames", type=int, default=512)
    args = parser.parse_args()
    if args.batch_frames <= 0:
        parser.error("--batch-frames must be positive")

    header = export_wav_native(
        args.wav,
        args.binarizer,
        args.output,
        reference_metadata=args.reference_metadata,
        batch_frames=args.batch_frames,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "frames": header.frame_count,
                "features": header.feature_count,
                "notes": header.note_count,
                "payload_sha256": header.payload_sha256.hex(),
                "metadata": str(
                    args.output.resolve().with_suffix(args.output.suffix + ".json")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
