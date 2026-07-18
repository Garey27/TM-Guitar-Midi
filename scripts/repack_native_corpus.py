from __future__ import annotations

import argparse
import json
from pathlib import Path

from tmgm_rt.native_repack import repack_native_corpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TMGMDAT corpus from one export's binary features and "
            "another row-aligned export's labels after fail-closed validation."
        )
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = repack_native_corpus(
        args.feature_root,
        args.label_root,
        args.output_root,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "splits": [
                    {
                        "path": str(result.path),
                        "frames": result.header.frame_count,
                        "features": result.header.feature_count,
                        "payload_sha256": result.header.payload_sha256.hex(),
                        "file_sha256": result.file_sha256,
                        "section_sha256": {
                            "features": result.feature_section_sha256,
                            "activity": result.activity_section_sha256,
                            "onset": result.onset_section_sha256,
                            "onset_indices": result.onset_indices_sha256,
                        },
                    }
                    for result in results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
