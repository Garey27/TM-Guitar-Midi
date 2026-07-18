from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tmgm_rt.native_score_eval import evaluate_score_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream native prediction TSVs and report held-out metrics."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--scores",
        type=Path,
        nargs="+",
        required=True,
        help="one activity and/or onset TSV emitted by tmgm_predict",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="defaults to <dataset>.json",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output file")
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args(argv)

    sidecar = args.sidecar or Path(str(args.dataset) + ".json")
    result = evaluate_score_files(
        args.dataset,
        args.scores,
        sidecar,
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
