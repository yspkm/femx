#!/usr/bin/env python3
"""Admit every process record from one physical Elmer/JAX waveguide FDTDX TPU run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femx.validation.tpu_fdtdx_waveguide_source_evidence import (
    aggregate_tpu_fdtdx_waveguide_source_process_evidence,
)
from scripts.aggregate_tpu_fdtdx_mode_source_evidence import (
    _load_process_record,
    _publish,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "process_metrics",
        nargs="+",
        type=Path,
        help="one raw waveguide process-metrics.json per initialized JAX process",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = tuple(path.resolve(strict=True) for path in args.process_metrics)
    if len(resolved) != len(set(resolved)):
        raise ValueError("waveguide process metrics paths must be unique")
    evidence = aggregate_tpu_fdtdx_waveguide_source_process_evidence(
        [_load_process_record(path) for path in args.process_metrics]
    )
    _publish(args.output, evidence.canonical_json())
    runtime = evidence.canonical_data()["runtime"]
    assert isinstance(runtime, dict)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": evidence.digest(),
                "process_count": runtime["process_count"],
                "global_device_count": runtime["global_device_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
