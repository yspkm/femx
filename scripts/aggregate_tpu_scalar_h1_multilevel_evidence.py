#!/usr/bin/env python3
"""Admit every process record from one physical scalar-H1 multilevel TPU run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project_root / "src"), str(project_root)]

from femx.validation.tpu_scalar_h1_multilevel_evidence import (
    aggregate_tpu_scalar_h1_multilevel_process_evidence,
)
from scripts.aggregate_tpu_scalar_h1_collective_evidence import (
    _load_process_record,
    _publish,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "process_metrics",
        nargs="+",
        type=Path,
        help="one raw results/process-metrics.json path per JAX process",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_inputs = tuple(path.resolve(strict=True) for path in args.process_metrics)
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError("process metrics paths must be unique")
    records = [_load_process_record(path) for path in args.process_metrics]
    evidence = aggregate_tpu_scalar_h1_multilevel_process_evidence(records)
    _publish(args.output, evidence.canonical_json())
    runtime = evidence.canonical_data()["runtime"]
    assert isinstance(runtime, dict)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "logical_sha256": evidence.digest(),
                "process_count": runtime["process_count"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
