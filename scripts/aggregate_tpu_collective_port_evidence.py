#!/usr/bin/env python3
"""Admit every process record from one physical TPU collective run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from femx.validation.tpu_collective_evidence import (
    aggregate_tpu_collective_process_evidence,
)

MAX_PROCESS_RECORD_BYTES = 4 * 1024 * 1024


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _load_process_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"process metrics must be a regular non-symlink file: {path}")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_PROCESS_RECORD_BYTES:
        raise ValueError(f"process metrics size is outside the admitted range: {path}")
    value = json.loads(
        resolved.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"process metrics root must be an object with string keys: {path}")
    return value


def _publish(path: Path, payload: str) -> None:
    destination = path.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite aggregate evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.incomplete")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"aggregate temporary path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


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
    records: list[dict[str, object]] = [_load_process_record(path) for path in args.process_metrics]
    evidence = aggregate_tpu_collective_process_evidence(records)
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
