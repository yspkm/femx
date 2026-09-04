#!/usr/bin/env python3
"""Admit all process records from one fine public-ring physical TPU forward run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    import_path = str(import_root)
    while import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

from femx.validation.tpu_public_ring_heater_evidence import (
    aggregate_tpu_public_ring_heater_process_evidence,
)


def _load(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"process evidence must be a regular non-symlink file: {path}")
    if path.stat().st_size <= 0 or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"process evidence size is outside the admitted range: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"process evidence must be a JSON object: {path}")
    return value


def _publish(path: Path, text: str) -> None:
    destination = path.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite aggregate evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process_metrics", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    inputs = tuple(path.resolve(strict=True) for path in arguments.process_metrics)
    if len(inputs) != len(set(inputs)):
        raise ValueError("public-ring process evidence paths must be unique")
    evidence = aggregate_tpu_public_ring_heater_process_evidence([_load(path) for path in inputs])
    _publish(arguments.output, evidence.canonical_json())
    runtime = evidence.canonical_data()["runtime"]
    assert isinstance(runtime, dict)
    print(
        json.dumps(
            {
                "output": str(arguments.output.absolute()),
                "logical_sha256": evidence.digest(),
                "process_count": runtime["process_count"],
                "global_device_count": runtime["global_device_count"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
