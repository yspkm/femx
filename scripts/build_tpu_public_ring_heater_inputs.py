#!/usr/bin/env python3
"""Build the immutable 32-partition M5d public-ring TPU input on a CPU controller."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("ring-heater TPU input preparation requires explicit JAX_PLATFORMS=cpu")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    import_path = str(import_root)
    while import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
import numpy as np

from femx.applications import prepare_public_ring_heater_forward_plan
from femx.backends.jax.partition import balanced_lexicographic_cell_owners
from femx.meshing.gmsh import (
    PublicRingHeater3D,
    read_gmsh_msh_3d,
    ring_heater_mesh_profile,
)
from scripts._tpu_public_ring_heater_plan import (
    read_public_ring_heater_cpu_authority,
    write_public_ring_heater_tpu_artifact,
)

EXPECTED_FINE_MSH_SHA256 = "c484d4be5f52a59b93ba0904f79bef98d7dea0aceb8976e269b49cdc739d0a69"
EXPECTED_FINE_NODE_COUNT = 521_442
EXPECTED_FINE_TETRAHEDRON_COUNT = 3_179_879
EXPECTED_FINE_CONDUCTOR_TETRAHEDRON_COUNT = 134_331


def _sha256_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


def build_inputs(
    mesh_path: Path,
    authority_record_path: Path,
    authority_state_path: Path,
    output_root: Path,
    *,
    partition_count: int,
    source_commit: str,
) -> dict[str, object]:
    """Prepare, validate, and publish one source-pinned fine-mesh artifact."""

    if partition_count != 32:
        raise ValueError("M5d public-ring input requires exactly 32 FEM partitions")
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("fine"))
    imported = read_gmsh_msh_3d(
        mesh_path.resolve(strict=True),
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )
    if imported.record.source_sha256 != EXPECTED_FINE_MSH_SHA256:
        raise RuntimeError("M5d source MSH does not match the admitted fine mesh")
    mesh = imported.mesh
    if (
        mesh.geometry.node_count != EXPECTED_FINE_NODE_COUNT
        or mesh.topology.cell_count != EXPECTED_FINE_TETRAHEDRON_COUNT
    ):
        raise RuntimeError("M5d source MSH counts do not match the admitted fine mesh")
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    owners = balanced_lexicographic_cell_owners(
        coordinates,
        cells,
        partition_count=partition_count,
    )
    forward = prepare_public_ring_heater_forward_plan(
        imported,
        recipe,
        owners,
        partition_count=partition_count,
    )
    if forward.tet4.current_layout.topology.cell_count != (
        EXPECTED_FINE_CONDUCTOR_TETRAHEDRON_COUNT
    ):
        raise RuntimeError("M5d conductor count does not match the admitted fine mesh")

    authority = read_public_ring_heater_cpu_authority(
        authority_record_path.resolve(strict=True),
        authority_state_path.resolve(strict=True),
    )
    authority.validate(forward)
    provenance = authority.record.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source_msh_sha256") != (
        imported.record.source_sha256
    ):
        raise RuntimeError("CPU authority does not bind the admitted fine source MSH")

    manifest = write_public_ring_heater_tpu_artifact(
        output_root.absolute(),
        forward,
        authority,
        source_commit=source_commit,
        source_msh_sha256=imported.record.source_sha256,
        partition_owner_sha256=_sha256_int64(owners),
        silicon_ring_cell_ids=np.asarray(mesh.tag("silicon_ring").entity_ids, dtype=np.int64),
        tin_heater_cell_ids=np.asarray(mesh.tag("tin_heater").entity_ids, dtype=np.int64),
    )
    runtime = manifest["runtime_plan"]
    model = manifest["model"]
    assert isinstance(runtime, dict) and isinstance(model, dict)
    return {
        "schema_version": "femx.public-ring-heater.tpu_forward_input_build/v1",
        "status": "passed",
        "output_root": str(output_root.absolute()),
        "artifact_logical_sha256": manifest["logical_sha256"],
        "runtime_plan_sha256": runtime["sha256"],
        "source_plan_sha256": runtime["source_plan_sha256"],
        "source_msh_sha256": manifest["source_msh_sha256"],
        "partition_owner_sha256": manifest["partition"]["owner_sha256"],  # type: ignore[index]
        "partition_count": partition_count,
        "node_count": model["node_count"],
        "tetrahedron_count": model["tetrahedron_count"],
        "conductor_tetrahedron_count": model["conductor_tetrahedron_count"],
        "total_array_file_bytes": manifest["total_array_file_bytes"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--authority-record", required=True, type=Path)
    parser.add_argument("--authority-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = build_inputs(
        arguments.mesh,
        arguments.authority_record,
        arguments.authority_state,
        arguments.output,
        partition_count=arguments.partitions,
        source_commit=arguments.source_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
