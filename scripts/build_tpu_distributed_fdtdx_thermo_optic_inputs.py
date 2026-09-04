#!/usr/bin/env python3
"""Build the immutable FEM-to-FDTDX input authority for the physical 8/32 TPU gate."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    import_path = str(import_root)
    while import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

from femx.backends.jax.distributed_electrothermal import (  # noqa: E402
    prepare_distributed_electrothermal_plan,
)
from femx.interop.fdtdx import (  # noqa: E402
    build_triangle_p1_sampling_plan,
    prepare_distributed_triangle_p1_sampling_plan,
)
from scripts import (  # noqa: E402
    build_tpu_distributed_electrothermal_inputs as electrothermal_builder,
)
from scripts._distributed_electrothermal_case import (  # noqa: E402
    bind_jax_self_consistent_microheater,
    distributed_electrothermal_iteration_policy,
)
from scripts._distributed_fdtdx_thermo_optic_case import (  # noqa: E402
    FDTDX_TIME_STEPS,
    MESH_AXIS_NAME,
    build_scene,
    device_contract,
    scene_metadata,
    thermo_optic_law,
    verify_locked_fdtdx,
)
from scripts._tpu_distributed_electrothermal_plan import (  # noqa: E402
    read_distributed_electrothermal_artifact,
)
from scripts._tpu_distributed_fdtdx_thermo_optic_plan import (  # noqa: E402
    write_distributed_fdtdx_thermo_optic_artifact,
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _source_commit(*, require_clean: bool) -> str:
    commit = _git("rev-parse", "HEAD")
    if require_clean and _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("physical TPU input authority requires a clean femx source worktree")
    return commit


def _cell_owners(
    coordinates: np.ndarray,
    cells: np.ndarray,
    partition_count: int,
) -> np.ndarray:
    centroids_x = np.mean(coordinates[cells, 0], axis=1)
    width = float(np.max(coordinates[:, 0]))
    owners = np.minimum(
        (partition_count * centroids_x / width).astype(np.int64),
        partition_count - 1,
    )
    if not np.array_equal(np.unique(owners), np.arange(partition_count, dtype=np.int64)):
        raise ValueError("bounded electrothermal mesh must assign cells to every partition")
    return np.asarray(owners, dtype=np.int64)


def build_inputs(
    output_root: Path,
    *,
    intervals: int,
    partition_count: int,
    require_clean: bool = True,
) -> dict[str, object]:
    """Build one nested float64 authority and exact float32 FDTDX transfer contract."""

    if intervals <= 0 or intervals % 2 != 0:
        raise ValueError("coupled TPU input intervals must be a positive even integer")
    if partition_count <= 1:
        raise ValueError("physical coupled TPU input requires multiple partitions")
    if jax.default_backend() != "cpu" or not bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("coupled TPU input authority requires local CPU JAX with x64 enabled")
    fdtdx = importlib.import_module("fdtdx")
    verify_locked_fdtdx(fdtdx)
    source_commit = _source_commit(require_clean=require_clean)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="femx-coupled-input-", dir=output_root.parent) as raw:
        nested_root = Path(raw) / "electrothermal"
        electrothermal_builder.build_inputs(
            nested_root,
            intervals=intervals,
            partition_count=partition_count,
        )
        electrothermal = read_distributed_electrothermal_artifact(nested_root)
        if electrothermal.manifest.get("source_commit") != source_commit:
            raise RuntimeError("nested electrothermal artifact used a different femx commit")

        system = bind_jax_self_consistent_microheater(
            intervals=intervals,
            iteration=distributed_electrothermal_iteration_policy(),
        )
        payload = system.current._engine.payload
        coordinates = np.asarray(payload.coordinates, dtype=np.float64)
        cells = np.asarray(payload.cells, dtype=np.int64)
        reconstructed_plan = prepare_distributed_electrothermal_plan(
            system,
            _cell_owners(coordinates, cells, partition_count),
            partition_count=partition_count,
        )
        if reconstructed_plan.digest() != electrothermal.plan.digest():
            raise RuntimeError("physical mesh does not reconstruct the nested electrothermal plan")

        scene = build_scene(fdtdx, jax, jnp, backend="cpu")
        sampling = build_triangle_p1_sampling_plan(
            coordinates,
            cells,
            tuple(np.asarray(axis, dtype=np.float64) for axis in scene.target_coordinates),
            plane_axes=(0, 2),
        )
        transfer = prepare_distributed_triangle_p1_sampling_plan(
            sampling,
            electrothermal.plan.layout.transport.cell_ids,
            source_layout_sha256=electrothermal.plan.layout.digest(),
            mesh_axis_name=MESH_AXIS_NAME,
        )
        law = thermo_optic_law()
        contract = device_contract(sampling, parameter_dtype="float32")
        manifest = write_distributed_fdtdx_thermo_optic_artifact(
            output_root,
            electrothermal_root=nested_root,
            source_commit=source_commit,
            sampling=sampling,
            transfer=transfer,
            law=law,
            contract=contract,
            scene=scene_metadata(time_steps=FDTDX_TIME_STEPS),
        )
    return dict(manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--intervals", type=int, default=16)
    parser.add_argument("--partitions", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = build_inputs(
        arguments.output,
        intervals=arguments.intervals,
        partition_count=arguments.partitions,
    )
    electrothermal = manifest["electrothermal"]
    transfer = manifest["transfer"]
    arrays = manifest["arrays"]
    assert isinstance(electrothermal, dict)
    assert isinstance(transfer, dict)
    assert isinstance(arrays, dict)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "electrothermal_plan_sha256": electrothermal["plan_sha256"],
                "transfer_operator_sha256": transfer["operator_sha256"],
                "arrays_sha256": arrays["sha256"],
                "partition_count": transfer["partition_count"],
                "status": "built",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
