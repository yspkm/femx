#!/usr/bin/env python3
"""Build the immutable float64 authority and plan for a physical coupled TPU run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version as package_version
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
from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.runtime import prepare  # noqa: E402
from scripts._distributed_electrothermal_case import (  # noqa: E402
    distributed_electrothermal_iteration_policy,
    parameterized_self_consistent_microheater,
)
from scripts._tpu_distributed_electrothermal_plan import (  # noqa: E402
    DistributedElectrothermalAuthority,
    write_distributed_electrothermal_artifact,
)


def _source_commit() -> str:
    completed = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _system(intervals: int) -> DifferentiableSelfConsistentElectrothermal:
    feedback, current_parameters, thermal_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(
            intervals=intervals,
            iteration=distributed_electrothermal_iteration_policy(),
        )
    )
    current_backend = JaxSteadyCurrentBackend()
    thermal_backend = JaxSteadyHeatBackend()
    current = current_backend.bind_differentiable(
        prepare(feedback.one_way.electrical_problem, current_backend),
        current_parameters,
    )
    thermal = thermal_backend.bind_differentiable(
        prepare(feedback.one_way.thermal_problem, thermal_backend),
        thermal_parameters,
    )
    return DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current,
        thermal,
        feedback_parameters,
    )


def _cell_owners(
    coordinates: np.ndarray,
    cells: np.ndarray,
    partition_count: int,
) -> np.ndarray:
    centroids = np.mean(coordinates[cells, 0], axis=1)
    width = float(np.max(coordinates[:, 0]))
    owners = np.minimum(
        (partition_count * centroids / width).astype(np.int64),
        partition_count - 1,
    )
    if not np.array_equal(np.unique(owners), np.arange(partition_count, dtype=np.int64)):
        raise ValueError("bounded electrothermal mesh must assign cells to every partition")
    return np.asarray(owners, dtype=np.int64)


def build_inputs(output_root: Path, *, intervals: int, partition_count: int) -> dict[str, object]:
    """Generate one controller-owned float64 authority without discovering TPU devices."""

    if intervals <= 0 or intervals % 2 != 0:
        raise ValueError("electrothermal TPU input intervals must be a positive even integer")
    if partition_count <= 1:
        raise ValueError("physical electrothermal TPU input requires multiple partitions")
    if jax.default_backend() != "cpu" or not bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("TPU input authority requires local CPU JAX with x64 enabled")
    system = _system(intervals)
    payload = system.current._engine.payload
    coordinates = np.asarray(payload.coordinates, dtype=np.float64)
    cells = np.asarray(payload.cells, dtype=np.int64)
    plan = prepare_distributed_electrothermal_plan(
        system,
        _cell_owners(coordinates, cells, partition_count),
        partition_count=partition_count,
    )
    current = system.initial_current_values
    thermal = system.initial_thermal_values
    feedback = system.initial_feedback_values
    forward = system.solve(current, thermal, feedback)
    weights = jnp.linspace(0.75, 1.25, coordinates.shape[0], dtype=jnp.float64)
    weights = weights.at[jnp.asarray(payload.dirichlet_nodes)].set(0.0)
    weights /= jnp.sum(weights)
    adjoint = system.vjp(current, thermal, feedback, weights)
    thermal_reference = float(
        np.asarray(system.thermal._engine.resolved_coefficients(thermal)[-1][0])
    )
    objective = float(jnp.vdot(weights, forward.temperature - thermal_reference))
    authority = DistributedElectrothermalAuthority(
        potential=np.asarray(forward.potential, dtype=np.float64),
        temperature=np.asarray(forward.temperature, dtype=np.float64),
        current_parameter_gradient=np.asarray(
            adjoint.current_parameter_gradient,
            dtype=np.float64,
        ),
        thermal_parameter_gradient=np.asarray(
            adjoint.thermal_parameter_gradient,
            dtype=np.float64,
        ),
        feedback_parameter_gradient=np.asarray(
            adjoint.feedback_parameter_gradient,
            dtype=np.float64,
        ),
        temperature_cotangent=np.asarray(weights, dtype=np.float64),
        objective=objective,
        forward_converged=bool(forward.converged),
        adjoint_converged=bool(jnp.isfinite(adjoint.adjoint_backward_error)),
        diagnostics={
            "iterations": int(forward.iterations),
            "update_error": float(forward.update_error),
            "current_residual_error": float(forward.current_residual_error),
            "heat_residual_error": float(forward.heat_residual_error),
            "electrical_joule_power_W_per_m": float(forward.electrical_joule_power),
            "thermal_joule_load_W_per_m": float(forward.thermal_joule_load),
            "transfer_relative_error": float(forward.transfer_relative_error),
            "heat_balance_relative_error": float(forward.heat_balance_relative_error),
            "adjoint_backward_error": float(adjoint.adjoint_backward_error),
        },
    )
    source_commit = _source_commit()
    manifest = write_distributed_electrothermal_artifact(
        output_root,
        plan,
        authority,
        source_commit=source_commit,
        case_metadata={
            "name": "bounded_same_mesh_siph_microheater",
            "intervals_per_axis": intervals,
            "width_m": 2.0e-6,
            "height_m": 0.5e-6,
            "partition_count": partition_count,
            "material_scope": "representative values; not foundry calibrated",
            "authority_runtime": {
                "backend": jax.default_backend(),
                "jax": jax.__version__,
                "jaxlib": package_version("jaxlib"),
                "numpy": np.__version__,
                "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            },
            "claim_scope": (
                "bounded 2D per-unit-depth same-mesh electrothermal authority for physical TPU "
                "correctness admission; not measured-device, scaling, or foundry evidence"
            ),
        },
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
    plan = manifest["plan"]
    arrays = manifest["arrays"]
    assert isinstance(plan, dict) and isinstance(arrays, dict)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "plan_sha256": plan["sha256"],
                "arrays_sha256": arrays["sha256"],
                "partition_count": plan["partition_count"],
                "status": "built",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
