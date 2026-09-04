"""Four-device CPU evidence probe for scalar H1 multilevel PCG."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.operators import (  # noqa: E402
    triangle_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    assert_scalar_h1_cg_converged,
    build_validation_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_multilevel import (  # noqa: E402
    ScalarH1MultilevelPolicy,
    build_validation_collective_scalar_h1_multilevel_pcg,
    prepare_scalar_h1_multilevel_hierarchy,
    prepare_scalar_h1_nested_prolongation,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from tests.support import structured_unit_square_mesh  # noqa: E402


def _mesh_arrays(intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates)
    cells = np.asarray(mesh.topology.connectivity)
    boundary = (
        np.isclose(coordinates[:, 0], 0.0)
        | np.isclose(coordinates[:, 0], 1.0)
        | np.isclose(coordinates[:, 1], 0.0)
        | np.isclose(coordinates[:, 1], 1.0)
    )
    return coordinates, cells, np.flatnonzero(~boundary).astype(np.int64)


def _cell_owners(coordinates: np.ndarray, cells: np.ndarray, count: int) -> np.ndarray:
    centroid_x = np.mean(coordinates[cells, 0], axis=1)
    return np.minimum((count * centroid_x).astype(np.int64), count - 1)


def _layout(intervals: int, partition_count: int):
    coordinates, cells, free_nodes = _mesh_arrays(intervals)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        _cell_owners(coordinates, cells, partition_count),
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=partition_count,
    )
    return prepare_collective_scalar_h1_layout(topology)


def _hierarchy(intervals: int, layout: object):
    prolongations = []
    fine = intervals
    while fine > 2:
        coarse = fine // 2
        fine_coordinates, _, fine_free = _mesh_arrays(fine)
        coarse_coordinates, coarse_cells, coarse_free = _mesh_arrays(coarse)
        prolongations.append(
            prepare_scalar_h1_nested_prolongation(
                fine_coordinates,
                fine_free,
                coarse_coordinates,
                coarse_cells,
                coarse_free,
            )
        )
        fine = coarse
    return prepare_scalar_h1_multilevel_hierarchy(
        layout,  # type: ignore[arg-type]
        prolongations,
        maximum_replicated_dofs=256,
    )


def _system(intervals: int, contrast: jax.Array) -> tuple[jax.Array, jax.Array]:
    coordinates, cells, _ = _mesh_arrays(intervals)
    vertices = coordinates[cells]
    first = vertices[:, 1] - vertices[:, 0]
    second = vertices[:, 2] - vertices[:, 0]
    areas = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    centroids_x = np.mean(vertices[:, :, 0], axis=1)
    coefficients = jnp.where(jnp.asarray(centroids_x) < 0.5, 1.0, contrast)
    stiffness = triangle_p1_diffusion_cell_matrices(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        coefficients,
    )
    rhs = jnp.asarray(np.repeat((areas / 3.0)[:, None], 3, axis=1))
    return stiffness, rhs


def _relative_difference(first: jax.Array, second: jax.Array) -> float:
    denominator = float(jnp.linalg.norm(second))
    numerator = float(jnp.linalg.norm(first - second))
    return numerator / max(denominator, np.finfo(np.float64).tiny)


def _solvers(intervals: int, devices: list[jax.Device]):
    layout = _layout(intervals, len(devices))
    hierarchy = _hierarchy(intervals, layout)
    mesh = Mesh(np.asarray(devices, dtype=object), ("partition",))
    cg_policy = ScalarH1CGPolicy(1.0e-9, 1.0e-14, 1_000)
    unpreconditioned = jax.jit(build_validation_collective_scalar_h1_cg(layout, mesh, cg_policy))
    multilevel = jax.jit(
        build_validation_collective_scalar_h1_multilevel_pcg(
            layout,
            mesh,
            hierarchy,
            ScalarH1MultilevelPolicy(maximum_coarse_condition_number=1.0e12),
            cg_policy,
            value_dtype=np.float64,
        )
    )
    return layout, hierarchy, unpreconditioned, multilevel


def _report_case(
    unpreconditioned: object,
    multilevel: object,
    stiffness: jax.Array,
    rhs: jax.Array,
) -> dict[str, object]:
    cg = unpreconditioned(stiffness, rhs)  # type: ignore[operator]
    pcg = multilevel(stiffness, rhs)  # type: ignore[operator]
    pcg.solution.block_until_ready()
    assert_scalar_h1_cg_converged(cg)
    assert_scalar_h1_cg_converged(pcg)
    return {
        "cg_iterations": int(cg.iterations),
        "pcg_iterations": int(pcg.iterations),
        "cg_relative_residual": float(cg.relative_residual),
        "pcg_relative_residual": float(pcg.relative_residual),
        "solution_relative_difference": _relative_difference(pcg.solution, cg.solution),
    }


def main() -> int:
    devices = jax.devices("cpu")
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("multilevel probe requires exactly four forced CPU devices")

    refinement: dict[str, object] = {}
    contrast: dict[str, object] = {}
    finest_solver = None
    finest_inputs = None
    hierarchy_hashes: dict[str, str] = {}

    for intervals in (8, 16, 32):
        _prepared_layout, hierarchy, cg, pcg = _solvers(intervals, devices)
        stiffness, rhs = _system(intervals, jnp.asarray(1.0))
        refinement[str(intervals)] = _report_case(cg, pcg, stiffness, rhs)
        hierarchy_hashes[str(intervals)] = hierarchy.digest()
        if intervals == 16:
            for coefficient_contrast in (1.0, 100.0, 10_000.0):
                case_stiffness, case_rhs = _system(
                    intervals,
                    jnp.asarray(coefficient_contrast),
                )
                contrast[f"{coefficient_contrast:.0e}"] = _report_case(
                    cg,
                    pcg,
                    case_stiffness,
                    case_rhs,
                )
        if intervals == 32:
            finest_solver = pcg
            finest_inputs = (stiffness, rhs)

    if finest_solver is None or finest_inputs is None:
        raise RuntimeError("finest multilevel witness was not constructed")

    _, _, _, gradient_solver = _solvers(16, devices)
    base_stiffness, gradient_rhs = _system(16, jnp.asarray(100.0))
    weights = jnp.linspace(0.2, 1.0, 225)

    def objective(scale: jax.Array) -> jax.Array:
        solution = gradient_solver(scale * base_stiffness, gradient_rhs).solution
        return jnp.vdot(weights, solution).real

    objective_value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.1))
    step = 2.0e-5
    finite_difference = (
        objective(jnp.asarray(1.1 + step)) - objective(jnp.asarray(1.1 - step))
    ) / (2.0 * step)
    gradient_relative_error = float(jnp.abs(derivative - finite_difference)) / max(
        abs(float(finite_difference)),
        np.finfo(np.float64).tiny,
    )

    stablehlo = str(finest_solver.lower(*finest_inputs).compiler_ir("stablehlo")).lower()
    payload = {
        "schema_version": "femx.jax.scalar_h1_multilevel_pcg.cpu_portability/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": jax.device_count(),
        "refinement_reports": refinement,
        "coefficient_contrast_reports": contrast,
        "hierarchy_sha256": hierarchy_hashes,
        "gradient_objective": float(objective_value),
        "gradient": float(derivative),
        "gradient_finite_difference": float(finite_difference),
        "gradient_relative_error": gradient_relative_error,
        "stablehlo_collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
        "stablehlo_all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
        "stablehlo_contains_all_gather": "all_gather" in stablehlo,
        "claim_scope": (
            "forced multi-CPU multilevel-PCG portability, refinement, contrast, and adjoint; "
            "not accelerator or multi-host evidence"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
