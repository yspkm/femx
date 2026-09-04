"""Four-device CPU portability probe for scalar H1 collective RHS and CG."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.operators import (  # noqa: E402
    assemble_scalar_h1_system,
    triangle_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    assert_scalar_h1_cg_converged,
    build_validation_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    prepare_collective_scalar_h1_layout,
    prepare_scalar_h1_boundary_facet_map,
    scalar_h1_reduced_cell_rhs,
    triangle_p1_scalar_cell_load_vectors,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from tests.support import structured_unit_square_mesh  # noqa: E402

DEFAULT_SCALE = jnp.asarray(1.0)


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    numerator = float(jnp.linalg.norm(observed - expected))
    denominator = float(jnp.linalg.norm(expected))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _slab_cell_owners(coordinates: np.ndarray, cells: np.ndarray, count: int) -> np.ndarray:
    normalized_x = np.mean(coordinates[cells, 0], axis=1) / np.max(coordinates[:, 0])
    return np.minimum((count * normalized_x).astype(np.int64), count - 1)


def _case(name: str):
    mesh = structured_unit_square_mesh(8)
    coordinates = np.asarray(mesh.geometry.coordinates) * np.asarray((2.0e-6, 0.8e-6))
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    width = float(np.max(coordinates[:, 0]))
    constrained_mask = np.isclose(coordinates[:, 0], 0.0) | np.isclose(
        coordinates[:, 0],
        width,
    )
    free_nodes = np.flatnonzero(~constrained_mask).astype(np.int64)
    constrained_nodes = np.flatnonzero(constrained_mask).astype(np.int64)
    centroid_x = np.mean(coordinates[cells, 0], axis=1)
    if name == "heat":
        diffusion = np.where(centroid_x < 1.0e-6, 148.0, 1.38)
        source = np.where(np.abs(centroid_x - 1.0e-6) < 0.3e-6, 5.0e13, 0.0)
        facet_load = np.where(
            np.isclose(np.mean(coordinates[facets, 1], axis=1), np.max(coordinates[:, 1])),
            2.0e5,
            0.0,
        )
        dirichlet_values = np.where(
            np.isclose(coordinates[constrained_nodes, 0], 0.0),
            300.0,
            310.0,
        )
    elif name == "current":
        diffusion = np.where(centroid_x < 1.0e-6, 2.0e5, 5.0e4)
        source = np.zeros(cells.shape[0])
        facet_load = np.zeros(facets.shape[0])
        dirichlet_values = np.where(
            np.isclose(coordinates[constrained_nodes, 0], 0.0),
            0.0,
            1.0,
        )
    else:
        raise ValueError("unknown scalar probe case")
    return (
        coordinates,
        cells,
        facets,
        free_nodes,
        diffusion,
        source,
        facet_load,
        dirichlet_values,
    )


def _local_system(name: str, partition_count: int, *, scale: jax.Array = DEFAULT_SCALE):
    (
        coordinates,
        cells,
        facets,
        free_nodes,
        diffusion,
        source,
        facet_load,
        dirichlet_values,
    ) = _case(name)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        _slab_cell_owners(coordinates, cells, partition_count),
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=partition_count,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    boundary_map = prepare_scalar_h1_boundary_facet_map(
        cells,
        facets,
        node_count=coordinates.shape[0],
    )
    coordinates_jax = jnp.asarray(coordinates)
    cells_jax = jnp.asarray(cells)
    facets_jax = jnp.asarray(facets)
    cell_stiffness = triangle_p1_diffusion_cell_matrices(
        coordinates_jax,
        cells_jax,
        scale * jnp.asarray(diffusion),
    )
    cell_load = triangle_p1_scalar_cell_load_vectors(
        coordinates_jax,
        cells_jax,
        jnp.asarray(source),
        facets_jax,
        jnp.asarray(facet_load),
        boundary_map,
    )
    cell_rhs = scalar_h1_reduced_cell_rhs(
        cell_stiffness,
        cell_load,
        topology,
        jnp.asarray(dirichlet_values),
    )
    return layout, cell_stiffness, cell_rhs


def _dense_authority(name: str, *, scale: jax.Array = DEFAULT_SCALE):
    (
        coordinates,
        cells,
        facets,
        free_nodes,
        diffusion,
        source,
        facet_load,
        dirichlet_values,
    ) = _case(name)
    dense = assemble_scalar_h1_system(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        scale * jnp.asarray(diffusion),
        jnp.asarray(source),
        jnp.asarray(facets),
        jnp.asarray(facet_load),
    )
    constrained_nodes = np.setdiff1d(np.arange(coordinates.shape[0]), free_nodes)
    matrix = dense.stiffness[jnp.ix_(jnp.asarray(free_nodes), jnp.asarray(free_nodes))]
    rhs = dense.load[jnp.asarray(free_nodes)] - dense.stiffness[
        jnp.ix_(jnp.asarray(free_nodes), jnp.asarray(constrained_nodes))
    ] @ jnp.asarray(dirichlet_values)
    return matrix, rhs


def main() -> int:
    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("scalar collective probe requires exactly four forced CPU devices")
    policy = ScalarH1CGPolicy(2.0e-13, 1.0e-14, 400)
    case_reports: dict[str, object] = {}
    maximum_solution_difference = 0.0
    maximum_rhs_difference = 0.0
    four_device_solver = None
    four_device_inputs = None
    four_device_layout = None

    for name in ("heat", "current"):
        dense_matrix, dense_rhs = _dense_authority(name)
        dense_solution = jnp.linalg.solve(dense_matrix, dense_rhs)
        partition_reports: dict[str, object] = {}
        for partition_count in (1, 2, 4):
            layout, cell_stiffness, cell_rhs = _local_system(name, partition_count)
            device_mesh = Mesh(
                np.asarray(devices[:partition_count], dtype=object),
                ("partition",),
            )
            solver = jax.jit(build_validation_collective_scalar_h1_cg(layout, device_mesh, policy))
            result = solver(cell_stiffness, cell_rhs)
            result.solution.block_until_ready()
            assert_scalar_h1_cg_converged(result)
            solution_difference = _relative_difference(result.solution, dense_solution)
            rhs_difference = _relative_difference(result.right_hand_side, dense_rhs)
            maximum_solution_difference = max(maximum_solution_difference, solution_difference)
            maximum_rhs_difference = max(maximum_rhs_difference, rhs_difference)
            partition_reports[str(partition_count)] = {
                "iterations": int(result.iterations),
                "relative_residual": float(result.relative_residual),
                "solution_relative_difference": solution_difference,
                "rhs_relative_difference": rhs_difference,
                "layout_sha256": layout.digest(),
                "halo_link_count": len(layout.transport.halo_links),
            }
            if name == "heat" and partition_count == 4:
                four_device_solver = solver
                four_device_inputs = (cell_stiffness, cell_rhs)
                four_device_layout = layout
        case_reports[name] = partition_reports

    if four_device_solver is None or four_device_inputs is None or four_device_layout is None:
        raise RuntimeError("four-device scalar witness was not constructed")

    def collective_objective(scale: jax.Array) -> jax.Array:
        _, stiffness, rhs = _local_system("heat", 4, scale=scale)
        result = four_device_solver(stiffness, rhs)
        return jnp.mean(result.solution)

    value, derivative = jax.jit(jax.value_and_grad(collective_objective))(jnp.asarray(1.0))
    step = 2.0e-5
    finite_difference = (
        collective_objective(jnp.asarray(1.0 + step))
        - collective_objective(jnp.asarray(1.0 - step))
    ) / (2.0 * step)
    gradient_relative_error = float(jnp.abs(derivative - finite_difference)) / max(
        abs(float(finite_difference)),
        np.finfo(np.float64).tiny,
    )

    stablehlo = str(four_device_solver.lower(*four_device_inputs).compiler_ir("stablehlo")).lower()
    payload = {
        "schema_version": "femx.jax.scalar_h1_collective_cg.cpu_portability/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": jax.device_count(),
        "case_reports": case_reports,
        "maximum_solution_relative_difference": maximum_solution_difference,
        "maximum_rhs_relative_difference": maximum_rhs_difference,
        "gradient_value": float(value),
        "gradient": float(derivative),
        "gradient_finite_difference": float(finite_difference),
        "gradient_relative_error": gradient_relative_error,
        "stablehlo_collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
        "stablehlo_all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
        "stablehlo_contains_all_gather": "all_gather" in stablehlo,
        "four_device_layout_sha256": four_device_layout.digest(),
        "claim_scope": (
            "forced multi-CPU scalar RHS/CG/adjoint portability; not accelerator or multi-host "
            "evidence"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
