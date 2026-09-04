"""Four-device CPU portability probe for the Tet4 scalar collective solve and VJP."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.elements.tetrahedron_h1 import (  # noqa: E402
    tetrahedron_p1_cell_nodal_load_vectors,
    tetrahedron_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    assert_scalar_h1_cg_converged,
    build_validation_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA,
    prepare_collective_scalar_h1_layout,
    scalar_h1_reduced_cell_rhs,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)


def _structured_tet4_mesh(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            (x / nx, y / ny, z / nz)
            for x in range(nx + 1)
            for y in range(ny + 1)
            for z in range(nz + 1)
        ],
        dtype=np.float64,
    )

    def node(x: int, y: int, z: int) -> int:
        return (x * (ny + 1) + y) * (nz + 1) + z

    cells: list[tuple[int, int, int, int]] = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                a = node(x, y, z)
                b = node(x + 1, y, z)
                c = node(x, y + 1, z)
                d = node(x + 1, y + 1, z)
                e = node(x, y, z + 1)
                f = node(x + 1, y, z + 1)
                g = node(x, y + 1, z + 1)
                h = node(x + 1, y + 1, z + 1)
                cells.extend(
                    (
                        (a, b, d, h),
                        (a, d, c, h),
                        (a, c, g, h),
                        (a, g, e, h),
                        (a, e, f, h),
                        (a, f, b, h),
                    )
                )
    return coordinates, np.asarray(cells, dtype=np.int64)


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    numerator = float(jnp.linalg.norm(observed - expected))
    denominator = max(float(jnp.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return numerator / denominator


def _system(partition_count: int, scale: jax.Array):
    coordinates, cells = _structured_tet4_mesh(4, 2, 2)
    constrained = np.isclose(coordinates[:, 0], 0.0) | np.isclose(coordinates[:, 0], 1.0)
    free_nodes = np.flatnonzero(~constrained).astype(np.int64)
    constrained_nodes = np.flatnonzero(constrained).astype(np.int64)
    centroids = np.mean(coordinates[cells], axis=1)
    owners = np.minimum((partition_count * centroids[:, 0]).astype(np.int64), partition_count - 1)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        owners,
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=partition_count,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    coordinates_jax = jnp.asarray(coordinates)
    cells_jax = jnp.asarray(cells)
    stiffness = tetrahedron_p1_diffusion_cell_matrices(
        coordinates_jax,
        cells_jax,
        jnp.full((cells.shape[0],), 2.0 * scale, dtype=jnp.float64),
    )
    load = tetrahedron_p1_cell_nodal_load_vectors(
        coordinates_jax,
        cells_jax,
        jnp.full((cells.shape[0], 4), 5.0, dtype=jnp.float64),
    )
    boundary_values = jnp.asarray(coordinates[constrained_nodes, 0])
    rhs = scalar_h1_reduced_cell_rhs(stiffness, load, topology, boundary_values)
    return (
        layout,
        stiffness,
        rhs,
        coordinates,
        cells,
        free_nodes,
        constrained_nodes,
        boundary_values,
    )


def _dense_authority(
    stiffness: jax.Array,
    coordinates: np.ndarray,
    cells: np.ndarray,
    free_nodes: np.ndarray,
    constrained_nodes: np.ndarray,
    boundary_values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    node_count = coordinates.shape[0]
    matrix = jnp.zeros((node_count, node_count), dtype=stiffness.dtype)
    rows = jnp.broadcast_to(jnp.asarray(cells)[:, :, None], stiffness.shape)
    columns = jnp.broadcast_to(jnp.asarray(cells)[:, None, :], stiffness.shape)
    matrix = matrix.at[rows, columns].add(stiffness)
    free = jnp.asarray(free_nodes)
    constrained = jnp.asarray(constrained_nodes)
    reduced_matrix = matrix[jnp.ix_(free, free)]
    source_load = tetrahedron_p1_cell_nodal_load_vectors(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.full((cells.shape[0], 4), 5.0, dtype=jnp.float64),
    )
    full_load = (
        jnp.zeros((node_count,), dtype=source_load.dtype).at[jnp.asarray(cells)].add(source_load)
    )
    reduced_rhs = full_load[free] - matrix[jnp.ix_(free, constrained)] @ boundary_values
    return reduced_matrix, reduced_rhs


def main() -> int:
    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("Tet4 collective probe requires exactly four forced CPU devices")
    policy = ScalarH1CGPolicy(2.0e-12, 1.0e-14, 300)
    reports: dict[str, object] = {}
    maximum_solution_difference = 0.0
    maximum_rhs_difference = 0.0
    four_device_solver = None
    four_device_inputs = None

    for partition_count in (1, 2, 4):
        system = _system(partition_count, jnp.asarray(1.0))
        layout, stiffness, rhs, coordinates, cells, free, constrained, boundary = system
        mesh = Mesh(np.asarray(devices[:partition_count], dtype=object), ("partition",))
        solver = jax.jit(build_validation_collective_scalar_h1_cg(layout, mesh, policy))
        result = solver(stiffness, rhs)
        result.solution.block_until_ready()
        assert_scalar_h1_cg_converged(result)
        dense_matrix, dense_rhs = _dense_authority(
            stiffness,
            coordinates,
            cells,
            free,
            constrained,
            boundary,
        )
        dense_solution = jnp.linalg.solve(dense_matrix, dense_rhs)
        solution_difference = _relative_difference(result.solution, dense_solution)
        rhs_difference = _relative_difference(result.right_hand_side, dense_rhs)
        maximum_solution_difference = max(maximum_solution_difference, solution_difference)
        maximum_rhs_difference = max(maximum_rhs_difference, rhs_difference)
        reports[str(partition_count)] = {
            "iterations": int(result.iterations),
            "relative_residual": float(result.relative_residual),
            "solution_relative_difference": solution_difference,
            "rhs_relative_difference": rhs_difference,
            "layout_sha256": layout.digest(),
            "halo_link_count": len(layout.transport.halo_links),
        }
        if partition_count == 4:
            four_device_solver = solver
            four_device_inputs = (stiffness, rhs)

    if four_device_solver is None or four_device_inputs is None:
        raise RuntimeError("four-device Tet4 witness was not constructed")

    def objective(scale: jax.Array) -> jax.Array:
        _, stiffness, rhs, *_ = _system(4, scale)
        return jnp.mean(four_device_solver(stiffness, rhs).solution)

    value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    step = 2.0e-5
    finite_difference = (
        objective(jnp.asarray(1.0 + step)) - objective(jnp.asarray(1.0 - step))
    ) / (2.0 * step)
    gradient_relative_error = float(jnp.abs(derivative - finite_difference)) / max(
        abs(float(finite_difference)),
        np.finfo(np.float64).tiny,
    )
    stablehlo = str(four_device_solver.lower(*four_device_inputs).compiler_ir("stablehlo")).lower()
    payload = {
        "schema_version": "femx.jax.tet4_scalar_collective.cpu_portability/v1",
        "layout_schema_version": SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA,
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": jax.device_count(),
        "node_count": 45,
        "tet4_cell_count": 96,
        "reports": reports,
        "maximum_solution_relative_difference": maximum_solution_difference,
        "maximum_rhs_relative_difference": maximum_rhs_difference,
        "objective": float(value),
        "gradient": float(derivative),
        "gradient_finite_difference": float(finite_difference),
        "gradient_relative_error": gradient_relative_error,
        "stablehlo_collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
        "stablehlo_all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
        "stablehlo_contains_all_gather": "all_gather" in stablehlo,
        "claim_scope": (
            "forced multi-CPU Tet4 scalar operator/RHS/CG/VJP portability; not accelerator, "
            "multi-host, ring-heater, or Elmer parity evidence"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
