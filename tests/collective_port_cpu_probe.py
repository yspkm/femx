"""Four-device CPU portability probe for the JAX collective port operator."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends._hcurl import (  # noqa: E402
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.backends.jax.port_collective import (  # noqa: E402
    build_packed_collective_port_matvec,
    build_validation_collective_port_matvec,
    collective_port_relative_difference,
    describe_collective_port_mesh,
    pack_collective_port_cell_matrix,
    pack_collective_port_owned_vector,
    prepare_collective_port_layout,
    unpack_collective_port_owned_vector,
)
from femx.backends.jax.port_collective_runtime import (  # noqa: E402
    make_collective_port_array_from_process_local_data,
)
from femx.backends.jax.port_matrix_free import (  # noqa: E402
    build_lossless_matrix_free_port_pencil,
    matrix_free_port_matvec,
    prepare_port_matrix_free_topology,
)
from femx.backends.jax.port_owned_ghost import (  # noqa: E402
    prepare_owned_ghost_port_topology,
)
from tests.support import structured_unit_square_mesh  # noqa: E402


def _physical_port_case() -> tuple[np.ndarray, np.ndarray, object, object]:
    mesh = structured_unit_square_mesh(4)
    coordinates = np.asarray(mesh.geometry.coordinates) * np.asarray((2.0e-6, 1.0e-6))
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)
    pec = canonical_mixed_port_dof_partition(
        facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    serial_topology = prepare_port_matrix_free_topology(
        cells,
        edge_map.cell_edge_dofs,
        pec.free_dofs,
        node_count=coordinates.shape[0],
        edge_dof_count=edge_map.dof_count,
    )
    centroids = np.mean(coordinates[cells], axis=1)
    silicon = (np.abs(centroids[:, 0] - 1.0e-6) <= 0.55e-6) & (
        np.abs(centroids[:, 1] - 0.5e-6) <= 0.24e-6
    )
    relative_permittivity = np.where(silicon, 3.48**2, 1.444**2)
    pencil = build_lossless_matrix_free_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(serial_topology.cell_reduced_dofs),
        jnp.asarray(relative_permittivity),
        jnp.ones(cells.shape[0]),
        jnp.asarray(193.414e12),
        free_dof_count=serial_topology.free_dof_count,
    )
    return coordinates, cells, serial_topology, pencil


def _slab_cell_owners(coordinates: np.ndarray, cells: np.ndarray, count: int) -> np.ndarray:
    normalized_x = np.mean(coordinates[cells, 0], axis=1) / np.max(coordinates[:, 0])
    return np.minimum((count * normalized_x).astype(np.int64), count - 1)


def main() -> int:
    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("portable collective probe requires exactly four forced CPU devices")
    coordinates, cells, serial_topology, pencil = _physical_port_case()
    cell_owners = _slab_cell_owners(coordinates, cells, 4)
    topology = prepare_owned_ghost_port_topology(
        serial_topology.cell_reduced_dofs,
        cell_owners,
        free_dof_count=serial_topology.free_dof_count,
        partition_count=4,
    )
    layout = prepare_collective_port_layout(topology)
    mesh = Mesh(np.asarray(devices, dtype=object), ("partition",))
    mesh_report = describe_collective_port_mesh(layout, mesh)
    packed_operator = build_packed_collective_port_matvec(layout, mesh)
    collective_operator = build_validation_collective_port_matvec(layout, mesh)

    rng = np.random.default_rng(20260901)
    real_vector = jnp.asarray(rng.normal(size=serial_topology.free_dof_count))
    complex_vector = real_vector + 1j * jnp.asarray(rng.normal(size=serial_topology.free_dof_count))
    shift = jnp.asarray(-2.1e14)
    matrices = {
        "stiffness": pencil.stiffness,
        "mass": pencil.mass,
        "shifted": pencil.stiffness - shift * pencil.mass,
    }
    action_differences: dict[str, dict[str, float]] = {}
    compiled_collective = jax.jit(collective_operator)
    for name, matrix in matrices.items():
        serial_real = matrix_free_port_matvec(matrix, pencil.cell_reduced_dofs, real_vector)
        serial_complex = matrix_free_port_matvec(matrix, pencil.cell_reduced_dofs, complex_vector)
        collective_real = compiled_collective(matrix, real_vector)
        collective_complex = compiled_collective(matrix, complex_vector)
        collective_real.block_until_ready()
        collective_complex.block_until_ready()
        action_differences[name] = {
            "real": collective_port_relative_difference(collective_real, serial_real),
            "complex": collective_port_relative_difference(collective_complex, serial_complex),
        }

    cotangent = jnp.asarray(rng.normal(size=serial_topology.free_dof_count))
    differentiated_matrix = matrices["shifted"]
    _, collective_pullback = jax.vjp(
        compiled_collective,
        differentiated_matrix,
        real_vector,
    )
    collective_matrix_vjp, collective_vector_vjp = collective_pullback(cotangent)

    def serial_operator(cell_matrix: jax.Array, vector: jax.Array) -> jax.Array:
        return matrix_free_port_matvec(cell_matrix, pencil.cell_reduced_dofs, vector)

    _, serial_pullback = jax.vjp(jax.jit(serial_operator), differentiated_matrix, real_vector)
    serial_matrix_vjp, serial_vector_vjp = serial_pullback(cotangent)

    complex_cotangent = cotangent + 1j * jnp.asarray(
        rng.normal(size=serial_topology.free_dof_count)
    )
    _, collective_complex_pullback = jax.vjp(
        compiled_collective,
        differentiated_matrix,
        complex_vector,
    )
    collective_complex_matrix_vjp, collective_complex_vector_vjp = collective_complex_pullback(
        complex_cotangent
    )
    _, serial_complex_pullback = jax.vjp(
        jax.jit(serial_operator),
        differentiated_matrix,
        complex_vector,
    )
    serial_complex_matrix_vjp, serial_complex_vector_vjp = serial_complex_pullback(
        complex_cotangent
    )

    packed_cells_host = np.asarray(
        jax.device_get(pack_collective_port_cell_matrix(layout, differentiated_matrix))
    )
    packed_owned_host = np.asarray(
        jax.device_get(pack_collective_port_owned_vector(layout, real_vector))
    )
    packed_map_host = np.asarray(layout.cell_local_dofs)
    packed_cells, cell_array_report = make_collective_port_array_from_process_local_data(
        "shifted-cell-blocks",
        packed_cells_host,
        mesh,
    )
    packed_map, map_array_report = make_collective_port_array_from_process_local_data(
        "cell-local-dof-map",
        packed_map_host,
        mesh,
    )
    packed_owned, owner_array_report = make_collective_port_array_from_process_local_data(
        "owned-vector",
        packed_owned_host,
        mesh,
    )
    packed_result = jax.jit(packed_operator)(packed_cells, packed_map, packed_owned)
    packed_result.block_until_ready()
    packed_canonical_result = unpack_collective_port_owned_vector(layout, packed_result)
    serial_shifted_result = matrix_free_port_matvec(
        differentiated_matrix,
        pencil.cell_reduced_dofs,
        real_vector,
    )
    stablehlo = str(
        jax.jit(packed_operator)
        .lower(packed_cells, packed_map, packed_owned)
        .compiler_ir("stablehlo")
    )
    report = layout.storage_report
    payload = {
        "schema_version": "femx.jax.port_collective.cpu_portability/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": jax.device_count(),
        "partition_count": layout.partition_count,
        "global_dof_count": topology.global_dof_count,
        "global_cell_count": topology.cell_count,
        "layout_sha256": layout.digest(),
        "mesh_report": mesh_report.canonical_data(),
        "array_reports": {
            "cell_blocks": cell_array_report.canonical_data(),
            "cell_local_dofs": map_array_report.canonical_data(),
            "owned_vector": owner_array_report.canonical_data(),
        },
        "halo_link_count": len(layout.halo_links),
        "halo_value_count": report.halo_value_count,
        "cell_padding_fraction": report.cell_padding_fraction,
        "owned_dof_padding_fraction": report.owned_dof_padding_fraction,
        "ghost_dof_padding_fraction": report.ghost_dof_padding_fraction,
        "action_relative_differences": action_differences,
        "maximum_real_action_relative_difference": max(
            value["real"] for value in action_differences.values()
        ),
        "maximum_complex_action_relative_difference": max(
            value["complex"] for value in action_differences.values()
        ),
        "matrix_vjp_relative_difference": collective_port_relative_difference(
            collective_matrix_vjp,
            serial_matrix_vjp,
        ),
        "vector_vjp_relative_difference": collective_port_relative_difference(
            collective_vector_vjp,
            serial_vector_vjp,
        ),
        "complex_matrix_vjp_relative_difference": collective_port_relative_difference(
            collective_complex_matrix_vjp,
            serial_complex_matrix_vjp,
        ),
        "complex_vector_vjp_relative_difference": collective_port_relative_difference(
            collective_complex_vector_vjp,
            serial_complex_vector_vjp,
        ),
        "packed_process_local_action_relative_difference": collective_port_relative_difference(
            packed_canonical_result,
            serial_shifted_result,
        ),
        "stablehlo_collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
        "expected_collective_permute_count": 2 * len(layout.halo_links),
        "stablehlo_contains_all_gather": "all_gather" in stablehlo.lower(),
        "claim_scope": "forced multi-CPU portability; not accelerator or multi-host evidence",
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
