"""Collective transport and right-hand-side assembly for real scalar H1/P1 systems.

Thermal conductivity and electrical conductivity retain separate units and physics adapters.  The
shared algebra here is only the reduced nodal diffusion system.  Preparation is host-only and does
not discover devices.  Runtime builders require an explicit JAX ``Mesh`` supplied by the caller.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from femx.core.errors import ContractError

from .collective import (
    CollectiveLayout,
    PackedCollectiveCellGather,
    PackedCollectiveMatvec,
    PackedCollectiveRowAssembly,
    build_packed_collective_cell_gather,
    build_packed_collective_matvec,
    build_packed_collective_row_assembly,
    pack_collective_cell_matrix,
    pack_collective_cell_vector,
    pack_collective_owned_mask,
    pack_collective_owned_vector,
    prepare_collective_layout,
    unpack_collective_owned_vector,
)
from .elements.tetrahedron_h1 import tetrahedron_p1_geometry
from .operators import triangle_p1_geometry
from .owned_ghost import _canonical_int64_array
from .scalar_owned_ghost import ScalarH1OwnedGhostTopology

SCALAR_H1_COLLECTIVE_LAYOUT_SCHEMA = "femx.jax.scalar_h1_collective/v1"
SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA = "femx.jax.scalar_h1_collective/v2"


def _layout_schema_for_topology(topology: ScalarH1OwnedGhostTopology) -> str:
    if topology.cell_dof_count == 3:
        return SCALAR_H1_COLLECTIVE_LAYOUT_SCHEMA
    return SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA


def _require_real_array(value: jax.Array, *, label: str) -> None:
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f"{label} must use a real floating dtype")


def _boundary_map(
    cells: np.ndarray,
    boundary_facets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_facets: tuple[tuple[int, ...], ...]
    if cells.shape[1] == 3:
        local_facets = ((0, 1), (1, 2), (2, 0))
        entity_name = "triangle edge"
    else:
        local_facets = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
        entity_name = "tetrahedron face"
    occurrences_by_facet: dict[
        tuple[int, ...],
        list[tuple[int, tuple[int, ...]]],
    ] = {}
    for cell_index, cell in enumerate(cells):
        for local_nodes in local_facets:
            key = tuple(sorted(int(cell[local]) for local in local_nodes))
            occurrences_by_facet.setdefault(key, []).append((cell_index, local_nodes))

    facet_cells = np.empty((boundary_facets.shape[0],), dtype=np.int64)
    facet_local_nodes = np.empty(boundary_facets.shape, dtype=np.int64)
    seen: set[tuple[int, ...]] = set()
    for facet_index, facet in enumerate(boundary_facets):
        key = tuple(sorted(int(node) for node in facet))
        if key in seen:
            raise ContractError(f"scalar H1 boundary facets cannot repeat a {entity_name}")
        seen.add(key)
        occurrences = occurrences_by_facet.get(key, ())
        if len(occurrences) != 1:
            raise ContractError(
                f"scalar H1 boundary facet must identify exactly one exterior {entity_name}"
            )
        cell_index, local_facet = occurrences[0]
        cell = cells[cell_index]
        local_by_node = {int(cell[local]): local for local in local_facet}
        facet_cells[facet_index] = cell_index
        facet_local_nodes[facet_index] = tuple(local_by_node[int(node)] for node in facet)
    facet_cells.setflags(write=False)
    facet_local_nodes.setflags(write=False)
    return facet_cells, facet_local_nodes


@dataclass(frozen=True, slots=True)
class ScalarH1BoundaryFacetMap:
    """Host-prepared simplex exterior-facet to incident-cell/local-node identity."""

    node_count: int
    cells: np.ndarray
    boundary_facets: np.ndarray
    facet_cells: np.ndarray
    facet_local_nodes: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int):
            raise ContractError("scalar H1 boundary-map node count must be an integer")
        if self.node_count <= 0:
            raise ContractError("scalar H1 boundary-map node count must be positive")
        cells = _canonical_int64_array(
            self.cells,
            label="scalar H1 boundary-map cells",
            rank=2,
        )
        if cells.shape[1] not in (3, 4):
            raise ContractError(
                "scalar H1 boundary-map cells must have 3 triangle or 4 tetrahedron nodes"
            )
        facet_width = cells.shape[1] - 1
        facets = _canonical_int64_array(
            self.boundary_facets,
            label="scalar H1 boundary facets",
            rank=2,
        )
        if facets.shape[1] != facet_width:
            raise ContractError(
                f"scalar H1 boundary facets must have {facet_width} nodes for the cell family"
            )
        if cells.shape[0] == 0:
            family = "triangle" if cells.shape[1] == 3 else "tetrahedron"
            raise ContractError(f"scalar H1 boundary map requires at least one {family}")
        if np.any(cells < 0) or np.any(cells >= self.node_count):
            raise ContractError("scalar H1 boundary-map cells contain an out-of-range node")
        if np.any(facets < 0) or np.any(facets >= self.node_count):
            raise ContractError("scalar H1 boundary facets contain an out-of-range node")
        if any(np.unique(row).shape[0] != row.shape[0] for row in (*cells, *facets)):
            raise ContractError("scalar H1 boundary-map entities cannot repeat a node")
        expected_cells, expected_local = _boundary_map(cells, facets)
        facet_cells = _canonical_int64_array(
            self.facet_cells,
            label="scalar H1 boundary facet cells",
            rank=1,
        )
        facet_local_nodes = _canonical_int64_array(
            self.facet_local_nodes,
            label="scalar H1 boundary facet local nodes",
            rank=2,
        )
        if facet_local_nodes.shape[1] != facet_width:
            raise ContractError(
                "scalar H1 boundary facet local nodes disagree with the simplex family"
            )
        if not np.array_equal(facet_cells, expected_cells):
            raise ContractError("scalar H1 boundary facet cells disagree with the mesh")
        if not np.array_equal(facet_local_nodes, expected_local):
            raise ContractError("scalar H1 boundary local nodes disagree with the mesh")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "boundary_facets", facets)
        object.__setattr__(self, "facet_cells", facet_cells)
        object.__setattr__(self, "facet_local_nodes", facet_local_nodes)


def prepare_scalar_h1_boundary_facet_map(
    cells: object,
    boundary_facets: object,
    *,
    node_count: int,
) -> ScalarH1BoundaryFacetMap:
    """Bind each requested exterior edge or face to its unique incident simplex."""

    canonical_cells = _canonical_int64_array(
        cells,
        label="scalar H1 boundary-map cells",
        rank=2,
    )
    if canonical_cells.shape[1] not in (3, 4):
        raise ContractError(
            "scalar H1 boundary-map cells must have 3 triangle or 4 tetrahedron nodes"
        )
    facet_width = canonical_cells.shape[1] - 1
    canonical_facets = _canonical_int64_array(
        boundary_facets,
        label="scalar H1 boundary facets",
        rank=2,
    )
    if canonical_facets.shape[1] != facet_width:
        raise ContractError(
            f"scalar H1 boundary facets must have {facet_width} nodes for the cell family"
        )
    facet_cells, local_nodes = _boundary_map(canonical_cells, canonical_facets)
    return ScalarH1BoundaryFacetMap(
        node_count=node_count,
        cells=canonical_cells,
        boundary_facets=canonical_facets,
        facet_cells=facet_cells,
        facet_local_nodes=local_nodes,
    )


@dataclass(frozen=True, slots=True)
class ScalarH1CollectiveLayout:
    """Scalar full-node identity plus generic fixed-capacity collective transport."""

    topology: ScalarH1OwnedGhostTopology
    transport: CollectiveLayout
    schema_version: str = SCALAR_H1_COLLECTIVE_LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.topology, ScalarH1OwnedGhostTopology):
            raise ContractError(
                "scalar H1 collective layout requires a scalar owned/ghost topology"
            )
        expected_schema = _layout_schema_for_topology(self.topology)
        if self.schema_version != expected_schema:
            raise ContractError(f"scalar H1 collective schema must be {expected_schema!r}")
        if not isinstance(self.transport, CollectiveLayout):
            raise ContractError("scalar H1 collective layout requires a collective transport")
        if self.transport.topology is not self.topology.owned_ghost:
            raise ContractError(
                "scalar H1 collective transport must bind the exact scalar topology"
            )
        if self.transport.schema_version != self.schema_version:
            raise ContractError("scalar H1 collective transport schema disagrees with the layout")

    @property
    def partition_count(self) -> int:
        return self.transport.partition_count

    @property
    def cell_capacity(self) -> int:
        return self.transport.cell_capacity

    @property
    def owned_dof_capacity(self) -> int:
        return self.transport.owned_dof_capacity

    @property
    def cell_dof_count(self) -> int:
        return self.topology.cell_dof_count

    def digest(self) -> str:
        """Bind the transport to full-node, cell, and free-node scalar identities."""

        hasher = hashlib.sha256(self.transport.digest().encode("ascii"))
        for name, array in (
            ("cells", self.topology.cells),
            ("free_nodes", self.topology.free_nodes),
            ("full_to_reduced", self.topology.full_to_reduced),
        ):
            canonical = np.asarray(array, dtype="<i8", order="C")
            hasher.update(name.encode("utf-8"))
            hasher.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            hasher.update(canonical.tobytes())
        return hasher.hexdigest()


def prepare_collective_scalar_h1_layout(
    topology: ScalarH1OwnedGhostTopology,
) -> ScalarH1CollectiveLayout:
    """Create scalar collective transport without selecting or discovering devices."""

    if not isinstance(topology, ScalarH1OwnedGhostTopology):
        raise ContractError("scalar H1 collective lowering requires a scalar owned/ghost topology")
    schema_version = _layout_schema_for_topology(topology)
    transport = prepare_collective_layout(
        topology.owned_ghost,
        schema_version=schema_version,
    )
    return ScalarH1CollectiveLayout(
        topology=topology,
        transport=transport,
        schema_version=schema_version,
    )


def triangle_p1_scalar_cell_load_vectors(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_source: jax.Array,
    boundary_facets: jax.Array,
    facet_load: jax.Array,
    boundary_map: ScalarH1BoundaryFacetMap,
) -> jax.Array:
    r"""Return cell-local ``integral N_i Q`` and boundary-load row contributions."""

    cell_count = boundary_map.cells.shape[0]
    facet_count = boundary_map.boundary_facets.shape[0]
    if coordinates.ndim != 2 or coordinates.shape != (boundary_map.node_count, 2):
        raise ValueError("scalar H1 load coordinates must be shaped (nodes, 2)")
    if cells.ndim != 2 or cells.shape != (cell_count, 3):
        raise ValueError("scalar H1 load cells must be shaped (cells, 3)")
    if boundary_facets.ndim != 2 or boundary_facets.shape != (facet_count, 2):
        raise ValueError("scalar H1 load facets must be shaped (facets, 2)")
    if cell_source.ndim != 1 or cell_source.shape != (cell_count,):
        raise ValueError("scalar H1 cell source must contain one value per triangle")
    if facet_load.ndim != 1 or facet_load.shape != (facet_count,):
        raise ValueError("scalar H1 facet load must contain one value per boundary facet")
    for value, label in (
        (coordinates, "scalar H1 coordinates"),
        (cell_source, "scalar H1 cell source"),
        (facet_load, "scalar H1 facet load"),
    ):
        _require_real_array(value, label=label)
    if not jnp.issubdtype(cells.dtype, jnp.integer):
        raise TypeError("scalar H1 cells must use an integer dtype")
    if not jnp.issubdtype(boundary_facets.dtype, jnp.integer):
        raise TypeError("scalar H1 boundary facets must use an integer dtype")

    areas, _ = triangle_p1_geometry(coordinates, cells)
    local = jnp.broadcast_to(cell_source[:, None] * areas[:, None] / 3.0, (cell_count, 3))
    facet_points = coordinates[boundary_facets]
    lengths = jnp.linalg.norm(facet_points[:, 1] - facet_points[:, 0], axis=1)
    contributions = facet_load * lengths / 2.0
    facet_cells = jnp.asarray(boundary_map.facet_cells)
    facet_local = jnp.asarray(boundary_map.facet_local_nodes)
    local = local.at[facet_cells, facet_local[:, 0]].add(contributions)
    return local.at[facet_cells, facet_local[:, 1]].add(contributions)


def triangle_p1_scalar_cell_nodal_load_vectors(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_nodal_source: jax.Array,
) -> jax.Array:
    r"""Return cell-local ``integral N_i q_h`` for a discontinuous P1 source."""

    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("scalar H1 nodal-load coordinates must be shaped (nodes, 2)")
    if cells.ndim != 2 or cells.shape[1:] != (3,):
        raise ValueError("scalar H1 nodal-load cells must be shaped (cells, 3)")
    if cell_nodal_source.ndim != 2 or cell_nodal_source.shape != cells.shape:
        raise ValueError("scalar H1 nodal source must be shaped (cells, 3)")
    _require_real_array(coordinates, label="scalar H1 nodal-load coordinates")
    _require_real_array(cell_nodal_source, label="scalar H1 nodal source")
    if not jnp.issubdtype(cells.dtype, jnp.integer):
        raise TypeError("scalar H1 nodal-load cells must use an integer dtype")
    areas, _ = triangle_p1_geometry(coordinates, cells)
    return (areas[:, None] / 12.0) * (
        jnp.sum(cell_nodal_source, axis=1, keepdims=True) + cell_nodal_source
    )


def tetrahedron_p1_scalar_cell_load_vectors(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_source: jax.Array,
    boundary_facets: jax.Array,
    facet_load: jax.Array,
    boundary_map: ScalarH1BoundaryFacetMap,
) -> jax.Array:
    r"""Return exact Tet4 constant-volume and constant-face load vectors."""

    cell_count = boundary_map.cells.shape[0]
    facet_count = boundary_map.boundary_facets.shape[0]
    if boundary_map.cells.shape[1:] != (4,) or boundary_map.boundary_facets.shape[1:] != (3,):
        raise ContractError("Tet4 scalar H1 load requires a tetrahedron boundary map")
    if coordinates.ndim != 2 or coordinates.shape != (boundary_map.node_count, 3):
        raise ValueError("Tet4 scalar H1 load coordinates must be shaped (nodes, 3)")
    if cells.ndim != 2 or cells.shape != (cell_count, 4):
        raise ValueError("Tet4 scalar H1 load cells must be shaped (cells, 4)")
    if boundary_facets.ndim != 2 or boundary_facets.shape != (facet_count, 3):
        raise ValueError("Tet4 scalar H1 load facets must be shaped (facets, 3)")
    if cell_source.ndim != 1 or cell_source.shape != (cell_count,):
        raise ValueError("Tet4 scalar H1 cell source must contain one value per tetrahedron")
    if facet_load.ndim != 1 or facet_load.shape != (facet_count,):
        raise ValueError("Tet4 scalar H1 facet load must contain one value per boundary face")
    for value, label in (
        (coordinates, "Tet4 scalar H1 coordinates"),
        (cell_source, "Tet4 scalar H1 cell source"),
        (facet_load, "Tet4 scalar H1 facet load"),
    ):
        _require_real_array(value, label=label)
    if not jnp.issubdtype(cells.dtype, jnp.integer):
        raise TypeError("Tet4 scalar H1 cells must use an integer dtype")
    if not jnp.issubdtype(boundary_facets.dtype, jnp.integer):
        raise TypeError("Tet4 scalar H1 boundary facets must use an integer dtype")

    volumes = tetrahedron_p1_geometry(coordinates, cells).volumes
    local = jnp.broadcast_to(cell_source[:, None] * volumes[:, None] / 4.0, (cell_count, 4))
    facet_points = coordinates[boundary_facets]
    facet_areas = 0.5 * jnp.linalg.norm(
        jnp.cross(
            facet_points[:, 1] - facet_points[:, 0],
            facet_points[:, 2] - facet_points[:, 0],
        ),
        axis=1,
    )
    contributions = facet_load * facet_areas / 3.0
    facet_cells = jnp.asarray(boundary_map.facet_cells)
    facet_local = jnp.asarray(boundary_map.facet_local_nodes)
    return local.at[facet_cells[:, None], facet_local].add(contributions[:, None])


def tetrahedron_p1_scalar_robin_cell_terms(
    coordinates: jax.Array,
    cells: jax.Array,
    boundary_facets: jax.Array,
    facet_transfer: jax.Array,
    facet_ambient: jax.Array,
    boundary_map: ScalarH1BoundaryFacetMap,
) -> tuple[jax.Array, jax.Array]:
    r"""Return exact local matrix/load terms for ``-k du/dn = h(u-u_inf)``."""

    cell_count = boundary_map.cells.shape[0]
    facet_count = boundary_map.boundary_facets.shape[0]
    if boundary_map.cells.shape[1:] != (4,) or boundary_map.boundary_facets.shape[1:] != (3,):
        raise ContractError("Tet4 Robin terms require a tetrahedron boundary map")
    if coordinates.ndim != 2 or coordinates.shape != (boundary_map.node_count, 3):
        raise ValueError("Tet4 Robin coordinates must be shaped (nodes, 3)")
    if cells.ndim != 2 or cells.shape != (cell_count, 4):
        raise ValueError("Tet4 Robin cells must be shaped (cells, 4)")
    if boundary_facets.ndim != 2 or boundary_facets.shape != (facet_count, 3):
        raise ValueError("Tet4 Robin facets must be shaped (facets, 3)")
    for value, label in (
        (facet_transfer, "Tet4 Robin transfer coefficient"),
        (facet_ambient, "Tet4 Robin ambient value"),
    ):
        if value.ndim != 1 or value.shape != (facet_count,):
            raise ValueError(f"{label} must contain one value per boundary face")
        _require_real_array(value, label=label)
    _require_real_array(coordinates, label="Tet4 Robin coordinates")
    if not jnp.issubdtype(cells.dtype, jnp.integer):
        raise TypeError("Tet4 Robin cells must use an integer dtype")
    if not jnp.issubdtype(boundary_facets.dtype, jnp.integer):
        raise TypeError("Tet4 Robin facets must use an integer dtype")

    facet_points = coordinates[boundary_facets]
    facet_areas = 0.5 * jnp.linalg.norm(
        jnp.cross(
            facet_points[:, 1] - facet_points[:, 0],
            facet_points[:, 2] - facet_points[:, 0],
        ),
        axis=1,
    )
    dtype = jnp.result_type(coordinates.dtype, facet_transfer.dtype, facet_ambient.dtype)
    reference_mass = (jnp.ones((3, 3), dtype=dtype) + jnp.eye(3, dtype=dtype)) / 12.0
    face_matrix = facet_transfer[:, None, None] * facet_areas[:, None, None] * reference_mass
    face_load = facet_transfer * facet_ambient * facet_areas / 3.0
    local_matrix = jnp.zeros((cell_count, 4, 4), dtype=dtype)
    local_load = jnp.zeros((cell_count, 4), dtype=dtype)
    facet_cells = jnp.asarray(boundary_map.facet_cells)
    facet_local = jnp.asarray(boundary_map.facet_local_nodes)
    rows = jnp.broadcast_to(facet_local[:, :, None], face_matrix.shape)
    columns = jnp.broadcast_to(facet_local[:, None, :], face_matrix.shape)
    local_matrix = local_matrix.at[facet_cells[:, None, None], rows, columns].add(face_matrix)
    local_load = local_load.at[facet_cells[:, None], facet_local].add(face_load[:, None])
    return local_matrix, local_load


def scalar_h1_reduced_cell_rhs(
    cell_stiffness: jax.Array,
    cell_load: jax.Array,
    topology: ScalarH1OwnedGhostTopology,
    dirichlet_values: jax.Array,
) -> jax.Array:
    r"""Return free-row cell RHS values, including ``-K_fc u_c`` strong-boundary terms."""

    width = topology.cell_dof_count
    expected_cell_shape = (topology.cell_count, width)
    if cell_stiffness.ndim != 3 or cell_stiffness.shape != (
        topology.cell_count,
        width,
        width,
    ):
        raise ValueError(
            f"scalar H1 cell stiffness must be shaped (global cells, {width}, {width})"
        )
    if cell_load.ndim != 2 or cell_load.shape != expected_cell_shape:
        raise ValueError(f"scalar H1 cell load must be shaped (global cells, {width})")
    if dirichlet_values.ndim != 1 or dirichlet_values.shape != (
        topology.constrained_nodes.shape[0],
    ):
        raise ValueError("scalar H1 Dirichlet values must follow canonical constrained nodes")
    for value, label in (
        (cell_stiffness, "scalar H1 cell stiffness"),
        (cell_load, "scalar H1 cell load"),
        (dirichlet_values, "scalar H1 Dirichlet values"),
    ):
        _require_real_array(value, label=label)
    dtype = jnp.result_type(cell_stiffness.dtype, cell_load.dtype, dirichlet_values.dtype)
    full_values = (
        jnp.zeros((topology.node_count,), dtype=dtype)
        .at[jnp.asarray(topology.constrained_nodes)]
        .set(dirichlet_values.astype(dtype))
    )
    local_values = full_values[jnp.asarray(topology.cells)]
    reduced = cell_load.astype(dtype) - jnp.einsum(
        "cij,cj->ci",
        cell_stiffness.astype(dtype),
        local_values,
    )
    free_rows = jnp.asarray(topology.cell_reduced_dofs) < topology.free_dof_count
    return jnp.where(free_rows, reduced, 0.0)


def reconstruct_scalar_h1_state(
    topology: ScalarH1OwnedGhostTopology,
    free_state: jax.Array,
    dirichlet_values: jax.Array,
) -> jax.Array:
    """Reconstruct canonical full-node state from disjoint free and constrained values."""

    if free_state.ndim != 1 or free_state.shape != (topology.free_dof_count,):
        raise ValueError("scalar H1 free state must match the canonical free nodes")
    if dirichlet_values.ndim != 1 or dirichlet_values.shape != (
        topology.constrained_nodes.shape[0],
    ):
        raise ValueError("scalar H1 Dirichlet values must follow canonical constrained nodes")
    _require_real_array(free_state, label="scalar H1 free state")
    _require_real_array(dirichlet_values, label="scalar H1 Dirichlet values")
    dtype = jnp.result_type(free_state.dtype, dirichlet_values.dtype)
    state = jnp.zeros((topology.node_count,), dtype=dtype)
    state = state.at[jnp.asarray(topology.free_nodes)].set(free_state.astype(dtype))
    return state.at[jnp.asarray(topology.constrained_nodes)].set(dirichlet_values.astype(dtype))


def pack_collective_scalar_h1_cell_matrix(
    layout: ScalarH1CollectiveLayout,
    cell_matrix: jax.Array,
) -> jax.Array:
    _require_real_array(cell_matrix, label="scalar H1 collective cell matrix")
    return pack_collective_cell_matrix(layout.transport, cell_matrix)


def pack_collective_scalar_h1_cell_vector(
    layout: ScalarH1CollectiveLayout,
    cell_vector: jax.Array,
) -> jax.Array:
    _require_real_array(cell_vector, label="scalar H1 collective cell vector")
    return pack_collective_cell_vector(layout.transport, cell_vector)


def pack_collective_scalar_h1_owned_vector(
    layout: ScalarH1CollectiveLayout,
    vector: jax.Array,
) -> jax.Array:
    _require_real_array(vector, label="scalar H1 collective owned vector")
    return pack_collective_owned_vector(layout.transport, vector)


def pack_collective_scalar_h1_owned_mask(layout: ScalarH1CollectiveLayout) -> jax.Array:
    return pack_collective_owned_mask(layout.transport)


def unpack_collective_scalar_h1_owned_vector(
    layout: ScalarH1CollectiveLayout,
    packed: jax.Array,
) -> jax.Array:
    _require_real_array(packed, label="scalar H1 collective packed owner vector")
    return unpack_collective_owned_vector(layout.transport, packed)


def _real_packed_matvec(operator: PackedCollectiveMatvec) -> PackedCollectiveMatvec:
    def apply(matrix: jax.Array, mapping: jax.Array, vector: jax.Array) -> jax.Array:
        _require_real_array(matrix, label="scalar H1 collective packed cell matrix")
        _require_real_array(vector, label="scalar H1 collective packed owner vector")
        return operator(matrix, mapping, vector)

    return apply


def _real_packed_row_assembly(
    assembly: PackedCollectiveRowAssembly,
) -> PackedCollectiveRowAssembly:
    def apply(cell_vector: jax.Array, mapping: jax.Array) -> jax.Array:
        _require_real_array(cell_vector, label="scalar H1 collective packed cell vector")
        return assembly(cell_vector, mapping)

    return apply


def build_packed_collective_scalar_h1_matvec(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PackedCollectiveMatvec:
    """Build the real scalar pairwise-halo action on an explicit Mesh."""

    return _real_packed_matvec(
        build_packed_collective_matvec(layout.transport, mesh, axis_name=axis_name)
    )


def build_packed_collective_scalar_h1_rhs_assembly(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PackedCollectiveRowAssembly:
    """Build cell-load scatter plus ghost-row-to-owner reduction on an explicit Mesh."""

    return _real_packed_row_assembly(
        build_packed_collective_row_assembly(layout.transport, mesh, axis_name=axis_name)
    )


def build_packed_collective_scalar_h1_cell_gather(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PackedCollectiveCellGather:
    """Build the real scalar owner/ghost gather to one vector per simplex cell."""

    generic = build_packed_collective_cell_gather(
        layout.transport,
        mesh,
        axis_name=axis_name,
    )

    def apply(mapping: jax.Array, vector: jax.Array) -> jax.Array:
        _require_real_array(vector, label="scalar H1 collective packed owner vector")
        return generic(mapping, vector)

    return apply


CanonicalScalarH1RhsAssembly = Callable[[jax.Array], jax.Array]


def build_validation_collective_scalar_h1_rhs_assembly(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> CanonicalScalarH1RhsAssembly:
    """Build a canonical small-problem wrapper for RHS parity and gradients."""

    packed = build_packed_collective_scalar_h1_rhs_assembly(
        layout,
        mesh,
        axis_name=axis_name,
    )
    mapping = jnp.asarray(layout.transport.cell_local_dofs)

    def apply(cell_rhs: jax.Array) -> jax.Array:
        packed_result = packed(
            pack_collective_scalar_h1_cell_vector(layout, cell_rhs),
            mapping,
        )
        return unpack_collective_scalar_h1_owned_vector(layout, packed_result)

    return apply
