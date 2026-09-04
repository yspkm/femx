from __future__ import annotations

import numpy as np

from femx.backends.elmer.tet4_case import lower_tagged_tet4_mesh
from femx.backends.elmer.tet4_electrothermal_case import (
    ElmerTet4BoundaryCondition,
    ElmerTet4ElectrothermalBody,
    ElmerTet4ElectrothermalCase,
)
from femx.mesh import CellType, EntityTag, Mesh, MeshGeometry, MeshTopology

_LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def structured_distinct_space_case() -> tuple[Mesh, ElmerTet4ElectrothermalCase]:
    """Return a tiny conformal conductor-over-dielectric Tet4 verification case."""

    x_values = (0.0, 0.5, 1.0)
    y_values = (0.0, 1.0)
    z_values = (-1.0, 0.0, 1.0)
    coordinates = np.asarray(
        tuple((x, y, z) for z in z_values for y in y_values for x in x_values),
        dtype=np.float64,
    )

    def node(ix: int, iy: int, iz: int) -> int:
        return iz * len(y_values) * len(x_values) + iy * len(x_values) + ix

    cells: list[tuple[int, int, int, int]] = []
    regions: list[int] = []
    for iz in range(len(z_values) - 1):
        for iy in range(len(y_values) - 1):
            for ix in range(len(x_values) - 1):
                v000 = node(ix, iy, iz)
                v100 = node(ix + 1, iy, iz)
                v010 = node(ix, iy + 1, iz)
                v110 = node(ix + 1, iy + 1, iz)
                v001 = node(ix, iy, iz + 1)
                v101 = node(ix + 1, iy, iz + 1)
                v011 = node(ix, iy + 1, iz + 1)
                v111 = node(ix + 1, iy + 1, iz + 1)
                raw = (
                    (v000, v100, v110, v111),
                    (v000, v110, v010, v111),
                    (v000, v010, v011, v111),
                    (v000, v011, v001, v111),
                    (v000, v001, v101, v111),
                    (v000, v101, v100, v111),
                )
                for tetrahedron in raw:
                    oriented = list(tetrahedron)
                    points = coordinates[oriented]
                    jacobian = np.stack(
                        (points[1] - points[0], points[2] - points[0], points[3] - points[0]),
                        axis=1,
                    )
                    if np.linalg.det(jacobian) < 0.0:
                        oriented[1], oriented[2] = oriented[2], oriented[1]
                    cells.append(tuple(oriented))
                    regions.append(0 if iz == 1 else 1)
    connectivity = np.asarray(cells, dtype=np.int64)

    occurrences: dict[tuple[int, int, int], int] = {}
    for cell in connectivity:
        for local_face in _LOCAL_FACES:
            face = tuple(sorted(int(cell[index]) for index in local_face))
            occurrences[face] = occurrences.get(face, 0) + 1
    external = tuple(sorted(face for face, count in occurrences.items() if count == 1))
    facets = np.asarray(external, dtype=np.int64)
    negative: list[int] = []
    positive: list[int] = []
    bottom: list[int] = []
    other: list[int] = []
    for facet_id, facet in enumerate(facets):
        points = coordinates[facet]
        if np.all(points[:, 0] == 0.0) and np.all(points[:, 2] >= 0.0):
            negative.append(facet_id)
        elif np.all(points[:, 0] == 1.0) and np.all(points[:, 2] >= 0.0):
            positive.append(facet_id)
        elif np.all(points[:, 2] == -1.0):
            bottom.append(facet_id)
        else:
            other.append(facet_id)

    mesh = Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(connectivity, CellType.TETRAHEDRON, len(coordinates)),
        tags=(
            EntityTag(
                "conductor",
                3,
                tuple(index for index, region in enumerate(regions) if region == 0),
            ),
            EntityTag(
                "dielectric",
                3,
                tuple(index for index, region in enumerate(regions) if region == 1),
            ),
            EntityTag("terminal_negative", 2, tuple(negative)),
            EntityTag("terminal_positive", 2, tuple(positive)),
            EntityTag("bottom_temperature", 2, tuple(bottom)),
            EntityTag("other", 2, tuple(other)),
        ),
        boundary_facets=MeshTopology(facets, CellType.TRIANGLE, len(coordinates)),
    )
    deck = lower_tagged_tet4_mesh(
        mesh,
        region_tags=("conductor", "dielectric"),
        boundary_tags=(
            "terminal_negative",
            "terminal_positive",
            "bottom_temperature",
            "other",
        ),
    )
    case = ElmerTet4ElectrothermalCase(
        mesh=deck,
        bodies=(
            ElmerTet4ElectrothermalBody(
                1,
                heat_conductivity_w_per_m_k=1.0,
                electric_conductivity_s_per_m=2.0,
            ),
            ElmerTet4ElectrothermalBody(2, heat_conductivity_w_per_m_k=1.0),
        ),
        boundaries=(
            ElmerTet4BoundaryCondition(1, potential_v=0.0),
            ElmerTet4BoundaryCondition(2, potential_v=1.0),
            ElmerTet4BoundaryCondition(3, temperature_k=300.0),
        ),
        initial_temperature_k=300.0,
    )
    return mesh, case


def structured_distinct_space_jax_plan(mesh: Mesh):
    """Bind the same synthetic mesh and coefficients to the native JAX Tet4 path."""

    from femx.backends.jax.tet4_electrothermal import prepare_tet4_electrothermal_plan

    if mesh.boundary_facets is None:
        raise AssertionError("synthetic distinct-space mesh must carry boundary facets")
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    facets = np.asarray(mesh.boundary_facets.connectivity, dtype=np.int64)
    current_cells = np.asarray(mesh.tag("conductor").entity_ids, dtype=np.int64)

    def boundary_nodes(name: str) -> np.ndarray:
        facet_ids = np.asarray(mesh.tag(name).entity_ids, dtype=np.int64)
        return np.unique(facets[facet_ids]).astype(np.int64)

    negative = boundary_nodes("terminal_negative")
    positive = boundary_nodes("terminal_positive")
    current_dirichlet = np.concatenate((negative, positive))
    current_order = np.argsort(current_dirichlet, kind="stable")
    current_dirichlet = current_dirichlet[current_order]
    current_scale = np.concatenate(
        (
            np.zeros(negative.shape, dtype=np.float64),
            np.ones(positive.shape, dtype=np.float64),
        )
    )[current_order]
    bottom = boundary_nodes("bottom_temperature")
    empty_facets = np.empty((0, 3), dtype=np.int64)
    empty_values = np.empty((0,), dtype=np.float64)
    return prepare_tet4_electrothermal_plan(
        coordinates,
        cells,
        np.zeros((cells.shape[0],), dtype=np.int64),
        current_cells,
        current_conductivity=np.full((current_cells.size,), 2.0, dtype=np.float64),
        current_cell_source=np.zeros((current_cells.size,), dtype=np.float64),
        current_flux_facets=empty_facets,
        current_facet_flux=empty_values,
        current_dirichlet_nodes=current_dirichlet,
        current_dirichlet_base=np.zeros(current_dirichlet.shape, dtype=np.float64),
        current_dirichlet_voltage_scale=current_scale,
        thermal_conductivity=np.ones((cells.shape[0],), dtype=np.float64),
        thermal_cell_source=np.zeros((cells.shape[0],), dtype=np.float64),
        thermal_flux_facets=empty_facets,
        thermal_facet_flux=empty_values,
        thermal_robin_facets=empty_facets,
        thermal_robin_transfer=empty_values,
        thermal_robin_ambient=empty_values,
        thermal_dirichlet_nodes=bottom,
        thermal_dirichlet_values=np.full(bottom.shape, 300.0, dtype=np.float64),
        thermal_reference=300.0,
        partition_count=1,
    )
