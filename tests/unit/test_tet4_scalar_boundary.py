from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.scalar_collective import (  # noqa: E402
    prepare_scalar_h1_boundary_facet_map,
    tetrahedron_p1_scalar_cell_load_vectors,
    tetrahedron_p1_scalar_robin_cell_terms,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


COORDINATES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
CELLS = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
FACETS = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)


def _map(facets: np.ndarray = FACETS):
    return prepare_scalar_h1_boundary_facet_map(CELLS, facets, node_count=4)


def test_tet4_boundary_map_preserves_requested_face_node_order() -> None:
    facets = FACETS[:, ::-1]
    boundary_map = _map(facets)

    np.testing.assert_array_equal(boundary_map.facet_cells, np.zeros((4,), dtype=np.int64))
    np.testing.assert_array_equal(boundary_map.facet_local_nodes, facets)
    assert not boundary_map.cells.flags.writeable
    assert not boundary_map.boundary_facets.flags.writeable


def test_tet4_constant_volume_and_face_loads_integrate_exactly() -> None:
    face = FACETS[1:2]
    boundary_map = _map(face)
    source = jnp.asarray((12.0,), dtype=jnp.float64)
    flux = jnp.asarray((6.0,), dtype=jnp.float64)

    apply = jax.jit(
        lambda cell_source, facet_load: tetrahedron_p1_scalar_cell_load_vectors(
            jnp.asarray(COORDINATES),
            jnp.asarray(CELLS),
            cell_source,
            jnp.asarray(face),
            facet_load,
            boundary_map,
        )
    )
    local = apply(source, flux)

    # Volume is 1/6 and the selected coordinate face has area 1/2.
    np.testing.assert_allclose(local, ((1.5, 0.5, 1.5, 1.5),), rtol=0.0, atol=1.0e-15)
    assert float(jnp.sum(local)) == pytest.approx(5.0)
    derivative = jax.grad(
        lambda value: jnp.sum(
            tetrahedron_p1_scalar_cell_load_vectors(
                jnp.asarray(COORDINATES),
                jnp.asarray(CELLS),
                source,
                jnp.asarray(face),
                value[None],
                boundary_map,
            )
        )
    )(jnp.asarray(6.0))
    assert float(derivative) == pytest.approx(0.5)


def test_tet4_robin_terms_match_exact_triangle_mass_and_ambient_load() -> None:
    face = FACETS[1:2]
    boundary_map = _map(face)
    apply = jax.jit(
        lambda transfer, ambient: tetrahedron_p1_scalar_robin_cell_terms(
            jnp.asarray(COORDINATES),
            jnp.asarray(CELLS),
            jnp.asarray(face),
            transfer,
            ambient,
            boundary_map,
        )
    )
    matrix, load = apply(jnp.asarray((8.0,)), jnp.asarray((300.0,)))

    expected_matrix = np.zeros((4, 4), dtype=np.float64)
    expected_matrix[np.ix_((0, 2, 3), (0, 2, 3))] = (np.ones((3, 3)) + np.eye(3)) / 3.0
    expected_load = np.asarray((400.0, 0.0, 400.0, 400.0))
    np.testing.assert_allclose(matrix[0], expected_matrix, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(load[0], expected_load, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(matrix[0] @ jnp.full((4,), 300.0), load[0])


def test_tet4_boundary_contracts_reject_nonfaces_duplicates_and_identity_drift() -> None:
    with pytest.raises(ContractError, match="3 nodes for the cell family"):
        prepare_scalar_h1_boundary_facet_map(CELLS, FACETS[:, :2], node_count=4)
    with pytest.raises(ContractError, match="cannot repeat a tetrahedron face"):
        prepare_scalar_h1_boundary_facet_map(CELLS, FACETS[[0, 0]], node_count=4)
    with pytest.raises(ContractError, match="exactly one exterior tetrahedron face"):
        prepare_scalar_h1_boundary_facet_map(CELLS, np.asarray(((0, 1, 4),)), node_count=5)

    boundary_map = _map(FACETS[:1])
    with pytest.raises(ContractError, match="facet cells disagree"):
        replace(boundary_map, facet_cells=np.asarray((1,), dtype=np.int64))
    with pytest.raises(ContractError, match="local nodes disagree"):
        replace(
            boundary_map,
            facet_local_nodes=boundary_map.facet_local_nodes[:, ::-1],
        )


def test_tet4_surface_kernels_reject_family_shape_and_dtype_drift() -> None:
    face = FACETS[:1]
    boundary_map = _map(face)
    coordinates = jnp.asarray(COORDINATES)
    cells = jnp.asarray(CELLS)
    facets = jnp.asarray(face)
    source = jnp.ones((1,), dtype=jnp.float64)
    load = jnp.ones((1,), dtype=jnp.float64)

    for name, value, message in (
        ("coordinates", coordinates[:, :2], "coordinates"),
        ("cells", jnp.ones((1, 3), dtype=jnp.int32), "cells"),
        ("boundary_facets", jnp.ones((1, 2), dtype=jnp.int32), "facets"),
        ("cell_source", jnp.ones((2,), dtype=jnp.float64), "cell source"),
        ("facet_load", jnp.ones((2,), dtype=jnp.float64), "facet load"),
    ):
        arguments = {
            "coordinates": coordinates,
            "cells": cells,
            "cell_source": source,
            "boundary_facets": facets,
            "facet_load": load,
            "boundary_map": boundary_map,
        }
        arguments[name] = value
        with pytest.raises(ValueError, match=message):
            tetrahedron_p1_scalar_cell_load_vectors(**arguments)

    with pytest.raises(TypeError, match="coordinates"):
        tetrahedron_p1_scalar_cell_load_vectors(
            coordinates.astype(jnp.complex128), cells, source, facets, load, boundary_map
        )
    with pytest.raises(TypeError, match="cells"):
        tetrahedron_p1_scalar_cell_load_vectors(
            coordinates, cells.astype(jnp.float64), source, facets, load, boundary_map
        )
    with pytest.raises(TypeError, match="boundary facets"):
        tetrahedron_p1_scalar_cell_load_vectors(
            coordinates, cells, source, facets.astype(jnp.float64), load, boundary_map
        )

    triangle_map = prepare_scalar_h1_boundary_facet_map(
        np.asarray(((0, 1, 2),)), np.asarray(((0, 1),)), node_count=3
    )
    with pytest.raises(ContractError, match="tetrahedron boundary map"):
        tetrahedron_p1_scalar_cell_load_vectors(
            coordinates,
            cells,
            source,
            facets,
            load,
            triangle_map,
        )
    with pytest.raises(ContractError, match="tetrahedron boundary map"):
        tetrahedron_p1_scalar_robin_cell_terms(
            coordinates,
            cells,
            facets,
            load,
            load,
            triangle_map,
        )

    for name, value, message in (
        ("coordinates", coordinates[:, :2], "coordinates"),
        ("cells", jnp.ones((1, 3), dtype=jnp.int32), "cells"),
        ("boundary_facets", jnp.ones((1, 2), dtype=jnp.int32), "facets"),
        ("facet_transfer", jnp.ones((2,), dtype=jnp.float64), "transfer"),
        ("facet_ambient", jnp.ones((2,), dtype=jnp.float64), "ambient"),
    ):
        arguments = {
            "coordinates": coordinates,
            "cells": cells,
            "boundary_facets": facets,
            "facet_transfer": load,
            "facet_ambient": load,
            "boundary_map": boundary_map,
        }
        arguments[name] = value
        with pytest.raises(ValueError, match=message):
            tetrahedron_p1_scalar_robin_cell_terms(**arguments)

    with pytest.raises(TypeError, match="coordinates"):
        tetrahedron_p1_scalar_robin_cell_terms(
            coordinates.astype(jnp.complex128), cells, facets, load, load, boundary_map
        )
    with pytest.raises(TypeError, match="cells"):
        tetrahedron_p1_scalar_robin_cell_terms(
            coordinates, cells.astype(jnp.float64), facets, load, load, boundary_map
        )
    with pytest.raises(TypeError, match="facets"):
        tetrahedron_p1_scalar_robin_cell_terms(
            coordinates, cells, facets.astype(jnp.float64), load, load, boundary_map
        )
