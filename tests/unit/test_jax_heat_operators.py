import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.operators import (  # noqa: E402
    assemble_steady_heat_system,
    assemble_triangle_p1_cell_nodal_load,
    impose_dirichlet_constraints,
    solve_steady_heat,
    triangle_p1_diffusion_cell_matrices,
    triangle_p1_geometry,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_reference_triangle_geometry_and_element_terms_are_exact() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    facets = jnp.asarray(((0, 1), (1, 2), (2, 0)), dtype=jnp.int32)

    areas, gradients = triangle_p1_geometry(coordinates, cells)
    system = assemble_steady_heat_system(
        coordinates,
        cells,
        jnp.asarray((2.0,)),
        jnp.asarray((6.0,)),
        facets,
        jnp.asarray((4.0, 0.0, 0.0)),
    )

    np.testing.assert_allclose(areas, (0.5,))
    np.testing.assert_allclose(gradients.sum(axis=1), 0.0)
    np.testing.assert_allclose(
        system.stiffness,
        ((2.0, -1.0, -1.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)),
    )
    np.testing.assert_allclose(system.load, (3.0, 3.0, 1.0))


def test_cell_diffusion_requires_one_scalar_per_triangle() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)

    with pytest.raises(ValueError, match="one scalar per triangle"):
        triangle_p1_diffusion_cell_matrices(coordinates, cells, jnp.ones((1, 1)))


def test_symmetric_dirichlet_elimination_preserves_the_prescribed_value() -> None:
    stiffness = jnp.asarray(((2.0, -1.0), (-1.0, 1.0)))
    load = jnp.asarray((0.0, 1.0))
    constrained = impose_dirichlet_constraints(
        stiffness,
        load,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((3.0,)),
    )

    np.testing.assert_allclose(constrained.stiffness, ((1.0, 0.0), (0.0, 1.0)))
    np.testing.assert_allclose(constrained.load, (3.0, 4.0))


def test_cell_local_p1_source_uses_the_exact_consistent_mass_and_preserves_integral() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    source = jnp.asarray(((1.0, 2.0, 4.0),), dtype=jnp.float64)

    load = jax.jit(assemble_triangle_p1_cell_nodal_load)(coordinates, cells, source)

    np.testing.assert_allclose(
        load, ((2.0 + 2.0 + 4.0) / 12.0, (1.0 + 4.0 + 4.0) / 12.0, (1.0 + 2.0 + 8.0) / 12.0)
    )
    np.testing.assert_allclose(jnp.sum(load), jnp.mean(source), rtol=0.0, atol=1.0e-15)


def test_compiled_jax_solve_reproduces_linear_temperature_field() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2), (0, 2, 3)), dtype=jnp.int32)
    facets = jnp.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=jnp.int32)
    temperature, system = solve_steady_heat(
        coordinates,
        cells,
        jnp.ones((2,)),
        jnp.zeros((2,)),
        facets,
        jnp.asarray((0.0, 1.0, 0.0, 0.0)),
        jnp.asarray((0, 3), dtype=jnp.int32),
        jnp.asarray((0.0, 0.0)),
    )

    np.testing.assert_allclose(temperature, coordinates[:, 0], rtol=1.0e-13, atol=1.0e-13)
    reaction = system.stiffness @ temperature - system.load
    np.testing.assert_allclose(reaction.sum(), -1.0, atol=1.0e-13)
