from dataclasses import replace

import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.core.problem import Problem  # noqa: E402
from femx.mesh import EntityTag, MeshGeometry  # noqa: E402
from femx.physics import (  # noqa: E402
    ConductiveRegion,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _solve_quadratic(intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = structured_unit_square_mesh(intervals)
    physics = SteadyCurrent(
        regions=(ConductiveRegion("domain", 1.0, volumetric_current_source=2.0),),
        potential_boundaries=(
            PotentialBoundary("left", 0.0),
            PotentialBoundary("right", 0.0),
        ),
    )
    backend = JaxSteadyCurrentBackend(relative_residual_tolerance=1.0e-11)
    solution = solve(
        prepare(Problem(f"quadratic-current-{intervals}", mesh, physics), backend),
        backend,
    )
    assert solution.convergence.status.value == "converged"
    assert solution.observables["energy_balance_relative_error"] < 2.0e-14
    return (
        np.asarray(mesh.geometry.coordinates),
        np.asarray(mesh.topology.connectivity),
        np.asarray(solution.fields["potential"].values),
    )


def _quadratic_l2_error(
    coordinates: np.ndarray,
    cells: np.ndarray,
    potential: np.ndarray,
) -> float:
    a = 0.059715871789770
    b = 0.470142064105115
    c = 0.797426985353087
    d = 0.101286507323456
    barycentric = np.asarray(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (a, b, b),
            (b, a, b),
            (b, b, a),
            (c, d, d),
            (d, c, d),
            (d, d, c),
        )
    )
    weights = np.asarray((0.225, *(0.132394152788506,) * 3, *(0.125939180544827,) * 3))
    vertices = coordinates[cells]
    quadrature_points = np.einsum("qi,cid->cqd", barycentric, vertices)
    interpolated = np.einsum("qi,ci->cq", barycentric, potential[cells])
    exact = quadrature_points[:, :, 0] * (1.0 - quadrature_points[:, :, 0])
    first = vertices[:, 1, :] - vertices[:, 0, :]
    second = vertices[:, 2, :] - vertices[:, 0, :]
    areas = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return float(np.sqrt(np.sum(areas[:, None] * weights * (interpolated - exact) ** 2)))


def test_manufactured_current_solution_has_second_order_potential_convergence() -> None:
    errors: list[float] = []
    for intervals in (4, 8, 16):
        coordinates, cells, potential = _solve_quadratic(intervals)
        exact_nodes = coordinates[:, 0] * (1.0 - coordinates[:, 0])
        np.testing.assert_allclose(potential, exact_nodes, rtol=0.0, atol=2.0e-14)
        errors.append(_quadratic_l2_error(coordinates, cells, potential))

    rates = [np.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert errors[-1] < 7.2e-4
    assert all(1.95 < rate < 2.05 for rate in rates)


def test_heater_and_doped_region_preserve_current_and_piecewise_joule_heating() -> None:
    mesh = structured_unit_square_mesh(8)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    length = 2.0e-6
    height = 0.5e-6
    coordinates[:, 0] *= length
    coordinates[:, 1] *= height
    cells = np.asarray(mesh.topology.connectivity)
    centroids_x = coordinates[cells, 0].mean(axis=1)
    heater_cells = tuple(np.flatnonzero(centroids_x < length / 2.0).tolist())
    doped_cells = tuple(np.flatnonzero(centroids_x >= length / 2.0).tolist())
    mesh = replace(
        mesh,
        geometry=MeshGeometry(coordinates),
        tags=(
            *mesh.tags,
            EntityTag("heater", 2, heater_cells),
            EntityTag("doped_region", 2, doped_cells),
        ),
    )
    heater_sigma = 2.0e5
    doped_sigma = 5.0e4
    applied_voltage = 1.0e-3
    physics = SteadyCurrent(
        regions=(
            ConductiveRegion("heater", heater_sigma),
            ConductiveRegion("doped_region", doped_sigma),
        ),
        potential_boundaries=(
            PotentialBoundary("left", 0.0),
            PotentialBoundary("right", applied_voltage),
        ),
    )
    backend = JaxSteadyCurrentBackend()
    solution = solve(
        prepare(Problem("siph-series-conductor", mesh, physics), backend),
        backend,
    )

    resistance_per_area = (length / 2.0) / heater_sigma + (length / 2.0) / doped_sigma
    current_magnitude = applied_voltage / resistance_per_area
    interface_potential = current_magnitude * (length / 2.0) / heater_sigma
    exact_potential = np.where(
        coordinates[:, 0] <= length / 2.0,
        current_magnitude * coordinates[:, 0] / heater_sigma,
        interface_potential + current_magnitude * (coordinates[:, 0] - length / 2.0) / doped_sigma,
    )
    np.testing.assert_allclose(
        solution.fields["potential"].values,
        exact_potential,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    current_density = np.asarray(solution.fields["current_density"].values)
    np.testing.assert_allclose(current_density[:, 0], -current_magnitude, rtol=2.0e-12)
    np.testing.assert_allclose(current_density[:, 1], 0.0, atol=2.0e-6)
    conductivity = np.where(centroids_x < length / 2.0, heater_sigma, doped_sigma)
    np.testing.assert_allclose(
        solution.fields["joule_heat_density"].values,
        current_magnitude**2 / conductivity,
        rtol=4.0e-12,
    )
    expected_power_per_depth = applied_voltage * current_magnitude * height
    assert solution.observables["joule_power_W_per_m"] == pytest.approx(
        expected_power_per_depth,
        rel=4.0e-12,
    )
    assert solution.observables["energy_balance_relative_error"] < 2.0e-14
