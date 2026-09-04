from dataclasses import replace

import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from femx.backends.jax.operators import triangle_p1_geometry  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import SolveRequest  # noqa: E402
from femx.core.problem import Problem  # noqa: E402
from femx.mesh import EntityTag, MeshGeometry  # noqa: E402
from femx.physics import SteadyHeat, TemperatureBoundary, ThermalRegion  # noqa: E402
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _solve_quadratic(intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = structured_unit_square_mesh(intervals)
    physics = SteadyHeat(
        regions=(ThermalRegion("domain", conductivity=1.0, volumetric_heat_source=2.0),),
        temperature_boundaries=(
            TemperatureBoundary("left", 0.0),
            TemperatureBoundary("right", 0.0),
        ),
    )
    backend = JaxSteadyHeatBackend(relative_residual_tolerance=1.0e-11)
    problem = Problem(f"quadratic-{intervals}", mesh, physics)
    solution = solve(prepare(problem, backend), backend, request=SolveRequest())
    assert solution.convergence.status.value == "converged"
    return (
        np.asarray(mesh.geometry.coordinates),
        np.asarray(mesh.topology.connectivity),
        np.asarray(solution.fields["temperature"].values),
    )


def _quadratic_l2_error(
    coordinates: np.ndarray,
    cells: np.ndarray,
    temperature: np.ndarray,
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
    interpolated = np.einsum("qi,ci->cq", barycentric, temperature[cells])
    exact = quadrature_points[:, :, 0] * (1.0 - quadrature_points[:, :, 0])
    first = vertices[:, 1, :] - vertices[:, 0, :]
    second = vertices[:, 2, :] - vertices[:, 0, :]
    areas = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return float(np.sqrt(np.sum(areas[:, None] * weights * (interpolated - exact) ** 2)))


def test_manufactured_quadratic_solution_has_second_order_l2_convergence() -> None:
    errors: list[float] = []
    for intervals in (4, 8, 16):
        coordinates, cells, temperature = _solve_quadratic(intervals)
        exact_nodes = coordinates[:, 0] * (1.0 - coordinates[:, 0])
        np.testing.assert_allclose(temperature, exact_nodes, rtol=0.0, atol=2.0e-14)
        errors.append(_quadratic_l2_error(coordinates, cells, temperature))

    rates = [np.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert errors[-1] < 7.2e-4
    assert all(1.95 < rate < 2.05 for rate in rates)


def test_silicon_silica_interface_matches_piecewise_analytic_flux() -> None:
    mesh = structured_unit_square_mesh(8)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    coordinates[:, 0] *= 2.0e-6
    coordinates[:, 1] *= 1.0e-6
    cells = np.asarray(mesh.topology.connectivity)
    centroids_x = coordinates[cells, 0].mean(axis=1)
    silicon_cells = tuple(np.flatnonzero(centroids_x < 1.0e-6).tolist())
    silica_cells = tuple(np.flatnonzero(centroids_x >= 1.0e-6).tolist())
    mesh = replace(
        mesh,
        geometry=MeshGeometry(coordinates),
        tags=(
            *mesh.tags,
            EntityTag("silicon", 2, silicon_cells),
            EntityTag("silica", 2, silica_cells),
        ),
    )
    silicon_k = 148.0
    silica_k = 1.38
    physics = SteadyHeat(
        regions=(
            ThermalRegion("silicon", silicon_k),
            ThermalRegion("silica", silica_k),
        ),
        temperature_boundaries=(
            TemperatureBoundary("left", 300.0),
            TemperatureBoundary("right", 301.0),
        ),
    )
    backend = JaxSteadyHeatBackend()
    solution = solve(prepare(Problem("siph-layered-thermal", mesh, physics), backend), backend)
    temperature = np.asarray(solution.fields["temperature"].values)

    resistance_per_area = 1.0e-6 / silicon_k + 1.0e-6 / silica_k
    gradient_heat_load = 1.0 / resistance_per_area
    interface_temperature = 300.0 + gradient_heat_load * 1.0e-6 / silicon_k
    exact = np.where(
        coordinates[:, 0] <= 1.0e-6,
        300.0 + gradient_heat_load * coordinates[:, 0] / silicon_k,
        interface_temperature + gradient_heat_load * (coordinates[:, 0] - 1.0e-6) / silica_k,
    )
    np.testing.assert_allclose(temperature, exact, rtol=0.0, atol=2.0e-12)

    _, basis_gradients = triangle_p1_geometry(
        jax.numpy.asarray(coordinates), jax.numpy.asarray(cells)
    )
    temperature_gradients = np.einsum("ci,cid->cd", temperature[cells], np.asarray(basis_gradients))
    cell_conductivity = np.where(centroids_x < 1.0e-6, silicon_k, silica_k)
    np.testing.assert_allclose(temperature_gradients[:, 1], 0.0, atol=2.0e-6)
    np.testing.assert_allclose(
        cell_conductivity * temperature_gradients[:, 0],
        gradient_heat_load,
        rtol=2.0e-12,
        atol=1.0e-3,
    )
