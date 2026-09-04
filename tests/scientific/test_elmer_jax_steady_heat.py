from dataclasses import replace

import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    ExecutionPolicy,
    PrepareRequest,
    SolveRequest,
)
from femx.core.problem import Problem  # noqa: E402
from femx.mesh import EntityTag, MeshGeometry  # noqa: E402
from femx.physics import (  # noqa: E402
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_jax,
]


def _solve_both(problem: Problem, elmer_backend, run_directory):
    jax_backend = JaxSteadyHeatBackend(relative_residual_tolerance=1.0e-10)
    jax_solution = solve(prepare(problem, jax_backend), jax_backend)
    elmer_solution = solve(
        prepare(
            problem,
            elmer_backend,
            request=PrepareRequest(run_directory=run_directory),
        ),
        elmer_backend,
        request=SolveRequest(
            run_directory=run_directory,
            policy=ExecutionPolicy(execution_authorized=True, allow_external_process=True),
        ),
    )
    assert jax_solution.convergence.status.value == "converged", jax_solution.convergence
    assert elmer_solution.convergence.status.value == "converged", elmer_solution.convergence
    assert len(elmer_solution.metadata["elmer_executable_sha256"]) == 64
    assert len(elmer_solution.metadata["elmer_heat_solve_sha256"]) == 64
    assert len(elmer_solution.metadata["elmer_source_commit"]) == 40
    assert len(elmer_solution.metadata["elmer_source_digest"]) == 64
    for key in (
        "startinfo_sha256",
        "input_sif_sha256",
        "mesh_header_sha256",
        "mesh_nodes_sha256",
        "mesh_elements_sha256",
        "mesh_boundary_sha256",
        "result_sha256",
        "raw_vtu_sha256",
        "stdout_sha256",
        "stderr_sha256",
    ):
        assert len(elmer_solution.metadata[key]) == 64
    return (
        np.asarray(jax_solution.fields["temperature"].values),
        np.asarray(elmer_solution.fields["temperature"].values),
    )


def test_elmer_and_jax_agree_on_nonzero_variational_heat_load(
    locked_elmer_backend, tmp_path
) -> None:
    mesh = structured_unit_square_mesh(4)
    problem = Problem(
        "cross-flux-sign",
        mesh,
        SteadyHeat(
            regions=(ThermalRegion("domain", 2.0),),
            temperature_boundaries=(TemperatureBoundary("left", 0.0),),
            heat_flux_boundaries=(HeatFluxBoundary("right", 2.0),),
        ),
    )

    jax_temperature, elmer_temperature = _solve_both(
        problem, locked_elmer_backend, tmp_path / "flux"
    )
    exact = np.asarray(mesh.geometry.coordinates)[:, 0]
    np.testing.assert_allclose(jax_temperature, exact, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(elmer_temperature, exact, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(elmer_temperature, jax_temperature, rtol=0.0, atol=3.0e-14)


def test_elmer_and_jax_agree_on_manufactured_volumetric_source(
    locked_elmer_backend, tmp_path
) -> None:
    mesh = structured_unit_square_mesh(8)
    problem = Problem(
        "cross-manufactured-source",
        mesh,
        SteadyHeat(
            regions=(ThermalRegion("domain", 1.0, volumetric_heat_source=2.0),),
            temperature_boundaries=(
                TemperatureBoundary("left", 0.0),
                TemperatureBoundary("right", 0.0),
            ),
        ),
    )

    jax_temperature, elmer_temperature = _solve_both(
        problem, locked_elmer_backend, tmp_path / "source"
    )
    x = np.asarray(mesh.geometry.coordinates)[:, 0]
    exact = x * (1.0 - x)
    np.testing.assert_allclose(jax_temperature, exact, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(elmer_temperature, exact, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(elmer_temperature, jax_temperature, rtol=0.0, atol=5.0e-14)


def test_elmer_and_jax_agree_across_silicon_silica_interface(
    locked_elmer_backend, tmp_path
) -> None:
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
    problem = Problem(
        "cross-siph-layered-thermal",
        mesh,
        SteadyHeat(
            regions=(
                ThermalRegion("silicon", silicon_k),
                ThermalRegion("silica", silica_k),
            ),
            temperature_boundaries=(
                TemperatureBoundary("left", 300.0),
                TemperatureBoundary("right", 301.0),
            ),
        ),
    )

    jax_temperature, elmer_temperature = _solve_both(
        problem, locked_elmer_backend, tmp_path / "silicon-silica"
    )
    resistance_per_area = 1.0e-6 / silicon_k + 1.0e-6 / silica_k
    gradient_heat_load = 1.0 / resistance_per_area
    interface_temperature = 300.0 + gradient_heat_load * 1.0e-6 / silicon_k
    exact = np.where(
        coordinates[:, 0] <= 1.0e-6,
        300.0 + gradient_heat_load * coordinates[:, 0] / silicon_k,
        interface_temperature + gradient_heat_load * (coordinates[:, 0] - 1.0e-6) / silica_k,
    )
    np.testing.assert_allclose(jax_temperature, exact, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(elmer_temperature, exact, rtol=0.0, atol=2.0e-11)
    np.testing.assert_allclose(elmer_temperature, jax_temperature, rtol=0.0, atol=2.0e-11)
