from dataclasses import replace

import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    ExecutionPolicy,
    PrepareRequest,
    SolveRequest,
)
from femx.core.problem import Problem  # noqa: E402
from femx.mesh import EntityTag, MeshGeometry  # noqa: E402
from femx.physics import (  # noqa: E402
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_jax,
]


def _solve_both(problem: Problem, elmer_backend, run_directory):
    jax_backend = JaxSteadyCurrentBackend(relative_residual_tolerance=1.0e-10)
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
    assert len(elmer_solution.metadata["elmer_stat_current_solve_sha256"]) == 64
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
    return jax_solution, elmer_solution


def _assert_field_parity(jax_solution, elmer_solution, *, potential_atol: float) -> None:
    np.testing.assert_allclose(
        elmer_solution.fields["potential"].values,
        jax_solution.fields["potential"].values,
        rtol=0.0,
        atol=potential_atol,
    )
    for field_name in ("electric_field", "current_density", "joule_heat_density"):
        jax_values = np.asarray(jax_solution.fields[field_name].values)
        elmer_values = np.asarray(elmer_solution.fields[field_name].values)
        normalized_l2_error = np.linalg.norm(elmer_values - jax_values) / max(
            np.linalg.norm(jax_values),
            1.0,
        )
        assert normalized_l2_error < 2.0e-12, (field_name, normalized_l2_error)
    for observable in (
        "joule_power_W_per_m",
        "variational_input_power_W_per_m",
    ):
        assert elmer_solution.observables[observable] == pytest.approx(
            jax_solution.observables[observable],
            rel=2.0e-11,
            abs=1.0e-14,
        )
    assert jax_solution.observables["energy_balance_relative_error"] < 2.0e-12
    assert elmer_solution.observables["energy_balance_relative_error"] < 2.0e-12


def test_elmer_and_jax_agree_on_nonzero_variational_current_load(
    locked_elmer_current_backend,
    tmp_path,
) -> None:
    mesh = structured_unit_square_mesh(4)
    problem = Problem(
        "cross-current-flux-sign",
        mesh,
        SteadyCurrent(
            regions=(ConductiveRegion("domain", 2.0),),
            potential_boundaries=(PotentialBoundary("left", 0.0),),
            current_flux_boundaries=(CurrentFluxBoundary("right", 2.0),),
        ),
    )

    jax_solution, elmer_solution = _solve_both(
        problem,
        locked_elmer_current_backend,
        tmp_path / "flux",
    )

    exact = np.asarray(mesh.geometry.coordinates)[:, 0]
    np.testing.assert_allclose(jax_solution.fields["potential"].values, exact, atol=3.0e-14)
    np.testing.assert_allclose(elmer_solution.fields["potential"].values, exact, atol=3.0e-14)
    _assert_field_parity(jax_solution, elmer_solution, potential_atol=3.0e-14)


def test_elmer_and_jax_agree_on_manufactured_current_source(
    locked_elmer_current_backend,
    tmp_path,
) -> None:
    mesh = structured_unit_square_mesh(8)
    problem = Problem(
        "cross-current-manufactured-source",
        mesh,
        SteadyCurrent(
            regions=(ConductiveRegion("domain", 1.0, volumetric_current_source=2.0),),
            potential_boundaries=(
                PotentialBoundary("left", 0.0),
                PotentialBoundary("right", 0.0),
            ),
        ),
    )

    jax_solution, elmer_solution = _solve_both(
        problem,
        locked_elmer_current_backend,
        tmp_path / "source",
    )

    x = np.asarray(mesh.geometry.coordinates)[:, 0]
    exact = x * (1.0 - x)
    np.testing.assert_allclose(jax_solution.fields["potential"].values, exact, atol=6.0e-14)
    np.testing.assert_allclose(elmer_solution.fields["potential"].values, exact, atol=6.0e-14)
    _assert_field_parity(jax_solution, elmer_solution, potential_atol=6.0e-14)


def test_elmer_and_jax_agree_for_a_synthetic_two_doping_heater(
    locked_elmer_current_backend,
    tmp_path,
) -> None:
    mesh = structured_unit_square_mesh(8)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    length = 2.0e-6
    width = 0.5e-6
    interface = 1.0e-6
    coordinates[:, 0] *= length
    coordinates[:, 1] *= width
    cells = np.asarray(mesh.topology.connectivity)
    centroids_x = coordinates[cells, 0].mean(axis=1)
    heater_cells = tuple(np.flatnonzero(centroids_x < interface).tolist())
    contact_cells = tuple(np.flatnonzero(centroids_x >= interface).tolist())
    mesh = replace(
        mesh,
        geometry=MeshGeometry(coordinates),
        tags=(
            *mesh.tags,
            EntityTag("doped_silicon_heater", 2, heater_cells),
            EntityTag("heavily_doped_contact", 2, contact_cells),
        ),
    )
    heater_conductivity = 2.0e3
    contact_conductivity = 2.0e5
    applied_voltage = 1.0
    problem = Problem(
        "cross-siph-two-doping-heater",
        mesh,
        SteadyCurrent(
            regions=(
                ConductiveRegion("doped_silicon_heater", heater_conductivity),
                ConductiveRegion("heavily_doped_contact", contact_conductivity),
            ),
            potential_boundaries=(
                PotentialBoundary("left", 0.0),
                PotentialBoundary("right", applied_voltage),
            ),
        ),
    )

    jax_solution, elmer_solution = _solve_both(
        problem,
        locked_elmer_current_backend,
        tmp_path / "two-doping-heater",
    )

    resistance_per_area = (
        interface / heater_conductivity + (length - interface) / contact_conductivity
    )
    current_magnitude = applied_voltage / resistance_per_area
    interface_potential = current_magnitude * interface / heater_conductivity
    exact_potential = np.where(
        coordinates[:, 0] <= interface,
        current_magnitude * coordinates[:, 0] / heater_conductivity,
        interface_potential
        + current_magnitude * (coordinates[:, 0] - interface) / contact_conductivity,
    )
    expected_joule = np.where(
        centroids_x < interface,
        current_magnitude**2 / heater_conductivity,
        current_magnitude**2 / contact_conductivity,
    )
    expected_power = width * applied_voltage**2 / resistance_per_area

    np.testing.assert_allclose(
        jax_solution.fields["potential"].values,
        exact_potential,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        elmer_solution.fields["potential"].values,
        exact_potential,
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        jax_solution.fields["joule_heat_density"].values,
        expected_joule,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(
        elmer_solution.fields["joule_heat_density"].values,
        expected_joule,
        rtol=2.0e-10,
    )
    assert jax_solution.observables["joule_power_W_per_m"] == pytest.approx(
        expected_power,
        rel=2.0e-11,
    )
    assert elmer_solution.observables["joule_power_W_per_m"] == pytest.approx(
        expected_power,
        rel=2.0e-10,
    )
    _assert_field_parity(jax_solution, elmer_solution, potential_atol=2.0e-11)
