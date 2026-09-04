from dataclasses import replace

import pytest
from tests.electrothermal_support import (
    parameterized_microheater_coupling,
    triangle_areas,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.electrothermal import (  # noqa: E402
    DifferentiableOneWayElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    ExecutionPolicy,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.problem import Problem  # noqa: E402
from femx.physics import SteadyHeat, ThermalRegion  # noqa: E402
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_jax,
]


def test_microheater_chain_and_adjoint_match_locked_elmer_finite_differences(
    locked_elmer_backend,
    locked_elmer_current_backend,
    tmp_path,
) -> None:
    coupling, current_parameters, heat_parameters = parameterized_microheater_coupling(intervals=4)
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    system = DifferentiableOneWayElectrothermal(
        coupling,
        current_backend.bind_differentiable(
            prepare(coupling.electrical_problem, current_backend),
            current_parameters,
        ),
        heat_backend.bind_differentiable(
            prepare(coupling.thermal_problem, heat_backend),
            heat_parameters,
        ),
    )
    current_initial = system.initial_current_values
    thermal_initial = system.initial_thermal_values
    areas = triangle_areas(coupling.thermal_problem)
    cells = np.asarray(coupling.thermal_problem.mesh.topology.connectivity)
    nodal_weights = np.zeros((coupling.thermal_problem.mesh.geometry.node_count,), dtype=np.float64)
    np.add.at(nodal_weights, cells.reshape(-1), np.repeat(areas / 3.0, 3))
    nodal_weights /= nodal_weights.sum()
    jax_temperature = np.asarray(system.temperature(current_initial, thermal_initial))
    jax_joule = np.asarray(system.current.joule_heat_density(current_initial))
    explicit = system.vjp(
        current_initial,
        thermal_initial,
        jnp.asarray(nodal_weights, dtype=jnp.float64),
    )

    current_reference = replace(
        coupling.electrical_problem,
        physics=replace(
            coupling.electrical_problem.physics,
            gradient_method=GradientMethod.NONE,
        ),
    )
    heat_reference = replace(
        coupling.thermal_problem,
        physics=replace(
            coupling.thermal_problem.physics,
            gradient_method=GradientMethod.NONE,
        ),
    )
    current_base = dict(current_parameters.values)
    heat_base = dict(heat_parameters.values)

    def current_values(active: np.ndarray):
        values = dict(current_base)
        values.update(
            (name, float(value))
            for name, value in zip(system.current.parameter_names, active, strict=True)
        )
        return coupling.electrical_problem.parameters.bind(values)

    def heat_values(active: np.ndarray):
        values = dict(heat_base)
        values.update(
            (name, float(value))
            for name, value in zip(system.thermal.parameter_names, active, strict=True)
        )
        return coupling.thermal_problem.parameters.bind(values)

    def run_current(active: np.ndarray, attempt: str):
        run_directory = tmp_path / attempt / "current"
        run_directory.parent.mkdir(exist_ok=True)
        solution = solve(
            prepare(
                current_reference,
                locked_elmer_current_backend,
                request=PrepareRequest(run_directory=run_directory),
            ),
            locked_elmer_current_backend,
            request=SolveRequest(
                parameters=current_values(active),
                run_directory=run_directory,
                policy=ExecutionPolicy(
                    execution_authorized=True,
                    allow_external_process=True,
                ),
            ),
        )
        assert solution.convergence.status.value == "converged"
        assert solution.observables["energy_balance_relative_error"] < 3.0e-12
        return solution

    def materialized_heat_problem(joule: np.ndarray, attempt: str) -> Problem:
        physics = heat_reference.physics
        assert isinstance(physics, SteadyHeat)
        regions = []
        transferred_power = 0.0
        for region in physics.regions:
            ids = np.asarray(heat_reference.mesh.tag(region.tag).entity_ids, dtype=np.int64)
            region_areas = areas[ids]
            region_joule = joule[ids]
            mean = float(np.vdot(region_areas, region_joule) / np.sum(region_areas))
            spread = float(np.max(np.abs(region_joule - mean)) / max(abs(mean), 1.0))
            assert spread < 5.0e-10, (region.tag, spread)
            transferred_power += mean * float(np.sum(region_areas))
            regions.append(
                ThermalRegion(
                    region.tag,
                    region.conductivity,
                    volumetric_heat_source=mean,
                )
            )
        electrical_power = float(np.vdot(areas, joule))
        assert transferred_power == pytest.approx(electrical_power, rel=3.0e-16, abs=1.0e-13)
        return replace(
            heat_reference,
            name=f"{heat_reference.name}-{attempt}",
            physics=replace(physics, regions=tuple(regions)),
        )

    def run_heat(joule: np.ndarray, active: np.ndarray, attempt: str):
        problem = materialized_heat_problem(joule, attempt)
        run_directory = tmp_path / attempt / "heat"
        run_directory.parent.mkdir(exist_ok=True)
        solution = solve(
            prepare(
                problem,
                locked_elmer_backend,
                request=PrepareRequest(run_directory=run_directory),
            ),
            locked_elmer_backend,
            request=SolveRequest(
                parameters=heat_values(active),
                run_directory=run_directory,
                policy=ExecutionPolicy(
                    execution_authorized=True,
                    allow_external_process=True,
                ),
            ),
        )
        assert solution.convergence.status.value == "converged"
        return solution

    def objective_from_chain(
        current_active: np.ndarray,
        thermal_active: np.ndarray,
        attempt: str,
    ) -> float:
        current_solution = run_current(current_active, attempt)
        joule = np.asarray(current_solution.fields["joule_heat_density"].values)
        heat_solution = run_heat(joule, thermal_active, attempt)
        return float(np.vdot(nodal_weights, heat_solution.fields["temperature"].values))

    current_numpy = np.asarray(current_initial)
    thermal_numpy = np.asarray(thermal_initial)
    baseline_current = run_current(current_numpy, "baseline")
    elmer_joule = np.asarray(baseline_current.fields["joule_heat_density"].values)
    baseline_heat = run_heat(elmer_joule, thermal_numpy, "baseline")
    elmer_temperature = np.asarray(baseline_heat.fields["temperature"].values)

    np.testing.assert_allclose(elmer_joule, jax_joule, rtol=3.0e-10, atol=2.0e-2)
    np.testing.assert_allclose(elmer_temperature, jax_temperature, rtol=0.0, atol=3.0e-9)

    elmer_current_gradient = []
    for index, value in enumerate(current_numpy):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = current_numpy.copy()
        minus = current_numpy.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = objective_from_chain(plus, thermal_numpy, f"current-{index}-plus")
        minus_value = objective_from_chain(minus, thermal_numpy, f"current-{index}-minus")
        elmer_current_gradient.append((plus_value - minus_value) / (2.0 * step))

    elmer_thermal_gradient = []
    for index, value in enumerate(thermal_numpy):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = thermal_numpy.copy()
        minus = thermal_numpy.copy()
        plus[index] += step
        minus[index] -= step
        plus_solution = run_heat(elmer_joule, plus, f"thermal-{index}-plus")
        minus_solution = run_heat(elmer_joule, minus, f"thermal-{index}-minus")
        plus_value = float(np.vdot(nodal_weights, plus_solution.fields["temperature"].values))
        minus_value = float(np.vdot(nodal_weights, minus_solution.fields["temperature"].values))
        elmer_thermal_gradient.append((plus_value - minus_value) / (2.0 * step))

    np.testing.assert_allclose(
        explicit.current.parameter_gradient,
        np.asarray(elmer_current_gradient),
        rtol=8.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        explicit.thermal.parameter_gradient,
        np.asarray(elmer_thermal_gradient),
        rtol=8.0e-6,
        atol=2.0e-8,
    )
