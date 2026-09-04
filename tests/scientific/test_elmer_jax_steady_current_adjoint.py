from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tests.current_adjoint_support import (  # noqa: E402
    parameterized_current_adjoint_problem,
    triangle_areas,
)

from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    ExecutionPolicy,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.problem import Problem  # noqa: E402
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_jax,
]


def test_jax_joule_adjoint_matches_locked_elmer_central_differences(
    locked_elmer_current_backend,
    tmp_path,
) -> None:
    adjoint_problem, initial_parameters = parameterized_current_adjoint_problem(intervals=3)
    elmer_problem = Problem(
        "cross-adjoint-current-elmer",
        adjoint_problem.mesh,
        replace(adjoint_problem.physics, gradient_method=GradientMethod.NONE),
        parameters=adjoint_problem.parameters,
    )
    jax_backend = JaxSteadyCurrentBackend()
    bound = jax_backend.bind_differentiable(
        prepare(adjoint_problem, jax_backend),
        initial_parameters,
    )
    initial = bound.initial_values
    areas = jnp.asarray(triangle_areas(adjoint_problem), dtype=jnp.float64)
    jax_joule = bound.joule_heat_density(initial)
    adjoint_gradient = np.asarray(bound.joule_vjp(initial, areas).parameter_gradient)

    initial_mapping = dict(initial_parameters.values)

    def elmer_objective(active: np.ndarray, attempt_name: str) -> float:
        values = dict(initial_mapping)
        values.update(
            (name, float(value)) for name, value in zip(bound.parameter_names, active, strict=True)
        )
        parameters = adjoint_problem.parameters.bind(values)
        run_directory = tmp_path / attempt_name
        solution = solve(
            prepare(
                elmer_problem,
                locked_elmer_current_backend,
                request=PrepareRequest(run_directory=run_directory),
            ),
            locked_elmer_current_backend,
            request=SolveRequest(
                parameters=parameters,
                run_directory=run_directory,
                policy=ExecutionPolicy(
                    execution_authorized=True,
                    allow_external_process=True,
                ),
            ),
        )
        assert solution.convergence.status.value == "converged"
        assert solution.observables["energy_balance_relative_error"] < 2.0e-13
        return float(solution.observables["joule_power_W_per_m"])

    initial_numpy = np.asarray(initial)
    elmer_gradient = []
    for index, value in enumerate(initial_numpy):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = initial_numpy.copy()
        minus = initial_numpy.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = elmer_objective(plus, f"parameter-{index}-plus")
        minus_value = elmer_objective(minus, f"parameter-{index}-minus")
        elmer_gradient.append((plus_value - minus_value) / (2.0 * step))

    baseline_jax_objective = float(jnp.vdot(areas, jax_joule))
    baseline_elmer_objective = elmer_objective(initial_numpy, "baseline")
    np.testing.assert_allclose(
        baseline_elmer_objective,
        baseline_jax_objective,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        adjoint_gradient,
        np.asarray(elmer_gradient),
        rtol=5.0e-7,
        atol=5.0e-9,
    )
