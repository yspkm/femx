from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.support import DummyPhysics, structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import femx.backends.jax.steady_current as backend_module  # noqa: E402
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    BackendDescriptor,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.errors import BackendError, ContractError  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem  # noqa: E402
from femx.mesh import EntityTag, Mesh  # noqa: E402
from femx.physics import (  # noqa: E402
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _basic_physics() -> SteadyCurrent:
    return SteadyCurrent(
        regions=(ConductiveRegion("domain", 2.0),),
        potential_boundaries=(PotentialBoundary("left", 0.0),),
        current_flux_boundaries=(CurrentFluxBoundary("right", 2.0),),
    )


def _problem(
    *,
    mesh: Mesh | None = None,
    physics: SteadyCurrent | None = None,
    parameters: ParameterSchema | None = None,
) -> Problem:
    return Problem(
        "jax-current",
        structured_unit_square_mesh(2) if mesh is None else mesh,
        _basic_physics() if physics is None else physics,
        parameters=ParameterSchema() if parameters is None else parameters,
    )


def test_parameterized_backend_reports_potential_current_joule_and_energy_balance() -> None:
    mesh = structured_unit_square_mesh(3)
    physics = SteadyCurrent(
        regions=(
            ConductiveRegion(
                "domain",
                ParameterReference("sigma"),
                ParameterReference("source"),
            ),
        ),
        potential_boundaries=(PotentialBoundary("left", ParameterReference("ground")),),
        current_flux_boundaries=(CurrentFluxBoundary("right", ParameterReference("current_load")),),
    )
    schema = ParameterSchema(
        (
            ParameterSpec("sigma", unit="S/m", lower_bound=0.1),
            ParameterSpec("source", unit="A/m^3"),
            ParameterSpec("ground", unit="V"),
            ParameterSpec("current_load", unit="A/m^2"),
        )
    )
    values = schema.bind({"sigma": 2.0, "source": 0.0, "ground": 0.0, "current_load": 2.0})
    backend = JaxSteadyCurrentBackend()

    solution = solve(
        prepare(_problem(mesh=mesh, physics=physics, parameters=schema), backend),
        backend,
        request=SolveRequest(parameters=values),
    )

    coordinates = np.asarray(mesh.geometry.coordinates)
    np.testing.assert_allclose(solution.fields["potential"].values, coordinates[:, 0], atol=2e-14)
    cell_count = mesh.topology.cell_count
    np.testing.assert_allclose(
        solution.fields["electric_field"].values,
        np.tile((-1.0, 0.0), (cell_count, 1)),
        atol=2e-14,
    )
    np.testing.assert_allclose(
        solution.fields["current_density"].values,
        np.tile((-2.0, 0.0), (cell_count, 1)),
        atol=4e-14,
    )
    np.testing.assert_allclose(solution.fields["joule_heat_density"].values, 2.0, atol=5e-14)
    assert solution.fields["potential"].unit == "V"
    assert solution.fields["electric_field"].unit == "V/m"
    assert solution.fields["current_density"].unit == "A/m^2"
    assert solution.fields["joule_heat_density"].unit == "W/m^3"
    assert solution.observables["joule_power_W_per_m"] == pytest.approx(2.0, abs=5e-14)
    assert solution.observables["variational_input_power_W_per_m"] == pytest.approx(2.0, abs=5e-14)
    assert solution.observables["energy_balance_relative_error"] < 2.0e-15
    assert solution.convergence.status.value == "converged"
    assert solution.metadata["physical_current_density"] == "J=-sigma*grad(phi)"
    assert solution.metadata["integrated_power_unit"] == "W/m"


def test_backend_reports_nonconvergence_separately_from_finite_energy_output() -> None:
    backend = JaxSteadyCurrentBackend(relative_residual_tolerance=1.0e-30)
    physics = SteadyCurrent(
        (ConductiveRegion("domain", 1.0, 2.0),),
        (PotentialBoundary("left", 0.0),),
    )

    solution = solve(prepare(_problem(physics=physics), backend), backend)

    assert solution.convergence.status.value == "not_converged"
    assert solution.convergence.residual_norm is not None
    assert solution.convergence.residual_norm > 1.0e-30
    assert np.isfinite(solution.observables["joule_power_W_per_m"])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1.0, "positive value"),
        (complex(1.0, 1.0), "finite real scalars"),
        (float("nan"), "finite real scalars"),
        (True, "finite real scalars"),
        (np.asarray((1.0,)), "finite real scalars"),
    ],
)
def test_backend_rejects_nonphysical_resolved_conductivity(value: object, message: str) -> None:
    physics = SteadyCurrent(
        (ConductiveRegion("domain", ParameterReference("sigma")),),
        (PotentialBoundary("left", 0.0),),
    )
    schema = ParameterSchema((ParameterSpec("sigma", unit="S/m"),))
    backend = JaxSteadyCurrentBackend()
    prepared = backend.prepare(_problem(physics=physics, parameters=schema), PrepareRequest())

    with pytest.raises(ContractError, match=message):
        backend.solve(prepared, SolveRequest(parameters=ParameterValues({"sigma": value})))


def test_backend_requires_tested_capability_cpu_and_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ContractError, match="tolerance"):
        JaxSteadyCurrentBackend(relative_residual_tolerance=0.0)

    adjoint = replace(_basic_physics(), gradient_method=GradientMethod.ADJOINT)
    prepared_adjoint = prepare(_problem(physics=adjoint), JaxSteadyCurrentBackend())
    assert prepared_adjoint.problem.physics.gradient_method is GradientMethod.ADJOINT

    backend = JaxSteadyCurrentBackend()
    monkeypatch.setattr(backend_module.jax, "default_backend", lambda: "gpu")
    with pytest.raises(BackendError, match="validated only on CPU"):
        backend.prepare(_problem(), PrepareRequest())
    monkeypatch.setattr(
        backend_module,
        "jax",
        SimpleNamespace(default_backend=lambda: "cpu", config=SimpleNamespace(x64_enabled=False)),
    )
    with pytest.raises(BackendError, match="float64"):
        backend.prepare(_problem(), PrepareRequest())


def test_backend_rejects_wrong_physics_payload_identity_and_parameter_keys() -> None:
    backend = JaxSteadyCurrentBackend()
    with pytest.raises(ContractError, match="SteadyCurrent"):
        backend.prepare(
            Problem("wrong", structured_unit_square_mesh(1), DummyPhysics()),
            PrepareRequest(),
        )

    valid = _problem()
    with pytest.raises(BackendError, match="identity"):
        backend.solve(
            PreparedProblem(BackendDescriptor("other", "1"), valid, object()),
            SolveRequest(),
        )
    with pytest.raises(BackendError, match="payload"):
        backend.solve(PreparedProblem(backend.descriptor, valid, object()), SolveRequest())

    physics = SteadyCurrent(
        (ConductiveRegion("domain", ParameterReference("sigma")),),
        (PotentialBoundary("left", 0.0),),
    )
    schema = ParameterSchema((ParameterSpec("sigma", unit="S/m"),))
    prepared = backend.prepare(_problem(physics=physics, parameters=schema), PrepareRequest())
    with pytest.raises(ContractError, match="parameter key mismatch"):
        backend.solve(prepared, SolveRequest())

    class StructuralMesh:
        geometry = object()
        topology = object()
        schema_version = "femx.mesh/v1"

    with pytest.raises(ContractError, match="concrete femx Mesh"):
        backend.prepare(
            Problem("structural", StructuralMesh(), _basic_physics()),
            PrepareRequest(),
        )


@pytest.mark.parametrize(
    ("physics", "parameters", "message"),
    [
        (
            SteadyCurrent(
                (ConductiveRegion("domain", ParameterReference("sigma")),),
                (PotentialBoundary("left", 0.0),),
            ),
            ParameterSchema(),
            "do not match",
        ),
        (
            SteadyCurrent(
                (ConductiveRegion("domain", ParameterReference("sigma")),),
                (PotentialBoundary("left", 0.0),),
            ),
            ParameterSchema((ParameterSpec("sigma", unit="V"),)),
            "must be a scalar with unit",
        ),
        (
            SteadyCurrent(
                (
                    ConductiveRegion(
                        "domain",
                        ParameterReference("shared"),
                        ParameterReference("shared"),
                    ),
                ),
                (PotentialBoundary("left", 0.0),),
            ),
            ParameterSchema((ParameterSpec("shared", unit="S/m"),)),
            "incompatible units",
        ),
    ],
)
def test_backend_rejects_parameter_schema_mismatches(
    physics: SteadyCurrent,
    parameters: ParameterSchema,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        JaxSteadyCurrentBackend().prepare(
            _problem(physics=physics, parameters=parameters),
            PrepareRequest(),
        )


def _with_tags(mesh: Mesh, *tags: EntityTag) -> Mesh:
    return replace(mesh, tags=(*mesh.tags, *tags))


@pytest.mark.parametrize(
    ("mesh_factory", "physics", "message"),
    [
        (
            lambda mesh: _with_tags(mesh, EntityTag("half", 2, (0,))),
            SteadyCurrent(
                (ConductiveRegion("half", 1.0),),
                (PotentialBoundary("left", 0.0),),
            ),
            "partition every cell",
        ),
        (
            lambda mesh: _with_tags(mesh, EntityTag("left_copy", 1, (3,))),
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0),),
                (PotentialBoundary("left", 0.0), PotentialBoundary("left_copy", 0.0)),
            ),
            "cannot overlap",
        ),
        (
            lambda mesh: _with_tags(mesh, EntityTag("left_copy", 1, (3,))),
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0),),
                (PotentialBoundary("left", 0.0),),
                (CurrentFluxBoundary("left_copy", 1.0),),
            ),
            "both potential and current flux",
        ),
        (
            lambda mesh: _with_tags(mesh, EntityTag("right_copy", 1, (1,))),
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0),),
                (PotentialBoundary("left", 0.0),),
                (
                    CurrentFluxBoundary("right", 1.0),
                    CurrentFluxBoundary("right_copy", 1.0),
                ),
            ),
            "cannot overlap",
        ),
    ],
)
def test_backend_rejects_invalid_region_and_boundary_ownership(
    mesh_factory,
    physics: SteadyCurrent,
    message: str,
) -> None:
    mesh = mesh_factory(structured_unit_square_mesh(1))
    with pytest.raises(ContractError, match=message):
        JaxSteadyCurrentBackend().prepare(_problem(mesh=mesh, physics=physics), PrepareRequest())


def test_backend_rejects_conflicting_potential_values_at_shared_corner() -> None:
    physics = SteadyCurrent(
        (ConductiveRegion("domain", 1.0),),
        (PotentialBoundary("left", 0.0), PotentialBoundary("bottom", 1.0)),
    )
    with pytest.raises(ContractError, match="conflict"):
        JaxSteadyCurrentBackend().prepare(_problem(physics=physics), PrepareRequest())


def test_zero_state_has_exact_zero_energy_balance_error() -> None:
    physics = SteadyCurrent(
        (ConductiveRegion("domain", 1.0),),
        (PotentialBoundary("left", 0.0),),
    )
    backend = JaxSteadyCurrentBackend()
    solution = solve(prepare(_problem(physics=physics), backend), backend)
    assert solution.observables["joule_power_W_per_m"] == 0.0
    assert solution.observables["energy_balance_relative_error"] == 0.0
