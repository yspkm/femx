from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tests.support import DummyPhysics, structured_unit_square_mesh  # noqa: E402

import femx.backends.jax.port_eigenmode as port_backend_module  # noqa: E402
from femx.backends.jax.port_eigen_adjoint import SimplePortEigenpairPolicy  # noqa: E402
from femx.backends.jax.port_eigenmode import (  # noqa: E402
    JaxPortEigenmodeBackend,
    PreparedJaxPortEigenmode,
)
from femx.backends.protocol import (  # noqa: E402
    Backend,
    BackendDescriptor,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import (  # noqa: E402
    AnalysisKind,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import BackendError, CapabilityError, ContractError  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem  # noqa: E402
from femx.core.solution import ConvergenceStatus  # noqa: E402
from femx.mesh import MeshGeometry, OrientationMap  # noqa: E402
from femx.physics import (  # noqa: E402
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _problem(
    *,
    intervals: int = 2,
    mode_count: int = 4,
    target_power_w: float = 2.0,
    relative_permittivity: float | ParameterReference = 1.0,
    gradient_method: GradientMethod = GradientMethod.NONE,
    parameters: ParameterSchema | None = None,
) -> Problem:
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    coordinates[:, 0] *= 2.0e-6
    coordinates[:, 1] *= 1.0e-6
    cells = np.asarray(mesh.topology.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    mesh = replace(
        mesh,
        geometry=MeshGeometry(coordinates),
        orientation=OrientationMap(edge_signs=signs),
    )
    return Problem(
        "jax-vacuum-port",
        mesh,
        PortEigenmode(
            regions=(IsotropicOpticalRegion("domain", relative_permittivity),),
            perfect_electric_boundaries=tuple(
                PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
            ),
            frequency_hz=1.0e14,
            eigenmode_count=mode_count,
            selected_mode_index=0,
            target_power_w=target_power_w,
            gradient_method=gradient_method,
        ),
        parameters=ParameterSchema() if parameters is None else parameters,
    )


def _parameterized_problem(
    *,
    gradient_method: GradientMethod,
    role: ParameterRole = ParameterRole.DESIGN,
) -> Problem:
    return _problem(
        relative_permittivity=ParameterReference("epsilon_r"),
        gradient_method=gradient_method,
        parameters=ParameterSchema(
            (
                ParameterSpec(
                    "epsilon_r",
                    unit="1",
                    role=role,
                    lower_bound=1.0,
                    upper_bound=16.0,
                ),
            )
        ),
    )


def test_jax_port_backend_advertises_only_executed_serial_capabilities() -> None:
    backend = JaxPortEigenmodeBackend()

    assert isinstance(backend, Backend)
    assert backend.capabilities.analyses == frozenset({AnalysisKind.EIGENMODE})
    assert backend.capabilities.function_spaces == frozenset(
        {FunctionSpaceFamily.HCURL, FunctionSpaceFamily.H1}
    )
    assert backend.capabilities.scalar_kinds == frozenset({ScalarKind.COMPLEX})
    assert backend.capabilities.gradients == frozenset(
        {GradientMethod.NONE, GradientMethod.ADJOINT}
    )
    assert backend.capabilities.parallel_models == frozenset({ParallelModel.SERIAL})
    assert backend.descriptor.name == "jax-port-eigenmode"
    assert jax.__version__ in backend.descriptor.version


@pytest.mark.parametrize("tolerance", (0.0, -1.0, float("inf")))
def test_jax_port_backend_rejects_invalid_residual_tolerance(tolerance: float) -> None:
    with pytest.raises(ContractError, match="finite and positive"):
        JaxPortEigenmodeBackend(relative_residual_tolerance=tolerance)


def test_jax_port_backend_prepares_static_topology_and_solves_physical_fields() -> None:
    backend = JaxPortEigenmodeBackend()
    problem = _problem()

    prepared = prepare(problem, backend)
    assert isinstance(prepared.payload, PreparedJaxPortEigenmode)
    payload = prepared.payload
    assert payload.coordinates.dtype == jnp.float64
    assert payload.cells.dtype == jnp.int32
    assert payload.cell_edge_signs.dtype == jnp.int8
    assert not payload.free_dofs.flags.writeable
    assert payload.scalar_dof_count == 1
    assert payload.free_dofs.size - payload.scalar_dof_count == 8

    solution = solve(prepared, backend)

    assert solution.convergence.status is ConvergenceStatus.CONVERGED
    assert solution.convergence.residual_norm is not None
    assert solution.convergence.residual_norm <= 1.0e-10
    electric = np.asarray(solution.fields["electric_field"].values)
    magnetic = np.asarray(solution.fields["magnetic_field"].values)
    assert electric.shape == magnetic.shape == (9, 3)
    assert electric.dtype == magnetic.dtype == np.complex128
    assert np.isfinite(electric).all()
    assert np.isfinite(magnetic).all()
    assert solution.fields["electric_field"].unit == "V/m"
    assert solution.fields["magnetic_field"].unit == "A/m"
    scalar_coefficients = solution.fields[PORT_LONGITUDINAL_POTENTIAL_FIELD]
    edge_coefficients = solution.fields[PORT_TRANSVERSE_ELECTRIC_FIELD]
    assert scalar_coefficients.values.shape == (9,)
    assert edge_coefficients.values.shape == (16,)
    assert scalar_coefficients.values.dtype == edge_coefficients.values.dtype == jnp.complex128
    assert scalar_coefficients.unit == PORT_LONGITUDINAL_POTENTIAL_UNIT
    assert edge_coefficients.unit == PORT_TRANSVERSE_ELECTRIC_DOF_UNIT
    assert scalar_coefficients.function_space.family is FunctionSpaceFamily.H1
    assert edge_coefficients.function_space.family is FunctionSpaceFamily.HCURL
    assert solution.observables["raw_forward_power_W"] > 0.0
    assert solution.observables["normalized_forward_power_W"] == pytest.approx(
        2.0,
        rel=2.0e-14,
    )
    assert solution.observables["target_forward_power_W"] == 2.0
    assert solution.observables["propagation_constant_rad_per_m"].real > 0.0
    assert solution.metadata["magnetic_field_convention"] == "physical_H"
    assert solution.metadata["fdtdx_mode_bundle_status"].startswith("requires_explicit_hashed")


def test_jax_port_backend_rejects_more_modes_than_the_finite_edge_spectrum() -> None:
    backend = JaxPortEigenmodeBackend()
    with pytest.raises(ContractError, match="exceeds the finite free-edge spectrum"):
        backend.prepare(_problem(intervals=1, mode_count=2), PrepareRequest())


def test_jax_port_backend_requires_explicit_cpu_and_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JaxPortEigenmodeBackend()
    monkeypatch.setattr(
        port_backend_module,
        "jax",
        SimpleNamespace(default_backend=lambda: "gpu", config=SimpleNamespace(x64_enabled=True)),
    )
    with pytest.raises(BackendError, match="validated only on CPU"):
        backend.prepare(_problem(), PrepareRequest())

    monkeypatch.setattr(
        port_backend_module,
        "jax",
        SimpleNamespace(default_backend=lambda: "cpu", config=SimpleNamespace(x64_enabled=False)),
    )
    with pytest.raises(BackendError, match="float64 is required"):
        backend.prepare(_problem(), PrepareRequest())


def test_jax_port_backend_rechecks_runtime_and_prepared_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(_problem(), PrepareRequest())

    wrong_identity = replace(prepared, backend=BackendDescriptor("other", "1"))
    with pytest.raises(BackendError, match="identity"):
        backend.solve(wrong_identity, SolveRequest())

    wrong_payload = replace(prepared, payload=object())
    with pytest.raises(BackendError, match="payload"):
        backend.solve(wrong_payload, SolveRequest())

    wrong_problem = replace(
        prepared,
        problem=Problem("wrong", prepared.problem.mesh, DummyPhysics()),
    )
    with pytest.raises(BackendError, match="not a port-eigenmode"):
        backend.solve(wrong_problem, SolveRequest())

    with pytest.raises(ContractError, match="parameter key mismatch"):
        backend.solve(
            prepared,
            SolveRequest(parameters=ParameterValues({"epsilon": 1.0})),
        )

    monkeypatch.setattr(
        port_backend_module,
        "jax",
        SimpleNamespace(default_backend=lambda: "tpu", config=SimpleNamespace(x64_enabled=True)),
    )
    with pytest.raises(BackendError, match="validated only on CPU"):
        backend.solve(prepared, SolveRequest())


def test_jax_port_backend_fails_closed_on_nonfinite_or_backward_spectra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(_problem(), PrepareRequest())
    original = port_backend_module.solve_lossless_port_eigenmode_fields

    def nonfinite(*args, **kwargs):
        modes, projected = original(*args, **kwargs)
        return (
            modes._replace(eigenvalues_per_m2=modes.eigenvalues_per_m2.at[0].set(jnp.nan + 0.0j)),
            projected,
        )

    monkeypatch.setattr(port_backend_module, "solve_lossless_port_eigenmode_fields", nonfinite)
    with pytest.raises(BackendError, match="non-finite spectrum"):
        backend.solve(prepared, SolveRequest())

    def backward(*args, **kwargs):
        modes, projected = original(*args, **kwargs)
        return (
            modes._replace(
                propagation_constants_per_m=modes.propagation_constants_per_m.at[0].set(
                    -jnp.abs(modes.propagation_constants_per_m[0]) + 0.0j
                )
            ),
            projected,
        )

    monkeypatch.setattr(port_backend_module, "solve_lossless_port_eigenmode_fields", backward)
    with pytest.raises(BackendError, match="non-forward"):
        backend.solve(prepared, SolveRequest())


def test_jax_port_backend_reports_nonconverged_without_weakening_tolerance() -> None:
    backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-30)
    prepared = backend.prepare(_problem(), PrepareRequest())

    solution = backend.solve(prepared, SolveRequest())

    assert solution.convergence.status is ConvergenceStatus.NOT_CONVERGED
    assert solution.convergence.residual_norm is not None
    assert solution.convergence.residual_norm > 1.0e-30


def test_jax_port_backend_fails_if_power_postcondition_is_not_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(_problem(), PrepareRequest())
    original = port_backend_module.normalize_projected_electromagnetic_mode

    def wrong_power(*args, **kwargs):
        normalized = original(*args, **kwargs)
        return replace(normalized, normalized_forward_power_w=3.0)

    monkeypatch.setattr(
        port_backend_module,
        "normalize_projected_electromagnetic_mode",
        wrong_power,
    )
    with pytest.raises(BackendError, match="requested forward power"):
        backend.solve(prepared, SolveRequest())


def test_jax_port_backend_rejects_a_non_port_prepared_envelope() -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = PreparedProblem(
        backend=backend.descriptor,
        problem=_problem(),
        payload=object(),
    )
    with pytest.raises(BackendError, match="payload"):
        backend.solve(prepared, SolveRequest())


def test_jax_port_backend_binds_materials_at_solve_time() -> None:
    backend = JaxPortEigenmodeBackend()
    parameterized = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.NONE),
        PrepareRequest(),
    )
    literal = backend.prepare(
        _problem(relative_permittivity=2.25),
        PrepareRequest(),
    )

    parameterized_solution = backend.solve(
        parameterized,
        SolveRequest(parameters=ParameterValues({"epsilon_r": 2.25})),
    )
    literal_solution = backend.solve(literal, SolveRequest())

    assert parameterized_solution.observables["propagation_constant_rad_per_m"] == pytest.approx(
        literal_solution.observables["propagation_constant_rad_per_m"],
        rel=2.0e-14,
    )


def test_jax_port_simple_mode_binding_is_jittable_and_matches_central_difference() -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )
    bound = backend.bind_differentiable(
        prepared,
        ParameterValues({"epsilon_r": 2.25}),
    )

    assert bound.problem is prepared.problem
    assert bound.parameter_names == ("epsilon_r",)
    assert bound.parameter_units == ("1",)
    assert np.asarray(bound.initial_values).tolist() == [2.25]
    assert bool(bound.baseline_diagnostics.is_valid)
    assert bound.phase_anchor_edge_dof >= 0
    assert float(bound.angular_frequency_rad_per_s) > 0.0
    assert bound.target_power_w == 2.0

    def beta(active: jax.Array) -> jax.Array:
        return bound.mode(active).propagation_constant_per_m

    initial = bound.initial_values
    mode = jax.jit(bound.mode)(initial)
    value, gradient = jax.jit(jax.value_and_grad(beta))(initial)
    step = 2.0e-4
    central_difference = (
        float(beta(initial.at[0].add(step))) - float(beta(initial.at[0].add(-step)))
    ) / (2.0 * step)

    assert bool(mode.is_valid)
    assert mode.scalar_coefficients.shape == (9,)
    assert mode.edge_coefficients.shape == (16,)
    assert mode.scalar_coefficients.dtype == mode.edge_coefficients.dtype == jnp.complex128
    assert mode.cell_relative_permittivity.shape == (8,)
    assert mode.cell_reluctivity_per_henry_m.shape == (8,)
    assert float(value) > 0.0
    assert float(gradient[0]) == pytest.approx(central_difference, rel=2.0e-8)


def test_jax_port_cluster_binding_is_invariant_jittable_and_differentiable() -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )
    bound = backend.bind_differentiable_cluster(
        prepared,
        ParameterValues({"epsilon_r": 2.25}),
        mode_indices=(0, 1),
        quadrature_point_count=96,
    )

    assert bound.problem is prepared.problem
    assert bound.parameter_names == ("epsilon_r",)
    assert bound.parameter_units == ("1",)
    assert np.asarray(bound.initial_values).tolist() == [2.25]
    assert bound.mode_indices == (0, 1)
    assert bound.contour.expected_cluster_size == 2
    assert bound.contour.quadrature_point_count == 96
    assert bool(bound.baseline_diagnostics.is_valid)

    weights = jnp.reshape(jnp.linspace(-0.2, 0.3, 64), (8, 8))

    def objective(active: jax.Array) -> jax.Array:
        cluster = bound.cluster(active)
        return cluster.mean_propagation_constant_per_m + 1.0e3 * jnp.sum(
            weights * cluster.reduced_edge_mass_projector
        )

    initial = bound.initial_values
    cluster = jax.jit(bound.cluster)(initial)
    value, gradient = jax.jit(jax.value_and_grad(objective))(initial)
    step = 2.0e-4
    central_difference = (
        float(objective(initial.at[0].add(step))) - float(objective(initial.at[0].add(-step)))
    ) / (2.0 * step)

    assert bool(cluster.is_valid)
    assert cluster.reduced_edge_mass_projector.shape == (8, 8)
    assert cluster.cell_relative_permittivity.shape == (8,)
    assert cluster.cell_relative_permeability.shape == (8,)
    assert float(cluster.dimensionless_eigenvalue_sum) < 0.0
    assert float(cluster.eigenvalue_sum_per_m2) < 0.0
    assert float(cluster.propagation_constant_sum_per_m) > 0.0
    assert float(cluster.mean_propagation_constant_per_m) > 0.0
    assert float(cluster.mean_effective_index) > 0.0
    assert math.isfinite(float(value))
    assert float(gradient[0]) == pytest.approx(central_difference, rel=2.0e-7)

    invalid = bound.cluster(jnp.asarray((0.5,), dtype=jnp.float64))
    assert not bool(invalid.is_valid)
    assert np.isnan(np.asarray(invalid.reduced_edge_mass_projector)).all()


@pytest.mark.parametrize(
    ("mode_indices", "message"),
    [
        ((0,), "at least two"),
        ((False, 1), "static integers"),
        ((1, 0), "strictly increasing"),
        ((0, 2), "consecutive"),
        ((-1, 0), "requested baseline spectrum"),
        ((3, 4), "requested baseline spectrum"),
    ],
)
def test_jax_port_cluster_binding_rejects_invalid_mode_indices(
    mode_indices: tuple[int, ...],
    message: str,
) -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )
    with pytest.raises(ContractError, match=message):
        backend.bind_differentiable_cluster(
            prepared,
            ParameterValues({"epsilon_r": 2.25}),
            mode_indices=mode_indices,
        )


def test_jax_port_cluster_binding_rechecks_envelope_and_admission() -> None:
    backend = JaxPortEigenmodeBackend()
    parameters = ParameterValues({"epsilon_r": 2.25})
    nonadjoint = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.NONE),
        PrepareRequest(),
    )
    with pytest.raises(CapabilityError, match="gradient_method=adjoint"):
        backend.bind_differentiable_cluster(nonadjoint, parameters, mode_indices=(0, 1))

    fixed = backend.prepare(
        _parameterized_problem(
            gradient_method=GradientMethod.ADJOINT,
            role=ParameterRole.FIXED,
        ),
        PrepareRequest(),
    )
    with pytest.raises(ContractError, match="DESIGN or CONTROL"):
        backend.bind_differentiable_cluster(fixed, parameters, mode_indices=(0, 1))

    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )
    with pytest.raises(BackendError, match="identity"):
        backend.bind_differentiable_cluster(
            replace(prepared, backend=BackendDescriptor("other", "1")),
            parameters,
            mode_indices=(0, 1),
        )
    with pytest.raises(BackendError, match="payload"):
        backend.bind_differentiable_cluster(
            replace(prepared, payload=object()),
            parameters,
            mode_indices=(0, 1),
        )
    with pytest.raises(BackendError, match="not a port-eigenmode"):
        backend.bind_differentiable_cluster(
            replace(
                prepared,
                problem=Problem("wrong", prepared.problem.mesh, DummyPhysics()),
            ),
            parameters,
            mode_indices=(0, 1),
        )
    with pytest.raises(CapabilityError, match="contour admission policy"):
        backend.bind_differentiable_cluster(
            prepared,
            parameters,
            mode_indices=(0, 1),
            quadrature_point_count=24,
        )
    with pytest.raises(CapabilityError, match="isolated finite cluster"):
        backend.bind_differentiable_cluster(
            prepared,
            parameters,
            mode_indices=(1, 2),
            quadrature_point_count=96,
        )

    all_finite_modes = backend.prepare(
        _problem(
            mode_count=8,
            relative_permittivity=ParameterReference("epsilon_r"),
            gradient_method=GradientMethod.ADJOINT,
            parameters=prepared.problem.parameters,
        ),
        PrepareRequest(),
    )
    with pytest.raises(CapabilityError, match="unselected finite mode"):
        backend.bind_differentiable_cluster(
            all_finite_modes,
            parameters,
            mode_indices=tuple(range(8)),
        )


def test_jax_port_differentiable_binding_fails_closed_outside_its_contract() -> None:
    backend = JaxPortEigenmodeBackend()
    nonadjoint = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.NONE),
        PrepareRequest(),
    )
    with pytest.raises(CapabilityError, match="gradient_method=adjoint"):
        backend.bind_differentiable(nonadjoint, ParameterValues({"epsilon_r": 2.25}))

    fixed = backend.prepare(
        _parameterized_problem(
            gradient_method=GradientMethod.ADJOINT,
            role=ParameterRole.FIXED,
        ),
        PrepareRequest(),
    )
    with pytest.raises(ContractError, match="DESIGN or CONTROL"):
        backend.bind_differentiable(fixed, ParameterValues({"epsilon_r": 2.25}))

    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )
    bound = backend.bind_differentiable(prepared, ParameterValues({"epsilon_r": 2.25}))
    with pytest.raises(ContractError, match="shape"):
        bound.mode(jnp.asarray((2.25, 2.5), dtype=jnp.float64))
    with pytest.raises(ContractError, match="float64"):
        bound.mode(jnp.asarray((2.25,), dtype=jnp.float32))
    invalid = bound.mode(jnp.asarray((0.5,), dtype=jnp.float64))
    assert not bool(invalid.is_valid)
    assert np.isnan(np.asarray(invalid.edge_coefficients)).all()


def test_jax_port_binding_rejects_an_insufficiently_separated_baseline_mode() -> None:
    backend = JaxPortEigenmodeBackend(
        eigen_adjoint_policy=SimplePortEigenpairPolicy(
            minimum_relative_eigenvalue_gap=10.0,
        )
    )
    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )

    with pytest.raises(CapabilityError, match="invariant subspace objective"):
        backend.bind_differentiable(
            prepared,
            ParameterValues({"epsilon_r": 2.25}),
        )


def test_jax_port_differentiable_binding_rechecks_prepared_envelope() -> None:
    backend = JaxPortEigenmodeBackend()
    prepared = backend.prepare(
        _parameterized_problem(gradient_method=GradientMethod.ADJOINT),
        PrepareRequest(),
    )
    parameters = ParameterValues({"epsilon_r": 2.25})

    with pytest.raises(BackendError, match="identity"):
        backend.bind_differentiable(
            replace(prepared, backend=BackendDescriptor("other", "1")),
            parameters,
        )
    with pytest.raises(BackendError, match="payload"):
        backend.bind_differentiable(replace(prepared, payload=object()), parameters)
    with pytest.raises(BackendError, match="not a port-eigenmode"):
        backend.bind_differentiable(
            replace(
                prepared,
                problem=Problem("wrong", prepared.problem.mesh, DummyPhysics()),
            ),
            parameters,
        )
