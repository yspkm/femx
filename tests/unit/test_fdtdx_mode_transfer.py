from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from femx.core.capabilities import FunctionSpaceFamily  # noqa: E402
from femx.core.errors import ContractError  # noqa: E402
from femx.core.solution import (  # noqa: E402
    ConvergenceReport,
    ConvergenceStatus,
    Field,
    Solution,
)
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXFingerprint,
    SamplingAmbiguityPolicy,
    SolverFingerprint,
    build_yee_grid,
    build_yee_port_sampling_plan,
    make_fdtdx_mode_function,
    port_mode_solution_to_bundle,
    sample_port_mode_to_yee,
)
from femx.mesh import FunctionSpace  # noqa: E402
from femx.physics import (  # noqa: E402
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

SHA = "a" * 64
FDTDX = FDTDXFingerprint("0.6.2", "b" * 40, "c" * 64)


def _triangle_plan():
    coordinates = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    signs = np.asarray(((1, 1, -1),), dtype=np.int8)
    grid = build_yee_grid(
        (
            np.asarray((0.2, 0.4, 0.6)),
            np.asarray((0.2, 0.4, 0.6)),
            np.asarray((0.0, 1.0)),
        )
    )
    return build_yee_port_sampling_plan(coordinates, cells, signs, grid)


def _constant_mode_coefficients() -> tuple[jax.Array, jax.Array]:
    scalar = jnp.full((3,), 1.0 + 2.0j, dtype=jnp.complex128)
    # Canonical edges are (0,1), (0,2), (1,2). These are edge moments of E=(2,3).
    edge = jnp.asarray((4.0, 6.0, 2.0), dtype=jnp.complex128)
    return scalar, edge


def _solution(plan, scalar, edge, *, target_power: float) -> Solution:
    beta = 4.0
    return Solution(
        backend_name="test-port",
        backend_version="1",
        fields={
            PORT_LONGITUDINAL_POTENTIAL_FIELD: Field(
                PORT_LONGITUDINAL_POTENTIAL_FIELD,
                scalar,
                PORT_LONGITUDINAL_POTENTIAL_UNIT,
                FunctionSpace(FunctionSpaceFamily.H1, order=1),
            ),
            PORT_TRANSVERSE_ELECTRIC_FIELD: Field(
                PORT_TRANSVERSE_ELECTRIC_FIELD,
                edge,
                PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
                FunctionSpace(FunctionSpaceFamily.HCURL, order=1, value_shape=(2,)),
            ),
        },
        observables={
            "propagation_constant_rad_per_m": beta + 0.0j,
            "effective_index": 2.0 + 0.0j,
            "target_forward_power_W": target_power,
        },
        convergence=ConvergenceReport(ConvergenceStatus.CONVERGED),
        metadata={"plan": plan.operator_sha256},
    )


def test_exact_yee_offsets_sample_constant_mixed_mode_and_conserve_signed_power() -> None:
    plan = _triangle_plan()
    scalar, edge = _constant_mode_coefficients()
    beta = 4.0
    omega = 8.0
    reluctivity = jnp.asarray((2.0,))
    target_power = 2.0

    samples = sample_port_mode_to_yee(
        plan,
        scalar,
        edge,
        beta,
        reluctivity,
        omega,
        target_power,
    )

    expected_flux = 0.5 * (2.0 * 2.0 + 3.0 * 3.0) * 2.0 * beta / omega
    expected_pre_power = expected_flux * 0.4 * 0.4
    expected_scale = math.sqrt(target_power / expected_pre_power)
    electric = np.asarray(samples.electric_v_per_m)
    magnetic = np.asarray(samples.magnetic_a_per_m)
    assert electric.shape == magnetic.shape == (3, 2, 2, 1)
    np.testing.assert_allclose(electric[0], 2.0 * expected_scale, rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(electric[1], 3.0 * expected_scale, rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(electric[2], (0.5 - 0.25j) * expected_scale, rtol=2e-14)
    np.testing.assert_allclose(magnetic[0], -3.0 * expected_scale, rtol=2e-14)
    np.testing.assert_allclose(magnetic[1], 2.0 * expected_scale, rtol=2e-14)
    np.testing.assert_allclose(magnetic[2], 0.0, atol=1e-14)
    assert float(samples.pre_correction_power_watts) == pytest.approx(expected_pre_power)
    assert float(samples.power_correction_scale) == pytest.approx(expected_scale)
    assert float(samples.transferred_power_watts) == pytest.approx(target_power, rel=2e-14)


def test_yee_transfer_is_jittable_and_reverse_differentiable() -> None:
    plan = _triangle_plan()
    scalar, base_edge = _constant_mode_coefficients()

    @jax.jit
    def objective(scale):
        samples = sample_port_mode_to_yee(
            plan,
            scalar,
            base_edge * scale,
            4.0,
            jnp.asarray((2.0,)),
            8.0,
            2.0,
        )
        return jnp.real(jnp.sum(jnp.abs(samples.electric_v_per_m) ** 2))

    value, derivative = jax.value_and_grad(objective)(jnp.asarray(1.1))
    step = 1.0e-5
    finite_difference = (objective(1.1 + step) - objective(1.1 - step)) / (2.0 * step)
    assert np.isfinite(value)
    assert np.isfinite(derivative)
    assert float(derivative) == pytest.approx(float(finite_difference), rel=2e-8, abs=2e-8)


def test_sampling_plan_hashes_all_six_locations_and_rejects_implicit_ties() -> None:
    first = _triangle_plan()
    second = _triangle_plan()
    assert first.operator_sha256 == second.operator_sha256
    assert first.target_grid.coordinate_sha256 == second.target_grid.coordinate_sha256
    assert first.ambiguous_target_point_count == 0

    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    signs = np.asarray(((1, 1, -1), (1, 1, -1)), dtype=np.int8)
    grid = build_yee_grid((np.asarray((0.0, 1.0)), np.asarray((0.0, 1.0)), np.asarray((0.0, 1.0))))
    with pytest.raises(ContractError, match="multiple source triangles"):
        build_yee_port_sampling_plan(coordinates, cells, signs, grid)
    explicit = build_yee_port_sampling_plan(
        coordinates,
        cells,
        signs,
        grid,
        ambiguity_policy=SamplingAmbiguityPolicy.LOWEST_CELL_ID,
    )
    assert explicit.ambiguous_target_point_count > 0
    assert explicit.operator_sha256 != first.operator_sha256


def test_solution_factory_builds_eta0_bundle_and_fdtdx_callback() -> None:
    plan = _triangle_plan()
    scalar, edge = _constant_mode_coefficients()
    target_power = 2.0
    solution = _solution(plan, scalar, edge, target_power=target_power)
    solver = SolverFingerprint("test-port", "1", SHA, plan.source_mesh_sha256, "revision")

    bundle = port_mode_solution_to_bundle(
        solution,
        plan,
        jnp.asarray((2.0,)),
        frequency_hz=8.0 / (2.0 * math.pi),
        solver=solver,
        fdtdx=FDTDX,
    )

    assert bundle.magnetic_convention.value == "eta0_H"
    assert bundle.transfer.operator_sha256 == plan.operator_sha256
    assert bundle.transfer.source_power_watts == target_power
    assert bundle.transfer.transferred_power_watts == pytest.approx(target_power)
    callback = make_fdtdx_mode_function(bundle)
    centers = [0.5 * (axis[:-1] + axis[1:]) for axis in plan.target_grid.edge_coordinates]
    coordinates = tuple(jnp.asarray(item) for item in np.meshgrid(*centers, indexing="ij"))
    electric, magnetic = callback(
        coordinates=coordinates,
        frequency=bundle.frequency_hz,
        propagation_axis=2,
        inv_permittivity=jnp.ones((3, *plan.target_grid.shape)),
    )
    np.testing.assert_array_equal(electric, bundle.electric.values)
    np.testing.assert_array_equal(magnetic, bundle.magnetic.values)


def test_mode_transfer_fails_closed_on_bad_grid_solution_and_callback_identity() -> None:
    with pytest.raises(ContractError, match="three edge-coordinate axes"):
        build_yee_grid((np.asarray((0.0, 1.0)),))
    with pytest.raises(ContractError, match="strictly increasing"):
        build_yee_grid(
            (
                np.asarray((0.0, 0.0)),
                np.asarray((0.0, 1.0)),
                np.asarray((0.0, 1.0)),
            )
        )

    plan = _triangle_plan()
    scalar, edge = _constant_mode_coefficients()
    solution = _solution(plan, scalar, edge, target_power=2.0)
    with pytest.raises(ContractError, match="mesh digest"):
        port_mode_solution_to_bundle(
            solution,
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=SolverFingerprint("test", "1", SHA, SHA),
            fdtdx=FDTDX,
        )

    bundle = port_mode_solution_to_bundle(
        solution,
        plan,
        jnp.asarray((2.0,)),
        frequency_hz=1.0,
        solver=SolverFingerprint("test", "1", SHA, plan.source_mesh_sha256),
        fdtdx=FDTDX,
    )
    callback = make_fdtdx_mode_function(bundle)
    centers = [0.5 * (axis[:-1] + axis[1:]) for axis in plan.target_grid.edge_coordinates]
    coordinates = tuple(jnp.asarray(item) for item in np.meshgrid(*centers, indexing="ij"))
    with pytest.raises(ContractError, match="propagation axis"):
        callback(
            coordinates=coordinates,
            frequency=1.0,
            propagation_axis=0,
            inv_permittivity=jnp.ones((3, *plan.target_grid.shape)),
        )
    with pytest.raises(ContractError, match="frequency"):
        callback(
            coordinates=coordinates,
            frequency=2.0,
            propagation_axis=2,
            inv_permittivity=jnp.ones((3, *plan.target_grid.shape)),
        )


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        (
            (
                np.asarray(((0.0, 1.0),)),
                np.asarray((0.0, 1.0)),
                np.asarray((0.0, 1.0)),
            ),
            "at least two coordinates",
        ),
        (
            (
                np.asarray((0.0, math.nan)),
                np.asarray((0.0, 1.0)),
                np.asarray((0.0, 1.0)),
            ),
            "finite real coordinates",
        ),
        (
            (
                np.asarray(("0", "1")),
                np.asarray((0.0, 1.0)),
                np.asarray((0.0, 1.0)),
            ),
            "finite real coordinates",
        ),
    ],
)
def test_yee_grid_builder_rejects_nonphysical_axes(edges, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        build_yee_grid(edges)


@pytest.mark.parametrize(
    ("coordinates", "cells", "signs", "kwargs", "message"),
    [
        (
            np.asarray((0.0, 1.0)),
            np.asarray(((0, 1, 2),)),
            np.asarray(((1, 1, -1),)),
            {},
            "coordinates",
        ),
        (
            np.asarray(((0.0, 0.0), (math.nan, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            np.asarray(((1, 1, -1),)),
            {},
            "finite real",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray((0, 1, 2)),
            np.asarray(((1, 1, -1),)),
            {},
            "source cells",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0.0, 1.0, 2.0),)),
            np.asarray(((1, 1, -1),)),
            {},
            "integer dtype",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 3),)),
            np.asarray(((1, 1, -1),)),
            {},
            "node range",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            np.asarray(((1, 1, -1),)),
            {"containment_tolerance": 0.0},
            "tolerance",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            np.asarray(((0, 1, 2),)),
            np.asarray(((1, 1, -1),)),
            {},
            "degenerate",
        ),
        (
            np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))),
            np.asarray(((0, 1, 2),)),
            None,
            {},
            "explicit cell-local",
        ),
        (
            np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))),
            np.asarray(((0, 1, 2),)),
            np.asarray(((1.0, 1.0, -1.0),)),
            {},
            "must be integer",
        ),
        (
            np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))),
            np.asarray(((0, 1, 2),)),
            np.asarray(((1, -1, -1),)),
            {},
            "canonical edge order",
        ),
    ],
)
def test_sampling_plan_builder_rejects_invalid_mesh_contracts(
    coordinates, cells, signs, kwargs: dict[str, object], message: str
) -> None:
    grid = build_yee_grid((np.asarray((0.2, 0.4)), np.asarray((0.2, 0.4)), np.asarray((0.0, 1.0))))
    with pytest.raises(ContractError, match=message):
        build_yee_port_sampling_plan(coordinates, cells, signs, grid, **kwargs)


def test_sampling_plan_rejects_thick_z_and_points_outside_source_mesh() -> None:
    coordinates = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    signs = np.asarray(((1, 1, -1),), dtype=np.int8)
    thick_grid = build_yee_grid(
        (np.asarray((0.2, 0.4)), np.asarray((0.2, 0.4)), np.asarray((0.0, 1.0, 2.0)))
    )
    with pytest.raises(ContractError, match="one cell along z"):
        build_yee_port_sampling_plan(coordinates, cells, signs, thick_grid)
    outside_grid = build_yee_grid(
        (np.asarray((3.0, 4.0)), np.asarray((3.0, 4.0)), np.asarray((0.0, 1.0)))
    )
    with pytest.raises(ContractError, match="outside the source mesh"):
        build_yee_port_sampling_plan(coordinates, cells, signs, outside_grid)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_coordinates": np.zeros((3, 3))}, "source coordinates"),
        ({"source_cells": np.zeros((1, 2), dtype=np.int64)}, "source cells"),
        ({"edge_nodes": np.zeros((3, 3), dtype=np.int64)}, "canonical edges"),
        ({"cell_edge_dofs": np.zeros((1, 2), dtype=np.int64)}, "edge topology"),
        (
            {"electric_cell_indices": np.zeros((2, 2, 2, 1), dtype=np.int64)},
            "electric cell indices",
        ),
        (
            {"magnetic_cell_indices": np.zeros((2, 2, 2, 1), dtype=np.int64)},
            "magnetic cell indices",
        ),
        ({"electric_barycentric_weights": np.zeros((3, 2, 2, 1, 2))}, "electric barycentric"),
        ({"magnetic_barycentric_weights": np.zeros((3, 2, 2, 1, 2))}, "magnetic barycentric"),
        ({"source_mesh_sha256": "bad"}, "source mesh digest"),
        ({"operator_sha256": "bad"}, "operator digest"),
        ({"plane_axes": (1, 2)}, "x-y plane"),
        ({"containment_tolerance": math.nan}, "containment tolerance"),
        ({"ambiguous_target_point_count": -1}, "ambiguous-point count"),
        ({"ambiguous_target_point_count": 1}, "rejecting Yee transfer"),
        ({"maximum_partition_error": -1.0}, "partition error"),
        ({"minimum_barycentric_weight": math.nan}, "minimum barycentric"),
        ({"minimum_barycentric_weight": -1.0}, "outside its assigned"),
        ({"schema_version": "v2"}, "unsupported Yee transfer schema"),
    ],
)
def test_sampling_plan_value_object_rejects_corrupt_provenance(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_triangle_plan(), **changes)


@pytest.mark.parametrize(
    ("scalar", "edge", "reluctivity", "beta", "omega", "power", "message"),
    [
        (
            jnp.ones((2,), dtype=jnp.complex128),
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((1,)),
            4.0,
            8.0,
            1.0,
            "scalar coefficients",
        ),
        (
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((2,), dtype=jnp.complex128),
            jnp.ones((1,)),
            4.0,
            8.0,
            1.0,
            "edge coefficients",
        ),
        (
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((2,)),
            4.0,
            8.0,
            1.0,
            "cell reluctivity",
        ),
        (
            jnp.ones((3,)),
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((1,)),
            4.0,
            8.0,
            1.0,
            "scalar coefficients must use a complex",
        ),
        (
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((3,)),
            jnp.ones((1,)),
            4.0,
            8.0,
            1.0,
            "edge coefficients must use a complex",
        ),
        (
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((1,), dtype=jnp.complex128),
            4.0,
            8.0,
            1.0,
            "reluctivity must use a real",
        ),
        (
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((3,), dtype=jnp.complex128),
            jnp.ones((1,)),
            jnp.ones((1,)),
            8.0,
            1.0,
            "must be scalars",
        ),
    ],
)
def test_sampling_kernel_rejects_bad_coefficient_and_scalar_shapes(
    scalar, edge, reluctivity, beta, omega, power, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        sample_port_mode_to_yee(_triangle_plan(), scalar, edge, beta, reluctivity, omega, power)


def test_solution_factory_rejects_bad_semantics_and_backward_power() -> None:
    plan = _triangle_plan()
    scalar, edge = _constant_mode_coefficients()
    base = _solution(plan, scalar, edge, target_power=2.0)
    solver = SolverFingerprint("test", "1", SHA, plan.source_mesh_sha256)

    with pytest.raises(ContractError, match="frequency"):
        port_mode_solution_to_bundle(
            base, plan, jnp.asarray((2.0,)), frequency_hz=math.nan, solver=solver, fdtdx=FDTDX
        )
    with pytest.raises(ContractError, match="lacks normalized mixed"):
        port_mode_solution_to_bundle(
            replace(base, fields={}),
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )
    complex_power = {**base.observables, "target_forward_power_W": 1.0 + 1.0j}
    with pytest.raises(ContractError, match="target power observable must be real"):
        port_mode_solution_to_bundle(
            replace(base, observables=complex_power),
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )
    wrong_scalar = replace(
        base.fields[PORT_LONGITUDINAL_POTENTIAL_FIELD],
        unit="V/m",
    )
    with pytest.raises(ContractError, match="longitudinal-potential"):
        port_mode_solution_to_bundle(
            replace(base, fields={**base.fields, PORT_LONGITUDINAL_POTENTIAL_FIELD: wrong_scalar}),
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )
    wrong_edge = replace(base.fields[PORT_TRANSVERSE_ELECTRIC_FIELD], unit="V/m")
    with pytest.raises(ContractError, match="transverse edge"):
        port_mode_solution_to_bundle(
            replace(base, fields={**base.fields, PORT_TRANSVERSE_ELECTRIC_FIELD: wrong_edge}),
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )
    invalid_power = replace(
        base,
        observables={**base.observables, "target_forward_power_W": math.nan},
    )
    with pytest.raises(ContractError, match="power observable must be finite"):
        port_mode_solution_to_bundle(
            invalid_power,
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )
    invalid_index = replace(
        base,
        observables={**base.observables, "effective_index": complex(math.nan, 0.0)},
    )
    with pytest.raises(ContractError, match="effective index must be finite"):
        port_mode_solution_to_bundle(
            invalid_index,
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )
    backward = replace(
        base,
        observables={**base.observables, "propagation_constant_rad_per_m": -4.0 + 0.0j},
    )
    with pytest.raises(ContractError, match="positive real beta"):
        port_mode_solution_to_bundle(
            backward,
            plan,
            jnp.asarray((2.0,)),
            frequency_hz=1.0,
            solver=solver,
            fdtdx=FDTDX,
        )


def test_fdtdx_callback_rejects_material_and_coordinate_mismatches() -> None:
    plan = _triangle_plan()
    scalar, edge = _constant_mode_coefficients()
    bundle = port_mode_solution_to_bundle(
        _solution(plan, scalar, edge, target_power=2.0),
        plan,
        jnp.asarray((2.0,)),
        frequency_hz=1.0,
        solver=SolverFingerprint("test", "1", SHA, plan.source_mesh_sha256),
        fdtdx=FDTDX,
    )
    callback = make_fdtdx_mode_function(bundle)
    centers = [0.5 * (axis[:-1] + axis[1:]) for axis in plan.target_grid.edge_coordinates]
    coordinates = tuple(jnp.asarray(item) for item in np.meshgrid(*centers, indexing="ij"))
    with pytest.raises(ContractError, match="inverse permittivity"):
        callback(
            coordinates=coordinates,
            frequency=1.0,
            propagation_axis=2,
            inv_permittivity=jnp.ones((2, *plan.target_grid.shape)),
        )
    with pytest.raises(ContractError, match="coordinates do not match"):
        callback(
            coordinates=coordinates[:2],
            frequency=1.0,
            propagation_axis=2,
            inv_permittivity=jnp.ones((3, *plan.target_grid.shape)),
        )
    shifted = (coordinates[0] + 1.0, coordinates[1], coordinates[2])
    with pytest.raises(ContractError, match="differ on axis 0"):
        callback(
            coordinates=shifted,
            frequency=1.0,
            propagation_axis=2,
            inv_permittivity=jnp.ones((3, *plan.target_grid.shape)),
        )
