from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends._port_eigenmode import (
    NormalizedProjectedMode,
    normalize_projected_electromagnetic_mode,
    normalize_projected_mode,
    resolve_port_materials,
    validate_port_eigenmode_problem,
)
from femx.core.errors import BackendError, ContractError
from femx.core.parameters import (
    ParameterReference,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem
from femx.mesh import EntityTag, Mesh, OrientationMap
from femx.physics import (
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    HeatFluxBoundary,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)

pytestmark = pytest.mark.unit


def _oriented_mesh() -> Mesh:
    mesh = structured_unit_square_mesh(1)
    cells = np.asarray(mesh.topology.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    return replace(mesh, orientation=OrientationMap(edge_signs=signs))


def _physics() -> PortEigenmode:
    return PortEigenmode(
        regions=(IsotropicOpticalRegion("domain", 12.0),),
        perfect_electric_boundaries=tuple(
            PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
        ),
        frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
        eigenmode_count=4,
        selected_mode_index=1,
        target_power_w=2.0,
    )


def _problem(*, mesh: Mesh | None = None) -> Problem:
    return Problem("port", _oriented_mesh() if mesh is None else mesh, _physics())


def test_port_problem_validation_preserves_mesh_orientation_and_material_order() -> None:
    validated = validate_port_eigenmode_problem(_problem())

    assert validated.coordinates.dtype == np.float64
    assert validated.cells.shape == (2, 3)
    assert validated.boundary_facets.shape == (4, 2)
    np.testing.assert_array_equal(
        validated.edge_nodes,
        ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3)),
    )
    np.testing.assert_array_equal(validated.cell_edge_dofs, ((0, 3, 2), (2, 4, 1)))
    np.testing.assert_array_equal(validated.edge_signs, ((1, 1, -1), (1, -1, -1)))
    np.testing.assert_array_equal(validated.dof_partition.scalar_dofs, (0, 1, 2, 3))
    np.testing.assert_array_equal(validated.dof_partition.edge_dofs, (0, 1, 3, 4))
    np.testing.assert_array_equal(
        validated.dof_partition.constrained_dofs,
        (0, 1, 2, 3, 4, 5, 7, 8),
    )
    np.testing.assert_array_equal(validated.dof_partition.free_dofs, (6,))
    np.testing.assert_array_equal(validated.region_cells[0], (0, 1))
    assert validated.relative_permittivity == (12.0,)
    assert validated.relative_permeability == (1.0,)
    assert tuple(group.tolist() for group in validated.pec_facets) == (
        [0],
        [1],
        [2],
        [3],
    )
    assert validated.eigenmode_count == 4
    assert validated.selected_mode_index == 1
    assert validated.target_power_w == 2.0


@dataclass(frozen=True)
class _ProtocolOnlyMesh:
    geometry: object
    topology: object
    schema_version: str = "femx.mesh/v1"


def test_port_problem_validation_rejects_wrong_problem_kinds_and_parameters() -> None:
    mesh = _oriented_mesh()
    faux_mesh = _ProtocolOnlyMesh(mesh.geometry, mesh.topology)
    with pytest.raises(ContractError, match="concrete femx Mesh"):
        validate_port_eigenmode_problem(Problem("faux", faux_mesh, _physics()))  # type: ignore[arg-type]

    heat = SteadyHeat(
        regions=(ThermalRegion("domain", 1.0),),
        temperature_boundaries=(TemperatureBoundary("left", 0.0),),
        heat_flux_boundaries=(HeatFluxBoundary("right", 0.0),),
    )
    with pytest.raises(ContractError, match="PortEigenmode"):
        validate_port_eigenmode_problem(Problem("heat", mesh, heat))

    parameterized = Problem(
        "parameterized",
        mesh,
        _physics(),
        parameters=ParameterSchema((ParameterSpec("epsilon", unit="1"),)),
    )
    with pytest.raises(ContractError, match="parameters do not match"):
        validate_port_eigenmode_problem(parameterized)


def test_port_problem_resolves_exact_dimensionless_material_parameters() -> None:
    physics = replace(
        _physics(),
        regions=(
            IsotropicOpticalRegion(
                "domain",
                ParameterReference("epsilon_r"),
                ParameterReference("mu_r"),
            ),
        ),
    )
    problem = Problem(
        "parameterized-port",
        _oriented_mesh(),
        physics,
        parameters=ParameterSchema(
            (
                ParameterSpec("epsilon_r", unit="1"),
                ParameterSpec("mu_r", unit="1"),
            )
        ),
    )

    validated = validate_port_eigenmode_problem(problem)
    resolved = resolve_port_materials(
        validated,
        ParameterValues({"epsilon_r": 12.1, "mu_r": 1.0}),
    )

    assert validated.relative_permittivity == (ParameterReference("epsilon_r"),)
    assert validated.relative_permeability == (ParameterReference("mu_r"),)
    assert validated.parameter_names == ("epsilon_r", "mu_r")
    assert resolved.relative_permittivity == (12.1,)
    assert resolved.relative_permeability == (1.0,)

    with pytest.raises(ContractError, match="parameter keys"):
        resolve_port_materials(validated, ParameterValues({"epsilon_r": 12.1}))
    with pytest.raises(ContractError, match="positive"):
        resolve_port_materials(
            validated,
            ParameterValues({"epsilon_r": -1.0, "mu_r": 1.0}),
        )
    with pytest.raises(ContractError, match="finite real scalars"):
        resolve_port_materials(
            validated,
            ParameterValues({"epsilon_r": float("nan"), "mu_r": 1.0}),
        )


@pytest.mark.parametrize(
    "spec",
    (
        ParameterSpec("epsilon_r", unit="F/m"),
        ParameterSpec("epsilon_r", unit="1", shape=(1,)),
    ),
)
def test_port_problem_rejects_non_dimensionless_or_array_material_parameter(
    spec: ParameterSpec,
) -> None:
    physics = replace(
        _physics(),
        regions=(IsotropicOpticalRegion("domain", ParameterReference("epsilon_r")),),
    )
    with pytest.raises(ContractError, match="dimensionless scalar"):
        validate_port_eigenmode_problem(
            Problem(
                "bad-parameterized-port",
                _oriented_mesh(),
                physics,
                parameters=ParameterSchema((spec,)),
            )
        )


def test_port_problem_validation_rejects_incomplete_and_overlapping_partitions() -> None:
    mesh = _oriented_mesh()
    incomplete_cells = replace(
        mesh,
        tags=tuple(
            EntityTag("domain", 2, (0,)) if tag.name == "domain" else tag for tag in mesh.tags
        ),
    )
    with pytest.raises(ContractError, match="partition every cell"):
        validate_port_eigenmode_problem(_problem(mesh=incomplete_cells))

    overlapping_cells = replace(
        mesh,
        tags=(*mesh.tags, EntityTag("second", 2, (0, 1))),
    )
    physics = replace(
        _physics(),
        regions=(
            IsotropicOpticalRegion("domain", 12.0),
            IsotropicOpticalRegion("second", 2.0),
        ),
    )
    with pytest.raises(ContractError, match="partition every cell"):
        validate_port_eigenmode_problem(Problem("overlap", overlapping_cells, physics))

    incomplete_boundary = replace(
        _physics(),
        perfect_electric_boundaries=(PerfectElectricBoundary("left"),),
    )
    with pytest.raises(ContractError, match="partition every external"):
        validate_port_eigenmode_problem(Problem("incomplete-boundary", mesh, incomplete_boundary))

    boundary_alias = replace(
        mesh,
        tags=(*mesh.tags, EntityTag("outer_alias", 1, (0, 1, 2, 3))),
    )
    overlapping_boundary = replace(
        _physics(),
        perfect_electric_boundaries=(
            *_physics().perfect_electric_boundaries,
            PerfectElectricBoundary("outer_alias"),
        ),
    )
    with pytest.raises(ContractError, match="partition every external"):
        validate_port_eigenmode_problem(
            Problem("overlap-boundary", boundary_alias, overlapping_boundary)
        )


@pytest.mark.parametrize(
    ("orientation", "message"),
    [
        (OrientationMap(), "explicit"),
        (OrientationMap(edge_signs=np.ones((2, 3), dtype=np.float64)), "integer"),
        (OrientationMap(edge_signs=np.ones((1, 3), dtype=np.int8)), "shaped"),
        (OrientationMap(edge_signs=np.ones((2, 3), dtype=np.int8)), "global edge ordering"),
    ],
)
def test_port_problem_validation_rejects_missing_or_wrong_edge_orientation(
    orientation: OrientationMap, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        validate_port_eigenmode_problem(
            _problem(mesh=replace(_oriented_mesh(), orientation=orientation))
        )


def test_projected_mode_normalization_scales_power_and_fixes_global_phase() -> None:
    field = np.asarray(
        [[1.0 + 1.0j, 0.0j, 0.0j], [0.0j, 0.0 + 3.0j, 0.0j]],
        dtype=np.complex128,
    )

    normalized = normalize_projected_mode(
        field,
        raw_forward_power_w=4.0,
        target_forward_power_w=1.0,
    )

    assert normalized.anchor_node == 1
    assert normalized.anchor_component == 1
    assert normalized.phase_factor == -1.0j
    assert normalized.amplitude_scale == 0.5
    assert normalized.electric_field[1, 1] == 1.5 + 0.0j
    np.testing.assert_allclose(normalized.electric_field, field * -0.5j)


@pytest.mark.parametrize(
    ("field", "raw_power", "target_power", "message"),
    [
        (np.ones((2, 2), dtype=np.complex128), 1.0, 1.0, r"complex \(nodes, 3\)"),
        (np.ones((2, 3), dtype=np.float64), 1.0, 1.0, r"complex \(nodes, 3\)"),
        (
            np.asarray([[complex(float("nan"), 0.0), 0.0j, 0.0j]]),
            1.0,
            1.0,
            "non-finite",
        ),
        (np.ones((1, 3), dtype=np.complex128), 0.0, 1.0, "raw forward power"),
        (np.ones((1, 3), dtype=np.complex128), 1.0, float("inf"), "target forward power"),
        (np.zeros((1, 3), dtype=np.complex128), 1.0, 1.0, "identically zero"),
    ],
)
def test_projected_mode_normalization_rejects_ambiguous_fields_and_power(
    field: np.ndarray,
    raw_power: float,
    target_power: float,
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        normalize_projected_mode(
            field,
            raw_forward_power_w=raw_power,
            target_forward_power_w=target_power,
        )


def test_projected_mode_normalization_fails_closed_if_phase_postcondition_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = np.asarray(
        [[4.986266153726417 - 9.087323683771121j, 0.0j, 0.0j]],
        dtype=np.complex128,
    )
    monkeypatch.setattr(
        "femx.backends._port_eigenmode.np.finfo",
        lambda _dtype: SimpleNamespace(eps=0.0),
    )

    with pytest.raises(BackendError, match="phase canonicalization failed"):
        normalize_projected_mode(
            field,
            raw_forward_power_w=1.0,
            target_forward_power_w=1.0,
        )


def test_projected_mode_normalization_rejects_scaling_overflow() -> None:
    field = np.ones((1, 3), dtype=np.complex128)
    with pytest.raises(BackendError, match="ratio overflowed"):
        normalize_projected_mode(
            field,
            raw_forward_power_w=np.finfo(np.float64).tiny,
            target_forward_power_w=np.finfo(np.float64).max,
        )

    field[0, 0] = np.finfo(np.float64).max + 0.0j
    with pytest.raises(BackendError, match="non-finite field values"):
        normalize_projected_mode(
            field,
            raw_forward_power_w=1.0,
            target_forward_power_w=4.0,
        )


def test_electromagnetic_mode_normalization_applies_one_factor_to_e_and_h() -> None:
    electric = np.asarray(((0.0j, 0.0 + 2.0j, 0.0j),), dtype=np.complex128)
    magnetic = np.asarray(((3.0 + 0.0j, 0.0j, 0.0j),), dtype=np.complex128)

    normalized = normalize_projected_electromagnetic_mode(
        electric,
        magnetic,
        raw_forward_power_w=4.0,
        target_forward_power_w=1.0,
    )

    np.testing.assert_allclose(normalized.electric_field, electric * -0.5j)
    np.testing.assert_allclose(normalized.magnetic_field, magnetic * -0.5j)
    assert normalized.normalized_forward_power_w == 1.0
    assert normalized.anchor_node == 0
    assert normalized.anchor_component == 1


@pytest.mark.parametrize(
    ("magnetic", "message"),
    (
        (np.ones((1, 2), dtype=np.complex128), "must match"),
        (np.ones((1, 3), dtype=np.float64), "must match"),
        (
            np.asarray(((complex(float("nan"), 0.0), 0.0j, 0.0j),)),
            "non-finite",
        ),
    ),
)
def test_electromagnetic_mode_normalization_rejects_invalid_magnetic_fields(
    magnetic: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        normalize_projected_electromagnetic_mode(
            np.ones((1, 3), dtype=np.complex128),
            magnetic,
            raw_forward_power_w=1.0,
            target_forward_power_w=1.0,
        )


def test_electromagnetic_mode_normalization_rejects_magnetic_scaling_overflow() -> None:
    magnetic = np.ones((1, 3), dtype=np.complex128)
    magnetic[0, 0] = np.finfo(np.float64).max + 0.0j
    with pytest.raises(BackendError, match="non-finite magnetic"):
        normalize_projected_electromagnetic_mode(
            np.ones((1, 3), dtype=np.complex128),
            magnetic,
            raw_forward_power_w=1.0,
            target_forward_power_w=4.0,
        )


def test_electromagnetic_mode_normalization_rejects_nonfinite_power_postcondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impossible = NormalizedProjectedMode(
        electric_field=np.ones((1, 3), dtype=np.complex128),
        anchor_node=0,
        anchor_component=0,
        phase_factor=1.0 + 0.0j,
        amplitude_scale=np.finfo(np.float64).max,
    )
    monkeypatch.setattr(
        "femx.backends._port_eigenmode.normalize_projected_mode",
        lambda *_args, **_kwargs: impossible,
    )

    with pytest.raises(BackendError, match="forward power is non-finite"):
        normalize_projected_electromagnetic_mode(
            np.ones((1, 3), dtype=np.complex128),
            np.zeros((1, 3), dtype=np.complex128),
            raw_forward_power_w=2.0,
            target_forward_power_w=1.0,
        )
