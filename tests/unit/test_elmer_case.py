from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends._steady_heat import validate_steady_heat_problem
from femx.backends.elmer.case import (
    _format_real,
    lower_elmer_mesh,
    lower_tagged_scalar_h1_mesh,
    render_steady_heat_sif,
)
from femx.core.errors import ContractError
from femx.core.parameters import (
    ParameterReference,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem
from femx.mesh import CellType, EntityTag, MeshGeometry, MeshTopology
from femx.physics import HeatFluxBoundary, SteadyHeat, TemperatureBoundary, ThermalRegion

pytestmark = pytest.mark.unit


def _validated_case(*, parameterized: bool = False):
    if parameterized:
        physics = SteadyHeat(
            regions=(
                ThermalRegion(
                    "domain",
                    ParameterReference("k"),
                    ParameterReference("source"),
                ),
            ),
            temperature_boundaries=(
                TemperatureBoundary("left", ParameterReference("temperature")),
            ),
            heat_flux_boundaries=(HeatFluxBoundary("right", ParameterReference("heat_load")),),
        )
        schema = ParameterSchema(
            (
                ParameterSpec("k", unit="W/(m*K)"),
                ParameterSpec("source", unit="W/m^3"),
                ParameterSpec("temperature", unit="K"),
                ParameterSpec("heat_load", unit="W/m^2"),
            )
        )
    else:
        physics = SteadyHeat(
            regions=(ThermalRegion("domain", 2.0, 3.0),),
            temperature_boundaries=(TemperatureBoundary("left", 300.0),),
            heat_flux_boundaries=(HeatFluxBoundary("right", 4.0),),
        )
        schema = ParameterSchema()
    problem = Problem("elmer-case", structured_unit_square_mesh(1), physics, parameters=schema)
    return validate_steady_heat_problem(problem)


def test_native_mesh_lowering_has_dense_ids_parents_and_oriented_boundary() -> None:
    deck = lower_elmer_mesh(_validated_case())

    assert deck.header == "4 2 4\n2\n202 4\n303 2\n"
    assert deck.elements == "1 1 303 1 2 4\n2 1 303 1 4 3\n"
    assert deck.boundary == ("1 3 1 0 202 1 2\n2 2 1 0 202 2 4\n3 3 2 0 202 4 3\n4 1 2 0 202 3 1\n")
    assert deck.essential_boundary_ids == (1,)
    assert deck.natural_boundary_ids == (2,)
    assert deck.nodes.splitlines()[0] == (
        "1 -1 0.00000000000000000e+00 0.00000000000000000e+00 0.00000000000000000e+00"
    )


def test_native_mesh_lowering_normalizes_clockwise_cells() -> None:
    validated = _validated_case()
    mesh = structured_unit_square_mesh(1)
    clockwise_cells = np.asarray(mesh.topology.connectivity).copy()
    clockwise_cells[:, [1, 2]] = clockwise_cells[:, [2, 1]]
    clockwise_mesh = replace(
        mesh,
        topology=MeshTopology(clockwise_cells, mesh.topology.cell_type, mesh.geometry.node_count),
    )
    clockwise_problem = Problem(
        "clockwise",
        clockwise_mesh,
        SteadyHeat(
            (ThermalRegion("domain", 2.0, 3.0),),
            (TemperatureBoundary("left", 300.0),),
            (HeatFluxBoundary("right", 4.0),),
        ),
    )

    normalized = lower_elmer_mesh(validate_steady_heat_problem(clockwise_problem))

    assert normalized.elements == lower_elmer_mesh(validated).elements
    assert normalized.boundary == lower_elmer_mesh(validated).boundary


def test_sif_renderer_binds_parameters_and_preserves_variational_flux_sign() -> None:
    validated = _validated_case(parameterized=True)
    deck = lower_elmer_mesh(validated)
    sif = render_steady_heat_sif(
        validated,
        deck,
        ParameterValues({"k": 2.0, "source": 3.0, "temperature": 300.0, "heat_load": -4.0}),
        convergence_tolerance=1.0e-12,
        heat_solve_module=Path("/locked/elmer/lib/HeatSolve.so"),
    )

    assert 'Mesh DB "." "mesh"' in sif
    assert 'Output File = "femx.result"' in sif
    assert 'Procedure = File "/locked/elmer/lib/HeatSolve.so" "HeatSolver"' in sif
    assert 'Procedure = "HeatSolve" "HeatSolver"' not in sif
    assert "vtu: Binary Output = Logical True" in sif
    assert "Heat Conductivity = Real 2.00000000000000000e+00" in sif
    assert "Volumetric Heat Source = Real 3.00000000000000000e+00" in sif
    assert "Temperature = Real 3.00000000000000000e+02" in sif
    assert "Heat Flux = Real -4.00000000000000000e+00" in sif
    assert sif.endswith("\n")


def test_elmer_real_formatter_rejects_nonfinite_values() -> None:
    with pytest.raises(ContractError, match="finite"):
        _format_real(float("nan"))


@pytest.mark.parametrize(
    "module",
    [Path("relative/HeatSolve.so"), Path("/locked/with space/HeatSolve.so")],
)
def test_sif_renderer_rejects_ambiguous_heat_solve_paths(module: Path) -> None:
    validated = _validated_case()
    with pytest.raises(ContractError, match="procedure path"):
        render_steady_heat_sif(
            validated,
            lower_elmer_mesh(validated),
            ParameterValues(),
            convergence_tolerance=1.0e-12,
            heat_solve_module=module,
        )


def test_native_mesh_needs_no_natural_group_when_every_facet_is_assigned() -> None:
    problem = Problem(
        "all-boundaries",
        structured_unit_square_mesh(1),
        SteadyHeat(
            (ThermalRegion("domain", 1.0),),
            (
                TemperatureBoundary("left", 0.0),
                TemperatureBoundary("bottom", 0.0),
                TemperatureBoundary("top", 0.0),
            ),
            (HeatFluxBoundary("right", 1.0),),
        ),
    )
    deck = lower_elmer_mesh(validate_steady_heat_problem(problem))

    assert {int(line.split()[1]) for line in deck.boundary.splitlines()} == {1, 2, 3, 4}


def test_tagged_mesh_lowering_preserves_complete_canonical_partitions() -> None:
    mesh = structured_unit_square_mesh(1)
    deck = lower_tagged_scalar_h1_mesh(
        mesh,
        region_tags=("domain",),
        essential_boundary_tags=("left", "bottom"),
        natural_boundary_tags=("right", "top"),
    )

    assert deck.elements == "1 1 303 1 2 4\n2 1 303 1 4 3\n"
    assert deck.essential_boundary_ids == (1, 2)
    assert deck.natural_boundary_ids == (3, 4)
    assert {int(line.split()[1]) for line in deck.boundary.splitlines()} == {1, 2, 3, 4}


def test_tagged_mesh_lowering_rejects_incompatible_topology_and_missing_facets() -> None:
    mesh = structured_unit_square_mesh(1)
    one_dimensional_geometry = replace(
        mesh,
        geometry=MeshGeometry(np.asarray(mesh.geometry.coordinates)[:, :1]),
    )
    with pytest.raises(ContractError, match="2D triangle"):
        lower_tagged_scalar_h1_mesh(
            one_dimensional_geometry,
            region_tags=("domain",),
            essential_boundary_tags=("left", "right", "bottom", "top"),
        )

    quadrilateral = replace(
        mesh,
        topology=MeshTopology(np.asarray([[0, 1, 3, 2]]), CellType.QUADRILATERAL, 4),
    )
    with pytest.raises(ContractError, match="2D triangle"):
        lower_tagged_scalar_h1_mesh(
            quadrilateral,
            region_tags=("domain",),
            essential_boundary_tags=("left", "right", "bottom", "top"),
        )

    with pytest.raises(ContractError, match="boundary segments"):
        lower_tagged_scalar_h1_mesh(
            replace(mesh, boundary_facets=None),
            region_tags=("domain",),
            essential_boundary_tags=("left", "right", "bottom", "top"),
        )


def test_tagged_mesh_lowering_rejects_ambiguous_or_incomplete_tag_partitions() -> None:
    mesh = structured_unit_square_mesh(1)
    calls = (
        (
            mesh,
            (),
            ("left", "right", "bottom", "top"),
            "must not be empty",
        ),
        (
            mesh,
            ("domain", "domain"),
            ("left", "right", "bottom", "top"),
            "duplicate names",
        ),
        (
            mesh,
            ("left",),
            ("left", "right", "bottom", "top"),
            "dimension 2",
        ),
        (
            replace(
                mesh,
                tags=tuple(
                    EntityTag("domain", 2, ()) if tag.name == "domain" else tag for tag in mesh.tags
                ),
            ),
            ("domain",),
            ("left", "right", "bottom", "top"),
            "must not be empty",
        ),
        (
            replace(
                mesh,
                tags=tuple(
                    EntityTag("domain", 2, (0, 99)) if tag.name == "domain" else tag
                    for tag in mesh.tags
                ),
            ),
            ("domain",),
            ("left", "right", "bottom", "top"),
            "out-of-range",
        ),
        (
            replace(mesh, tags=(*mesh.tags, EntityTag("domain_alias", 2, (0, 1)))),
            ("domain", "domain_alias"),
            ("left", "right", "bottom", "top"),
            "overlaps",
        ),
        (
            replace(
                mesh,
                tags=tuple(
                    EntityTag("domain", 2, (0,)) if tag.name == "domain" else tag
                    for tag in mesh.tags
                ),
            ),
            ("domain",),
            ("left", "right", "bottom", "top"),
            "unmapped",
        ),
        (
            mesh,
            ("domain",),
            (),
            "must not be empty",
        ),
    )
    for candidate, regions, boundaries, message in calls:
        with pytest.raises(ContractError, match=message):
            lower_tagged_scalar_h1_mesh(
                candidate,
                region_tags=regions,
                essential_boundary_tags=boundaries,
            )
