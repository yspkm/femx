from pathlib import Path

import pytest
from tests.support import structured_unit_square_mesh

from femx.backends._steady_current import validate_steady_current_problem
from femx.backends.elmer.case import lower_scalar_h1_mesh
from femx.backends.elmer.current_case import render_steady_current_sif
from femx.core.errors import ContractError
from femx.core.parameters import (
    ParameterReference,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem
from femx.physics import (
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)

pytestmark = pytest.mark.unit


def _validated_case(*, parameterized: bool = False):
    if parameterized:
        physics = SteadyCurrent(
            (
                ConductiveRegion(
                    "domain",
                    ParameterReference("sigma"),
                    ParameterReference("source"),
                ),
            ),
            (PotentialBoundary("left", ParameterReference("potential")),),
            (CurrentFluxBoundary("right", ParameterReference("current_load")),),
        )
        schema = ParameterSchema(
            (
                ParameterSpec("sigma", unit="S/m"),
                ParameterSpec("source", unit="A/m^3"),
                ParameterSpec("potential", unit="V"),
                ParameterSpec("current_load", unit="A/m^2"),
            )
        )
    else:
        physics = SteadyCurrent(
            (ConductiveRegion("domain", 2.0, 3.0),),
            (PotentialBoundary("left", 0.0),),
            (CurrentFluxBoundary("right", 4.0),),
        )
        schema = ParameterSchema()
    return validate_steady_current_problem(
        Problem("elmer-current-case", structured_unit_square_mesh(1), physics, parameters=schema)
    )


def _mesh(validated):
    return lower_scalar_h1_mesh(
        coordinates=validated.coordinates,
        cells=validated.cells,
        boundary_facets=validated.boundary_facets,
        region_cells=validated.region_cells,
        essential_facets=validated.potential_facets,
        natural_facets=validated.flux_facets,
    )


def test_current_mesh_and_sif_bind_the_locked_scalar_contract() -> None:
    validated = _validated_case(parameterized=True)
    deck = _mesh(validated)
    sif = render_steady_current_sif(
        validated,
        deck,
        ParameterValues({"sigma": 2.0, "source": 3.0, "potential": 0.0, "current_load": -4.0}),
        convergence_tolerance=1.0e-12,
        stat_current_module=Path("/locked/elmer/lib/StatCurrentSolve.so"),
    )

    assert deck.essential_boundary_ids == (1,)
    assert deck.natural_boundary_ids == (2,)
    assert 'Output Variable 1 = String "Potential"' in sif
    assert 'Procedure = File "/locked/elmer/lib/StatCurrentSolve.so" "StatCurrentSolver"' in sif
    assert 'Procedure = "StatCurrentSolve" "StatCurrentSolver"' not in sif
    assert "Electric Conductivity = Real 2.00000000000000000e+00" in sif
    assert "Current Source = Real 3.00000000000000000e+00" in sif
    assert "Potential = Real 0.00000000000000000e+00" in sif
    assert "Current Density BC = Logical True" in sif
    assert "Current Density = Real -4.00000000000000000e+00" in sif
    assert sif.endswith("\n")


@pytest.mark.parametrize(
    "module",
    [Path("relative/StatCurrentSolve.so"), Path("/locked/with space/StatCurrentSolve.so")],
)
def test_current_sif_rejects_ambiguous_module_paths(module: Path) -> None:
    validated = _validated_case()
    with pytest.raises(ContractError, match="procedure path"):
        render_steady_current_sif(
            validated,
            _mesh(validated),
            ParameterValues(),
            convergence_tolerance=1.0e-12,
            stat_current_module=module,
        )
