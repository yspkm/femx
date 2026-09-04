from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.electrothermal_support import parameterized_self_consistent_microheater

from femx.backends._steady_current import validate_steady_current_problem
from femx.backends._steady_heat import validate_steady_heat_problem
from femx.backends.elmer.electrothermal_case import (
    ElmerCoupledBoundary,
    _boundary_signatures,
    _resolve_feedback_scalar,
    lower_self_consistent_mesh,
    render_self_consistent_sif,
)
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference, ParameterValues

pytestmark = pytest.mark.unit


def _case():
    feedback, current_parameters, heat_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=2)
    )
    current = validate_steady_current_problem(feedback.one_way.electrical_problem)
    heat = validate_steady_heat_problem(feedback.one_way.thermal_problem)
    mesh = lower_self_consistent_mesh(current, heat)
    return (
        feedback,
        current,
        heat,
        mesh,
        current_parameters,
        heat_parameters,
        feedback_parameters,
    )


def test_coupled_mesh_and_sif_preserve_combined_partitions_and_feedback_law() -> None:
    feedback, current, heat, mesh, current_parameters, heat_parameters, feedback_parameters = (
        _case()
    )
    sif = render_self_consistent_sif(
        feedback,
        current,
        heat,
        mesh,
        current_parameters,
        heat_parameters,
        feedback_parameters,
        stat_current_module=Path("/locked/elmer/lib/StatCurrentSolve.so"),
        heat_solve_module=Path("/locked/elmer/lib/HeatSolve.so"),
        convergence_tolerance=1.0e-12,
    )

    assert [(body.electrical_region_index, body.thermal_region_index) for body in mesh.bodies] == [
        (0, 0),
        (1, 1),
    ]
    assert len(mesh.boundaries) == 2
    assert mesh.boundaries[0].potential_index == 1
    assert mesh.boundaries[0].temperature_index == 1
    assert mesh.boundaries[1].potential_index == 0
    assert mesh.boundaries[1].temperature_index == 0
    assert 'Output Variable 1 = String "Potential"' in sif
    assert 'Output Variable 2 = String "Temperature"' in sif
    assert "Steady State Max Iterations = 100" in sif
    assert "Steady State Min Iterations = 2" in sif
    assert 'Procedure = File "/locked/elmer/lib/StatCurrentSolve.so" "StatCurrentSolver"' in sif
    assert 'Procedure = File "/locked/elmer/lib/HeatSolve.so" "HeatSolver"' in sif
    assert "Electric Conductivity = Variable Temperature" in sif
    assert (
        'Real MATC "2.00000000000000000e+03/'
        '(1.0+3.00000000000000006e-03*(tx(0)-3.00000000000000000e+02))"' in sif
    )
    assert "Electric Conductivity = Real 2.00000000000000000e+05" in sif
    assert sif.count("Heat Conductivity = Real 1.20000000000000000e+02") == 2
    assert sif.count("Joule Heat = Logical True") == 2
    assert "Steady State Relaxation Factor = 5.00000000000000000e-01" in sif
    assert "Potential = Real 2.00000000000000011e-01" in sif
    assert sif.count("Temperature = Real 3.00000000000000000e+02") == 3
    assert sif.endswith("\n")


@pytest.mark.parametrize(
    ("stat_module", "heat_module"),
    [
        (Path("relative/StatCurrentSolve.so"), Path("/locked/HeatSolve.so")),
        (Path("/locked/StatCurrentSolve.so"), Path("/locked/with space/HeatSolve.so")),
    ],
)
def test_coupled_sif_rejects_ambiguous_module_paths(
    stat_module: Path,
    heat_module: Path,
) -> None:
    feedback, current, heat, mesh, current_parameters, heat_parameters, feedback_parameters = (
        _case()
    )
    with pytest.raises(ContractError, match="procedure path"):
        render_self_consistent_sif(
            feedback,
            current,
            heat,
            mesh,
            current_parameters,
            heat_parameters,
            feedback_parameters,
            stat_current_module=stat_module,
            heat_solve_module=heat_module,
            convergence_tolerance=1.0e-12,
        )


def test_coupled_lowering_and_renderer_fail_closed_on_invalid_internal_inputs() -> None:
    feedback, current, heat, mesh, current_parameters, heat_parameters, feedback_parameters = (
        _case()
    )
    with pytest.raises(ContractError, match="identical validated coordinates"):
        lower_self_consistent_mesh(
            current,
            replace(heat, coordinates=heat.coordinates + 1.0),
        )
    with pytest.raises(ContractError, match="overlapping equation data"):
        _boundary_signatures(
            2,
            ((np.asarray((0,), dtype=np.int64), np.asarray((0,), dtype=np.int64)),),
        )
    with pytest.raises(ContractError, match="finite real scalars"):
        _resolve_feedback_scalar(
            ParameterReference("alpha"),
            ParameterValues({"alpha": 1.0 + 0.0j}),
        )
    with pytest.raises(ContractError, match="finite and positive"):
        render_self_consistent_sif(
            feedback,
            current,
            heat,
            mesh,
            current_parameters,
            heat_parameters,
            feedback_parameters,
            stat_current_module=Path("/locked/StatCurrentSolve.so"),
            heat_solve_module=Path("/locked/HeatSolve.so"),
            convergence_tolerance=float("nan"),
        )


def test_coupled_sif_combines_natural_conditions_without_fake_essential_data() -> None:
    feedback, current, heat, mesh, current_parameters, heat_parameters, feedback_parameters = (
        _case()
    )
    current = replace(current, flux_values=(1.25,))
    heat = replace(heat, flux_values=(-2.5,))
    mesh = replace(
        mesh,
        boundaries=(
            *mesh.boundaries,
            ElmerCoupledBoundary(
                boundary_id=99,
                potential_index=None,
                current_flux_index=0,
                temperature_index=None,
                heat_flux_index=0,
            ),
        ),
    )

    sif = render_self_consistent_sif(
        feedback,
        current,
        heat,
        mesh,
        current_parameters,
        heat_parameters,
        feedback_parameters,
        stat_current_module=Path("/locked/StatCurrentSolve.so"),
        heat_solve_module=Path("/locked/HeatSolve.so"),
        convergence_tolerance=1.0e-12,
    )

    assert "Target Boundaries(1) = 99" in sif
    assert "Current Density BC = Logical True" in sif
    assert "Current Density = Real 1.25000000000000000e+00" in sif
    assert "Heat Flux = Real -2.50000000000000000e+00" in sif
