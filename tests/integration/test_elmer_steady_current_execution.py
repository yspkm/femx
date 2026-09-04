import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends.protocol import ExecutionPolicy, PrepareRequest, SolveRequest
from femx.core.problem import Problem
from femx.physics import (
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.runtime import prepare, solve

pytestmark = [pytest.mark.integration, pytest.mark.requires_elmer]


def test_generated_current_case_executes_and_preserves_flux_sign(
    locked_elmer_current_backend,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    shadow_modules = tmp_path / "untrusted-modules"
    shadow_modules.mkdir()
    (shadow_modules / "StatCurrentSolve.so").write_bytes(b"must never be loaded")
    monkeypatch.setenv("ELMER_LIB", str(shadow_modules))
    monkeypatch.setenv("ELMER_MODULES_PATH", str(shadow_modules))
    mesh = structured_unit_square_mesh(2)
    problem = Problem(
        "elmer-linear-current-flux",
        mesh,
        SteadyCurrent(
            regions=(ConductiveRegion("domain", electric_conductivity=2.0),),
            potential_boundaries=(PotentialBoundary("left", 0.0),),
            current_flux_boundaries=(CurrentFluxBoundary("right", 2.0),),
        ),
    )
    run_directory = tmp_path / "attempt-001"
    solution = solve(
        prepare(
            problem,
            locked_elmer_current_backend,
            request=PrepareRequest(run_directory=run_directory),
        ),
        locked_elmer_current_backend,
        request=SolveRequest(
            run_directory=run_directory,
            policy=ExecutionPolicy(execution_authorized=True, allow_external_process=True),
        ),
    )

    coordinates = np.asarray(mesh.geometry.coordinates)
    np.testing.assert_allclose(
        solution.fields["potential"].values,
        coordinates[:, 0],
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(solution.fields["current_density"].values[:, 0], -2.0, atol=4e-14)
    np.testing.assert_allclose(solution.fields["joule_heat_density"].values, 2.0, atol=5e-14)
    assert solution.convergence.status.value == "converged"
    case = (run_directory / "case.sif").read_text(encoding="utf-8")
    assert "Current Density = Real 2.00000000000000000e+00" in case
    assert (
        f'Procedure = File "{solution.metadata["elmer_stat_current_solve_module"]}" '
        '"StatCurrentSolver"' in case
    )
    assert (run_directory / "mesh" / "femx.result").is_file()
    assert (run_directory / "mesh" / "femx.vtu").is_file()
