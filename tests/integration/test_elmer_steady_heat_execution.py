import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends.protocol import ExecutionPolicy, PrepareRequest, SolveRequest
from femx.core.problem import Problem
from femx.physics import HeatFluxBoundary, SteadyHeat, TemperatureBoundary, ThermalRegion
from femx.runtime import prepare, solve

pytestmark = [pytest.mark.integration, pytest.mark.requires_elmer]


def test_generated_case_executes_and_preserves_heat_load_sign(
    locked_elmer_backend, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    shadow_modules = tmp_path / "untrusted-modules"
    shadow_modules.mkdir()
    (shadow_modules / "HeatSolve.so").write_bytes(b"must never be loaded")
    monkeypatch.setenv("ELMER_LIB", str(shadow_modules))
    monkeypatch.setenv("ELMER_MODULES_PATH", str(shadow_modules))
    mesh = structured_unit_square_mesh(2)
    problem = Problem(
        "elmer-linear-flux",
        mesh,
        SteadyHeat(
            regions=(ThermalRegion("domain", conductivity=2.0),),
            temperature_boundaries=(TemperatureBoundary("left", 0.0),),
            heat_flux_boundaries=(HeatFluxBoundary("right", 2.0),),
        ),
    )
    run_directory = tmp_path / "attempt-001"
    prepared = prepare(
        problem,
        locked_elmer_backend,
        request=PrepareRequest(run_directory=run_directory),
    )
    solution = solve(
        prepared,
        locked_elmer_backend,
        request=SolveRequest(
            run_directory=run_directory,
            policy=ExecutionPolicy(execution_authorized=True, allow_external_process=True),
        ),
    )

    coordinates = np.asarray(mesh.geometry.coordinates)
    np.testing.assert_allclose(
        solution.fields["temperature"].values,
        coordinates[:, 0],
        rtol=0.0,
        atol=2.0e-14,
    )
    assert solution.convergence.status.value == "converged"
    assert "Heat Flux = Real 2.00000000000000000e+00" in (run_directory / "case.sif").read_text(
        encoding="utf-8"
    )
    assert f'Procedure = File "{solution.metadata["elmer_heat_solve_module"]}" "HeatSolver"' in (
        run_directory / "case.sif"
    ).read_text(encoding="utf-8")
    assert (run_directory / "mesh" / "femx.result").is_file()
    assert (run_directory / "mesh" / "femx.vtu").is_file()
