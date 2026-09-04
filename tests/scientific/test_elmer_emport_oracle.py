from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from femx.backends.elmer.runner import ElmerCommand, ElmerInstallation, ElmerRunner
from femx.backends.protocol import ExecutionPolicy, PrepareRequest, SolveRequest
from femx.core.problem import Problem
from femx.core.solution import ConvergenceStatus
from femx.meshing.gmsh import (
    GmshMeshingRequest,
    RectangularWaveguideCrossSection,
    read_gmsh_msh,
)
from femx.physics import (
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.runtime import prepare, solve

pytestmark = [pytest.mark.scientific, pytest.mark.requires_elmer]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
_SOURCE_HASHES = {
    "case.sif": "b7ba1dd60aebb5bb63061d1a487fc6875bb3871979f50820aac41eb0aad57cfb",
    "port.grd": "cafaec5dc9e7b3f380d4e7b1b7a4b1430e7c42c6e12ef67b403ac72f20418443",
    "ELMERSOLVER_STARTINFO": ("3ac39d30e3fabb5f926a646b635279bdb79955ae3efac437195f9fb85a591c2b"),
}


def _configured_elmer_solver() -> Path:
    configured = os.environ.get("FEMX_ELMER_EXECUTABLE")
    if configured is not None:
        path = Path(configured)
        if not path.is_absolute():
            pytest.fail("FEMX_ELMER_EXECUTABLE must be absolute")
        return path.resolve()
    discovered = shutil.which("ElmerSolver")
    if discovered is None:
        pytest.skip("ElmerSolver is not available on PATH")
    return Path(discovered).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_locked_elmer_reproduces_its_registered_emport_reference(
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    del locked_elmer_port_backend  # fixture verifies the full locked runtime identity first
    repository_root = Path(__file__).resolve().parents[2]
    source = repository_root.parent / "elmerfem/fem/tests/EM_port_eigen"
    if not source.is_dir():
        pytest.skip(f"locked Elmer source test is absent: {source}")
    for filename, expected_hash in _SOURCE_HASHES.items():
        source_path = source / filename
        assert _sha256(source_path) == expected_hash
        shutil.copyfile(source_path, tmp_path / filename)

    solver = _configured_elmer_solver()
    grid = solver.parent / "ElmerGrid"
    if not grid.is_file():
        pytest.skip(f"matching ElmerGrid is absent: {grid}")
    elmer_home = solver.parent.parent
    module_directory = elmer_home / "share/elmersolver/lib"
    environment = {
        "ELMER_HOME": str(elmer_home),
        "ELMER_LIB": str(module_directory),
        "ELMER_MODULES_PATH": str(module_directory),
    }
    grid_result = ElmerRunner(ElmerInstallation(grid.resolve())).run(
        ElmerCommand(
            arguments=("1", "2", "port.grd"),
            environment=environment,
            timeout_seconds=120.0,
        ),
        working_directory=tmp_path,
        policy=_AUTHORIZED,
    )
    assert grid_result.process_succeeded, grid_result.stderr
    solver_result = ElmerRunner(ElmerInstallation(solver)).run(
        ElmerCommand(environment=environment, timeout_seconds=300.0),
        working_directory=tmp_path,
        policy=_AUTHORIZED,
    )
    assert solver_result.process_succeeded, solver_result.stderr
    assert "MAIN: *** Elmer Solver: ALL DONE ***" in solver_result.stdout
    assert (tmp_path / "TEST.PASSED").is_file()

    eigenvalue_match = re.search(
        r"^EigenSolveComplex:\s+1\s+\(\s*([^,]+),\s*([^\)]+)\)",
        solver_result.stdout,
        re.MULTILINE,
    )
    beta_match = re.search(
        r"SaveScalars:\s+\d+:\s+res:\s+port beta 1\s+(\S+)",
        solver_result.stdout,
    )
    residual_matches = re.findall(
        r"^CheckResidualsComplex:\s+L\^2 Norm of the residual:\s+\d+\s+(\S+)",
        solver_result.stdout,
        re.MULTILINE,
    )
    assert eigenvalue_match is not None
    assert beta_match is not None
    assert len(residual_matches) == 30
    eigenvalue = complex(float(eigenvalue_match.group(1)), float(eigenvalue_match.group(2)))
    beta = float(beta_match.group(1))
    residuals = np.asarray([float(value) for value in residual_matches])

    assert eigenvalue.real == pytest.approx(-14.647970391372228, abs=2.0e-11)
    assert abs(eigenvalue.imag) <= 1.0e-11
    assert beta == pytest.approx(3.82726670, abs=5.0e-9)
    assert beta**2 == pytest.approx(-eigenvalue.real, rel=2.0e-9)
    assert beta / 3.0 == pytest.approx(1.27576, abs=1.0e-5)
    assert np.max(residuals) <= 1.0e-10


def _representative_silicon_port(mesh, *, name: str) -> Problem:
    return Problem(
        name,
        mesh,
        PortEigenmode(
            regions=(
                IsotropicOpticalRegion("cladding", 1.444**2),
                IsotropicOpticalRegion("core", 3.48**2),
            ),
            perfect_electric_boundaries=tuple(
                PerfectElectricBoundary(boundary) for boundary in ("bottom", "right", "top", "left")
            ),
            frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
            eigenmode_count=8,
            selected_mode_index=0,
            target_power_w=1.0,
        ),
    )


@pytest.mark.requires_gmsh
def test_elmer_rectangular_waveguide_effective_index_converges_under_mesh_halving(
    locked_gmsh_runner,
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    effective_indices: list[complex] = []
    node_counts: list[int] = []
    for level, factor in (("coarse", 1.0), ("medium", 0.5), ("fine", 0.25)):
        level_directory = tmp_path / level
        level_directory.mkdir()
        recipe = RectangularWaveguideCrossSection(
            cladding_mesh_size_m=0.44e-6 * factor,
            core_mesh_size_m=0.09e-6 * factor,
        )
        (level_directory / "waveguide.geo").write_text(recipe.render_geo(), encoding="utf-8")
        meshing = locked_gmsh_runner.run(
            GmshMeshingRequest("waveguide.geo"),
            working_directory=level_directory,
            policy=_AUTHORIZED,
        )
        assert meshing.process_succeeded, meshing.stderr
        imported = read_gmsh_msh(
            level_directory / "mesh.msh",
            coordinate_scale_to_m=recipe.coordinate_scale_to_m,
        )
        node_counts.append(imported.mesh.geometry.node_count)
        run_directory = level_directory / "elmer-attempt-001"
        prepared = prepare(
            _representative_silicon_port(imported.mesh, name=f"refinement-{level}"),
            locked_elmer_port_backend,
            request=PrepareRequest(run_directory=run_directory),
        )
        solution = solve(
            prepared,
            locked_elmer_port_backend,
            request=SolveRequest(run_directory=run_directory, policy=_AUTHORIZED),
        )
        assert solution.convergence.status is ConvergenceStatus.CONVERGED
        effective_indices.append(complex(solution.observables["effective_index"]))

    assert node_counts[0] < node_counts[1] < node_counts[2]
    real_indices = np.asarray([value.real for value in effective_indices])
    assert np.all(np.diff(real_indices) > 0.0)
    coarse_change = abs(real_indices[1] - real_indices[0])
    fine_change = abs(real_indices[2] - real_indices[1])
    observed_order = math.log2(coarse_change / fine_change)
    assert fine_change < 0.35 * coarse_change
    assert observed_order > 1.5
    assert real_indices[-1] == pytest.approx(2.44731439529414, abs=5.0e-8)
    assert max(abs(value.imag) for value in effective_indices) <= 1.0e-12
