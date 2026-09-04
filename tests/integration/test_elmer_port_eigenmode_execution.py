from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_elmer,
    pytest.mark.requires_gmsh,
]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def test_locked_elmer_solves_one_canonical_gmsh_silicon_waveguide_port(
    locked_gmsh_runner,
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    recipe = RectangularWaveguideCrossSection()
    meshing_directory = tmp_path / "meshing"
    meshing_directory.mkdir()
    (meshing_directory / "waveguide.geo").write_text(recipe.render_geo(), encoding="utf-8")
    meshing = locked_gmsh_runner.run(
        GmshMeshingRequest("waveguide.geo"),
        working_directory=meshing_directory,
        policy=_AUTHORIZED,
    )
    assert meshing.process_succeeded, meshing.stderr
    imported = read_gmsh_msh(
        meshing_directory / "mesh.msh",
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )

    wavelength_m = 1.55e-6
    cladding_index = 1.444
    core_index = 3.48
    problem = Problem(
        "locked-elmer-rectangular-silicon-waveguide-port",
        imported.mesh,
        PortEigenmode(
            regions=(
                IsotropicOpticalRegion("cladding", cladding_index**2),
                IsotropicOpticalRegion("core", core_index**2),
            ),
            perfect_electric_boundaries=tuple(
                PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
            ),
            frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / wavelength_m,
            eigenmode_count=8,
            selected_mode_index=0,
            target_power_w=1.0,
        ),
    )
    run_directory = tmp_path / "elmer-attempt-001"
    prepared = prepare(
        problem,
        locked_elmer_port_backend,
        request=PrepareRequest(run_directory=run_directory),
    )
    solution = solve(
        prepared,
        locked_elmer_port_backend,
        request=SolveRequest(run_directory=run_directory, policy=_AUTHORIZED),
    )

    assert solution.convergence.status is ConvergenceStatus.CONVERGED
    beta = complex(solution.observables["propagation_constant_rad_per_m"])
    effective_index = complex(solution.observables["effective_index"])
    assert beta.real > 0.0
    assert abs(beta.imag) <= 1.0e-10 * beta.real
    assert cladding_index < effective_index.real < core_index
    assert abs(effective_index.imag) <= 1.0e-10
    assert solution.observables["raw_forward_power_W"] > 0.0
    assert solution.observables["target_forward_power_W"] == 1.0
    electric_field = np.asarray(solution.fields["electric_field"].values)
    assert electric_field.shape == (imported.mesh.geometry.node_count, 3)
    anchor = np.unravel_index(int(np.argmax(np.abs(electric_field))), electric_field.shape)
    assert electric_field[anchor].real > 0.0
    assert abs(electric_field[anchor].imag) <= 1.0e-12 * abs(electric_field[anchor])
    assert solution.metadata["field_projection_limitation"].startswith("projected_H1_E_returned")
    assert (run_directory / "port-spectrum.json").is_file()
    assert (run_directory / "mesh/femx-mode_t0001.vtu").is_file()
