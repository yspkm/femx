from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from femx.backends.elmer.case import lower_tagged_scalar_h1_mesh  # noqa: E402
from femx.backends.jax.steady_heat import (  # noqa: E402
    JaxSteadyHeatBackend,
    PreparedSteadyHeat,
)
from femx.backends.protocol import PrepareRequest  # noqa: E402
from femx.core.execution import ExecutionPolicy  # noqa: E402
from femx.core.problem import Problem  # noqa: E402
from femx.meshing.gmsh import (  # noqa: E402
    GmshMeshingRequest,
    RectangularWaveguideCrossSection,
    read_gmsh_msh,
)
from femx.physics import (  # noqa: E402
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.runtime import prepare  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_gmsh,
    pytest.mark.requires_jax,
]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def test_real_gmsh_repeat_is_one_canonical_jax_elmer_mesh(locked_gmsh_runner, tmp_path) -> None:
    recipe = RectangularWaveguideCrossSection()
    attempts = []
    for attempt_number in (1, 2):
        attempt = tmp_path / f"attempt-{attempt_number:03d}"
        attempt.mkdir()
        (attempt / "waveguide.geo").write_text(recipe.render_geo(), encoding="utf-8")
        process = locked_gmsh_runner.run(
            GmshMeshingRequest("waveguide.geo"),
            working_directory=attempt,
            policy=_AUTHORIZED,
        )
        assert process.process_succeeded, process.stderr
        assert process.mesh_sha256 is not None
        imported = read_gmsh_msh(
            attempt / "mesh.msh",
            coordinate_scale_to_m=recipe.coordinate_scale_to_m,
        )
        attempts.append((process, imported))

    first_process, first = attempts[0]
    second_process, second = attempts[1]
    assert first_process.identity == second_process.identity
    assert first_process.geometry_sha256 == second_process.geometry_sha256
    assert first_process.mesh_sha256 == second_process.mesh_sha256
    assert first.record.digest() == second.record.digest()
    assert first.record.canonical_mesh_sha256 == second.record.canonical_mesh_sha256

    mesh = first.mesh
    assert tuple(tag.name for tag in mesh.tags) == (
        "bottom",
        "cladding",
        "core",
        "left",
        "right",
        "top",
    )
    assert all(mesh.tag(name).entity_ids for name in ("cladding", "core"))
    assert all(mesh.tag(name).entity_ids for name in ("bottom", "right", "top", "left"))

    problem = Problem(
        "gmsh-waveguide-handoff",
        mesh,
        SteadyHeat(
            regions=(
                ThermalRegion("cladding", 1.0),
                ThermalRegion("core", 2.0),
            ),
            temperature_boundaries=(
                TemperatureBoundary("left", 0.0),
                TemperatureBoundary("right", 0.0),
            ),
            heat_flux_boundaries=(
                HeatFluxBoundary("bottom", 0.0),
                HeatFluxBoundary("top", 0.0),
            ),
        ),
    )
    jax_prepared = prepare(problem, JaxSteadyHeatBackend(), request=PrepareRequest())
    assert isinstance(jax_prepared.payload, PreparedSteadyHeat)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(jax_prepared.payload.coordinates)),
        np.asarray(mesh.geometry.coordinates),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(jax_prepared.payload.cells)),
        np.asarray(mesh.topology.connectivity),
    )

    elmer = lower_tagged_scalar_h1_mesh(
        mesh,
        region_tags=("cladding", "core"),
        essential_boundary_tags=("left", "right"),
        natural_boundary_tags=("bottom", "top"),
    )
    elmer_coordinates = np.asarray(
        [[float(field) for field in line.split()[2:4]] for line in elmer.nodes.splitlines()]
    )
    elmer_cells = np.asarray(
        [[int(field) - 1 for field in line.split()[3:6]] for line in elmer.elements.splitlines()]
    )
    np.testing.assert_array_equal(elmer_coordinates, np.asarray(mesh.geometry.coordinates))
    np.testing.assert_array_equal(elmer_cells, np.asarray(mesh.topology.connectivity))

    body_ids = np.asarray([int(line.split()[1]) for line in elmer.elements.splitlines()])
    np.testing.assert_array_equal(body_ids[list(mesh.tag("cladding").entity_ids)], 1)
    np.testing.assert_array_equal(body_ids[list(mesh.tag("core").entity_ids)], 2)
    boundary_ids_by_edge = {
        tuple(sorted((int(fields[-2]) - 1, int(fields[-1]) - 1))): int(fields[1])
        for fields in (line.split() for line in elmer.boundary.splitlines())
    }
    boundary_facets = np.asarray(mesh.boundary_facets.connectivity)
    for expected_id, name in enumerate(("left", "right", "bottom", "top"), start=1):
        for facet_id in mesh.tag(name).entity_ids:
            edge = tuple(sorted(int(node) for node in boundary_facets[facet_id]))
            assert boundary_ids_by_edge[edge] == expected_id
