from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.support import DummyPhysics, FakeArray, structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import femx.backends.jax.steady_heat as backend_module  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.errors import BackendError, CapabilityError, ContractError  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem  # noqa: E402
from femx.mesh import (  # noqa: E402
    CellType,
    EntityTag,
    Mesh,
    MeshGeometry,
    MeshTopology,
)
from femx.physics import (  # noqa: E402
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _basic_physics() -> SteadyHeat:
    return SteadyHeat(
        regions=(ThermalRegion("domain", 1.0),),
        temperature_boundaries=(TemperatureBoundary("left", 0.0),),
    )


def _problem(
    *,
    mesh: Mesh | None = None,
    physics: SteadyHeat | None = None,
    parameters: ParameterSchema | None = None,
) -> Problem:
    return Problem(
        "jax-heat",
        structured_unit_square_mesh(2) if mesh is None else mesh,
        _basic_physics() if physics is None else physics,
        parameters=ParameterSchema() if parameters is None else parameters,
    )


def _rebuild_mesh(
    mesh: Mesh,
    *,
    coordinates: object | None = None,
    cells: object | None = None,
    facets: object | None = None,
    cell_type: CellType | None = None,
    tags: tuple[EntityTag, ...] | None = None,
    include_facets: bool = True,
) -> Mesh:
    actual_coordinates = mesh.geometry.coordinates if coordinates is None else coordinates
    actual_cells = mesh.topology.connectivity if cells is None else cells
    actual_facets = (
        mesh.boundary_facets.connectivity
        if facets is None and mesh.boundary_facets is not None
        else facets
    )
    node_count = int(actual_coordinates.shape[0])
    return Mesh(
        geometry=MeshGeometry(actual_coordinates),
        topology=MeshTopology(
            actual_cells,
            mesh.topology.cell_type if cell_type is None else cell_type,
            node_count,
        ),
        tags=mesh.tags if tags is None else tags,
        boundary_facets=(
            MeshTopology(actual_facets, CellType.SEGMENT, node_count)
            if include_facets and actual_facets is not None
            else None
        ),
    )


def test_parameterized_backend_lifecycle_reproduces_linear_solution() -> None:
    mesh = structured_unit_square_mesh(3)
    physics = SteadyHeat(
        regions=(
            ThermalRegion(
                "domain",
                ParameterReference("conductivity"),
                ParameterReference("source"),
            ),
        ),
        temperature_boundaries=(TemperatureBoundary("left", 0.0),),
        heat_flux_boundaries=(HeatFluxBoundary("right", ParameterReference("boundary_load")),),
    )
    schema = ParameterSchema(
        (
            ParameterSpec("conductivity", unit="W/(m*K)", lower_bound=0.1),
            ParameterSpec("source", unit="W/m^3"),
            ParameterSpec("boundary_load", unit="W/m^2"),
        )
    )
    problem = _problem(mesh=mesh, physics=physics, parameters=schema)
    values = schema.bind({"conductivity": 2.0, "source": 0.0, "boundary_load": 2.0})
    backend = JaxSteadyHeatBackend()

    solution = solve(
        prepare(problem, backend, request=PrepareRequest()),
        backend,
        request=SolveRequest(parameters=values),
    )

    coordinates = np.asarray(mesh.geometry.coordinates)
    np.testing.assert_allclose(
        solution.fields["temperature"].values,
        coordinates[:, 0],
        atol=2.0e-14,
        rtol=0.0,
    )
    assert solution.convergence.status.value == "converged"
    assert solution.observables["variational_heat_load_W_per_m"] == pytest.approx(2.0)
    assert solution.metadata["precision"] == "float64"
    assert solution.metadata["dof_count"] == str(coordinates.shape[0])


def test_backend_reports_nonconvergence_without_relabeling_process_success() -> None:
    backend = JaxSteadyHeatBackend(relative_residual_tolerance=1.0e-30)
    problem = _problem(
        physics=SteadyHeat(
            regions=(ThermalRegion("domain", 1.0, 2.0),),
            temperature_boundaries=(TemperatureBoundary("left", 0.0),),
        )
    )

    solution = solve(prepare(problem, backend), backend)

    assert solution.convergence.status.value == "not_converged"
    assert solution.convergence.residual_norm is not None
    assert solution.convergence.residual_norm > 1.0e-30


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1.0, "positive value"),
        (complex(1.0, 1.0), "finite real scalars"),
        (float("nan"), "finite real scalars"),
        (True, "finite real scalars"),
        (np.asarray((1.0,)), "finite real scalars"),
    ],
)
def test_backend_rejects_nonphysical_resolved_conductivity(value: object, message: str) -> None:
    physics = SteadyHeat(
        (ThermalRegion("domain", ParameterReference("conductivity")),),
        (TemperatureBoundary("left", 0.0),),
    )
    schema = ParameterSchema((ParameterSpec("conductivity", unit="W/(m*K)"),))
    backend = JaxSteadyHeatBackend()
    prepared = backend.prepare(_problem(physics=physics, parameters=schema), PrepareRequest())

    with pytest.raises(ContractError, match=message):
        backend.solve(
            prepared,
            SolveRequest(parameters=ParameterValues({"conductivity": value})),
        )


def test_backend_validates_tolerance_and_advertises_only_tested_gradients() -> None:
    with pytest.raises(ContractError, match="tolerance"):
        JaxSteadyHeatBackend(relative_residual_tolerance=0.0)

    adjoint = replace(_basic_physics(), gradient_method=backend_module.GradientMethod.ADJOINT)
    prepared = prepare(_problem(physics=adjoint), JaxSteadyHeatBackend())
    assert prepared.problem.physics.requirements.gradient is backend_module.GradientMethod.ADJOINT

    forward = replace(_basic_physics(), gradient_method=backend_module.GradientMethod.FORWARD)
    with pytest.raises(CapabilityError, match="gradient=forward"):
        prepare(_problem(physics=forward), JaxSteadyHeatBackend())


def test_backend_requires_cpu_and_explicit_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = JaxSteadyHeatBackend()
    problem = _problem()
    monkeypatch.setattr(backend_module.jax, "default_backend", lambda: "gpu")
    with pytest.raises(BackendError, match="validated only on CPU"):
        backend.prepare(problem, PrepareRequest())

    monkeypatch.setattr(
        backend_module,
        "jax",
        SimpleNamespace(
            default_backend=lambda: "cpu",
            config=SimpleNamespace(x64_enabled=False),
        ),
    )
    with pytest.raises(BackendError, match="float64"):
        backend.prepare(problem, PrepareRequest())


def test_backend_rejects_wrong_problem_and_prepared_payload_types() -> None:
    backend = JaxSteadyHeatBackend()
    with pytest.raises(ContractError, match="SteadyHeat"):
        backend.prepare(
            Problem("wrong-physics", structured_unit_square_mesh(1), DummyPhysics()),
            PrepareRequest(),
        )

    valid_problem = _problem()
    invalid_prepared = PreparedProblem(backend.descriptor, valid_problem, payload=object())
    with pytest.raises(BackendError, match="payload"):
        backend.solve(invalid_prepared, SolveRequest())


def test_backend_rejects_structural_mesh_without_concrete_mesh_type() -> None:
    class StructuralMesh:
        geometry = object()
        topology = object()
        schema_version = "femx.mesh/v1"

    backend = JaxSteadyHeatBackend()
    problem = Problem("structural", StructuralMesh(), _basic_physics())
    with pytest.raises(ContractError, match="concrete femx Mesh"):
        backend.prepare(problem, PrepareRequest())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda mesh: _rebuild_mesh(
                mesh,
                coordinates=np.column_stack(
                    (np.asarray(mesh.geometry.coordinates), np.zeros(mesh.geometry.node_count))
                ),
            ),
            "two-dimensional",
        ),
        (
            lambda mesh: Mesh(
                mesh.geometry,
                MeshTopology(FakeArray((1, 4)), CellType.QUADRILATERAL, mesh.geometry.node_count),
                tags=mesh.tags,
                boundary_facets=mesh.boundary_facets,
            ),
            "P1 triangle",
        ),
        (lambda mesh: _rebuild_mesh(mesh, include_facets=False), "explicit boundary facets"),
        (
            lambda mesh: _rebuild_mesh(
                mesh,
                coordinates=np.asarray(mesh.geometry.coordinates).astype(float)
                * np.asarray((np.nan, 1.0)),
            ),
            "finite",
        ),
        (
            lambda mesh: _rebuild_mesh(
                mesh,
                coordinates=np.asarray(mesh.geometry.coordinates, dtype=np.float32),
            ),
            "exact float64 dtype",
        ),
        (
            lambda mesh: _rebuild_mesh(
                mesh,
                coordinates=np.asarray(mesh.geometry.coordinates, dtype=np.complex128) + 1.0j,
            ),
            "exact float64 dtype",
        ),
        (
            lambda mesh: _rebuild_mesh(
                mesh, cells=np.asarray(mesh.topology.connectivity, dtype=float)
            ),
            "integer dtype",
        ),
    ],
)
def test_backend_rejects_invalid_mesh_classes(mutator, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        JaxSteadyHeatBackend().prepare(
            _problem(mesh=mutator(structured_unit_square_mesh(1))), PrepareRequest()
        )


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    [
        ("cells", np.asarray(((0, 1, 9), (0, 2, 3)), dtype=np.int32), "out-of-range"),
        ("cells", np.asarray(((0, 1, 1), (0, 2, 3)), dtype=np.int32), "repeated node"),
        ("facets", np.asarray(((0, 0), (1, 2), (2, 3), (3, 0)), dtype=np.int32), "repeated node"),
        ("facets", np.asarray(((0, 1), (1, 9), (2, 3), (3, 0)), dtype=np.int32), "out-of-range"),
        (
            "facets",
            np.asarray(((0, 1), (0, 1), (2, 3), (3, 0)), dtype=np.int32),
            "duplicate facets",
        ),
        ("facets", np.asarray(((0, 1), (1, 3), (2, 3), (3, 0)), dtype=np.int32), "mesh boundary"),
    ],
)
def test_backend_rejects_invalid_connectivity(
    target: str, replacement: np.ndarray, message: str
) -> None:
    mesh = structured_unit_square_mesh(1)
    kwargs = {target: replacement}
    with pytest.raises(ContractError, match=message):
        JaxSteadyHeatBackend().prepare(
            _problem(mesh=_rebuild_mesh(mesh, **kwargs)),
            PrepareRequest(),
        )


def test_backend_rejects_numerically_singular_triangle() -> None:
    mesh = structured_unit_square_mesh(1)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    coordinates[3] = (0.5, 0.0)
    with pytest.raises(ContractError, match="singular triangle"):
        JaxSteadyHeatBackend().prepare(
            _problem(mesh=_rebuild_mesh(mesh, coordinates=coordinates)),
            PrepareRequest(),
        )


def test_backward_error_is_invariant_to_uniform_equation_scaling() -> None:
    backend = JaxSteadyHeatBackend(relative_residual_tolerance=1.0e-20)

    def solve_scaled(scale: float):
        return solve(
            prepare(
                _problem(
                    physics=SteadyHeat(
                        regions=(ThermalRegion("domain", scale, 2.0 * scale),),
                        temperature_boundaries=(TemperatureBoundary("left", 0.0),),
                    )
                ),
                backend,
            ),
            backend,
        )

    unscaled = solve_scaled(1.0)
    scaled = solve_scaled(1.0e-10)

    assert scaled.convergence.status is unscaled.convergence.status
    assert scaled.convergence.residual_norm is not None
    assert unscaled.convergence.residual_norm is not None
    roundoff_ratio = scaled.convergence.residual_norm / unscaled.convergence.residual_norm
    assert 0.1 < roundoff_ratio < 10.0


def test_backend_rejects_non_manifold_triangle_connectivity() -> None:
    mesh = structured_unit_square_mesh(1)
    cells = np.asarray(((0, 1, 3), (0, 3, 2), (0, 3, 1)), dtype=np.int32)
    with pytest.raises(ContractError, match="non-manifold"):
        JaxSteadyHeatBackend().prepare(
            _problem(mesh=_rebuild_mesh(mesh, cells=cells)),
            PrepareRequest(),
        )


@pytest.mark.parametrize(
    ("physics", "parameters", "message"),
    [
        (
            SteadyHeat(
                (ThermalRegion("domain", ParameterReference("k")),),
                (TemperatureBoundary("left", 0.0),),
            ),
            ParameterSchema(),
            "do not match",
        ),
        (
            SteadyHeat(
                (ThermalRegion("domain", ParameterReference("k")),),
                (TemperatureBoundary("left", 0.0),),
            ),
            ParameterSchema((ParameterSpec("k", unit="K"),)),
            "must be a scalar with unit",
        ),
        (
            SteadyHeat(
                (
                    ThermalRegion(
                        "domain", ParameterReference("shared"), ParameterReference("shared")
                    ),
                ),
                (TemperatureBoundary("left", 0.0),),
            ),
            ParameterSchema((ParameterSpec("shared", unit="W/(m*K)"),)),
            "incompatible units",
        ),
    ],
)
def test_backend_rejects_parameter_schema_mismatches(
    physics: SteadyHeat, parameters: ParameterSchema, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        JaxSteadyHeatBackend().prepare(
            _problem(physics=physics, parameters=parameters),
            PrepareRequest(),
        )


def test_backend_rejects_bad_or_ambiguous_entity_tags() -> None:
    mesh = structured_unit_square_mesh(1)
    backend = JaxSteadyHeatBackend()

    wrong_dimension = replace(
        mesh,
        tags=tuple(
            EntityTag("domain", 1, tag.entity_ids) if tag.name == "domain" else tag
            for tag in mesh.tags
        ),
    )
    with pytest.raises(ContractError, match="dimension"):
        backend.prepare(_problem(mesh=wrong_dimension), PrepareRequest())

    empty_domain = replace(
        mesh,
        tags=tuple(
            EntityTag("domain", 2, ()) if tag.name == "domain" else tag for tag in mesh.tags
        ),
    )
    with pytest.raises(ContractError, match="cannot be empty"):
        backend.prepare(_problem(mesh=empty_domain), PrepareRequest())

    out_of_range = replace(
        mesh,
        tags=tuple(
            EntityTag("domain", 2, (99,)) if tag.name == "domain" else tag for tag in mesh.tags
        ),
    )
    with pytest.raises(ContractError, match="out-of-range entity"):
        backend.prepare(_problem(mesh=out_of_range), PrepareRequest())

    missing_region = replace(_basic_physics(), regions=(ThermalRegion("missing", 1.0),))
    with pytest.raises(ContractError, match="does not define"):
        backend.prepare(_problem(mesh=mesh, physics=missing_region), PrepareRequest())


def test_backend_rejects_region_flux_and_temperature_overlap() -> None:
    mesh = structured_unit_square_mesh(1)
    domain_ids = mesh.tag("domain").entity_ids
    right_ids = mesh.tag("right").entity_ids
    tags = (
        *mesh.tags,
        EntityTag("domain_alias", 2, domain_ids),
        EntityTag("right_alias", 1, right_ids),
    )
    tagged_mesh = replace(mesh, tags=tags)
    backend = JaxSteadyHeatBackend()

    regions = SteadyHeat(
        (ThermalRegion("domain", 1.0), ThermalRegion("domain_alias", 2.0)),
        (TemperatureBoundary("left", 0.0),),
    )
    with pytest.raises(ContractError, match="partition every cell"):
        backend.prepare(_problem(mesh=tagged_mesh, physics=regions), PrepareRequest())

    fluxes = SteadyHeat(
        (ThermalRegion("domain", 1.0),),
        (TemperatureBoundary("left", 0.0),),
        (HeatFluxBoundary("right", 1.0), HeatFluxBoundary("right_alias", 1.0)),
    )
    with pytest.raises(ContractError, match="cannot overlap"):
        backend.prepare(_problem(mesh=tagged_mesh, physics=fluxes), PrepareRequest())

    overlapping_temperatures = SteadyHeat(
        (ThermalRegion("domain", 1.0),),
        (TemperatureBoundary("right", 0.0), TemperatureBoundary("right_alias", 0.0)),
    )
    with pytest.raises(ContractError, match="temperature boundary tags cannot overlap"):
        backend.prepare(
            _problem(mesh=tagged_mesh, physics=overlapping_temperatures), PrepareRequest()
        )

    mixed = SteadyHeat(
        (ThermalRegion("domain", 1.0),),
        (TemperatureBoundary("right", 0.0),),
        (HeatFluxBoundary("right_alias", 1.0),),
    )
    with pytest.raises(ContractError, match="both temperature and heat flux"):
        backend.prepare(_problem(mesh=tagged_mesh, physics=mixed), PrepareRequest())

    temperatures = SteadyHeat(
        (ThermalRegion("domain", 1.0),),
        (TemperatureBoundary("left", 0.0), TemperatureBoundary("bottom", 1.0)),
    )
    with pytest.raises(ContractError, match="conflict at mesh node"):
        backend.prepare(_problem(mesh=mesh, physics=temperatures), PrepareRequest())
