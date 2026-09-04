from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.elmer.port_eigenmode import PreparedElmerPortEigenmode  # noqa: E402
from femx.backends.elmer.port_result import (  # noqa: E402
    read_port_eigenmode_result,
    reorder_elmer_edge_coefficients,
)
from femx.backends.jax.port_eigenmode import (  # noqa: E402
    JaxPortEigenmodeBackend,
    PreparedJaxPortEigenmode,
)
from femx.backends.jax.port_eigensolver import (  # noqa: E402
    compare_port_mode_subspaces,
    solve_dense_port_eigenmodes,
)
from femx.backends.jax.port_krylov import (  # noqa: E402
    MatrixFreePortArnoldiPolicy,
    solve_matrix_free_port_eigenmodes,
)
from femx.backends.jax.port_matrix_free import (  # noqa: E402
    MatrixFreePortSolvePolicy,
    build_lossless_matrix_free_port_pencil,
    prepare_port_matrix_free_topology,
)
from femx.backends.jax.port_operator import (  # noqa: E402
    assemble_lossless_port_pencil,
    lossless_port_coefficients,
    reduce_port_pencil,
)
from femx.backends.jax.port_projection import (  # noqa: E402
    project_port_electric_field_to_nodes,
    project_port_electromagnetic_fields_to_nodes,
)
from femx.backends.protocol import ExecutionPolicy, PrepareRequest, SolveRequest  # noqa: E402
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXFingerprint,
    SolverFingerprint,
    build_yee_grid,
    build_yee_port_sampling_plan,
    port_mode_solution_to_bundle,
    read_mode_bundle_hdf5,
    sample_port_mode_to_yee,
    write_mode_bundle_hdf5,
)
from femx.meshing.gmsh import (  # noqa: E402
    GmshMeshingRequest,
    RectangularWaveguideCrossSection,
    read_gmsh_msh,
)
from femx.physics import (  # noqa: E402
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_gmsh,
    pytest.mark.requires_jax,
]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
_MODE_COUNT = 8
_FDTDX_FINGERPRINT = FDTDXFingerprint(
    "0.6.2",
    "eaab78a42cd1351b7f447f312fa50c9febfe4b99",
    "cf7bf29a1aa22911e1b3a523a65cf267079562d408238f45c19bb9cd01b51d70",
)


def _cell_materials(validated) -> tuple[np.ndarray, np.ndarray]:
    relative_permittivity = np.empty(validated.cells.shape[0], dtype=np.float64)
    relative_permeability = np.empty_like(relative_permittivity)
    for cell_ids, epsilon_r, mu_r in zip(
        validated.region_cells,
        validated.relative_permittivity,
        validated.relative_permeability,
        strict=True,
    ):
        relative_permittivity[cell_ids] = epsilon_r
        relative_permeability[cell_ids] = mu_r
    return relative_permittivity, relative_permeability


def _mass_norm(values: np.ndarray, mass: np.ndarray) -> float:
    return math.sqrt(float(np.einsum("nc,nk,kc->", values.conj(), mass, values).real))


def test_locked_elmer_and_jax_match_same_mesh_silicon_port_spectrum_and_field(
    locked_gmsh_runner,
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    recipe = RectangularWaveguideCrossSection(
        cladding_mesh_size_m=0.44e-6,
        core_mesh_size_m=0.09e-6,
    )
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
    problem = Problem(
        "locked-elmer-jax-same-mesh-silicon-port",
        imported.mesh,
        PortEigenmode(
            regions=(
                IsotropicOpticalRegion("cladding", 1.444**2),
                IsotropicOpticalRegion("core", 3.48**2),
            ),
            perfect_electric_boundaries=tuple(
                PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
            ),
            frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
            eigenmode_count=_MODE_COUNT,
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
    assert isinstance(prepared.payload, PreparedElmerPortEigenmode)
    elmer_solution = solve(
        prepared,
        locked_elmer_port_backend,
        request=SolveRequest(run_directory=run_directory, policy=_AUTHORIZED),
    )
    validated = prepared.payload.validated
    np.testing.assert_array_equal(validated.coordinates, imported.mesh.geometry.coordinates)
    np.testing.assert_array_equal(validated.cells, imported.mesh.topology.connectivity)

    relative_permittivity, relative_permeability = _cell_materials(validated)
    assembled = assemble_lossless_port_pencil(
        jnp.asarray(validated.coordinates),
        jnp.asarray(validated.cells),
        jnp.asarray(validated.cell_edge_dofs),
        jnp.asarray(validated.edge_signs),
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
        jnp.asarray(validated.frequency_hz),
        edge_dof_count=validated.edge_nodes.shape[0],
    )
    reduced = reduce_port_pencil(
        assembled.stiffness,
        assembled.mass,
        jnp.asarray(validated.dof_partition.free_dofs),
    )
    node_count = validated.coordinates.shape[0]
    scalar_dof_count = int(np.count_nonzero(reduced.full_dofs < node_count))
    angular_frequency = 2.0 * math.pi * validated.frequency_hz
    propagation_scale = angular_frequency * math.sqrt(
        VACUUM_PERMITTIVITY_F_PER_M
        * float(np.max(relative_permittivity))
        * VACUUM_PERMEABILITY_H_PER_M
        * float(np.max(relative_permeability))
    )
    modes = solve_dense_port_eigenmodes(
        reduced.stiffness,
        reduced.mass,
        jnp.asarray(propagation_scale),
        scalar_dof_count=scalar_dof_count,
        mode_count=_MODE_COUNT,
    )

    spectrum = json.loads((run_directory / "port-spectrum.json").read_text(encoding="utf-8"))
    elmer_eigenvalues = np.asarray(
        [complex(value["real"], value["imag"]) for value in spectrum["eigenvalues_per_m2"]]
    )
    elmer_beta = np.sqrt(-elmer_eigenvalues)
    jax_beta = np.asarray(modes.propagation_constants_per_m)
    beta_relative_error = np.abs(jax_beta - elmer_beta) / np.maximum(np.abs(elmer_beta), 1.0)
    assert float(np.max(beta_relative_error)) <= 5.0e-9
    assert float(np.max(np.asarray(modes.residuals.maximum_mixed))) <= 1.0e-12

    matrix_free_topology = prepare_port_matrix_free_topology(
        validated.cells,
        validated.cell_edge_dofs,
        validated.dof_partition.free_dofs,
        node_count=node_count,
        edge_dof_count=validated.edge_nodes.shape[0],
    )
    matrix_free_pencil = build_lossless_matrix_free_port_pencil(
        jnp.asarray(validated.coordinates),
        jnp.asarray(validated.cells),
        jnp.asarray(validated.edge_signs),
        jnp.asarray(matrix_free_topology.cell_reduced_dofs),
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
        jnp.asarray(validated.frequency_hz),
        free_dof_count=matrix_free_topology.free_dof_count,
    )
    matrix_free_modes = solve_matrix_free_port_eigenmodes(
        matrix_free_pencil,
        jnp.asarray(-(propagation_scale**2)),
        jnp.asarray(np.random.default_rng(1801).normal(size=matrix_free_topology.free_dof_count)),
        free_scalar_dof_count=scalar_dof_count,
        mode_count=_MODE_COUNT,
        arnoldi_policy=MatrixFreePortArnoldiPolicy(krylov_dimension=241),
        linear_policy=MatrixFreePortSolvePolicy(
            relative_tolerance=1.0e-11,
            restart=min(160, matrix_free_topology.free_dof_count),
            maximum_restart_cycles=100,
            maximum_relative_residual=5.0e-10,
        ),
    )
    assert bool(matrix_free_modes.diagnostics.is_valid)
    matrix_free_beta = np.asarray(matrix_free_modes.propagation_constants_per_m)
    matrix_free_dense_error = np.abs(matrix_free_beta - jax_beta) / np.maximum(
        np.abs(jax_beta),
        1.0,
    )
    matrix_free_elmer_error = np.abs(matrix_free_beta - elmer_beta) / np.maximum(
        np.abs(elmer_beta),
        1.0,
    )
    assert float(np.max(matrix_free_dense_error)) <= 5.0e-8
    assert float(np.max(matrix_free_elmer_error)) <= 5.0e-8
    assert float(np.max(np.asarray(matrix_free_modes.residuals.maximum_mixed))) <= 5.0e-8

    raw = read_port_eigenmode_result(
        run_directory / "mesh/femx.result",
        expected_node_count=node_count,
        expected_edge_count=validated.edge_nodes.shape[0],
        expected_mode_count=_MODE_COUNT,
    )
    elmer_edges = reorder_elmer_edge_coefficients(
        raw.mixed.edge_coefficients,
        elmer_edge_nodes=prepared.payload.mesh.edge_nodes,
        canonical_edge_nodes=validated.edge_nodes,
    )
    canonical_elmer_mixed = np.vstack((raw.mixed.nodal_coefficients, elmer_edges))
    np.testing.assert_array_equal(
        canonical_elmer_mixed[validated.dof_partition.constrained_dofs],
        0.0,
    )
    free_edge_dofs = np.asarray(reduced.full_dofs)[scalar_dof_count:] - node_count
    comparison = compare_port_mode_subspaces(
        jnp.asarray(elmer_edges[free_edge_dofs]),
        modes.edge_coefficients,
        reduced.mass[scalar_dof_count:, scalar_dof_count:],
    )
    assert float(comparison.projector_distance) <= 1.0e-5
    matrix_free_comparison = compare_port_mode_subspaces(
        modes.edge_coefficients,
        matrix_free_modes.edge_coefficients,
        reduced.mass[scalar_dof_count:, scalar_dof_count:],
    )
    assert float(matrix_free_comparison.projector_distance) <= 1.0e-5

    projected = project_port_electric_field_to_nodes(
        jnp.asarray(validated.coordinates),
        jnp.asarray(validated.cells),
        jnp.asarray(validated.cell_edge_dofs),
        jnp.asarray(validated.edge_signs),
        modes.scalar_coefficients,
        modes.edge_coefficients,
        modes.propagation_constants_per_m,
        np.asarray(reduced.full_dofs),
        edge_dof_count=validated.edge_nodes.shape[0],
    )
    elmer_field = np.asarray(elmer_solution.fields["electric_field"].values)
    jax_field = np.asarray(projected.values[:, 0, :])
    nodal_mass = np.asarray(projected.nodal_mass)
    alignment = np.einsum(
        "nc,nk,kc->",
        jax_field.conj(),
        nodal_mass,
        elmer_field,
    ) / np.einsum("nc,nk,kc->", jax_field.conj(), nodal_mass, jax_field)
    difference = elmer_field - alignment * jax_field
    aligned_relative_error = _mass_norm(difference, nodal_mass) / _mass_norm(
        elmer_field,
        nodal_mass,
    )
    assert aligned_relative_error <= 1.0e-11

    _, cell_reluctivity = lossless_port_coefficients(
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
    )
    free_elmer_mixed = canonical_elmer_mixed[np.asarray(reduced.full_dofs)]
    selected = prepared.payload.validated.selected_mode_index
    elmer_physical = project_port_electromagnetic_fields_to_nodes(
        jnp.asarray(validated.coordinates),
        jnp.asarray(validated.cells),
        jnp.asarray(validated.cell_edge_dofs),
        jnp.asarray(validated.edge_signs),
        jnp.asarray(free_elmer_mixed[:scalar_dof_count, selected : selected + 1]),
        jnp.asarray(free_elmer_mixed[scalar_dof_count:, selected : selected + 1]),
        jnp.asarray(elmer_beta[selected : selected + 1]),
        cell_reluctivity,
        jnp.asarray(angular_frequency),
        np.asarray(reduced.full_dofs),
        edge_dof_count=validated.edge_nodes.shape[0],
    )
    reconstructed_elmer_power = float(elmer_physical.raw_forward_power_w[0])
    printed_elmer_power = float(elmer_solution.observables["raw_forward_power_W"])
    assert abs(reconstructed_elmer_power - printed_elmer_power) / printed_elmer_power <= 2e-12

    jax_backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-12)
    jax_prepared = prepare(problem, jax_backend)
    jax_solution = solve(jax_prepared, jax_backend)
    assert isinstance(jax_prepared.payload, PreparedJaxPortEigenmode)
    assert (
        abs(
            complex(jax_solution.observables["propagation_constant_rad_per_m"])
            - complex(elmer_solution.observables["propagation_constant_rad_per_m"])
        )
        / abs(complex(elmer_solution.observables["propagation_constant_rad_per_m"]))
        <= 5e-9
    )
    public_jax_electric = np.asarray(jax_solution.fields["electric_field"].values)
    direct_difference = elmer_field - public_jax_electric
    direct_relative_error = _mass_norm(direct_difference, nodal_mass) / _mass_norm(
        elmer_field,
        nodal_mass,
    )
    assert direct_relative_error <= 1.0e-10
    public_jax_magnetic = np.asarray(jax_solution.fields["magnetic_field"].values)
    assert np.isfinite(public_jax_magnetic).all()
    assert jax_solution.observables["normalized_forward_power_W"] == pytest.approx(
        validated.target_power_w,
        rel=2e-14,
    )

    coordinate_minimum = np.min(validated.coordinates, axis=0)
    coordinate_maximum = np.max(validated.coordinates, axis=0)
    elmer_refinement_errors: list[float] = []
    jax_refinement_errors: list[float] = []
    for cells_x, cells_y in ((16, 12), (32, 24), (64, 48), (128, 96), (256, 192)):
        refinement_grid = build_yee_grid(
            (
                np.linspace(
                    coordinate_minimum[0] + 1.1e-12,
                    coordinate_maximum[0] - 2.3e-12,
                    cells_x + 1,
                ),
                np.linspace(
                    coordinate_minimum[1] + 1.7e-12,
                    coordinate_maximum[1] - 2.9e-12,
                    cells_y + 1,
                ),
                np.asarray((0.0, 20.0e-9)),
            )
        )
        refinement_plan = build_yee_port_sampling_plan(
            validated.coordinates,
            validated.cells,
            validated.edge_signs,
            refinement_grid,
        )
        assert refinement_plan.ambiguous_target_point_count == 0
        elmer_refinement_samples = sample_port_mode_to_yee(
            refinement_plan,
            elmer_solution.fields[PORT_LONGITUDINAL_POTENTIAL_FIELD].values,
            elmer_solution.fields[PORT_TRANSVERSE_ELECTRIC_FIELD].values,
            complex(elmer_solution.observables["propagation_constant_rad_per_m"]),
            cell_reluctivity,
            angular_frequency,
            validated.target_power_w,
        )
        jax_refinement_samples = sample_port_mode_to_yee(
            refinement_plan,
            jax_solution.fields[PORT_LONGITUDINAL_POTENTIAL_FIELD].values,
            jax_solution.fields[PORT_TRANSVERSE_ELECTRIC_FIELD].values,
            complex(jax_solution.observables["propagation_constant_rad_per_m"]),
            cell_reluctivity,
            angular_frequency,
            validated.target_power_w,
        )
        elmer_pre_power = float(np.asarray(elmer_refinement_samples.pre_correction_power_watts))
        jax_pre_power = float(np.asarray(jax_refinement_samples.pre_correction_power_watts))
        assert elmer_pre_power == pytest.approx(jax_pre_power, rel=1.0e-10, abs=1.0e-12)
        elmer_refinement_errors.append(
            abs(elmer_pre_power - validated.target_power_w) / validated.target_power_w
        )
        jax_refinement_errors.append(
            abs(jax_pre_power - validated.target_power_w) / validated.target_power_w
        )

    for refinement_errors in (elmer_refinement_errors, jax_refinement_errors):
        errors = np.asarray(refinement_errors)
        assert np.all(np.diff(errors) < 0.0)
        observed_orders = np.log2(errors[:-1] / errors[1:])
        assert float(np.min(observed_orders)) >= 1.0
        assert errors[-1] <= 5.0e-3

    yee_grid = build_yee_grid(
        (
            np.linspace(coordinate_minimum[0] + 1.1e-12, coordinate_maximum[0] - 2.3e-12, 34),
            np.linspace(coordinate_minimum[1] + 1.7e-12, coordinate_maximum[1] - 2.9e-12, 26),
            np.asarray((0.0, 20.0e-9)),
        )
    )
    transfer_plan = build_yee_port_sampling_plan(
        validated.coordinates,
        validated.cells,
        validated.edge_signs,
        yee_grid,
    )
    config_sha256 = hashlib.sha256(
        json.dumps(
            problem.physics.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    elmer_bundle = port_mode_solution_to_bundle(
        elmer_solution,
        transfer_plan,
        cell_reluctivity,
        frequency_hz=validated.frequency_hz,
        solver=SolverFingerprint(
            name=elmer_solution.backend_name,
            version=elmer_solution.backend_version,
            config_sha256=str(elmer_solution.metadata["input_sif_sha256"]),
            mesh_sha256=transfer_plan.source_mesh_sha256,
            source_revision=str(elmer_solution.metadata["elmer_source_commit"]),
        ),
        fdtdx=_FDTDX_FINGERPRINT,
    )
    jax_bundle = port_mode_solution_to_bundle(
        jax_solution,
        transfer_plan,
        cell_reluctivity,
        frequency_hz=validated.frequency_hz,
        solver=SolverFingerprint(
            name=jax_solution.backend_name,
            version=jax_solution.backend_version,
            config_sha256=config_sha256,
            mesh_sha256=transfer_plan.source_mesh_sha256,
        ),
        fdtdx=_FDTDX_FINGERPRINT,
    )
    artifact_root = tmp_path / "mode-artifacts"
    artifact_root.mkdir()
    elmer_artifact = write_mode_bundle_hdf5(
        artifact_root,
        "elmer-mode.h5",
        elmer_bundle,
    )
    jax_artifact = write_mode_bundle_hdf5(
        artifact_root,
        "jax-mode.h5",
        jax_bundle,
    )
    elmer_bundle = read_mode_bundle_hdf5(artifact_root, elmer_artifact.reference).bundle
    jax_bundle = read_mode_bundle_hdf5(artifact_root, jax_artifact.reference).bundle
    transferred_elmer_e = np.asarray(elmer_bundle.electric.values)
    transferred_jax_e = np.asarray(jax_bundle.electric.values)
    transferred_elmer_h = np.asarray(elmer_bundle.magnetic.values)
    transferred_jax_h = np.asarray(jax_bundle.magnetic.values)
    electric_relative_error = np.linalg.norm(
        transferred_jax_e - transferred_elmer_e
    ) / np.linalg.norm(transferred_elmer_e)
    magnetic_relative_error = np.linalg.norm(
        transferred_jax_h - transferred_elmer_h
    ) / np.linalg.norm(transferred_elmer_h)
    assert electric_relative_error <= 1.0e-8
    assert magnetic_relative_error <= 1.0e-8
    assert elmer_bundle.transfer.relative_power_error <= 2.0e-14
    assert jax_bundle.transfer.relative_power_error <= 2.0e-14
    assert elmer_bundle.transfer.power_correction_scale == pytest.approx(
        jax_bundle.transfer.power_correction_scale,
        rel=1.0e-8,
    )
    assert transfer_plan.target_grid.shape == (33, 25, 1)
    assert transfer_plan.ambiguous_target_point_count == 0
    assert elmer_bundle.transfer.relative_pre_correction_power_error is not None
    assert elmer_bundle.transfer.relative_pre_correction_power_error <= 0.10


def test_silicon_core_eigen_adjoint_matches_jax_and_elmer_central_differences(
    locked_gmsh_runner,
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    """Validate one actual Si/SiO2 simple-mode derivative through the exact Yee transfer."""

    recipe = RectangularWaveguideCrossSection(
        cladding_mesh_size_m=0.44e-6,
        core_mesh_size_m=0.09e-6,
    )
    meshing_directory = tmp_path / "adjoint-meshing"
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
    core_permittivity = 3.48**2
    parameter_schema = ParameterSchema(
        (
            ParameterSpec(
                "core_relative_permittivity",
                unit="1",
                role=ParameterRole.DESIGN,
                lower_bound=8.0,
                upper_bound=16.0,
            ),
        )
    )

    def problem(gradient_method: GradientMethod) -> Problem:
        return Problem(
            f"silicon-port-core-epsilon-{gradient_method.value}",
            imported.mesh,
            PortEigenmode(
                regions=(
                    IsotropicOpticalRegion("cladding", 1.444**2),
                    IsotropicOpticalRegion(
                        "core",
                        ParameterReference("core_relative_permittivity"),
                    ),
                ),
                perfect_electric_boundaries=tuple(
                    PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
                ),
                frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
                eigenmode_count=_MODE_COUNT,
                selected_mode_index=0,
                target_power_w=1.0,
                gradient_method=gradient_method,
            ),
            parameters=parameter_schema,
        )

    baseline_parameters = ParameterValues({"core_relative_permittivity": core_permittivity})
    jax_backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-12)
    jax_prepared = prepare(problem(GradientMethod.ADJOINT), jax_backend)
    bound = jax_backend.bind_differentiable(jax_prepared, baseline_parameters)
    initial = bound.initial_values

    def beta(active: jax.Array) -> jax.Array:
        return bound.mode(active).propagation_constant_per_m

    jax_beta, jax_adjoint = jax.jit(jax.value_and_grad(beta))(initial)
    step = 2.0e-3
    jax_central_difference = (
        float(beta(initial.at[0].add(step))) - float(beta(initial.at[0].add(-step)))
    ) / (2.0 * step)
    assert float(jax_adjoint[0]) == pytest.approx(jax_central_difference, rel=2.0e-7)

    elmer_prepared = prepare(problem(GradientMethod.NONE), locked_elmer_port_backend)

    def elmer_beta(relative_permittivity: float, label: str) -> float:
        solution = solve(
            elmer_prepared,
            locked_elmer_port_backend,
            request=SolveRequest(
                parameters=ParameterValues({"core_relative_permittivity": relative_permittivity}),
                run_directory=tmp_path / f"elmer-adjoint-{label}",
                policy=_AUTHORIZED,
            ),
        )
        eigenvalue = complex(solution.observables["selected_eigenvalue_per_m2"])
        propagation_constant = np.sqrt(-eigenvalue)
        assert abs(propagation_constant.imag) <= 1.0e-8 * abs(propagation_constant.real)
        return float(propagation_constant.real)

    elmer_minus = elmer_beta(core_permittivity - step, "minus")
    elmer_plus = elmer_beta(core_permittivity + step, "plus")
    elmer_central_difference = (elmer_plus - elmer_minus) / (2.0 * step)
    assert float(jax_beta) == pytest.approx(
        0.5 * (elmer_plus + elmer_minus),
        rel=5.0e-8,
    )
    assert float(jax_adjoint[0]) == pytest.approx(
        elmer_central_difference,
        rel=2.0e-6,
    )

    payload = jax_prepared.payload
    assert isinstance(payload, PreparedJaxPortEigenmode)
    minimum = np.min(payload.validated.coordinates, axis=0)
    maximum = np.max(payload.validated.coordinates, axis=0)
    yee_grid = build_yee_grid(
        (
            np.linspace(minimum[0] + 1.1e-12, maximum[0] - 2.3e-12, 34),
            np.linspace(minimum[1] + 1.7e-12, maximum[1] - 2.9e-12, 26),
            np.asarray((0.0, 20.0e-9)),
        )
    )
    transfer_plan = build_yee_port_sampling_plan(
        payload.validated.coordinates,
        payload.validated.cells,
        payload.validated.edge_signs,
        yee_grid,
    )

    def yee_energy(active: jax.Array) -> jax.Array:
        mode = bound.mode(active)
        samples = sample_port_mode_to_yee(
            transfer_plan,
            mode.scalar_coefficients,
            mode.edge_coefficients,
            mode.propagation_constant_per_m,
            mode.cell_reluctivity_per_henry_m,
            bound.angular_frequency_rad_per_s,
            bound.target_power_w,
        )
        return jnp.mean(jnp.abs(samples.electric_v_per_m) ** 2)

    yee_value, yee_adjoint = jax.jit(jax.value_and_grad(yee_energy))(initial)
    yee_central_difference = (
        float(yee_energy(initial.at[0].add(step))) - float(yee_energy(initial.at[0].add(-step)))
    ) / (2.0 * step)
    assert math.isfinite(float(yee_value)) and float(yee_value) > 0.0
    assert float(yee_adjoint[0]) == pytest.approx(yee_central_difference, rel=2.0e-6)

    evidence = {
        "schema_version": "femx.port-eigen-adjoint-evidence/v1",
        "claim_scope": "same-mesh discrete Si/SiO2 simple-mode material derivative",
        "node_count": int(payload.validated.coordinates.shape[0]),
        "triangle_count": int(payload.validated.cells.shape[0]),
        "parameter": "core_relative_permittivity",
        "parameter_value": core_permittivity,
        "central_difference_step": step,
        "jax_beta_per_m": float(jax_beta),
        "jax_adjoint_beta_derivative": float(jax_adjoint[0]),
        "jax_central_beta_derivative": jax_central_difference,
        "jax_adjoint_vs_central_relative_error": abs(float(jax_adjoint[0]) - jax_central_difference)
        / abs(jax_central_difference),
        "elmer_central_beta_derivative": elmer_central_difference,
        "jax_adjoint_vs_elmer_central_relative_error": abs(
            float(jax_adjoint[0]) - elmer_central_difference
        )
        / abs(elmer_central_difference),
        "yee_mean_electric_intensity": float(yee_value),
        "yee_adjoint_derivative": float(yee_adjoint[0]),
        "yee_central_derivative": yee_central_difference,
        "yee_adjoint_vs_central_relative_error": abs(float(yee_adjoint[0]) - yee_central_difference)
        / abs(yee_central_difference),
        "simple_mode_relative_gap": float(bound.baseline_diagnostics.relative_eigenvalue_gap),
        "simple_mode_relative_residual": float(bound.baseline_diagnostics.relative_residual),
        "simple_mode_phase_anchor_relative_magnitude": float(
            bound.baseline_diagnostics.phase_anchor_relative_magnitude
        ),
        "phase_anchor_edge_dof_zero_based": bound.phase_anchor_edge_dof,
        "elmer_backend": locked_elmer_port_backend.descriptor.version,
        "gmsh_executable_sha256": meshing.identity.executable_sha256,
    }
    (tmp_path / "port-eigen-adjoint-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_square_silicon_core_cluster_adjoint_matches_jax_and_elmer_central_differences(
    locked_gmsh_runner,
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    """Validate one guided polarization-pair aggregate without choosing either basis vector."""

    recipe = RectangularWaveguideCrossSection(
        cladding_height_m=4.0e-6,
        core_height_m=0.5e-6,
        cladding_mesh_size_m=0.60e-6,
        core_mesh_size_m=0.12e-6,
    )
    meshing_directory = tmp_path / "cluster-meshing"
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
    core_permittivity = 3.48**2
    parameter_schema = ParameterSchema(
        (
            ParameterSpec(
                "core_relative_permittivity",
                unit="1",
                role=ParameterRole.DESIGN,
                lower_bound=8.0,
                upper_bound=16.0,
            ),
        )
    )

    def problem(gradient_method: GradientMethod) -> Problem:
        return Problem(
            f"square-silicon-port-cluster-{gradient_method.value}",
            imported.mesh,
            PortEigenmode(
                regions=(
                    IsotropicOpticalRegion("cladding", 1.444**2),
                    IsotropicOpticalRegion(
                        "core",
                        ParameterReference("core_relative_permittivity"),
                    ),
                ),
                perfect_electric_boundaries=tuple(
                    PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
                ),
                frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
                eigenmode_count=_MODE_COUNT,
                selected_mode_index=0,
                target_power_w=1.0,
                gradient_method=gradient_method,
            ),
            parameters=parameter_schema,
        )

    parameters = ParameterValues({"core_relative_permittivity": core_permittivity})
    jax_backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-12)
    jax_prepared = prepare(problem(GradientMethod.ADJOINT), jax_backend)
    bound = jax_backend.bind_differentiable_cluster(
        jax_prepared,
        parameters,
        mode_indices=(0, 1),
        quadrature_point_count=32,
    )
    initial = bound.initial_values

    def beta_sum(active: jax.Array) -> jax.Array:
        return bound.cluster(active).propagation_constant_sum_per_m

    jax_beta_sum, jax_adjoint = jax.jit(jax.value_and_grad(beta_sum))(initial)
    baseline_cluster = bound.cluster(initial)
    step = 2.0e-3
    jax_central_difference = (
        float(beta_sum(initial.at[0].add(step))) - float(beta_sum(initial.at[0].add(-step)))
    ) / (2.0 * step)
    assert bool(baseline_cluster.is_valid)
    assert float(baseline_cluster.mean_effective_index) > 1.444
    assert float(jax_adjoint[0]) == pytest.approx(jax_central_difference, rel=2.0e-7)

    elmer_prepared = prepare(problem(GradientMethod.NONE), locked_elmer_port_backend)

    def elmer_cluster(relative_permittivity: float, label: str) -> np.ndarray:
        run_directory = tmp_path / f"elmer-cluster-{label}"
        solve(
            elmer_prepared,
            locked_elmer_port_backend,
            request=SolveRequest(
                parameters=ParameterValues({"core_relative_permittivity": relative_permittivity}),
                run_directory=run_directory,
                policy=_AUTHORIZED,
            ),
        )
        spectrum = json.loads((run_directory / "port-spectrum.json").read_text(encoding="utf-8"))
        eigenvalues = np.asarray(
            [complex(value["real"], value["imag"]) for value in spectrum["eigenvalues_per_m2"]]
        )
        propagation_constants = np.sqrt(-eigenvalues[:2])
        assert np.max(np.abs(propagation_constants.imag)) <= 1.0e-8 * np.max(
            np.abs(propagation_constants.real)
        )
        vacuum_wavenumber = 2.0 * math.pi / 1.55e-6
        assert float(np.min(propagation_constants.real / vacuum_wavenumber)) > 1.444
        return propagation_constants.real

    elmer_minus = elmer_cluster(core_permittivity - step, "minus")
    elmer_plus = elmer_cluster(core_permittivity + step, "plus")
    elmer_central_difference = float((np.sum(elmer_plus) - np.sum(elmer_minus)) / (2.0 * step))
    elmer_baseline_estimate = 0.5 * (elmer_plus + elmer_minus)
    relative_pair_split = float(
        abs(elmer_baseline_estimate[0] - elmer_baseline_estimate[1])
        / np.mean(elmer_baseline_estimate)
    )
    assert relative_pair_split < 3.0e-4
    assert float(jax_beta_sum) == pytest.approx(
        float(np.sum(elmer_baseline_estimate)),
        rel=5.0e-8,
    )
    assert float(jax_adjoint[0]) == pytest.approx(elmer_central_difference, rel=2.0e-6)

    payload = jax_prepared.payload
    assert isinstance(payload, PreparedJaxPortEigenmode)
    evidence = {
        "schema_version": "femx.port-cluster-adjoint-evidence/v1",
        "claim_scope": (
            "same-mesh discrete square-Si-core guided polarization-pair propagation-sum "
            "material derivative"
        ),
        "node_count": int(payload.validated.coordinates.shape[0]),
        "triangle_count": int(payload.validated.cells.shape[0]),
        "canonical_mesh_sha256": imported.record.canonical_mesh_sha256,
        "core_width_m": recipe.core_width_m,
        "core_height_m": recipe.core_height_m,
        "cladding_width_m": recipe.cladding_width_m,
        "cladding_height_m": recipe.cladding_height_m,
        "parameter": "core_relative_permittivity",
        "parameter_value": core_permittivity,
        "central_difference_step": step,
        "mode_indices_zero_based": [0, 1],
        "relative_pair_split": relative_pair_split,
        "contour_center": bound.contour.center,
        "contour_radius": bound.contour.radius,
        "quadrature_point_count": bound.contour.quadrature_point_count,
        "observed_cluster_size": int(bound.baseline_diagnostics.observed_cluster_size),
        "relative_contour_clearance": float(bound.baseline_diagnostics.relative_contour_clearance),
        "relative_quadrature_error": float(bound.baseline_diagnostics.relative_quadrature_error),
        "projector_idempotency_error": float(
            bound.baseline_diagnostics.projector_idempotency_error
        ),
        "mean_effective_index": float(baseline_cluster.mean_effective_index),
        "jax_beta_sum_per_m": float(jax_beta_sum),
        "jax_adjoint_beta_sum_derivative": float(jax_adjoint[0]),
        "jax_central_beta_sum_derivative": jax_central_difference,
        "jax_adjoint_vs_central_relative_error": abs(float(jax_adjoint[0]) - jax_central_difference)
        / abs(jax_central_difference),
        "elmer_central_beta_sum_derivative": elmer_central_difference,
        "jax_adjoint_vs_elmer_central_relative_error": abs(
            float(jax_adjoint[0]) - elmer_central_difference
        )
        / abs(elmer_central_difference),
        "elmer_backend": locked_elmer_port_backend.descriptor.version,
        "gmsh_executable_sha256": meshing.identity.executable_sha256,
    }
    (tmp_path / "port-cluster-adjoint-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
