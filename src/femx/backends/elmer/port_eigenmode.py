"""Locked external-process Elmer oracle for a lossless Silicon Photonics port mode."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from femx.backends._port_eigenmode import (
    ELECTRIC_FIELD_UNIT,
    PROPAGATION_CONSTANT_UNIT,
    ValidatedPortEigenmode,
    normalize_projected_mode,
    resolve_port_materials,
    validate_port_eigenmode_problem,
)
from femx.backends.elmer._oracle import (
    GIT_COMMIT_PATTERN,
    SHA256_PATTERN,
    SOURCE_STATES,
    file_digest,
    installation_digest,
    parse_elmer_identity,
    prepare_run_directory,
    validate_identity_part,
    write_text,
)
from femx.backends.elmer.case import ElmerTriangleMeshDeck, lower_tagged_triangle_mesh
from femx.backends.elmer.port_case import render_port_eigenmode_sif
from femx.backends.elmer.port_result import (
    parse_port_eigenmode_log,
    read_port_eigenmode_result,
    reorder_elmer_edge_coefficients,
)
from femx.backends.elmer.runner import ElmerCommand, ElmerInstallation, ElmerRunner
from femx.backends.protocol import (
    BackendDescriptor,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import (
    AnalysisKind,
    CapabilitySet,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import BackendError, BackendUnavailableError, ContractError
from femx.core.problem import Problem
from femx.core.solution import ConvergenceReport, ConvergenceStatus, Field, Solution
from femx.mesh import FunctionSpace, Mesh
from femx.physics.port_eigenmode import (
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    PortEigenmode,
)

_ADAPTER_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ElmerPortEigenmodeIdentity:
    """Exact source, executable, and runtime-module identity for the EMPort oracle."""

    version: str
    revision: str
    executable_sha256: str
    em_port_sha256: str
    result_output_sha256: str
    save_data_sha256: str
    source_commit: str
    source_digest: str
    source_worktree_state: str = "not_checked"

    def __post_init__(self) -> None:
        validate_identity_part(self.version, label="version")
        validate_identity_part(self.revision, label="revision")
        for label, value in (
            ("executable SHA-256", self.executable_sha256),
            ("EMPort SHA-256", self.em_port_sha256),
            ("ResultOutputSolve SHA-256", self.result_output_sha256),
            ("SaveData SHA-256", self.save_data_sha256),
            ("source digest", self.source_digest),
        ):
            if SHA256_PATTERN.fullmatch(value) is None:
                raise ContractError(f"Elmer {label} must be a lowercase SHA-256 digest")
        if GIT_COMMIT_PATTERN.fullmatch(self.source_commit) is None:
            raise ContractError("Elmer source commit must be a full lowercase Git SHA-1")
        if self.source_worktree_state not in SOURCE_STATES:
            raise ContractError(
                "Elmer source worktree state must be one of "
                f"{sorted(SOURCE_STATES)}, got {self.source_worktree_state!r}"
            )


@dataclass(frozen=True, slots=True)
class PreparedElmerPortEigenmode:
    """Pure in-memory EMPort lowering awaiting explicitly authorized execution."""

    validated: ValidatedPortEigenmode
    mesh: ElmerTriangleMeshDeck
    sif: str | None
    default_run_directory: Path | None


class ElmerPortEigenmodeBackend:
    """Pinned-identity Elmer EMPort reference for the v1 mixed edge/nodal formulation."""

    capabilities = CapabilitySet(
        analyses=frozenset({AnalysisKind.EIGENMODE}),
        function_spaces=frozenset({FunctionSpaceFamily.HCURL, FunctionSpaceFamily.H1}),
        scalar_kinds=frozenset({ScalarKind.COMPLEX}),
        gradients=frozenset({GradientMethod.NONE}),
        parallel_models=frozenset({ParallelModel.SERIAL}),
    )

    def __init__(
        self,
        installation: ElmerInstallation,
        identity: ElmerPortEigenmodeIdentity,
        *,
        timeout_seconds: float = 300.0,
        convergence_tolerance: float = 1.0e-10,
    ) -> None:
        if not installation.executable.is_file():
            raise BackendUnavailableError(
                f"Elmer executable does not exist: {installation.executable}"
            )
        for label, value in (
            ("timeout", timeout_seconds),
            ("eigen convergence tolerance", convergence_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"Elmer port {label} must be finite and positive")
        self._identity = identity
        self._timeout_seconds = float(timeout_seconds)
        self._convergence_tolerance = float(convergence_tolerance)
        self._runner = ElmerRunner(installation)
        self._elmer_home = installation.executable.resolve().parent.parent
        module_directory = self._elmer_home / "share" / "elmersolver" / "lib"
        self._em_port_module = module_directory / "EMPort.so"
        self._result_output_module = module_directory / "ResultOutputSolve.so"
        self._save_data_module = module_directory / "SaveData.so"
        self._verify_installation_identity()
        self._descriptor = BackendDescriptor(
            name="elmer-port-eigenmode",
            version=(
                f"adapter-{_ADAPTER_VERSION}+elmer-{identity.version}."
                f"rev-{identity.revision}.sha256-{identity.executable_sha256[:12]}"
            ),
        )

    def _verify_installation_identity(self) -> tuple[str, str, str, str]:
        observed = (
            installation_digest(
                self._runner.installation.executable,
                label="executable",
            ),
            installation_digest(self._em_port_module, label="EMPort module"),
            installation_digest(
                self._result_output_module,
                label="ResultOutputSolve module",
            ),
            installation_digest(self._save_data_module, label="SaveData module"),
        )
        expected = (
            self._identity.executable_sha256,
            self._identity.em_port_sha256,
            self._identity.result_output_sha256,
            self._identity.save_data_sha256,
        )
        labels = ("executable", "EMPort", "ResultOutputSolve", "SaveData")
        for label, actual, locked in zip(labels, observed, expected, strict=True):
            if actual != locked:
                raise BackendUnavailableError(
                    f"Elmer {label} SHA-256 differs from the locked port-oracle identity"
                )
        return observed

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the exact adapter and installed-Elmer identity."""

        return self._descriptor

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        """Validate and lower one canonical mesh without filesystem or process side effects."""

        validated = validate_port_eigenmode_problem(problem)
        mesh_contract = cast(Mesh, problem.mesh)
        physics = cast(PortEigenmode, problem.physics)
        mesh = lower_tagged_triangle_mesh(
            mesh_contract,
            region_tags=tuple(region.tag for region in physics.regions),
            boundary_tags=tuple(boundary.tag for boundary in physics.perfect_electric_boundaries),
        )
        sif = None
        if not validated.parameter_names:
            sif = render_port_eigenmode_sif(
                validated,
                mesh,
                convergence_tolerance=self._convergence_tolerance,
                em_port_module=self._em_port_module,
                result_output_module=self._result_output_module,
                save_data_module=self._save_data_module,
            )
        payload = PreparedElmerPortEigenmode(
            validated=validated,
            mesh=mesh,
            sif=sif,
            default_run_directory=request.run_directory,
        )
        return PreparedProblem(backend=self.descriptor, problem=problem, payload=payload)

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        """Execute a fresh locked EMPort case and ingest every required evidence artifact."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared Elmer port identity does not match this executable")
        if not isinstance(prepared.payload, PreparedElmerPortEigenmode):
            raise BackendError("prepared payload is not an Elmer port-eigenmode lowering")
        payload = prepared.payload
        default_directory = payload.default_run_directory
        requested_directory = request.run_directory
        if default_directory is not None and requested_directory is not None:
            if default_directory.resolve() != requested_directory.resolve():
                raise ContractError("prepare and solve selected different Elmer run directories")
        run_directory = (
            requested_directory if requested_directory is not None else default_directory
        )

        request.policy.require_external_process(component_name=self.descriptor.name)
        self._verify_installation_identity()
        if run_directory is None:
            raise ContractError("Elmer port solve requires an explicit durable run directory")
        run_directory = prepare_run_directory(run_directory)
        mesh_directory = run_directory / "mesh"
        materials = resolve_port_materials(payload.validated, request.parameters)
        sif = payload.sif
        if sif is None:
            sif = render_port_eigenmode_sif(
                payload.validated,
                payload.mesh,
                convergence_tolerance=self._convergence_tolerance,
                em_port_module=self._em_port_module,
                result_output_module=self._result_output_module,
                save_data_module=self._save_data_module,
                materials=materials,
            )
        input_files = {
            run_directory / "ELMERSOLVER_STARTINFO": "case.sif\n",
            run_directory / "case.sif": sif,
            mesh_directory / "mesh.header": payload.mesh.header,
            mesh_directory / "mesh.nodes": payload.mesh.nodes,
            mesh_directory / "mesh.elements": payload.mesh.elements,
            mesh_directory / "mesh.boundary": payload.mesh.boundary,
        }
        for path, content in input_files.items():
            write_text(path, content)

        result = self._runner.run(
            ElmerCommand(
                environment={
                    "ELMER_HOME": str(self._elmer_home),
                    "ELMER_LIB": str(self._em_port_module.parent),
                    "ELMER_MODULES_PATH": str(self._em_port_module.parent),
                },
                timeout_seconds=self._timeout_seconds,
            ),
            working_directory=run_directory,
            policy=request.policy,
        )
        stdout_path = run_directory / "elmer.stdout.log"
        stderr_path = run_directory / "elmer.stderr.log"
        write_text(stdout_path, result.stdout)
        write_text(stderr_path, result.stderr)
        if not result.process_succeeded:
            raise BackendError(f"ElmerSolver exited with return code {result.return_code}")
        if "MAIN: *** Elmer Solver: ALL DONE ***" not in result.stdout:
            raise BackendError("ElmerSolver returned zero without its completion marker")
        module_hashes_after = self._verify_installation_identity()

        actual_version, actual_revision = parse_elmer_identity(result.stdout)
        if (actual_version, actual_revision) != (
            self._identity.version,
            self._identity.revision,
        ):
            raise BackendError(
                "executed Elmer identity differs from the locked port backend: "
                f"expected={self._identity.version}/{self._identity.revision}, "
                f"actual={actual_version}/{actual_revision}"
            )

        log = parse_port_eigenmode_log(
            result.stdout,
            selected_mode_index=payload.validated.selected_mode_index,
        )
        requested_count = payload.validated.eigenmode_count
        if log.eigenvalues.size != requested_count or log.residuals.size != requested_count:
            raise BackendError("Elmer port spectrum does not contain every requested eigenpair")
        tolerance_scale = max(1.0, abs(self._convergence_tolerance))
        if (
            abs(log.reported_tolerance - self._convergence_tolerance)
            > 32.0 * np.finfo(np.float64).eps * tolerance_scale
        ):
            raise BackendError("Elmer port used a different eigen convergence tolerance")

        result_path = mesh_directory / "femx.result"
        raw_vtu_path = mesh_directory / "femx-mode_t0001.vtu"
        port_result = read_port_eigenmode_result(
            result_path,
            expected_node_count=payload.validated.coordinates.shape[0],
            expected_edge_count=payload.validated.edge_nodes.shape[0],
            expected_mode_count=requested_count,
        )
        projected = port_result.projected
        canonical_edge_coefficients = reorder_elmer_edge_coefficients(
            port_result.mixed.edge_coefficients,
            elmer_edge_nodes=payload.mesh.edge_nodes,
            canonical_edge_nodes=payload.validated.edge_nodes,
        )
        canonical_mixed_coefficients = np.vstack(
            (port_result.mixed.nodal_coefficients, canonical_edge_coefficients)
        )
        constrained = payload.validated.dof_partition.constrained_dofs
        if np.any(canonical_mixed_coefficients[constrained] != 0.0):
            raise BackendError("Elmer port raw PEC-constrained coefficients are not exactly zero")
        if not raw_vtu_path.is_file():
            raise BackendError("Elmer port did not produce the required raw VTU artifact")
        normalized = normalize_projected_mode(
            projected.values,
            raw_forward_power_w=log.raw_forward_power_w,
            target_forward_power_w=payload.validated.target_power_w,
        )

        selected_index = payload.validated.selected_mode_index
        coefficient_scale = normalized.phase_factor * normalized.amplitude_scale
        normalized_mixed_coefficients = (
            canonical_mixed_coefficients[:, selected_index] * coefficient_scale
        )
        node_count = payload.validated.coordinates.shape[0]
        selected_residual = float(log.residuals[selected_index])
        maximum_residual = float(np.max(log.residuals))
        converged = log.converged_count == requested_count
        convergence_status = (
            ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.NOT_CONVERGED
        )
        propagation_constant = log.selected_beta
        if propagation_constant.real <= 0.0:
            raise BackendError("Elmer port selected a non-forward propagation constant")
        vacuum_wavenumber = (
            2.0 * math.pi * payload.validated.frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S
        )
        effective_index = propagation_constant / vacuum_wavenumber

        spectrum_path = run_directory / "port-spectrum.json"
        spectrum_data = {
            "schema_version": "femx.elmer-port-spectrum/v1",
            "mode_ordering": "decreasing_real_propagation_constant",
            "selected_mode_index_zero_based": selected_index,
            "eigenvalues_per_m2": [
                {"real": float(value.real), "imag": float(value.imag)} for value in log.eigenvalues
            ],
            "residuals": [float(value) for value in log.residuals],
            "reported_tolerance": log.reported_tolerance,
            "iterations": log.iterations,
            "converged_count": log.converged_count,
        }
        write_text(
            spectrum_path,
            json.dumps(spectrum_data, sort_keys=True, separators=(",", ":")) + "\n",
        )

        executable_sha256, em_port_sha256, result_output_sha256, save_data_sha256 = (
            module_hashes_after
        )
        metadata = {
            "element": "Elmer first-family first-order Hcurl triangle plus H1 constraint",
            "mode_ordering": "decreasing_real_propagation_constant",
            "linear_solver": "Elmer EMPort / UMFPACK",
            "field_representation": "nodal_cartesian_projection_of_hcurl_solution",
            "field_components": "x,y,z",
            "field_global_phase": "largest_magnitude_component_positive_real",
            "field_power_normalization": "Elmer_EMPort_edge_power_to_requested_forward_power",
            "field_projection_limitation": (
                "projected_H1_E_returned; normalized_mixed_coefficients_exposed; "
                "H_requires_explicit_reconstruction"
            ),
            "printed_residual_semantics": (
                "scale_dependent_absolute_L2_evidence; not_compared_to_Ritz_tolerance"
            ),
            "fdtdx_mode_bundle_status": (
                "requires_explicit_hashed_FEM_to_Yee_transfer_plan_and_target_identity"
            ),
            "propagation_axis": "+z",
            "magnetic_field_convention": "not_emitted",
            "propagation_constant_unit": PROPAGATION_CONSTANT_UNIT,
            "elmer_version": actual_version,
            "elmer_revision": actual_revision,
            "elmer_executable_sha256": executable_sha256,
            "elmer_em_port_module": str(self._em_port_module),
            "elmer_em_port_sha256": em_port_sha256,
            "elmer_result_output_module": str(self._result_output_module),
            "elmer_result_output_sha256": result_output_sha256,
            "elmer_save_data_module": str(self._save_data_module),
            "elmer_save_data_sha256": save_data_sha256,
            "elmer_source_commit": self._identity.source_commit,
            "elmer_source_digest": self._identity.source_digest,
            "elmer_source_worktree_state": self._identity.source_worktree_state,
            "startinfo_sha256": file_digest(run_directory / "ELMERSOLVER_STARTINFO"),
            "input_sif_sha256": file_digest(run_directory / "case.sif"),
            "mesh_header_sha256": file_digest(mesh_directory / "mesh.header"),
            "mesh_nodes_sha256": file_digest(mesh_directory / "mesh.nodes"),
            "mesh_elements_sha256": file_digest(mesh_directory / "mesh.elements"),
            "mesh_boundary_sha256": file_digest(mesh_directory / "mesh.boundary"),
            "result_sha256": file_digest(result_path),
            "raw_vtu_sha256": file_digest(raw_vtu_path),
            "stdout_sha256": file_digest(stdout_path),
            "stderr_sha256": file_digest(stderr_path),
            "spectrum_sha256": file_digest(spectrum_path),
            "result_numeric_source": "mesh/femx.result ASCII 3",
            "raw_vtu_artifact": "mesh/femx-mode_t0001.vtu",
            "spectrum_artifact": "port-spectrum.json",
            "projected_field_record_count": str(projected.record_count),
            "projected_field_final_save_count": str(projected.final_save_count),
            "projected_field_final_timestep": str(projected.final_timestep),
            "projected_field_permutation_size": str(projected.permutation_size),
            "raw_mixed_mode_count": str(port_result.mixed.nodal_coefficients.shape[1]),
            "raw_mixed_edge_count": str(port_result.mixed.edge_coefficients.shape[0]),
            "raw_mixed_record_count": str(port_result.mixed.record_count),
            "raw_mixed_final_zero_verified": str(
                port_result.mixed.final_zero_record_verified
            ).lower(),
            "raw_mixed_pec_zero_verified": "true",
            "phase_anchor_node_zero_based": str(normalized.anchor_node),
            "phase_anchor_component_zero_based": str(normalized.anchor_component),
            "phase_factor_real": format(normalized.phase_factor.real, ".17e"),
            "phase_factor_imag": format(normalized.phase_factor.imag, ".17e"),
        }
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={
                "electric_field": Field(
                    name="electric_field",
                    values=normalized.electric_field,
                    unit=ELECTRIC_FIELD_UNIT,
                    function_space=FunctionSpace(
                        FunctionSpaceFamily.H1,
                        order=1,
                        value_shape=(3,),
                    ),
                ),
                PORT_LONGITUDINAL_POTENTIAL_FIELD: Field(
                    name=PORT_LONGITUDINAL_POTENTIAL_FIELD,
                    values=normalized_mixed_coefficients[:node_count],
                    unit=PORT_LONGITUDINAL_POTENTIAL_UNIT,
                    function_space=FunctionSpace(FunctionSpaceFamily.H1, order=1),
                ),
                PORT_TRANSVERSE_ELECTRIC_FIELD: Field(
                    name=PORT_TRANSVERSE_ELECTRIC_FIELD,
                    values=normalized_mixed_coefficients[node_count:],
                    unit=PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
                    function_space=FunctionSpace(
                        FunctionSpaceFamily.HCURL,
                        order=1,
                        value_shape=(2,),
                    ),
                ),
            },
            observables={
                "propagation_constant_rad_per_m": propagation_constant,
                "effective_index": effective_index,
                "selected_eigenvalue_per_m2": complex(log.eigenvalues[selected_index]),
                "vacuum_wavenumber_rad_per_m": vacuum_wavenumber,
                "raw_forward_power_W": log.raw_forward_power_w,
                "target_forward_power_W": payload.validated.target_power_w,
                "port_impedance_ohm": log.port_impedance_ohm,
                "field_amplitude_scale": normalized.amplitude_scale,
                "selected_eigen_residual": selected_residual,
                "maximum_requested_eigen_residual": maximum_residual,
            },
            convergence=ConvergenceReport(
                status=convergence_status,
                iterations=log.iterations,
                residual_norm=None,
                tolerance=self._convergence_tolerance,
                message=(
                    "Elmer complex Ritz-count convergence; printed absolute L2 residuals are "
                    "retained separately, and scientific validity requires a port-mode "
                    "ValidationReport"
                ),
            ),
            metadata=metadata,
        )
