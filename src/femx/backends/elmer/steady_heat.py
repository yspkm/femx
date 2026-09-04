"""External-process Elmer reference backend for steady H1 heat conduction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from femx.backends._steady_heat import (
    TEMPERATURE_UNIT,
    ValidatedSteadyHeat,
    validate_steady_heat_problem,
)
from femx.backends.elmer._oracle import (
    GIT_COMMIT_PATTERN,
    SHA256_PATTERN,
    SOURCE_STATES,
    file_digest,
    installation_digest,
    parse_elmer_identity,
    parse_steady_change,
    prepare_run_directory,
    validate_identity_part,
    write_text,
)
from femx.backends.elmer.case import ElmerMeshDeck, lower_elmer_mesh, render_steady_heat_sif
from femx.backends.elmer.result import read_temperature_result
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
from femx.mesh import FunctionSpace

_ADAPTER_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ElmerSteadyHeatIdentity:
    """Expected source and installed-binary identity for one Elmer oracle build."""

    version: str
    revision: str
    executable_sha256: str
    heat_solve_sha256: str
    source_commit: str
    source_digest: str
    source_worktree_state: str = "not_checked"

    def __post_init__(self) -> None:
        validate_identity_part(self.version, label="version")
        validate_identity_part(self.revision, label="revision")
        for label, value in (
            ("executable SHA-256", self.executable_sha256),
            ("HeatSolve SHA-256", self.heat_solve_sha256),
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
class PreparedElmerSteadyHeat:
    """Pure in-memory lowering awaiting parameter binding and authorized execution."""

    validated: ValidatedSteadyHeat
    mesh: ElmerMeshDeck
    default_run_directory: Path | None


class ElmerSteadyHeatBackend:
    """Locked-identity Elmer oracle for the same P1 slice as the JAX backend."""

    capabilities = CapabilitySet(
        analyses=frozenset({AnalysisKind.STEADY}),
        function_spaces=frozenset({FunctionSpaceFamily.H1}),
        scalar_kinds=frozenset({ScalarKind.REAL}),
        gradients=frozenset({GradientMethod.NONE}),
        parallel_models=frozenset({ParallelModel.SERIAL}),
    )

    def __init__(
        self,
        installation: ElmerInstallation,
        identity: ElmerSteadyHeatIdentity,
        *,
        timeout_seconds: float = 120.0,
        convergence_tolerance: float = 1.0e-12,
    ) -> None:
        if not installation.executable.is_file():
            raise BackendUnavailableError(
                f"Elmer executable does not exist: {installation.executable}"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ContractError("Elmer timeout must be positive")
        if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
            raise ContractError("Elmer convergence tolerance must be positive")
        self._identity = identity
        self._timeout_seconds = float(timeout_seconds)
        self._convergence_tolerance = float(convergence_tolerance)
        self._runner = ElmerRunner(installation)
        self._elmer_home = installation.executable.resolve().parent.parent
        self._heat_solve_module = (
            self._elmer_home / "share" / "elmersolver" / "lib" / "HeatSolve.so"
        )
        self._verify_installation_identity()
        self._descriptor = BackendDescriptor(
            name="elmer-steady-heat",
            version=(
                f"adapter-{_ADAPTER_VERSION}+elmer-{identity.version}."
                f"rev-{identity.revision}.sha256-{identity.executable_sha256[:12]}"
            ),
        )

    def _verify_installation_identity(self) -> tuple[str, str]:
        executable_sha256 = installation_digest(
            self._runner.installation.executable,
            label="executable",
        )
        if executable_sha256 != self._identity.executable_sha256:
            raise BackendUnavailableError(
                "Elmer executable SHA-256 differs from the locked oracle identity"
            )
        heat_solve_sha256 = installation_digest(
            self._heat_solve_module,
            label="HeatSolve module",
        )
        if heat_solve_sha256 != self._identity.heat_solve_sha256:
            raise BackendUnavailableError(
                "Elmer HeatSolve SHA-256 differs from the locked oracle identity"
            )
        return executable_sha256, heat_solve_sha256

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return adapter, expected Elmer identity, and executable fingerprint."""

        return self._descriptor

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        """Validate and lower the case without filesystem writes or process execution."""

        validated = validate_steady_heat_problem(problem)
        payload = PreparedElmerSteadyHeat(
            validated=validated,
            mesh=lower_elmer_mesh(validated),
            default_run_directory=request.run_directory,
        )
        return PreparedProblem(backend=self.descriptor, problem=problem, payload=payload)

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        """Write one fresh case, execute locked Elmer, and ingest its full-precision result."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared Elmer backend identity does not match this executable")
        if not isinstance(prepared.payload, PreparedElmerSteadyHeat):
            raise BackendError("prepared payload is not an Elmer steady-heat lowering")
        payload = prepared.payload
        default_directory = payload.default_run_directory
        requested_directory = request.run_directory
        if default_directory is not None and requested_directory is not None:
            if default_directory.resolve() != requested_directory.resolve():
                raise ContractError("prepare and solve selected different Elmer run directories")
        run_directory = (
            requested_directory if requested_directory is not None else default_directory
        )

        sif = render_steady_heat_sif(
            payload.validated,
            payload.mesh,
            request.parameters,
            convergence_tolerance=self._convergence_tolerance,
            heat_solve_module=self._heat_solve_module,
        )
        request.policy.require_external_process(component_name=self.descriptor.name)
        self._verify_installation_identity()
        if run_directory is None:
            raise ContractError("Elmer solve requires an explicit durable run directory")
        run_directory = prepare_run_directory(run_directory)
        mesh_directory = run_directory / "mesh"
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
                    "ELMER_LIB": str(self._heat_solve_module.parent),
                    "ELMER_MODULES_PATH": str(self._heat_solve_module.parent),
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
        executable_sha256, heat_solve_sha256 = self._verify_installation_identity()

        actual_version, actual_revision = parse_elmer_identity(result.stdout)
        if (actual_version, actual_revision) != (
            self._identity.version,
            self._identity.revision,
        ):
            raise BackendError(
                "executed Elmer identity differs from the locked backend descriptor: "
                f"expected={self._identity.version}/{self._identity.revision}, "
                f"actual={actual_version}/{actual_revision}"
            )

        result_path = mesh_directory / "femx.result"
        vtu_path = mesh_directory / "femx.vtu"
        temperature_result = read_temperature_result(
            result_path,
            expected_node_count=payload.validated.coordinates.shape[0],
        )
        if not vtu_path.is_file():
            raise BackendError("Elmer did not produce the required raw VTU artifact")

        steady_change = parse_steady_change(result.stdout, equation_name="heat equation")
        if steady_change is None:
            status = ConvergenceStatus.NOT_EVALUATED
            iterations = None
            relative_change = None
            message = "Elmer direct solve completed; steady convergence record was absent"
        else:
            iterations, relative_change = steady_change
            status = (
                ConvergenceStatus.CONVERGED
                if relative_change <= self._convergence_tolerance
                else ConvergenceStatus.NOT_CONVERGED
            )
            message = (
                "Elmer direct UMFPACK solve; tolerance applies to steady relative change, "
                "not a linear residual"
            )

        temperature = temperature_result.values
        metadata = {
            "element": "H1 P1 triangle",
            "linear_solver": "Elmer HeatSolve / UMFPACK",
            "out_of_plane_convention": "per_unit_depth",
            "heat_flux_sign": "positive_variational_rhs",
            "elmer_version": actual_version,
            "elmer_revision": actual_revision,
            "elmer_executable_sha256": executable_sha256,
            "elmer_heat_solve_module": str(self._heat_solve_module),
            "elmer_heat_solve_sha256": heat_solve_sha256,
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
            "raw_vtu_sha256": file_digest(vtu_path),
            "stdout_sha256": file_digest(stdout_path),
            "stderr_sha256": file_digest(stderr_path),
            "result_numeric_source": "mesh/femx.result ASCII 3",
            "raw_vtu_artifact": "mesh/femx.vtu",
            "result_save_count": str(temperature_result.save_count),
            "result_timestep": str(temperature_result.timestep),
        }
        if relative_change is not None:
            metadata["steady_relative_change"] = format(relative_change, ".17e")
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={
                "temperature": Field(
                    name="temperature",
                    values=temperature,
                    unit=TEMPERATURE_UNIT,
                    function_space=FunctionSpace(FunctionSpaceFamily.H1, order=1),
                )
            },
            observables={
                "temperature_min_K": float(np.min(temperature)),
                "temperature_max_K": float(np.max(temperature)),
            },
            convergence=ConvergenceReport(
                status=status,
                iterations=iterations,
                residual_norm=None,
                tolerance=self._convergence_tolerance,
                message=message,
            ),
            metadata=metadata,
        )
