"""Locked external Elmer oracle for one-way 3D Tet4 electrothermal cases."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from femx.backends.elmer._oracle import (
    file_digest,
    installation_digest,
    parse_elmer_identity,
    parse_steady_change,
    prepare_run_directory,
    write_text,
)
from femx.backends.elmer.result import read_indexed_scalar_fields_result
from femx.backends.elmer.runner import (
    ElmerCommand,
    ElmerInstallation,
    ElmerProcessResult,
    ElmerRunner,
)
from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity
from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity
from femx.backends.elmer.tet4_electrothermal_case import (
    ElmerTet4ElectrothermalCase,
    render_tet4_electrothermal_sif,
)
from femx.core.errors import BackendError, BackendUnavailableError, ContractError
from femx.core.execution import ExecutionPolicy

_ADAPTER_VERSION = "0.1.0"


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ElmerTet4ElectrothermalResult:
    """Numeric fields and provenance from one completed external oracle attempt."""

    potential_node_ids: np.ndarray
    potential_v: np.ndarray
    temperature_k: np.ndarray
    current_steady_change: tuple[int, float] | None
    heat_steady_change: tuple[int, float] | None
    convergence_tolerance: float
    process: ElmerProcessResult
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        node_ids = _readonly(np.asarray(self.potential_node_ids, dtype=np.int64))
        potential = _readonly(np.asarray(self.potential_v, dtype=np.float64))
        temperature = _readonly(np.asarray(self.temperature_k, dtype=np.float64))
        if (
            node_ids.ndim != 1
            or potential.shape != node_ids.shape
            or temperature.ndim != 1
            or node_ids.size == 0
            or not np.array_equal(node_ids, np.unique(node_ids))
            or np.any(node_ids < 0)
            or np.any(node_ids >= temperature.size)
            or not np.isfinite(potential).all()
            or not np.isfinite(temperature).all()
        ):
            raise ContractError("Elmer Tet4 result fields must be finite canonical nodal vectors")
        if not math.isfinite(self.convergence_tolerance) or self.convergence_tolerance <= 0.0:
            raise ContractError("Elmer Tet4 result convergence tolerance must be positive")
        if not isinstance(self.process, ElmerProcessResult):
            raise ContractError("Elmer Tet4 result requires external-process evidence")
        object.__setattr__(self, "potential_node_ids", node_ids)
        object.__setattr__(self, "potential_v", potential)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def numerical_convergence_evaluated(self) -> bool:
        """Whether Elmer reported both steady relative-change records."""

        return self.current_steady_change is not None and self.heat_steady_change is not None

    @property
    def numerically_converged(self) -> bool:
        """Whether both reported changes satisfy the configured threshold."""

        return bool(
            self.numerical_convergence_evaluated
            and self.current_steady_change is not None
            and self.heat_steady_change is not None
            and self.current_steady_change[1] <= self.convergence_tolerance
            and self.heat_steady_change[1] <= self.convergence_tolerance
        )


class ElmerTet4ElectrothermalOracle:
    """Identity-locked serial Elmer runner for the typed distinct-space Tet4 case."""

    def __init__(
        self,
        installation: ElmerInstallation,
        current_identity: ElmerSteadyCurrentIdentity,
        heat_identity: ElmerSteadyHeatIdentity,
        *,
        timeout_seconds: float = 120.0,
        convergence_tolerance: float = 1.0e-12,
    ) -> None:
        if not installation.executable.is_file():
            raise BackendUnavailableError(
                f"Elmer executable does not exist: {installation.executable}"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ContractError("Elmer Tet4 timeout must be positive")
        if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
            raise ContractError("Elmer Tet4 convergence tolerance must be positive")
        current_common = (
            current_identity.version,
            current_identity.revision,
            current_identity.executable_sha256,
            current_identity.source_commit,
            current_identity.source_digest,
            current_identity.source_worktree_state,
        )
        heat_common = (
            heat_identity.version,
            heat_identity.revision,
            heat_identity.executable_sha256,
            heat_identity.source_commit,
            heat_identity.source_digest,
            heat_identity.source_worktree_state,
        )
        if current_common != heat_common:
            raise ContractError("Elmer Tet4 module identities must share one executable and source")
        self._current_identity = current_identity
        self._heat_identity = heat_identity
        self._timeout_seconds = float(timeout_seconds)
        self._convergence_tolerance = float(convergence_tolerance)
        self._runner = ElmerRunner(installation)
        self._elmer_home = installation.executable.resolve().parent.parent
        module_directory = self._elmer_home / "share" / "elmersolver" / "lib"
        self._current_module = module_directory / "StatCurrentSolve.so"
        self._heat_module = module_directory / "HeatSolve.so"
        self._verify_installation_identity()

    @property
    def version(self) -> str:
        """Return a stable adapter plus external solver identity string."""

        identity = self._current_identity
        return (
            f"adapter-{_ADAPTER_VERSION}+elmer-{identity.version}."
            f"rev-{identity.revision}.sha256-{identity.executable_sha256[:12]}"
        )

    def _verify_installation_identity(self) -> tuple[str, str, str]:
        executable_sha256 = installation_digest(
            self._runner.installation.executable,
            label="executable",
        )
        current_sha256 = installation_digest(
            self._current_module,
            label="StatCurrentSolve module",
        )
        heat_sha256 = installation_digest(self._heat_module, label="HeatSolve module")
        if executable_sha256 != self._current_identity.executable_sha256:
            raise BackendUnavailableError(
                "Elmer executable SHA-256 differs from the Tet4 oracle identity"
            )
        if current_sha256 != self._current_identity.stat_current_solve_sha256:
            raise BackendUnavailableError(
                "Elmer StatCurrentSolve SHA-256 differs from the Tet4 oracle identity"
            )
        if heat_sha256 != self._heat_identity.heat_solve_sha256:
            raise BackendUnavailableError(
                "Elmer HeatSolve SHA-256 differs from the Tet4 oracle identity"
            )
        return executable_sha256, current_sha256, heat_sha256

    def run(
        self,
        case: ElmerTet4ElectrothermalCase,
        *,
        run_directory: Path,
        policy: ExecutionPolicy,
    ) -> ElmerTet4ElectrothermalResult:
        """Execute one fresh case and retain raw input, output, VTU, and logs."""

        if not isinstance(case, ElmerTet4ElectrothermalCase):
            raise ContractError("Elmer Tet4 oracle requires a typed electrothermal case")
        policy.require_external_process(component_name="elmer-tet4-electrothermal")
        self._verify_installation_identity()
        selected = prepare_run_directory(run_directory)
        mesh_directory = selected / "mesh"
        sif = render_tet4_electrothermal_sif(
            case,
            stat_current_module=self._current_module,
            heat_solve_module=self._heat_module,
            convergence_tolerance=self._convergence_tolerance,
        )
        input_files = {
            selected / "ELMERSOLVER_STARTINFO": "case.sif\n",
            selected / "case.sif": sif,
            mesh_directory / "mesh.header": case.mesh.header,
            mesh_directory / "mesh.nodes": case.mesh.nodes,
            mesh_directory / "mesh.elements": case.mesh.elements,
            mesh_directory / "mesh.boundary": case.mesh.boundary,
        }
        for path, content in input_files.items():
            write_text(path, content)

        process = self._runner.run(
            ElmerCommand(
                environment={
                    "ELMER_HOME": str(self._elmer_home),
                    "ELMER_LIB": str(self._current_module.parent),
                    "ELMER_MODULES_PATH": str(self._current_module.parent),
                },
                timeout_seconds=self._timeout_seconds,
            ),
            working_directory=selected,
            policy=policy,
        )
        stdout_path = selected / "elmer.stdout.log"
        stderr_path = selected / "elmer.stderr.log"
        write_text(stdout_path, process.stdout)
        write_text(stderr_path, process.stderr)
        if not process.process_succeeded:
            raise BackendError(f"ElmerSolver exited with return code {process.return_code}")
        if "MAIN: *** Elmer Solver: ALL DONE ***" not in process.stdout:
            raise BackendError("ElmerSolver returned zero without its completion marker")
        executable_sha256, current_sha256, heat_sha256 = self._verify_installation_identity()
        actual_version, actual_revision = parse_elmer_identity(process.stdout)
        if (actual_version, actual_revision) != (
            self._current_identity.version,
            self._current_identity.revision,
        ):
            raise BackendError("executed Elmer identity differs from the Tet4 oracle descriptor")

        result_path = mesh_directory / "femx.result"
        vtu_path = mesh_directory / "femx.vtu"
        fields = read_indexed_scalar_fields_result(
            result_path,
            expected_node_count=case.mesh.node_count,
            field_node_ids={
                "potential": case.potential_node_ids,
                "temperature": tuple(range(case.mesh.node_count)),
            },
        )
        if not vtu_path.is_file():
            raise BackendError("Elmer did not produce the required raw Tet4 VTU artifact")
        potential = fields.fields["potential"]
        temperature = fields.fields["temperature"]
        current_change = parse_steady_change(process.stdout, equation_name="static current")
        heat_change = parse_steady_change(process.stdout, equation_name="heat equation")
        provenance = {
            "adapter_version": self.version,
            "case_sha256": case.digest(),
            "elmer_version": actual_version,
            "elmer_revision": actual_revision,
            "elmer_executable_sha256": executable_sha256,
            "elmer_stat_current_solve_sha256": current_sha256,
            "elmer_heat_solve_sha256": heat_sha256,
            "elmer_source_commit": self._current_identity.source_commit,
            "elmer_source_digest": self._current_identity.source_digest,
            "elmer_source_worktree_state": self._current_identity.source_worktree_state,
            "startinfo_sha256": file_digest(selected / "ELMERSOLVER_STARTINFO"),
            "input_sif_sha256": file_digest(selected / "case.sif"),
            "mesh_header_sha256": file_digest(mesh_directory / "mesh.header"),
            "mesh_nodes_sha256": file_digest(mesh_directory / "mesh.nodes"),
            "mesh_elements_sha256": file_digest(mesh_directory / "mesh.elements"),
            "mesh_boundary_sha256": file_digest(mesh_directory / "mesh.boundary"),
            "result_sha256": file_digest(result_path),
            "raw_vtu_sha256": file_digest(vtu_path),
            "stdout_sha256": file_digest(stdout_path),
            "stderr_sha256": file_digest(stderr_path),
            "result_save_count": str(fields.save_count),
            "result_timestep": str(fields.timestep),
            "result_numeric_source": "mesh/femx.result ASCII 3",
            "raw_vtu_artifact": "mesh/femx.vtu",
        }
        return ElmerTet4ElectrothermalResult(
            potential_node_ids=potential.source_node_ids,
            potential_v=potential.values,
            temperature_k=temperature.values,
            current_steady_change=current_change,
            heat_steady_change=heat_change,
            convergence_tolerance=self._convergence_tolerance,
            process=process,
            provenance=provenance,
        )


__all__ = [
    "ElmerTet4ElectrothermalOracle",
    "ElmerTet4ElectrothermalResult",
]
