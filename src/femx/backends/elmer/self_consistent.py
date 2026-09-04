"""Locked external Elmer oracle for self-consistent electrothermal feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from femx.backends._steady_current import (
    CURRENT_DENSITY_UNIT,
    ELECTRIC_FIELD_UNIT,
    JOULE_HEAT_DENSITY_UNIT,
    POTENTIAL_UNIT,
    POWER_PER_DEPTH_UNIT,
    ValidatedSteadyCurrent,
    resolve_current_scalar,
    validate_steady_current_problem,
)
from femx.backends._steady_heat import (
    TEMPERATURE_UNIT,
    ValidatedSteadyHeat,
    resolve_scalar,
    validate_steady_heat_problem,
)
from femx.backends.elmer._oracle import (
    file_digest,
    installation_digest,
    parse_elmer_identity,
    parse_steady_change,
    prepare_run_directory,
    write_text,
)
from femx.backends.elmer.electrothermal_case import (
    ElmerCoupledMeshDeck,
    _resolve_feedback_scalar,
    lower_self_consistent_mesh,
    render_self_consistent_sif,
)
from femx.backends.elmer.result import read_scalar_fields_result
from femx.backends.elmer.runner import ElmerCommand, ElmerInstallation, ElmerRunner
from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity
from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity
from femx.backends.protocol import BackendDescriptor, PrepareRequest
from femx.core.capabilities import FunctionSpaceFamily
from femx.core.errors import BackendError, BackendUnavailableError, ContractError
from femx.core.execution import ExecutionPolicy
from femx.core.parameters import ParameterValues
from femx.core.solution import ConvergenceReport, ConvergenceStatus, Field, Solution
from femx.mesh import FunctionSpace
from femx.physics.steady_current import SteadyCurrent
from femx.workflows.electrothermal import SelfConsistentJouleHeating

_ADAPTER_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ElmerSelfConsistentSolveRequest:
    """Three typed parameter namespaces and one explicitly authorized attempt path."""

    current_parameters: ParameterValues
    thermal_parameters: ParameterValues
    feedback_parameters: ParameterValues
    run_directory: Path
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)


@dataclass(frozen=True, slots=True)
class PreparedElmerSelfConsistent:
    """Pure coupled lowering awaiting binding and authorized external execution."""

    feedback: SelfConsistentJouleHeating
    current: ValidatedSteadyCurrent
    heat: ValidatedSteadyHeat
    mesh: ElmerCoupledMeshDeck
    default_run_directory: Path | None
    backend: BackendDescriptor


@dataclass(frozen=True, slots=True)
class _ElectrothermalAudit:
    electric_field: np.ndarray
    cell_nodal_conductivity: np.ndarray
    cell_nodal_current_density: np.ndarray
    cell_nodal_joule: np.ndarray
    current_relative_residual: float
    heat_relative_residual: float
    electrical_joule_power: float
    thermal_joule_load: float
    transfer_relative_error: float
    current_energy_relative_error: float
    heat_balance_relative_error: float


def _relative_error(difference: float, left: float, right: float) -> float:
    scale = abs(left) + abs(right)
    if scale > 0.0:
        return difference / scale
    return 0.0 if difference == 0.0 else math.inf


def _triangle_geometry(
    coordinates: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = coordinates[cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    areas = 0.5 * np.abs(determinant)
    numerators = np.stack(
        (
            np.stack(
                (points[:, 1, 1] - points[:, 2, 1], points[:, 2, 0] - points[:, 1, 0]),
                axis=1,
            ),
            np.stack(
                (points[:, 2, 1] - points[:, 0, 1], points[:, 0, 0] - points[:, 2, 0]),
                axis=1,
            ),
            np.stack(
                (points[:, 0, 1] - points[:, 1, 1], points[:, 1, 0] - points[:, 0, 0]),
                axis=1,
            ),
        ),
        axis=1,
    )
    return areas, numerators / determinant[:, None, None]


def _assemble_stiffness(
    cells: np.ndarray,
    areas: np.ndarray,
    basis_gradients: np.ndarray,
    cell_coefficient: np.ndarray,
    node_count: int,
) -> np.ndarray:
    local = (
        cell_coefficient[:, None, None]
        * areas[:, None, None]
        * np.einsum("cid,cjd->cij", basis_gradients, basis_gradients)
    )
    rows = np.repeat(cells, 3, axis=1).reshape(-1)
    columns = np.tile(cells, (1, 3)).reshape(-1)
    matrix = np.zeros((node_count, node_count), dtype=np.float64)
    np.add.at(matrix, (rows, columns), local.reshape(-1))
    return matrix


def _assemble_scalar_load(
    coordinates: np.ndarray,
    cells: np.ndarray,
    boundary_facets: np.ndarray,
    areas: np.ndarray,
    cell_source: np.ndarray,
    facet_load: np.ndarray,
) -> np.ndarray:
    load = np.zeros((coordinates.shape[0],), dtype=np.float64)
    local_source = np.broadcast_to(cell_source[:, None] * areas[:, None] / 3.0, cells.shape)
    np.add.at(load, cells.reshape(-1), local_source.reshape(-1))
    facet_points = coordinates[boundary_facets]
    lengths = np.linalg.norm(facet_points[:, 1, :] - facet_points[:, 0, :], axis=1)
    local_facet = facet_load * lengths / 2.0
    np.add.at(load, boundary_facets.reshape(-1), np.repeat(local_facet, 2))
    return load


def _resolved_current_arrays(
    problem: ValidatedSteadyCurrent,
    parameters: ParameterValues,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conductivity = np.zeros((problem.cells.shape[0],), dtype=np.float64)
    source = np.zeros_like(conductivity)
    for ids, coefficient, current_source in zip(
        problem.region_cells,
        problem.region_conductivity,
        problem.region_source,
        strict=True,
    ):
        conductivity[ids] = resolve_current_scalar(
            coefficient,
            parameters,
            strictly_positive=True,
        )
        source[ids] = resolve_current_scalar(current_source, parameters)
    facet_load = np.zeros((problem.boundary_facets.shape[0],), dtype=np.float64)
    for ids, value in zip(problem.flux_facets, problem.flux_values, strict=True):
        facet_load[ids] = resolve_current_scalar(value, parameters)
    return conductivity, source, facet_load


def _resolved_heat_arrays(
    problem: ValidatedSteadyHeat,
    parameters: ParameterValues,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conductivity = np.zeros((problem.cells.shape[0],), dtype=np.float64)
    source = np.zeros_like(conductivity)
    for ids, coefficient, heat_source in zip(
        problem.region_cells,
        problem.region_conductivity,
        problem.region_source,
        strict=True,
    ):
        conductivity[ids] = resolve_scalar(coefficient, parameters, strictly_positive=True)
        source[ids] = resolve_scalar(heat_source, parameters)
    facet_load = np.zeros((problem.boundary_facets.shape[0],), dtype=np.float64)
    for ids, value in zip(problem.flux_facets, problem.flux_values, strict=True):
        facet_load[ids] = resolve_scalar(value, parameters)
    return conductivity, source, facet_load


def _shifted_free_residual(
    matrix: np.ndarray,
    state: np.ndarray,
    load: np.ndarray,
    free_nodes: np.ndarray,
    reference: float,
) -> float:
    shifted = state - reference
    shifted_load = load - matrix @ np.full_like(state, reference)
    left = matrix[free_nodes, :] @ shifted
    right = shifted_load[free_nodes]
    residual = float(np.linalg.norm(left - right))
    scale = float(np.linalg.norm(left) + np.linalg.norm(right))
    return residual / scale if scale > 0.0 else (0.0 if residual == 0.0 else math.inf)


def _audit_fields(
    feedback: SelfConsistentJouleHeating,
    current: ValidatedSteadyCurrent,
    heat: ValidatedSteadyHeat,
    current_parameters: ParameterValues,
    heat_parameters: ParameterValues,
    feedback_parameters: ParameterValues,
    potential: np.ndarray,
    temperature: np.ndarray,
) -> _ElectrothermalAudit:
    node_count = current.coordinates.shape[0]
    expected = (node_count,)
    if (
        potential.shape != expected
        or temperature.shape != expected
        or not np.isfinite(potential).all()
        or not np.isfinite(temperature).all()
    ):
        raise ContractError("Elmer coupled fields must be finite nodal scalar arrays")
    areas, gradients = _triangle_geometry(current.coordinates, current.cells)
    base_sigma, current_source, current_flux = _resolved_current_arrays(
        current,
        current_parameters,
    )
    local_sigma = np.broadcast_to(base_sigma[:, None], current.cells.shape).copy()
    physics = feedback.one_way.electrical_problem.physics
    assert isinstance(physics, SteadyCurrent)
    region_by_tag = {
        region.tag: ids for region, ids in zip(physics.regions, current.region_cells, strict=True)
    }
    for law in feedback.conductivity_laws:
        ids = region_by_tag[law.tag]
        reference = _resolve_feedback_scalar(law.reference_temperature, feedback_parameters)
        coefficient = _resolve_feedback_scalar(
            law.temperature_coefficient,
            feedback_parameters,
        )
        denominator = 1.0 + coefficient * (temperature[current.cells[ids]] - reference)
        local_sigma[ids] = base_sigma[ids, None] / denominator
    if not np.isfinite(local_sigma).all() or np.any(local_sigma <= 0.0):
        raise ContractError("Elmer coupled conductivity law left its positive finite domain")

    potential_gradient = np.einsum("ci,cid->cd", potential[current.cells], gradients)
    electric_field = -potential_gradient
    local_current = local_sigma[:, :, None] * electric_field[:, None, :]
    electric_norm_squared = np.einsum("cd,cd->c", electric_field, electric_field)
    local_joule = local_sigma * electric_norm_squared[:, None]
    mean_sigma = np.mean(local_sigma, axis=1)
    current_matrix = _assemble_stiffness(
        current.cells,
        areas,
        gradients,
        mean_sigma,
        node_count,
    )
    current_load = _assemble_scalar_load(
        current.coordinates,
        current.cells,
        current.boundary_facets,
        areas,
        current_source,
        current_flux,
    )
    current_residual = current_matrix @ potential - current_load
    current_reference = resolve_current_scalar(current.dirichlet_values[0], current_parameters)
    current_relative_residual = _shifted_free_residual(
        current_matrix,
        potential,
        current_load,
        current.free_nodes,
        current_reference,
    )
    electrical_power = float(np.vdot(areas, np.mean(local_joule, axis=1)))
    current_reaction_power = float(
        np.vdot(potential[current.dirichlet_nodes], current_residual[current.dirichlet_nodes])
    )
    current_input_power = float(np.vdot(potential, current_load)) + current_reaction_power
    current_energy_error = _relative_error(
        abs(electrical_power - current_input_power),
        electrical_power,
        current_input_power,
    )

    heat_k, heat_source, heat_flux = _resolved_heat_arrays(heat, heat_parameters)
    heat_matrix = _assemble_stiffness(
        heat.cells,
        areas,
        gradients,
        heat_k,
        node_count,
    )
    heat_load = _assemble_scalar_load(
        heat.coordinates,
        heat.cells,
        heat.boundary_facets,
        areas,
        heat_source,
        heat_flux,
    )
    local_joule_load = areas[:, None] * (local_joule + np.sum(local_joule, axis=1)[:, None]) / 12.0
    np.add.at(heat_load, heat.cells.reshape(-1), local_joule_load.reshape(-1))
    thermal_joule_load = float(np.sum(local_joule_load))
    transfer_error = _relative_error(
        abs(electrical_power - thermal_joule_load),
        electrical_power,
        thermal_joule_load,
    )
    heat_residual = heat_matrix @ temperature - heat_load
    heat_reference = resolve_scalar(heat.dirichlet_values[0], heat_parameters)
    heat_relative_residual = _shifted_free_residual(
        heat_matrix,
        temperature,
        heat_load,
        heat.free_nodes,
        heat_reference,
    )
    total_heat_load = float(np.sum(heat_load))
    reaction = float(np.sum(heat_residual[heat.dirichlet_nodes]))
    heat_balance_error = _relative_error(
        abs(total_heat_load + reaction),
        total_heat_load,
        reaction,
    )
    return _ElectrothermalAudit(
        electric_field=electric_field,
        cell_nodal_conductivity=local_sigma,
        cell_nodal_current_density=local_current,
        cell_nodal_joule=local_joule,
        current_relative_residual=current_relative_residual,
        heat_relative_residual=heat_relative_residual,
        electrical_joule_power=electrical_power,
        thermal_joule_load=thermal_joule_load,
        transfer_relative_error=transfer_error,
        current_energy_relative_error=current_energy_error,
        heat_balance_relative_error=heat_balance_error,
    )


class ElmerSelfConsistentElectrothermalBackend:
    """Locked Elmer current/heat oracle with independent field and balance audits."""

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
            raise ContractError("Elmer timeout must be positive")
        if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
            raise ContractError("Elmer convergence tolerance must be positive")
        common_current = (
            current_identity.version,
            current_identity.revision,
            current_identity.executable_sha256,
            current_identity.source_commit,
            current_identity.source_digest,
            current_identity.source_worktree_state,
        )
        common_heat = (
            heat_identity.version,
            heat_identity.revision,
            heat_identity.executable_sha256,
            heat_identity.source_commit,
            heat_identity.source_digest,
            heat_identity.source_worktree_state,
        )
        if common_current != common_heat:
            raise ContractError(
                "coupled Elmer module identities must share one executable and source"
            )
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
        self._descriptor = BackendDescriptor(
            name="elmer-self-consistent-electrothermal",
            version=(
                f"adapter-{_ADAPTER_VERSION}+elmer-{current_identity.version}."
                f"rev-{current_identity.revision}.sha256-{current_identity.executable_sha256[:12]}"
            ),
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the adapter and locked executable identity."""

        return self._descriptor

    def _verify_installation_identity(self) -> tuple[str, str, str]:
        executable = installation_digest(self._runner.installation.executable, label="executable")
        current_module = installation_digest(
            self._current_module,
            label="StatCurrentSolve module",
        )
        heat_module = installation_digest(self._heat_module, label="HeatSolve module")
        if executable != self._current_identity.executable_sha256:
            raise BackendUnavailableError(
                "Elmer executable SHA-256 differs from the locked identity"
            )
        if current_module != self._current_identity.stat_current_solve_sha256:
            raise BackendUnavailableError(
                "Elmer StatCurrentSolve SHA-256 differs from the locked identity"
            )
        if heat_module != self._heat_identity.heat_solve_sha256:
            raise BackendUnavailableError(
                "Elmer HeatSolve SHA-256 differs from the locked identity"
            )
        return executable, current_module, heat_module

    def prepare(
        self,
        feedback: SelfConsistentJouleHeating,
        request: PrepareRequest | None = None,
    ) -> PreparedElmerSelfConsistent:
        """Validate and lower the closed workflow without filesystem or process side effects."""

        request = PrepareRequest() if request is None else request
        current = validate_steady_current_problem(feedback.one_way.electrical_problem)
        heat = validate_steady_heat_problem(feedback.one_way.thermal_problem)
        mesh = lower_self_consistent_mesh(current, heat)
        return PreparedElmerSelfConsistent(
            feedback=feedback,
            current=current,
            heat=heat,
            mesh=mesh,
            default_run_directory=request.run_directory,
            backend=self.descriptor,
        )

    def solve(
        self,
        prepared: PreparedElmerSelfConsistent,
        request: ElmerSelfConsistentSolveRequest,
    ) -> Solution:
        """Execute one fresh coupled attempt and return numeric plus provenance evidence."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared coupled Elmer identity does not match this executable")
        prepared.feedback.one_way.electrical_problem.parameters.bind(
            request.current_parameters.values
        )
        prepared.feedback.one_way.thermal_problem.parameters.bind(request.thermal_parameters.values)
        prepared.feedback.parameters.bind(request.feedback_parameters.values)
        selected = request.run_directory
        if prepared.default_run_directory is not None:
            if prepared.default_run_directory.resolve() != selected.resolve():
                raise ContractError("prepare and solve selected different Elmer run directories")
        request.policy.require_external_process(component_name=self.descriptor.name)
        self._verify_installation_identity()
        run_directory = prepare_run_directory(selected)
        mesh_directory = run_directory / "mesh"
        sif = render_self_consistent_sif(
            prepared.feedback,
            prepared.current,
            prepared.heat,
            prepared.mesh,
            request.current_parameters,
            request.thermal_parameters,
            request.feedback_parameters,
            stat_current_module=self._current_module,
            heat_solve_module=self._heat_module,
            convergence_tolerance=self._convergence_tolerance,
        )
        input_files = {
            run_directory / "ELMERSOLVER_STARTINFO": "case.sif\n",
            run_directory / "case.sif": sif,
            mesh_directory / "mesh.header": prepared.mesh.native.header,
            mesh_directory / "mesh.nodes": prepared.mesh.native.nodes,
            mesh_directory / "mesh.elements": prepared.mesh.native.elements,
            mesh_directory / "mesh.boundary": prepared.mesh.native.boundary,
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
            working_directory=run_directory,
            policy=request.policy,
        )
        stdout_path = run_directory / "elmer.stdout.log"
        stderr_path = run_directory / "elmer.stderr.log"
        write_text(stdout_path, process.stdout)
        write_text(stderr_path, process.stderr)
        if not process.process_succeeded:
            raise BackendError(f"ElmerSolver exited with return code {process.return_code}")
        if "MAIN: *** Elmer Solver: ALL DONE ***" not in process.stdout:
            raise BackendError("ElmerSolver returned zero without its completion marker")
        executable_hash, current_hash, heat_hash = self._verify_installation_identity()
        actual_version, actual_revision = parse_elmer_identity(process.stdout)
        if (actual_version, actual_revision) != (
            self._current_identity.version,
            self._current_identity.revision,
        ):
            raise BackendError("executed Elmer identity differs from the locked descriptor")

        result_path = mesh_directory / "femx.result"
        vtu_path = mesh_directory / "femx.vtu"
        fields = read_scalar_fields_result(
            result_path,
            expected_node_count=prepared.current.coordinates.shape[0],
            field_names=("potential", "temperature"),
        )
        if not vtu_path.is_file():
            raise BackendError("Elmer did not produce the required raw coupled VTU artifact")
        potential = fields.values["potential"]
        temperature = fields.values["temperature"]
        audit = _audit_fields(
            prepared.feedback,
            prepared.current,
            prepared.heat,
            request.current_parameters,
            request.thermal_parameters,
            request.feedback_parameters,
            potential,
            temperature,
        )
        current_change = parse_steady_change(process.stdout, equation_name="static current")
        heat_change = parse_steady_change(process.stdout, equation_name="heat equation")
        if current_change is None or heat_change is None:
            status = ConvergenceStatus.NOT_EVALUATED
            iterations = None
            reported_change = None
            message = "Elmer coupled run completed; one or both steady-change records were absent"
        else:
            iterations = max(current_change[0], heat_change[0])
            reported_change = max(current_change[1], heat_change[1])
            status = (
                ConvergenceStatus.CONVERGED
                if reported_change <= self._convergence_tolerance
                else ConvergenceStatus.NOT_CONVERGED
            )
            message = (
                "Elmer serial block current/heat iteration; convergence is Elmer's steady "
                "relative change, while independent residual and balance audits are separate"
            )

        h1 = FunctionSpace(FunctionSpaceFamily.H1, order=1)
        p0_vector = FunctionSpace(
            FunctionSpaceFamily.L2,
            order=0,
            value_shape=(2,),
            continuity="discontinuous",
        )
        local_p1_scalar = FunctionSpace(
            FunctionSpaceFamily.L2,
            order=1,
            continuity="discontinuous",
        )
        local_p1_vector = FunctionSpace(
            FunctionSpaceFamily.L2,
            order=1,
            value_shape=(2,),
            continuity="discontinuous",
        )
        metadata = {
            "element": "H1 P1 triangle",
            "conductivity_evaluation": "cell-local nodal law and P1 interpolation",
            "joule_load_integration": "consistent cell-local P1 triangle",
            "linear_solver": "Elmer StatCurrentSolve + HeatSolve / UMFPACK",
            "coupling": "serial steady block iteration",
            "out_of_plane_convention": "per_unit_depth",
            "integrated_power_unit": POWER_PER_DEPTH_UNIT,
            "independent_current_relative_residual": format(
                audit.current_relative_residual,
                ".17e",
            ),
            "independent_heat_relative_residual": format(audit.heat_relative_residual, ".17e"),
            "elmer_version": actual_version,
            "elmer_revision": actual_revision,
            "elmer_executable_sha256": executable_hash,
            "elmer_stat_current_solve_sha256": current_hash,
            "elmer_heat_solve_sha256": heat_hash,
            "elmer_source_commit": self._current_identity.source_commit,
            "elmer_source_digest": self._current_identity.source_digest,
            "elmer_source_worktree_state": self._current_identity.source_worktree_state,
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
            "result_save_count": str(fields.save_count),
            "result_timestep": str(fields.timestep),
        }
        if reported_change is not None:
            metadata["steady_relative_change"] = format(reported_change, ".17e")
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={
                "potential": Field("potential", potential, POTENTIAL_UNIT, h1),
                "temperature": Field("temperature", temperature, TEMPERATURE_UNIT, h1),
                "electric_field": Field(
                    "electric_field",
                    audit.electric_field,
                    ELECTRIC_FIELD_UNIT,
                    p0_vector,
                ),
                "electric_conductivity": Field(
                    "electric_conductivity",
                    audit.cell_nodal_conductivity,
                    "S/m",
                    local_p1_scalar,
                ),
                "current_density": Field(
                    "current_density",
                    audit.cell_nodal_current_density,
                    CURRENT_DENSITY_UNIT,
                    local_p1_vector,
                ),
                "joule_heat_density": Field(
                    "joule_heat_density",
                    audit.cell_nodal_joule,
                    JOULE_HEAT_DENSITY_UNIT,
                    local_p1_scalar,
                ),
            },
            observables={
                "potential_min_V": float(np.min(potential)),
                "potential_max_V": float(np.max(potential)),
                "temperature_min_K": float(np.min(temperature)),
                "temperature_max_K": float(np.max(temperature)),
                "joule_power_W_per_m": audit.electrical_joule_power,
                "thermal_joule_load_W_per_m": audit.thermal_joule_load,
                "transfer_relative_error": audit.transfer_relative_error,
                "current_energy_balance_relative_error": audit.current_energy_relative_error,
                "heat_balance_relative_error": audit.heat_balance_relative_error,
            },
            convergence=ConvergenceReport(
                status=status,
                iterations=iterations,
                residual_norm=reported_change,
                tolerance=self._convergence_tolerance,
                message=message,
            ),
            metadata=metadata,
        )
