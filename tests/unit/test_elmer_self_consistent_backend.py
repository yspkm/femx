from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.electrothermal_support import parameterized_self_consistent_microheater

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from femx.backends.elmer.runner import (  # noqa: E402
    ElmerInstallation,
    ElmerProcessResult,
    ElmerRunner,
)
from femx.backends.elmer.self_consistent import (  # noqa: E402
    ElmerSelfConsistentElectrothermalBackend,
    ElmerSelfConsistentSolveRequest,
    _audit_fields,
    _relative_error,
    _shifted_free_residual,
)
from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity  # noqa: E402
from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity  # noqa: E402
from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    BackendDescriptor,
    ExecutionPolicy,
    PrepareRequest,
)
from femx.core.errors import (  # noqa: E402
    BackendError,
    BackendUnavailableError,
    ContractError,
    ExecutionNotAuthorizedError,
)
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

EXPECTED_VERSION = "26.2-devel"
EXPECTED_REVISION = "abc123"
SOURCE_COMMIT = "a" * 40
SOURCE_DIGEST = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_installation(
    tmp_path: Path,
) -> tuple[ElmerInstallation, ElmerSteadyCurrentIdentity, ElmerSteadyHeatIdentity]:
    root = tmp_path / "fake-elmer-install"
    executable = root / "bin" / "ElmerSolver"
    modules = root / "share" / "elmersolver" / "lib"
    current_module = modules / "StatCurrentSolve.so"
    heat_module = modules / "HeatSolve.so"
    executable.parent.mkdir(parents=True, exist_ok=True)
    modules.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"locked fake ElmerSolver\n")
    current_module.write_bytes(b"locked fake StatCurrentSolve\n")
    heat_module.write_bytes(b"locked fake HeatSolve\n")
    common = {
        "version": EXPECTED_VERSION,
        "revision": EXPECTED_REVISION,
        "executable_sha256": _sha256(executable),
        "source_commit": SOURCE_COMMIT,
        "source_digest": SOURCE_DIGEST,
    }
    return (
        ElmerInstallation(executable.resolve()),
        ElmerSteadyCurrentIdentity(
            **common,
            stat_current_solve_sha256=_sha256(current_module),
        ),
        ElmerSteadyHeatIdentity(
            **common,
            heat_solve_sha256=_sha256(heat_module),
        ),
    )


def _backend(tmp_path: Path, **kwargs) -> ElmerSelfConsistentElectrothermalBackend:
    installation, current, heat = _fake_installation(tmp_path)
    current = replace(
        current,
        executable_sha256=kwargs.pop("current_executable_sha256", current.executable_sha256),
        stat_current_solve_sha256=kwargs.pop(
            "current_module_sha256",
            current.stat_current_solve_sha256,
        ),
    )
    heat = replace(
        heat,
        revision=kwargs.pop("heat_revision", heat.revision),
        executable_sha256=kwargs.pop("heat_executable_sha256", heat.executable_sha256),
        heat_solve_sha256=kwargs.pop("heat_module_sha256", heat.heat_solve_sha256),
    )
    return ElmerSelfConsistentElectrothermalBackend(
        installation,
        current,
        heat,
        **kwargs,
    )


def _bound_case():
    feedback, current_parameters, heat_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=2)
    )
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    system = DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current_backend.bind_differentiable(
            prepare(feedback.one_way.electrical_problem, current_backend),
            current_parameters,
        ),
        heat_backend.bind_differentiable(
            prepare(feedback.one_way.thermal_problem, heat_backend),
            heat_parameters,
        ),
        feedback_parameters,
    )
    state = system.solve(
        system.initial_current_values,
        system.initial_thermal_values,
        system.initial_feedback_values,
    )
    assert bool(state.converged)
    return feedback, current_parameters, heat_parameters, feedback_parameters, state


def _result_text(potential: np.ndarray, temperature: np.ndarray) -> str:
    count = potential.size
    pairs = "\n".join(f"{index} {count + 1 - index}" for index in range(1, count + 1))
    potential_values = "\n".join(format(float(value), ".17e") for value in potential)
    temperature_values = "\n".join(format(float(value), ".17e") for value in temperature)
    return (
        "ASCII 3\n"
        "!dynamic timestamp\n"
        "Degrees of freedom:\n"
        f"Potential : {count} {count} 1 : static current\n"
        f"Temperature : {count} {count} 1 : heat equation\n"
        "Total DOFs: 2\n"
        f"Number Of Nodes: {count}\n"
        "Time: 1 9 0.0\n"
        "Potential\n"
        f"Perm: {count} {count}\n"
        f"{pairs}\n"
        f"{potential_values}\n"
        "Temperature\n"
        "Perm: use previous\n"
        f"{temperature_values}\n"
    )


def _stdout(
    *,
    version: str = EXPECTED_VERSION,
    revision: str = EXPECTED_REVISION,
    current_change: str | None = "0.0",
    heat_change: str | None = "0.0",
    completed: bool = True,
) -> str:
    lines = [f"MAIN: Version: {version} (Rev: {revision}, Compiled: test)"]
    if current_change is not None:
        lines.append(
            f"ComputeChange: SS (ITER=7) (NRM,RELC): ( 0.5 {current_change} ) :: static current"
        )
    if heat_change is not None:
        lines.append(
            f"ComputeChange: SS (ITER=8) (NRM,RELC): ( 300.0 {heat_change} ) :: heat equation"
        )
    if completed:
        lines.append("MAIN: *** Elmer Solver: ALL DONE ***")
    return "\n".join(lines) + "\n"


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    potential: np.ndarray,
    temperature: np.ndarray,
    *,
    return_code: int = 0,
    stdout: str | None = None,
    write_result: bool = True,
    write_vtu: bool = True,
) -> None:
    def fake_run(self, command, *, working_directory, policy):
        root = self.installation.executable.resolve().parent.parent
        modules = root / "share" / "elmersolver" / "lib"
        assert command.environment == {
            "ELMER_HOME": str(root),
            "ELMER_LIB": str(modules),
            "ELMER_MODULES_PATH": str(modules),
        }
        policy.require_external_process(component_name="elmer")
        if write_result:
            (working_directory / "mesh" / "femx.result").write_text(
                _result_text(potential, temperature),
                encoding="utf-8",
            )
        if write_vtu:
            (working_directory / "mesh" / "femx.vtu").write_bytes(b"raw-vtu")
        return ElmerProcessResult(
            argv=(str(self.installation.executable),),
            return_code=return_code,
            stdout=_stdout() if stdout is None else stdout,
            stderr="fake-stderr",
            elapsed_seconds=0.25,
        )

    monkeypatch.setattr(ElmerRunner, "run", fake_run)


def _request(
    run_directory: Path,
    current_parameters,
    heat_parameters,
    feedback_parameters,
    *,
    authorized: bool = True,
) -> ElmerSelfConsistentSolveRequest:
    return ElmerSelfConsistentSolveRequest(
        current_parameters=current_parameters,
        thermal_parameters=heat_parameters,
        feedback_parameters=feedback_parameters,
        run_directory=run_directory,
        policy=ExecutionPolicy(
            execution_authorized=authorized,
            allow_external_process=authorized,
        ),
    )


def test_backend_prepare_is_pure_and_success_retains_coupled_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters, state = _bound_case()
    _install_fake_run(
        monkeypatch,
        np.asarray(state.potential),
        np.asarray(state.temperature),
    )
    backend = _backend(tmp_path)
    run_directory = tmp_path / "attempt"
    prepared = backend.prepare(feedback, PrepareRequest(run_directory=run_directory))
    assert backend.descriptor.name == "elmer-self-consistent-electrothermal"
    assert not run_directory.exists()

    solution = backend.solve(
        prepared,
        _request(
            run_directory,
            current_parameters,
            heat_parameters,
            feedback_parameters,
        ),
    )

    assert solution.convergence.status.value == "converged"
    assert solution.convergence.iterations == 8
    assert solution.convergence.residual_norm == 0.0
    np.testing.assert_allclose(solution.fields["potential"].values, state.potential)
    np.testing.assert_allclose(solution.fields["temperature"].values, state.temperature)
    np.testing.assert_allclose(
        solution.fields["electric_conductivity"].values,
        state.cell_nodal_conductivity,
    )
    np.testing.assert_allclose(
        solution.fields["joule_heat_density"].values,
        state.cell_nodal_joule_heat_density,
    )
    assert solution.observables["transfer_relative_error"] < 2.0e-15
    assert solution.observables["current_energy_balance_relative_error"] < 2.0e-11
    assert solution.observables["heat_balance_relative_error"] < 2.0e-9
    assert solution.metadata["elmer_source_commit"] == SOURCE_COMMIT
    assert solution.metadata["result_save_count"] == "1"
    assert solution.metadata["result_timestep"] == "9"
    for key in (
        "startinfo_sha256",
        "input_sif_sha256",
        "mesh_header_sha256",
        "mesh_nodes_sha256",
        "mesh_elements_sha256",
        "mesh_boundary_sha256",
        "result_sha256",
        "raw_vtu_sha256",
        "stdout_sha256",
        "stderr_sha256",
    ):
        assert len(solution.metadata[key]) == 64
    assert "Electric Conductivity = Variable Temperature" in (run_directory / "case.sif").read_text(
        encoding="utf-8"
    )


def test_backend_requires_authority_and_exact_prepared_identity(tmp_path) -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters, _state = _bound_case()
    backend = _backend(tmp_path)
    run_directory = tmp_path / "denied"
    prepared = backend.prepare(feedback, PrepareRequest(run_directory=run_directory))
    with pytest.raises(ExecutionNotAuthorizedError, match="requires"):
        backend.solve(
            prepared,
            _request(
                run_directory,
                current_parameters,
                heat_parameters,
                feedback_parameters,
                authorized=False,
            ),
        )
    assert not run_directory.exists()

    wrong = replace(prepared, backend=BackendDescriptor("other", "1"))
    with pytest.raises(BackendError, match="identity"):
        backend.solve(
            wrong,
            _request(
                tmp_path / "wrong",
                current_parameters,
                heat_parameters,
                feedback_parameters,
            ),
        )
    different = backend.prepare(feedback, PrepareRequest(run_directory=tmp_path / "prepared"))
    with pytest.raises(ContractError, match="different"):
        backend.solve(
            different,
            _request(
                tmp_path / "solved",
                current_parameters,
                heat_parameters,
                feedback_parameters,
            ),
        )


def test_backend_constructor_and_installation_identity_are_fail_closed(tmp_path) -> None:
    installation, current, heat = _fake_installation(tmp_path)
    with pytest.raises(BackendUnavailableError, match="does not exist"):
        ElmerSelfConsistentElectrothermalBackend(
            ElmerInstallation((tmp_path / "missing").resolve()),
            current,
            heat,
        )
    with pytest.raises(ContractError, match="timeout"):
        _backend(tmp_path, timeout_seconds=0.0)
    with pytest.raises(ContractError, match="convergence"):
        _backend(tmp_path, convergence_tolerance=float("nan"))
    with pytest.raises(ContractError, match="share one executable"):
        _backend(tmp_path, heat_revision="other")
    with pytest.raises(BackendUnavailableError, match="executable SHA-256"):
        _backend(tmp_path, current_executable_sha256="0" * 64, heat_executable_sha256="0" * 64)
    with pytest.raises(BackendUnavailableError, match="StatCurrentSolve SHA-256"):
        _backend(tmp_path, current_module_sha256="0" * 64)
    with pytest.raises(BackendUnavailableError, match="HeatSolve SHA-256"):
        _backend(tmp_path, heat_module_sha256="0" * 64)
    assert installation.executable.is_file()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"return_code": 4}, "return code 4"),
        ({"stdout": _stdout(completed=False)}, "completion marker"),
        ({"stdout": "MAIN: *** Elmer Solver: ALL DONE ***\n"}, "version and revision"),
        ({"stdout": _stdout(version="wrong")}, "differs from the locked"),
        ({"write_result": False}, "does not exist"),
        ({"write_vtu": False}, "raw coupled VTU"),
    ],
)
def test_backend_rejects_process_identity_and_artifact_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    options: dict[str, object],
    message: str,
) -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters, state = _bound_case()
    _install_fake_run(
        monkeypatch,
        np.asarray(state.potential),
        np.asarray(state.temperature),
        **options,  # type: ignore[arg-type]
    )
    backend = _backend(tmp_path)
    with pytest.raises(BackendError, match=message):
        backend.solve(
            backend.prepare(feedback),
            _request(
                tmp_path / message.replace(" ", "-"),
                current_parameters,
                heat_parameters,
                feedback_parameters,
            ),
        )


def test_backend_reports_missing_and_nonconverged_coupled_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters, state = _bound_case()
    backend = _backend(tmp_path)
    _install_fake_run(
        monkeypatch,
        np.asarray(state.potential),
        np.asarray(state.temperature),
        stdout=_stdout(heat_change=None),
    )
    missing = backend.solve(
        backend.prepare(feedback),
        _request(
            tmp_path / "missing",
            current_parameters,
            heat_parameters,
            feedback_parameters,
        ),
    )
    assert missing.convergence.status.value == "not_evaluated"
    assert missing.convergence.iterations is None

    _install_fake_run(
        monkeypatch,
        np.asarray(state.potential),
        np.asarray(state.temperature),
        stdout=_stdout(current_change="1.0e-3", heat_change="2.0e-3"),
    )
    nonconverged = backend.solve(
        backend.prepare(feedback),
        _request(
            tmp_path / "nonconverged",
            current_parameters,
            heat_parameters,
            feedback_parameters,
        ),
    )
    assert nonconverged.convergence.status.value == "not_converged"
    assert nonconverged.convergence.residual_norm == pytest.approx(2.0e-3)
    assert nonconverged.metadata["steady_relative_change"] == "2.00000000000000004e-03"


def test_independent_audit_rejects_invalid_fields_and_law_domain(tmp_path) -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters, state = _bound_case()
    backend = _backend(tmp_path)
    prepared = backend.prepare(feedback)
    with pytest.raises(ContractError, match="finite nodal scalar"):
        _audit_fields(
            feedback,
            prepared.current,
            prepared.heat,
            current_parameters,
            heat_parameters,
            feedback_parameters,
            np.zeros((1,)),
            np.asarray(state.temperature),
        )
    invalid_temperature = np.full_like(np.asarray(state.temperature), -100.0)
    with pytest.raises(ContractError, match="positive finite domain"):
        _audit_fields(
            feedback,
            prepared.current,
            prepared.heat,
            current_parameters,
            heat_parameters,
            feedback_parameters,
            np.asarray(state.potential),
            invalid_temperature,
        )

    zero_facet = np.asarray((0,), dtype=np.int64)
    current_with_flux = replace(
        prepared.current,
        flux_facets=(zero_facet,),
        flux_values=(0.0,),
    )
    heat_with_flux = replace(
        prepared.heat,
        flux_facets=(zero_facet,),
        flux_values=(0.0,),
    )
    audited = _audit_fields(
        feedback,
        current_with_flux,
        heat_with_flux,
        current_parameters,
        heat_parameters,
        feedback_parameters,
        np.asarray(state.potential),
        np.asarray(state.temperature),
    )
    assert audited.transfer_relative_error < 2.0e-15


def test_numeric_error_helpers_cover_zero_and_degenerate_scales() -> None:
    assert _relative_error(0.0, 0.0, 0.0) == 0.0
    assert np.isinf(_relative_error(1.0, 0.0, 0.0))
    matrix = np.zeros((2, 2), dtype=np.float64)
    state = np.zeros((2,), dtype=np.float64)
    load = np.zeros((2,), dtype=np.float64)
    free = np.asarray((0, 1), dtype=np.int64)
    assert _shifted_free_residual(matrix, state, load, free, 0.0) == 0.0
    load[0] = 1.0
    assert _shifted_free_residual(matrix, state, load, free, 0.0) == pytest.approx(1.0)
