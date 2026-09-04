from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends._steady_current import (
    postprocess_current_potential,
    validate_steady_current_problem,
)
from femx.backends.elmer.runner import (
    ElmerInstallation,
    ElmerProcessResult,
    ElmerRunner,
)
from femx.backends.elmer.steady_current import (
    ElmerSteadyCurrentBackend,
    ElmerSteadyCurrentIdentity,
)
from femx.backends.protocol import (
    BackendDescriptor,
    ExecutionPolicy,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.errors import (
    BackendError,
    BackendUnavailableError,
    ContractError,
    ExecutionNotAuthorizedError,
)
from femx.core.problem import Problem
from femx.physics import (
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.runtime import prepare, solve

pytestmark = pytest.mark.unit

EXPECTED_VERSION = "26.2-devel"
EXPECTED_REVISION = "abc123"
SOURCE_COMMIT = "a" * 40
SOURCE_DIGEST = "b" * 64


def _problem() -> Problem:
    return Problem(
        "elmer-current",
        structured_unit_square_mesh(1),
        SteadyCurrent(
            regions=(ConductiveRegion("domain", 2.0),),
            potential_boundaries=(PotentialBoundary("left", 0.0),),
            current_flux_boundaries=(CurrentFluxBoundary("right", 2.0),),
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_installation(
    tmp_path: Path,
) -> tuple[ElmerInstallation, ElmerSteadyCurrentIdentity]:
    root = tmp_path / "fake-elmer-install"
    executable = root / "bin" / "ElmerSolver"
    stat_current = root / "share" / "elmersolver" / "lib" / "StatCurrentSolve.so"
    executable.parent.mkdir(parents=True, exist_ok=True)
    stat_current.parent.mkdir(parents=True, exist_ok=True)
    if not executable.exists():
        executable.write_bytes(b"locked fake ElmerSolver\n")
    if not stat_current.exists():
        stat_current.write_bytes(b"locked fake StatCurrentSolve\n")
    identity = ElmerSteadyCurrentIdentity(
        version=EXPECTED_VERSION,
        revision=EXPECTED_REVISION,
        executable_sha256=_sha256(executable),
        stat_current_solve_sha256=_sha256(stat_current),
        source_commit=SOURCE_COMMIT,
        source_digest=SOURCE_DIGEST,
    )
    return ElmerInstallation(executable.resolve()), identity


def _backend(tmp_path: Path, **kwargs) -> ElmerSteadyCurrentBackend:
    installation, default_identity = _fake_installation(tmp_path)
    identity = ElmerSteadyCurrentIdentity(
        version=kwargs.pop("expected_version", default_identity.version),
        revision=kwargs.pop("expected_revision", default_identity.revision),
        executable_sha256=kwargs.pop(
            "expected_executable_sha256",
            default_identity.executable_sha256,
        ),
        stat_current_solve_sha256=kwargs.pop(
            "expected_stat_current_sha256",
            default_identity.stat_current_solve_sha256,
        ),
        source_commit=kwargs.pop("source_commit", default_identity.source_commit),
        source_digest=kwargs.pop("source_digest", default_identity.source_digest),
        source_worktree_state=kwargs.pop(
            "source_worktree_state",
            default_identity.source_worktree_state,
        ),
    )
    return ElmerSteadyCurrentBackend(installation, identity, **kwargs)


def _result_text(values: tuple[float, ...] = (0.0, 1.0, 0.0, 1.0)) -> str:
    count = len(values)
    pairs = "\n".join(f"{index} {count + 1 - index}" for index in range(1, count + 1))
    encoded_values = "\n".join(format(value, ".17e") for value in values)
    return (
        " ASCII 3\n"
        "!dynamic timestamp\n"
        " Degrees of freedom:\n"
        f"Potential : {count} {count} 1 : static current\n"
        " Total DOFs: 1\n"
        f" Number Of Nodes: {count}\n"
        "Time: 1 1 1.00000000E+000\n"
        "Potential\n"
        f"Perm: {count} {count}\n"
        f"{pairs}\n"
        f"{encoded_values}\n"
    )


def _stdout(
    *,
    version: str = EXPECTED_VERSION,
    revision: str = EXPECTED_REVISION,
    relative_change: str | None = "0.0000000",
    completed: bool = True,
) -> str:
    lines = [
        f"MAIN: Version: {version} (Rev: {revision}, Compiled: test)",
        "ComputeChange: SS (ITER=1) (NRM,RELC): ( 0.5 2.0 ) :: static current",
    ]
    if relative_change is not None:
        lines.append(
            f"ComputeChange: SS (ITER=2) (NRM,RELC): ( 0.5 {relative_change} ) :: static current"
        )
    if completed:
        lines.append("MAIN: *** Elmer Solver: ALL DONE ***")
    return "\n".join(lines) + "\n"


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    return_code: int = 0,
    stdout: str | None = None,
    write_result: bool = True,
    write_vtu: bool = True,
) -> None:
    def fake_run(self, command, *, working_directory, policy):
        assert command.arguments == ()
        root = self.installation.executable.resolve().parent.parent
        modules = root / "share" / "elmersolver" / "lib"
        assert command.environment["ELMER_HOME"] == str(root)
        assert command.environment["ELMER_LIB"] == str(modules)
        assert command.environment["ELMER_MODULES_PATH"] == str(modules)
        policy.require_external_process(component_name="elmer-steady-current")
        if write_result:
            (working_directory / "mesh" / "femx.result").write_text(
                _result_text(),
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


def _authorized_request(run_directory: Path) -> SolveRequest:
    return SolveRequest(
        run_directory=run_directory,
        policy=ExecutionPolicy(execution_authorized=True, allow_external_process=True),
    )


def test_backend_prepare_is_pure_and_solve_reconstructs_physical_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_run(monkeypatch)
    run_directory = tmp_path / "attempt-001"
    backend = _backend(tmp_path)
    prepared = prepare(
        _problem(),
        backend,
        request=PrepareRequest(run_directory=run_directory),
    )
    assert not run_directory.exists()

    solution = solve(prepared, backend, request=_authorized_request(run_directory))

    np.testing.assert_array_equal(
        solution.fields["potential"].values,
        np.asarray((0.0, 1.0, 0.0, 1.0)),
    )
    np.testing.assert_allclose(
        solution.fields["electric_field"].values,
        np.tile((-1.0, 0.0), (2, 1)),
    )
    np.testing.assert_allclose(
        solution.fields["current_density"].values,
        np.tile((-2.0, 0.0), (2, 1)),
    )
    np.testing.assert_allclose(solution.fields["joule_heat_density"].values, 2.0)
    assert solution.observables["joule_power_W_per_m"] == pytest.approx(2.0)
    assert solution.observables["variational_input_power_W_per_m"] == pytest.approx(2.0)
    assert solution.observables["energy_balance_relative_error"] < 1.0e-15
    assert solution.convergence.status.value == "converged"
    assert solution.convergence.iterations == 2
    assert solution.metadata["elmer_revision"] == EXPECTED_REVISION
    assert len(solution.metadata["elmer_stat_current_solve_sha256"]) == 64
    assert solution.metadata["elmer_source_commit"] == SOURCE_COMMIT
    assert solution.metadata["elmer_source_digest"] == SOURCE_DIGEST
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
    case = (run_directory / "case.sif").read_text(encoding="utf-8")
    assert 'StatCurrentSolve.so" "StatCurrentSolver"' in case
    assert "Current Density = Real 2.00000000000000000e+00" in case
    assert (run_directory / "elmer.stdout.log").read_text(encoding="utf-8") == _stdout()
    assert (run_directory / "elmer.stderr.log").read_text(encoding="utf-8") == "fake-stderr"


def test_backend_requires_authorization_before_writing(tmp_path) -> None:
    backend = _backend(tmp_path)
    run_directory = tmp_path / "denied"
    prepared = backend.prepare(_problem(), PrepareRequest(run_directory=run_directory))

    with pytest.raises(ExecutionNotAuthorizedError, match="requires"):
        backend.solve(prepared, SolveRequest(run_directory=run_directory))
    assert not run_directory.exists()


def test_backend_validates_constructor_identity_and_payload(tmp_path) -> None:
    missing = ElmerInstallation((tmp_path / "missing").resolve())
    _, identity = _fake_installation(tmp_path)
    with pytest.raises(BackendUnavailableError, match="does not exist"):
        ElmerSteadyCurrentBackend(missing, identity)
    with pytest.raises(ContractError, match="timeout"):
        _backend(tmp_path, timeout_seconds=0.0)
    with pytest.raises(ContractError, match="convergence"):
        _backend(tmp_path, convergence_tolerance=float("nan"))
    with pytest.raises(ContractError, match="version"):
        _backend(tmp_path, expected_version=" bad ")
    with pytest.raises(ContractError, match="revision"):
        _backend(tmp_path, expected_revision="bad\nrevision")
    with pytest.raises(ContractError, match="SHA-256"):
        _backend(tmp_path, expected_stat_current_sha256="bad")
    with pytest.raises(ContractError, match="Git SHA-1"):
        _backend(tmp_path, source_commit="short")
    with pytest.raises(ContractError, match="worktree state"):
        _backend(tmp_path, source_worktree_state="unknown")
    with pytest.raises(BackendUnavailableError, match="executable SHA-256"):
        _backend(tmp_path, expected_executable_sha256="0" * 64)
    with pytest.raises(BackendUnavailableError, match="StatCurrentSolve SHA-256"):
        _backend(tmp_path, expected_stat_current_sha256="0" * 64)

    backend = _backend(tmp_path)
    valid = backend.prepare(_problem(), PrepareRequest())
    with pytest.raises(BackendError, match="identity"):
        backend.solve(
            PreparedProblem(BackendDescriptor("other", "1"), valid.problem, valid.payload),
            _authorized_request(tmp_path / "wrong-id"),
        )
    with pytest.raises(BackendError, match="payload"):
        backend.solve(
            PreparedProblem(backend.descriptor, valid.problem, object()),
            _authorized_request(tmp_path / "wrong-payload"),
        )


def test_backend_translates_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    installation, identity = _fake_installation(tmp_path)
    executable = installation.executable
    original_open = Path.open

    def unreadable_executable(self, *args, **kwargs):
        if self == executable:
            raise OSError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable_executable)
    with pytest.raises(BackendUnavailableError, match="cannot read"):
        ElmerSteadyCurrentBackend(installation, identity)
    monkeypatch.setattr(Path, "open", original_open)

    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    original_write_text = Path.write_text

    def unwritable_case(self, *args, **kwargs):
        if self.name == "case.sif":
            raise OSError("denied")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", unwritable_case)
    with pytest.raises(BackendError, match="cannot write"):
        backend.solve(prepared, _authorized_request(tmp_path / "unwritable"))


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        (Path("bin/ElmerSolver"), "executable SHA-256"),
        (Path("share/elmersolver/lib/StatCurrentSolve.so"), "StatCurrentSolve SHA-256"),
    ],
)
def test_backend_reverifies_installed_files_before_execution(
    tmp_path,
    relative_path: Path,
    message: str,
) -> None:
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    (tmp_path / "fake-elmer-install" / relative_path).write_bytes(b"replaced\n")
    run_directory = tmp_path / "identity-mismatch"

    with pytest.raises(BackendUnavailableError, match=message):
        backend.solve(prepared, _authorized_request(run_directory))
    assert not run_directory.exists()


def test_backend_requires_one_fresh_absolute_attempt_directory(tmp_path) -> None:
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    with pytest.raises(ContractError, match="explicit durable"):
        backend.solve(prepared, SolveRequest(policy=policy))

    different = backend.prepare(_problem(), PrepareRequest(run_directory=tmp_path / "prepared"))
    with pytest.raises(ContractError, match="different"):
        backend.solve(different, _authorized_request(tmp_path / "solved"))

    relative = Path("relative-attempt")
    with pytest.raises(ContractError, match="absolute"):
        backend.solve(prepared, _authorized_request(relative))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(BackendError, match="empty"):
        backend.solve(prepared, _authorized_request(occupied))

    regular_file = tmp_path / "regular-file"
    regular_file.write_text("owned", encoding="utf-8")
    with pytest.raises(BackendError, match="not a directory"):
        backend.solve(prepared, _authorized_request(regular_file))

    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink = tmp_path / "symlink-attempt"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(BackendError, match="symbolic link"):
        backend.solve(prepared, _authorized_request(symlink))

    with pytest.raises(BackendError, match="cannot create"):
        backend.solve(prepared, _authorized_request(tmp_path / "missing-parent" / "attempt"))


def test_backend_accepts_existing_empty_attempt_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_run(monkeypatch)
    run_directory = tmp_path / "empty-attempt"
    run_directory.mkdir()
    backend = _backend(tmp_path)
    solution = backend.solve(
        backend.prepare(_problem(), PrepareRequest()),
        _authorized_request(run_directory),
    )
    assert solution.convergence.status.value == "converged"


@pytest.mark.parametrize(
    ("run_options", "message"),
    [
        ({"return_code": 3}, "return code 3"),
        ({"stdout": _stdout(completed=False)}, "completion marker"),
        ({"stdout": "MAIN: *** Elmer Solver: ALL DONE ***\n"}, "version and revision"),
        ({"stdout": _stdout(version="wrong", revision="wrong")}, "differs from the locked"),
        ({"write_result": False}, "does not exist"),
        ({"write_vtu": False}, "raw VTU"),
    ],
)
def test_backend_fails_closed_on_process_identity_and_artifact_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    run_options: dict[str, object],
    message: str,
) -> None:
    _install_fake_run(monkeypatch, **run_options)  # type: ignore[arg-type]
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    run_directory = tmp_path / message.replace(" ", "-")
    with pytest.raises(BackendError, match=message):
        backend.solve(prepared, _authorized_request(run_directory))
    assert (run_directory / "case.sif").is_file()
    assert (run_directory / "elmer.stdout.log").is_file()


def test_backend_reports_missing_and_nonconverged_steady_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_run(
        monkeypatch,
        stdout=(
            f"MAIN: Version: {EXPECTED_VERSION} (Rev: {EXPECTED_REVISION}, Compiled: test)\n"
            "MAIN: *** Elmer Solver: ALL DONE ***\n"
        ),
    )
    backend = _backend(tmp_path)
    missing = backend.solve(
        backend.prepare(_problem(), PrepareRequest()),
        _authorized_request(tmp_path / "missing-change"),
    )
    assert missing.convergence.status.value == "not_evaluated"
    assert missing.convergence.iterations is None

    _install_fake_run(monkeypatch, stdout=_stdout(relative_change="1.0E-3"))
    nonconverged = backend.solve(
        backend.prepare(_problem(), PrepareRequest()),
        _authorized_request(tmp_path / "nonconverged"),
    )
    assert nonconverged.convergence.status.value == "not_converged"
    assert nonconverged.metadata["steady_relative_change"] == "1.00000000000000002e-03"


@pytest.mark.parametrize("relative_change", ["NaN", "-1.0", "broken"])
def test_backend_rejects_invalid_steady_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    relative_change: str,
) -> None:
    _install_fake_run(monkeypatch, stdout=_stdout(relative_change=relative_change))
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    message = "invalid" if relative_change == "broken" else "non-finite or negative"
    with pytest.raises(BackendError, match=message):
        backend.solve(
            prepared,
            _authorized_request(tmp_path / f"bad-change-{relative_change}"),
        )


def test_independent_postprocessor_rejects_invalid_fields_and_handles_zero_energy() -> None:
    problem = Problem(
        "zero-current",
        structured_unit_square_mesh(1),
        SteadyCurrent(
            regions=(ConductiveRegion("domain", 1.0),),
            potential_boundaries=(PotentialBoundary("left", 0.0),),
        ),
    )
    validated = validate_steady_current_problem(problem)
    parameters = problem.parameters.bind({})

    zero = postprocess_current_potential(
        validated,
        parameters,
        np.zeros(4, dtype=np.float64),
    )

    assert zero.relative_backward_error == 0.0
    assert zero.joule_power == 0.0
    assert zero.energy_balance_relative_error == 0.0
    for invalid in (np.zeros(3), np.asarray((0.0, 0.0, np.nan, 0.0))):
        with pytest.raises(ContractError, match="finite scalar"):
            postprocess_current_potential(validated, parameters, invalid)
