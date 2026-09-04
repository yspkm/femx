from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.unit.test_elmer_tet4_case import _case

from femx.backends.elmer.runner import (
    ElmerInstallation,
    ElmerProcessResult,
    ElmerRunner,
)
from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity
from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity
from femx.backends.elmer.tet4_electrothermal import (
    ElmerTet4ElectrothermalOracle,
    ElmerTet4ElectrothermalResult,
)
from femx.core.errors import (
    BackendError,
    BackendUnavailableError,
    ContractError,
    ExecutionNotAuthorizedError,
)
from femx.core.execution import ExecutionPolicy

pytestmark = pytest.mark.unit

EXPECTED_VERSION = "26.2-devel"
EXPECTED_REVISION = "abc123"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_installation(
    tmp_path: Path,
) -> tuple[ElmerInstallation, ElmerSteadyCurrentIdentity, ElmerSteadyHeatIdentity]:
    root = tmp_path / "fake-elmer"
    executable = root / "bin" / "ElmerSolver"
    modules = root / "share" / "elmersolver" / "lib"
    current_module = modules / "StatCurrentSolve.so"
    heat_module = modules / "HeatSolve.so"
    executable.parent.mkdir(parents=True, exist_ok=True)
    modules.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"locked executable")
    current_module.write_bytes(b"locked current")
    heat_module.write_bytes(b"locked heat")
    common = {
        "version": EXPECTED_VERSION,
        "revision": EXPECTED_REVISION,
        "executable_sha256": _sha256(executable),
        "source_commit": "a" * 40,
        "source_digest": "b" * 64,
        "source_worktree_state": "clean",
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


def _oracle(tmp_path: Path, **kwargs: object) -> ElmerTet4ElectrothermalOracle:
    installation, current, heat = _fake_installation(tmp_path)
    current = replace(
        current,
        executable_sha256=kwargs.pop("current_executable_sha256", current.executable_sha256),
        stat_current_solve_sha256=kwargs.pop(
            "current_module_sha256", current.stat_current_solve_sha256
        ),
    )
    heat = replace(
        heat,
        revision=kwargs.pop("heat_revision", heat.revision),
        executable_sha256=kwargs.pop("heat_executable_sha256", heat.executable_sha256),
        heat_solve_sha256=kwargs.pop("heat_module_sha256", heat.heat_solve_sha256),
    )
    return ElmerTet4ElectrothermalOracle(
        installation,
        current,
        heat,
        **kwargs,  # type: ignore[arg-type]
    )


def _result_text() -> str:
    return (
        "ASCII 3\n"
        "!dynamic timestamp\n"
        "Degrees of freedom:\n"
        "Potential : 4 5 1 : static current\n"
        "Temperature : 5 5 1 : heat equation\n"
        "Total DOFs: 2\n"
        "Number Of Nodes: 5\n"
        "Time: 1 2 0.0\n"
        "Potential\n"
        "Perm: 5 4\n"
        "1 4\n2 3\n3 2\n4 1\n"
        "0.0\n0.25\n0.75\n1.0\n"
        "Temperature\n"
        "Perm: 5 5\n"
        "1 5\n2 4\n3 3\n4 2\n5 1\n"
        "300.0\n301.0\n302.0\n303.0\n300.0\n"
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
            f"ComputeChange: SS (ITER=2) (NRM,RELC): ( 1.0 {current_change} ) :: static current"
        )
    if heat_change is not None:
        lines.append(
            f"ComputeChange: SS (ITER=2) (NRM,RELC): ( 300.0 {heat_change} ) :: heat equation"
        )
    if completed:
        lines.append("MAIN: *** Elmer Solver: ALL DONE ***")
    return "\n".join(lines) + "\n"


def _fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    return_code: int = 0,
    stdout: str | None = None,
    write_result: bool = True,
    write_vtu: bool = True,
) -> None:
    def run(self, command, *, working_directory, policy):
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
                _result_text(), encoding="utf-8"
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

    monkeypatch.setattr(ElmerRunner, "run", run)


def _authorized() -> ExecutionPolicy:
    return ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def test_tet4_oracle_retains_partial_fields_convergence_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fake_run(monkeypatch)
    oracle = _oracle(tmp_path)
    run_directory = tmp_path / "attempt"

    result = oracle.run(_case(), run_directory=run_directory, policy=_authorized())

    assert oracle.version.startswith("adapter-0.1.0+elmer-26.2-devel")
    assert result.process.process_succeeded
    assert result.numerical_convergence_evaluated
    assert result.numerically_converged
    np.testing.assert_array_equal(result.potential_node_ids, (0, 1, 2, 3))
    np.testing.assert_allclose(result.potential_v, (0.0, 0.25, 0.75, 1.0))
    np.testing.assert_allclose(result.temperature_k, (300.0, 301.0, 302.0, 303.0, 300.0))
    assert not result.potential_v.flags.writeable
    assert result.provenance["elmer_source_worktree_state"] == "clean"
    assert result.provenance["result_save_count"] == "1"
    assert result.provenance["result_timestep"] == "2"
    for name in (
        "case_sha256",
        "elmer_executable_sha256",
        "elmer_stat_current_solve_sha256",
        "elmer_heat_solve_sha256",
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
        assert len(result.provenance[name]) == 64
    with pytest.raises(TypeError):
        result.provenance["other"] = "value"  # type: ignore[index]
    sif = (run_directory / "case.sif").read_text(encoding="utf-8")
    assert "Coordinate System = Cartesian 3D" in sif
    assert sif.count("Joule Heat = Logical True") == 1


def test_tet4_oracle_distinguishes_missing_and_failed_convergence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    oracle = _oracle(tmp_path)
    _fake_run(monkeypatch, stdout=_stdout(heat_change=None))
    missing = oracle.run(_case(), run_directory=tmp_path / "missing", policy=_authorized())
    assert not missing.numerical_convergence_evaluated
    assert not missing.numerically_converged

    _fake_run(monkeypatch, stdout=_stdout(current_change="1.0e-3"))
    failed = oracle.run(_case(), run_directory=tmp_path / "failed", policy=_authorized())
    assert failed.numerical_convergence_evaluated
    assert not failed.numerically_converged


def test_tet4_oracle_requires_authorization_before_filesystem_mutation(tmp_path: Path) -> None:
    oracle = _oracle(tmp_path)
    attempt = tmp_path / "denied"
    with pytest.raises(ExecutionNotAuthorizedError, match="requires"):
        oracle.run(_case(), run_directory=attempt, policy=ExecutionPolicy())
    assert not attempt.exists()
    with pytest.raises(ContractError, match="typed electrothermal case"):
        oracle.run(  # type: ignore[arg-type]
            object(), run_directory=tmp_path / "wrong", policy=_authorized()
        )


def test_tet4_oracle_constructor_checks_shared_locked_installation(tmp_path: Path) -> None:
    installation, current, heat = _fake_installation(tmp_path)
    with pytest.raises(BackendUnavailableError, match="does not exist"):
        ElmerTet4ElectrothermalOracle(
            ElmerInstallation((tmp_path / "missing").resolve()), current, heat
        )
    with pytest.raises(ContractError, match="timeout"):
        _oracle(tmp_path, timeout_seconds=0.0)
    with pytest.raises(ContractError, match="convergence tolerance"):
        _oracle(tmp_path, convergence_tolerance=float("nan"))
    with pytest.raises(ContractError, match="share one executable"):
        _oracle(tmp_path, heat_revision="other")
    with pytest.raises(BackendUnavailableError, match="executable SHA-256"):
        _oracle(
            tmp_path,
            current_executable_sha256="0" * 64,
            heat_executable_sha256="0" * 64,
        )
    with pytest.raises(BackendUnavailableError, match="StatCurrentSolve SHA-256"):
        _oracle(tmp_path, current_module_sha256="0" * 64)
    with pytest.raises(BackendUnavailableError, match="HeatSolve SHA-256"):
        _oracle(tmp_path, heat_module_sha256="0" * 64)
    assert installation.executable.is_file()


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"return_code": 3}, "return code 3"),
        ({"stdout": _stdout(completed=False)}, "completion marker"),
        ({"stdout": "MAIN: *** Elmer Solver: ALL DONE ***\n"}, "version and revision"),
        ({"stdout": _stdout(version="wrong")}, "differs from the Tet4"),
        ({"write_result": False}, "does not exist"),
        ({"write_vtu": False}, "raw Tet4 VTU"),
    ),
)
def test_tet4_oracle_rejects_process_identity_and_artifact_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: dict[str, object],
    message: str,
) -> None:
    _fake_run(monkeypatch, **options)  # type: ignore[arg-type]
    oracle = _oracle(tmp_path)
    with pytest.raises(BackendError, match=message):
        oracle.run(
            _case(),
            run_directory=tmp_path / message.replace(" ", "-"),
            policy=_authorized(),
        )


def test_tet4_result_rejects_invalid_numeric_contract() -> None:
    process = ElmerProcessResult(("ElmerSolver",), 0, "", "", 0.1)
    valid = ElmerTet4ElectrothermalResult(
        potential_node_ids=np.asarray((0, 2)),
        potential_v=np.asarray((0.0, 1.0)),
        temperature_k=np.asarray((300.0, 301.0, 302.0)),
        current_steady_change=(1, 0.0),
        heat_steady_change=(1, 0.0),
        convergence_tolerance=1.0e-12,
        process=process,
        provenance={},
    )
    for changes in (
        {"potential_node_ids": np.asarray(((0, 2),))},
        {"potential_v": np.asarray((0.0,))},
        {"temperature_k": np.asarray(((300.0,),))},
        {"potential_node_ids": np.asarray(())},
        {"potential_node_ids": np.asarray((0, 0))},
        {"potential_node_ids": np.asarray((-1, 2))},
        {"potential_node_ids": np.asarray((0, 3))},
        {"potential_v": np.asarray((0.0, np.nan))},
        {"temperature_k": np.asarray((300.0, np.inf, 302.0))},
    ):
        with pytest.raises(ContractError, match="finite canonical nodal vectors"):
            replace(valid, **changes)
    with pytest.raises(ContractError, match="convergence tolerance"):
        replace(valid, convergence_tolerance=0.0)
    with pytest.raises(ContractError, match="external-process evidence"):
        replace(valid, process=object())  # type: ignore[arg-type]
