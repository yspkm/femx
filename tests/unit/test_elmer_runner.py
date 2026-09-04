import sys
from pathlib import Path

import pytest

from femx.backends.elmer import ElmerCommand, ElmerInstallation, ElmerRunner
from femx.backends.protocol import ExecutionPolicy
from femx.core.errors import (
    BackendError,
    BackendUnavailableError,
    ContractError,
    ExecutionNotAuthorizedError,
)

pytestmark = pytest.mark.unit


def test_elmer_runner_denies_execution_by_default(tmp_path) -> None:
    runner = ElmerRunner(ElmerInstallation(Path(sys.executable).resolve()))
    with pytest.raises(ExecutionNotAuthorizedError, match=r"two independent gates|requires"):
        runner.run(ElmerCommand(), working_directory=tmp_path, policy=ExecutionPolicy())


def test_elmer_command_rejects_unsafe_arguments_and_timeouts() -> None:
    with pytest.raises(ContractError, match="NUL"):
        ElmerCommand(("bad\x00argument",))
    with pytest.raises(ContractError, match="positive"):
        ElmerCommand(timeout_seconds=0)
    with pytest.raises(ContractError, match="variable name"):
        ElmerCommand(environment={"BAD=NAME": "value"})
    with pytest.raises(ContractError, match="environment contains a NUL"):
        ElmerCommand(environment={"NAME": "bad\x00value"})


def test_elmer_runner_checks_resolved_executable_after_authorization(tmp_path) -> None:
    runner = ElmerRunner(ElmerInstallation((tmp_path / "missing").resolve()))
    policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    with pytest.raises(BackendUnavailableError, match="does not exist"):
        runner.run(ElmerCommand(), working_directory=tmp_path, policy=policy)


def test_elmer_installation_discovery_and_path_validation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("femx.backends.elmer.runner.shutil.which", lambda _name: None)
    assert ElmerInstallation.discover() is None

    monkeypatch.setattr(
        "femx.backends.elmer.runner.shutil.which", lambda _name: str(Path(sys.executable).resolve())
    )
    assert ElmerInstallation.discover() == ElmerInstallation(Path(sys.executable).resolve())
    with pytest.raises(ContractError, match="absolute"):
        ElmerInstallation(Path("relative/ElmerSolver"))

    runner = ElmerRunner(ElmerInstallation(Path(sys.executable).resolve()))
    policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    with pytest.raises(BackendError, match="working directory"):
        runner.run(ElmerCommand(), working_directory=tmp_path / "missing", policy=policy)
