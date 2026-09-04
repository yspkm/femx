import sys
from pathlib import Path

import pytest

from femx.backends.elmer import ElmerCommand, ElmerInstallation, ElmerRunner
from femx.backends.protocol import ExecutionPolicy
from femx.core.errors import BackendError

pytestmark = pytest.mark.integration


def test_guarded_runner_executes_without_a_shell(tmp_path) -> None:
    runner = ElmerRunner(ElmerInstallation(Path(sys.executable).resolve()))
    result = runner.run(
        ElmerCommand(("-c", "print('process-boundary-ok')"), timeout_seconds=5),
        working_directory=tmp_path,
        policy=ExecutionPolicy(execution_authorized=True, allow_external_process=True),
    )

    assert result.process_succeeded
    assert result.stdout.strip() == "process-boundary-ok"
    assert result.stderr == ""
    assert result.argv[0] == str(Path(sys.executable).resolve())


def test_guarded_runner_reports_nonzero_and_timeout_without_claiming_convergence(tmp_path) -> None:
    runner = ElmerRunner(ElmerInstallation(Path(sys.executable).resolve()))
    policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    failed = runner.run(
        ElmerCommand(("-c", "raise SystemExit(3)"), timeout_seconds=5),
        working_directory=tmp_path,
        policy=policy,
    )
    assert not failed.process_succeeded
    assert failed.return_code == 3

    with pytest.raises(BackendError, match="timed out"):
        runner.run(
            ElmerCommand(("-c", "import time; time.sleep(1)"), timeout_seconds=0.01),
            working_directory=tmp_path,
            policy=policy,
        )
