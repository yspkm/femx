import subprocess
import sys
from pathlib import Path

import pytest

from femx.core.errors import (
    ContractError,
    ExecutionNotAuthorizedError,
    MesherUnavailableError,
    MeshingError,
)
from femx.core.execution import ExecutionPolicy
from femx.meshing.gmsh import (
    GmshInstallation,
    GmshMeshingRequest,
    GmshProcessResult,
    GmshRunner,
    GmshToolIdentity,
)

pytestmark = pytest.mark.unit

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def test_gmsh_runner_denies_external_execution_by_default(tmp_path) -> None:
    runner = GmshRunner(GmshInstallation(Path(sys.executable).resolve()))
    with pytest.raises(ExecutionNotAuthorizedError, match=r"two independent gates|requires"):
        runner.run(
            GmshMeshingRequest("model.geo"),
            working_directory=tmp_path,
            policy=ExecutionPolicy(),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"geometry_filename": "../model.geo"}, "leaf"),
        ({"geometry_filename": "model.txt"}, ".geo"),
        ({"geometry_filename": "model.geo", "mesh_filename": "mesh.txt"}, ".msh"),
        ({"geometry_filename": "model.geo", "timeout_seconds": 0.0}, "positive"),
        ({"geometry_filename": "model.geo", "dimension": 4}, "two or three dimensions"),
    ],
)
def test_gmsh_request_rejects_ambiguous_scope(kwargs, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        GmshMeshingRequest(**kwargs)


def test_gmsh_identity_and_installation_validate_exact_values() -> None:
    with pytest.raises(ContractError, match="absolute"):
        GmshInstallation(Path("gmsh"))
    with pytest.raises(ContractError, match="version"):
        GmshToolIdentity("unknown", "a" * 64)
    with pytest.raises(ContractError, match="SHA-256"):
        GmshToolIdentity("4.12.1", "bad")


def test_gmsh_discovery_is_read_only(monkeypatch) -> None:
    monkeypatch.setattr("femx.meshing.gmsh.runner.shutil.which", lambda _name: None)
    assert GmshInstallation.discover() is None
    executable = Path(sys.executable).resolve()
    monkeypatch.setattr("femx.meshing.gmsh.runner.shutil.which", lambda _name: str(executable))
    assert GmshInstallation.discover() == GmshInstallation(executable)


def test_gmsh_runner_checks_paths_before_process_execution(tmp_path) -> None:
    executable = Path(sys.executable).resolve()
    runner = GmshRunner(GmshInstallation(executable))
    missing_runner = GmshRunner(GmshInstallation((tmp_path / "missing-gmsh").resolve()))
    with pytest.raises(MesherUnavailableError, match="does not exist"):
        missing_runner.run(
            GmshMeshingRequest("model.geo"),
            working_directory=tmp_path,
            policy=_AUTHORIZED,
        )
    with pytest.raises(MeshingError, match="working directory"):
        runner.run(
            GmshMeshingRequest("model.geo"),
            working_directory=tmp_path / "missing",
            policy=_AUTHORIZED,
        )
    with pytest.raises(MeshingError, match="geometry file"):
        runner.run(
            GmshMeshingRequest("model.geo"),
            working_directory=tmp_path,
            policy=_AUTHORIZED,
        )
    (tmp_path / "model.geo").write_text("// model", encoding="utf-8")
    (tmp_path / "mesh.msh").write_text("existing", encoding="utf-8")
    with pytest.raises(MeshingError, match="overwrite"):
        runner.run(
            GmshMeshingRequest("model.geo"),
            working_directory=tmp_path,
            policy=_AUTHORIZED,
        )


@pytest.mark.parametrize(("dimension", "dimension_flag"), ((2, "-2"), (3, "-3")))
def test_gmsh_runner_records_shell_free_deterministic_process_evidence(
    tmp_path, monkeypatch, dimension: int, dimension_flag: str
) -> None:
    executable = Path(sys.executable).resolve()
    geometry = tmp_path / "model.geo"
    geometry.write_text("// geometry", encoding="utf-8")
    observed: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        normalized = tuple(argv)
        observed.append((normalized, kwargs))
        if normalized[-1] == "--version":
            return subprocess.CompletedProcess(normalized, 0, "4.12.1\n", "")
        (tmp_path / "mesh.msh").write_text("mesh payload", encoding="utf-8")
        return subprocess.CompletedProcess(normalized, 0, "generated", "")

    monkeypatch.setattr("femx.meshing.gmsh.runner.subprocess.run", fake_run)
    runner = GmshRunner(GmshInstallation(executable))
    result = runner.run(
        GmshMeshingRequest("model.geo", dimension=dimension),
        working_directory=tmp_path,
        policy=_AUTHORIZED,
    )

    assert result.process_succeeded
    assert result.return_code == 0
    assert result.identity.version == "4.12.1"
    assert len(result.identity.executable_sha256) == 64
    assert len(result.geometry_sha256) == 64
    assert result.mesh_sha256 is not None and len(result.mesh_sha256) == 64
    assert result.environment_overrides == (("LC_ALL", "C"), ("OMP_NUM_THREADS", "1"))
    assert result.argv[1:] == (
        "model.geo",
        dimension_flag,
        "-format",
        "msh41",
        "-order",
        "1",
        "-setnumber",
        "Mesh.Binary",
        "0",
        "-nt",
        "1",
        "-o",
        "mesh.msh",
    )
    assert all(call[1]["shell"] is False for call in observed)
    assert all(call[1]["check"] is False for call in observed)
    assert all(call[1]["cwd"] == tmp_path for call in observed)


def test_gmsh_runner_keeps_nonzero_process_status_separate(tmp_path, monkeypatch) -> None:
    executable = Path(sys.executable).resolve()
    (tmp_path / "model.geo").write_text("// geometry", encoding="utf-8")

    def fake_run(argv, **_kwargs):
        normalized = tuple(argv)
        if normalized[-1] == "--version":
            return subprocess.CompletedProcess(normalized, 0, "4.12.1\n", "")
        return subprocess.CompletedProcess(normalized, 2, "", "meshing failed")

    monkeypatch.setattr("femx.meshing.gmsh.runner.subprocess.run", fake_run)
    result = GmshRunner(GmshInstallation(executable)).run(
        GmshMeshingRequest("model.geo"), working_directory=tmp_path, policy=_AUTHORIZED
    )
    assert not result.process_succeeded
    assert result.mesh_sha256 is None
    assert result.stderr == "meshing failed"


@pytest.mark.parametrize("during_version", [True, False])
def test_gmsh_runner_reports_timeouts(tmp_path, monkeypatch, during_version: bool) -> None:
    executable = Path(sys.executable).resolve()
    (tmp_path / "model.geo").write_text("// geometry", encoding="utf-8")
    calls = 0

    def fake_run(argv, **_kwargs):
        nonlocal calls
        calls += 1
        if during_version or calls == 2:
            raise subprocess.TimeoutExpired(argv, timeout=1.0)
        return subprocess.CompletedProcess(tuple(argv), 0, "4.12.1\n", "")

    monkeypatch.setattr("femx.meshing.gmsh.runner.subprocess.run", fake_run)
    with pytest.raises(MeshingError, match=r"version probe timed out|timed out"):
        GmshRunner(GmshInstallation(executable)).run(
            GmshMeshingRequest("model.geo", timeout_seconds=1.0),
            working_directory=tmp_path,
            policy=_AUTHORIZED,
        )


@pytest.mark.parametrize(
    ("return_code", "stdout"),
    [(1, "4.12.1\n"), (0, "not-a-version\n")],
)
def test_gmsh_runner_rejects_unusable_version_identity(
    tmp_path, monkeypatch, return_code: int, stdout: str
) -> None:
    executable = Path(sys.executable).resolve()
    (tmp_path / "model.geo").write_text("// geometry", encoding="utf-8")
    monkeypatch.setattr(
        "femx.meshing.gmsh.runner.subprocess.run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(tuple(argv), return_code, stdout, ""),
    )
    with pytest.raises(MeshingError, match="version probe failed"):
        GmshRunner(GmshInstallation(executable)).run(
            GmshMeshingRequest("model.geo"), working_directory=tmp_path, policy=_AUTHORIZED
        )


def test_gmsh_runner_rejects_executable_identity_change(tmp_path, monkeypatch) -> None:
    executable = Path(sys.executable).resolve()
    (tmp_path / "model.geo").write_text("// geometry", encoding="utf-8")
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr("femx.meshing.gmsh.runner.sha256_file", lambda _path: next(digests))

    def fake_run(argv, **_kwargs):
        normalized = tuple(argv)
        if normalized[-1] == "--version":
            return subprocess.CompletedProcess(normalized, 0, "4.12.1\n", "")
        return subprocess.CompletedProcess(normalized, 0, "", "")

    monkeypatch.setattr("femx.meshing.gmsh.runner.subprocess.run", fake_run)
    with pytest.raises(MeshingError, match="changed"):
        GmshRunner(GmshInstallation(executable)).run(
            GmshMeshingRequest("model.geo"), working_directory=tmp_path, policy=_AUTHORIZED
        )


def test_process_result_distinguishes_return_code_only() -> None:
    result = GmshProcessResult(
        argv=("gmsh",),
        return_code=0,
        stdout="",
        stderr="",
        elapsed_seconds=0.1,
        identity=GmshToolIdentity("4.12.1", "a" * 64),
        geometry_sha256="b" * 64,
        mesh_sha256=None,
        environment_overrides=(),
    )
    assert result.process_succeeded
