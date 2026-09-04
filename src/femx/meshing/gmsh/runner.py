"""Guarded, deterministic Gmsh command-line process boundary."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePath

from femx.artifacts import sha256_file
from femx.core.errors import ContractError, MesherUnavailableError, MeshingError
from femx.core.execution import ExecutionPolicy

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9.-]+)?$")


@dataclass(frozen=True, slots=True)
class GmshInstallation:
    """Resolved Gmsh executable without executing it."""

    executable: Path

    @classmethod
    def discover(cls, executable_name: str = "gmsh") -> GmshInstallation | None:
        """Locate Gmsh on PATH without starting a process."""

        resolved = shutil.which(executable_name)
        return None if resolved is None else cls(Path(resolved).resolve())

    def __post_init__(self) -> None:
        if not self.executable.is_absolute():
            raise ContractError("Gmsh executable must be an absolute path")


@dataclass(frozen=True, slots=True)
class GmshToolIdentity:
    """Exact executable identity observed during one meshing attempt."""

    version: str
    executable_sha256: str

    def __post_init__(self) -> None:
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ContractError(f"invalid Gmsh version: {self.version!r}")
        if not _SHA256_PATTERN.fullmatch(self.executable_sha256):
            raise ContractError("Gmsh executable identity requires a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class GmshMeshingRequest:
    """Fixed-scope request for an ASCII MSH 4.1 first-order surface or volume mesh."""

    geometry_filename: str
    mesh_filename: str = "mesh.msh"
    timeout_seconds: float | None = 120.0
    dimension: int = 2

    def __post_init__(self) -> None:
        for label, name, suffix in (
            ("geometry", self.geometry_filename, ".geo"),
            ("mesh", self.mesh_filename, ".msh"),
        ):
            if not name or PurePath(name).name != name or "\x00" in name:
                raise ContractError(f"Gmsh {label} filename must be one safe leaf name")
            if not name.lower().endswith(suffix):
                raise ContractError(f"Gmsh {label} filename must end with {suffix}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0.0:
            raise ContractError("Gmsh timeout must be positive")
        if self.dimension not in (2, 3):
            raise ContractError("the Gmsh adapter supports exactly two or three dimensions")


@dataclass(frozen=True, slots=True)
class GmshProcessResult:
    """Process evidence separate from mesh ingestion and scientific validation."""

    argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    identity: GmshToolIdentity
    geometry_sha256: str
    mesh_sha256: str | None
    environment_overrides: tuple[tuple[str, str], ...]

    @property
    def process_succeeded(self) -> bool:
        """Whether Gmsh returned zero; this is not mesh-validity evidence."""

        return self.return_code == 0


class GmshRunner:
    """Generate only the narrow deterministic mesh format accepted by femx v1."""

    _ENVIRONMENT_OVERRIDES = (("LC_ALL", "C"), ("OMP_NUM_THREADS", "1"))

    def __init__(self, installation: GmshInstallation) -> None:
        self._installation = installation

    @property
    def installation(self) -> GmshInstallation:
        """Return the immutable executable location."""

        return self._installation

    def run(
        self,
        request: GmshMeshingRequest,
        *,
        working_directory: Path,
        policy: ExecutionPolicy,
    ) -> GmshProcessResult:
        """Generate one mesh without a shell after authorization and path checks."""

        policy.require_external_process(component_name="gmsh")
        executable = self.installation.executable
        if not executable.is_file():
            raise MesherUnavailableError(f"Gmsh executable does not exist: {executable}")
        if not working_directory.is_dir():
            raise MeshingError(f"Gmsh working directory does not exist: {working_directory}")

        geometry_path = working_directory / request.geometry_filename
        mesh_path = working_directory / request.mesh_filename
        if not geometry_path.is_file():
            raise MeshingError(f"Gmsh geometry file does not exist: {geometry_path}")
        if mesh_path.exists():
            raise MeshingError(f"Gmsh refuses to overwrite an existing mesh: {mesh_path}")

        executable_sha256 = sha256_file(executable)
        identity = GmshToolIdentity(
            version=self._read_version(
                executable,
                working_directory=working_directory,
                timeout_seconds=request.timeout_seconds,
            ),
            executable_sha256=executable_sha256,
        )
        argv = (
            str(executable),
            request.geometry_filename,
            f"-{request.dimension}",
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
            request.mesh_filename,
        )
        environment = os.environ.copy()
        environment.update(dict(self._ENVIRONMENT_OVERRIDES))
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise MeshingError(f"Gmsh timed out after {request.timeout_seconds} seconds") from error
        elapsed = time.monotonic() - started

        if sha256_file(executable) != executable_sha256:
            raise MeshingError("Gmsh executable changed during mesh generation")
        mesh_sha256 = sha256_file(mesh_path) if mesh_path.is_file() else None
        return GmshProcessResult(
            argv=argv,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=elapsed,
            identity=identity,
            geometry_sha256=sha256_file(geometry_path),
            mesh_sha256=mesh_sha256,
            environment_overrides=self._ENVIRONMENT_OVERRIDES,
        )

    @staticmethod
    def _read_version(
        executable: Path,
        *,
        working_directory: Path,
        timeout_seconds: float | None,
    ) -> str:
        environment = os.environ.copy()
        environment.update(dict(GmshRunner._ENVIRONMENT_OVERRIDES))
        try:
            completed = subprocess.run(
                (str(executable), "--version"),
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise MeshingError(
                f"Gmsh version probe timed out after {timeout_seconds} seconds"
            ) from error
        version = completed.stdout.strip()
        if completed.returncode != 0 or not _VERSION_PATTERN.fullmatch(version):
            raise MeshingError(
                "Gmsh version probe failed or returned an unsupported identity: "
                f"return_code={completed.returncode}, stdout={version!r}"
            )
        return version
