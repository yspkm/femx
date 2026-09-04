"""Guarded, shell-free execution boundary for a separately installed Elmer."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from femx.core.errors import BackendError, BackendUnavailableError, ContractError
from femx.core.execution import ExecutionPolicy


@dataclass(frozen=True, slots=True)
class ElmerInstallation:
    """Resolved Elmer executable without executing or probing it."""

    executable: Path

    @classmethod
    def discover(cls, executable_name: str = "ElmerSolver") -> ElmerInstallation | None:
        """Locate Elmer on PATH without starting a process."""

        resolved = shutil.which(executable_name)
        return None if resolved is None else cls(Path(resolved).resolve())

    def __post_init__(self) -> None:
        if not self.executable.is_absolute():
            raise ContractError("Elmer executable must be an absolute path")


@dataclass(frozen=True, slots=True)
class ElmerCommand:
    """One shell-free Elmer invocation."""

    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if any(not argument or "\x00" in argument for argument in self.arguments):
            raise ContractError("Elmer arguments must be non-empty and contain no NUL bytes")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ContractError("Elmer timeout must be positive")
        if any(not key or "\x00" in key or "=" in key for key in self.environment):
            raise ContractError("Elmer environment contains an invalid variable name")
        if any("\x00" in value for value in self.environment.values()):
            raise ContractError("Elmer environment contains a NUL byte")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True, slots=True)
class ElmerProcessResult:
    """Process evidence, explicitly separate from convergence and validation."""

    argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def process_succeeded(self) -> bool:
        """Whether the executable returned zero; not a scientific validity claim."""

        return self.return_code == 0


class ElmerRunner:
    """Execute only a previously resolved Elmer installation."""

    def __init__(self, installation: ElmerInstallation) -> None:
        self._installation = installation

    @property
    def installation(self) -> ElmerInstallation:
        """Return the immutable executable identity."""

        return self._installation

    def run(
        self,
        command: ElmerCommand,
        *,
        working_directory: Path,
        policy: ExecutionPolicy,
    ) -> ElmerProcessResult:
        """Run Elmer with no shell after explicit policy and path checks."""

        policy.require_external_process(component_name="elmer")
        executable = self.installation.executable
        if not executable.is_file():
            raise BackendUnavailableError(f"Elmer executable does not exist: {executable}")
        if not working_directory.is_dir():
            raise BackendError(f"Elmer working directory does not exist: {working_directory}")

        argv = (str(executable), *command.arguments)
        environment = os.environ.copy()
        environment.update(command.environment)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise BackendError(
                f"Elmer timed out after {command.timeout_seconds} seconds"
            ) from error
        elapsed = time.monotonic() - started
        return ElmerProcessResult(
            argv=argv,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=elapsed,
        )
