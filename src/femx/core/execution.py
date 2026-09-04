"""Runtime-independent authorization for side-effecting execution."""

from dataclasses import dataclass

from femx.core.errors import ExecutionNotAuthorizedError


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Explicit permission gate for side-effecting or topology-specific work."""

    execution_authorized: bool = False
    allow_external_process: bool = False
    allow_accelerator: bool = False
    allow_network: bool = False

    def require_external_process(self, *, component_name: str) -> None:
        """Require two independent gates for one external-process component."""

        if not self.execution_authorized or not self.allow_external_process:
            raise ExecutionNotAuthorizedError(
                f"external execution for component {component_name!r} requires "
                "execution_authorized=True and allow_external_process=True"
            )
