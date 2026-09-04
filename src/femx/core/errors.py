"""Stable exception hierarchy used across public femx contracts."""


class FemxError(Exception):
    """Base class for expected femx errors."""


class ContractError(FemxError, ValueError):
    """A public schema or semantic contract was violated."""


class CapabilityError(FemxError):
    """A backend cannot satisfy a problem's requested capabilities."""


class BackendError(FemxError):
    """A backend failed before producing a valid solution."""


class BackendUnavailableError(BackendError):
    """A requested backend or executable is not available."""


class ExecutionNotAuthorizedError(BackendError):
    """A side-effecting execution was requested without explicit authorization."""


class MeshingError(FemxError):
    """Mesh generation or ingestion failed before producing a valid mesh."""


class MesherUnavailableError(MeshingError):
    """A requested external mesher executable is unavailable."""


class ArtifactError(FemxError):
    """A durable artifact or its provenance is invalid."""


class ValidationError(FemxError):
    """Evidence is insufficient for the requested scientific claim."""
