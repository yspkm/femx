"""Explicit backend capability negotiation."""

from dataclasses import dataclass, field
from enum import StrEnum

from femx.core.errors import CapabilityError


class AnalysisKind(StrEnum):
    """Supported high-level mathematical analysis categories."""

    STEADY = "steady"
    TRANSIENT = "transient"
    HARMONIC = "harmonic"
    EIGENMODE = "eigenmode"


class FunctionSpaceFamily(StrEnum):
    """Finite-element conformity family."""

    H1 = "H1"
    HCURL = "Hcurl"
    HDIV = "Hdiv"
    L2 = "L2"
    DG = "DG"


class ScalarKind(StrEnum):
    """Scalar representation required by an analysis."""

    REAL = "real"
    COMPLEX = "complex"


class GradientMethod(StrEnum):
    """Differentiation guarantee exposed by a backend."""

    NONE = "none"
    FORWARD = "forward"
    REVERSE = "reverse"
    IMPLICIT = "implicit"
    ADJOINT = "adjoint"


class ParallelModel(StrEnum):
    """Execution topology understood by a backend."""

    SERIAL = "serial"
    SHARED_MEMORY = "shared-memory"
    MPI = "mpi"
    JAX_SPMD = "jax-spmd"


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    """The minimum mathematical and execution capabilities a problem requires."""

    analysis: AnalysisKind
    function_spaces: frozenset[FunctionSpaceFamily]
    scalar_kind: ScalarKind = ScalarKind.REAL
    gradient: GradientMethod = GradientMethod.NONE
    parallel: ParallelModel = ParallelModel.SERIAL

    def __post_init__(self) -> None:
        if not self.function_spaces:
            raise ValueError("a capability request must include at least one function space")


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Capabilities a backend promises for the current implementation and version."""

    analyses: frozenset[AnalysisKind] = field(default_factory=frozenset)
    function_spaces: frozenset[FunctionSpaceFamily] = field(default_factory=frozenset)
    scalar_kinds: frozenset[ScalarKind] = field(default_factory=frozenset)
    gradients: frozenset[GradientMethod] = field(
        default_factory=lambda: frozenset({GradientMethod.NONE})
    )
    parallel_models: frozenset[ParallelModel] = field(
        default_factory=lambda: frozenset({ParallelModel.SERIAL})
    )

    def missing(self, request: CapabilityRequest) -> tuple[str, ...]:
        """Return deterministic descriptions of unmet requirements."""

        missing: list[str] = []
        if request.analysis not in self.analyses:
            missing.append(f"analysis={request.analysis.value}")
        unavailable_spaces = request.function_spaces - self.function_spaces
        if unavailable_spaces:
            names = ",".join(sorted(space.value for space in unavailable_spaces))
            missing.append(f"function_spaces={names}")
        if request.scalar_kind not in self.scalar_kinds:
            missing.append(f"scalar_kind={request.scalar_kind.value}")
        if request.gradient not in self.gradients:
            missing.append(f"gradient={request.gradient.value}")
        if request.parallel not in self.parallel_models:
            missing.append(f"parallel={request.parallel.value}")
        return tuple(missing)

    def require(self, request: CapabilityRequest, *, backend_name: str) -> None:
        """Raise when this set does not satisfy ``request``."""

        missing = self.missing(request)
        if missing:
            detail = "; ".join(missing)
            raise CapabilityError(f"backend {backend_name!r} lacks required capabilities: {detail}")
