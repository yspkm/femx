"""Evidence reports that distinguish validation from execution and convergence."""

from dataclasses import dataclass
from enum import StrEnum

from femx.core.errors import ContractError, ValidationError


class CheckStatus(StrEnum):
    """Outcome of one scientific validation check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One named piece of validation evidence."""

    name: str
    status: CheckStatus
    required: bool = True
    metric: float | None = None
    threshold: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("validation check name must be non-empty and trimmed")
        if (
            self.status is CheckStatus.PASSED
            and self.required
            and self.detail.startswith("blocked")
        ):
            raise ContractError("a passed check cannot describe itself as blocked")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A strict collection of evidence supporting one explicit claim."""

    claim: str
    checks: tuple[ValidationCheck, ...]
    schema_version: str = "femx.validation/v1"

    def __post_init__(self) -> None:
        if not self.claim or self.claim.strip() != self.claim:
            raise ContractError("validation claim must be non-empty and trimmed")
        if not self.checks:
            raise ContractError("validation report must contain at least one check")
        names = tuple(check.name for check in self.checks)
        if len(names) != len(set(names)):
            raise ContractError("validation check names must be unique")
        if self.schema_version != "femx.validation/v1":
            raise ContractError(f"unsupported validation schema {self.schema_version!r}")

    @property
    def publishable(self) -> bool:
        """Whether all required evidence passed."""

        return all(
            not check.required or check.status is CheckStatus.PASSED for check in self.checks
        )

    def require_publishable(self) -> None:
        """Raise with deterministic detail when required evidence is incomplete."""

        incomplete = [
            f"{check.name}={check.status.value}"
            for check in self.checks
            if check.required and check.status is not CheckStatus.PASSED
        ]
        if incomplete:
            raise ValidationError(
                f"claim {self.claim!r} lacks required evidence: {', '.join(incomplete)}"
            )
