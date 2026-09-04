import pytest

from femx.core.errors import ContractError, ValidationError
from femx.validation import CheckStatus, ValidationCheck, ValidationReport

pytestmark = pytest.mark.unit


def test_validation_report_requires_all_required_evidence() -> None:
    report = ValidationReport(
        "steady heat is mesh converged",
        (
            ValidationCheck("manufactured_solution", CheckStatus.PASSED, metric=1e-8),
            ValidationCheck("refinement_slope", CheckStatus.BLOCKED),
            ValidationCheck("optional_plot", CheckStatus.SKIPPED, required=False),
        ),
    )

    assert not report.publishable
    with pytest.raises(ValidationError, match="refinement_slope=blocked"):
        report.require_publishable()


def test_validation_report_passes_only_complete_required_checks() -> None:
    report = ValidationReport(
        "mode normalization is valid",
        (ValidationCheck("power", CheckStatus.PASSED, metric=1.0, threshold=1e-8),),
    )

    assert report.publishable
    report.require_publishable()


def test_validation_report_rejects_empty_or_duplicate_evidence() -> None:
    with pytest.raises(ContractError, match="at least one"):
        ValidationReport("claim", ())
    with pytest.raises(ContractError, match="unique"):
        ValidationReport(
            "claim",
            (
                ValidationCheck("same", CheckStatus.PASSED),
                ValidationCheck("same", CheckStatus.PASSED),
            ),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ValidationCheck(" bad", CheckStatus.FAILED), "check name"),
        (
            lambda: ValidationCheck("bad", CheckStatus.PASSED, detail="blocked by input"),
            "cannot describe",
        ),
        (
            lambda: ValidationReport(" bad", (ValidationCheck("x", CheckStatus.PASSED),)),
            "claim",
        ),
        (
            lambda: ValidationReport(
                "claim",
                (ValidationCheck("x", CheckStatus.PASSED),),
                schema_version="femx.validation/v2",
            ),
            "unsupported validation schema",
        ),
    ],
)
def test_validation_schema_rejects_ambiguous_evidence(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()
