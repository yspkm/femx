from pathlib import Path

import pytest
from scripts.check_markdown_math import find_violations, markdown_files

pytestmark = pytest.mark.architecture

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_supported_dollar_math_and_code_examples_are_accepted(tmp_path: Path) -> None:
    document = tmp_path / "supported.md"
    document.write_text(
        "Inline $x^2$ and display math:\n\n$$\nA x = b\n$$\n\n"
        "Literal examples `\\operatorname`, `\\(x\\)`, and `\\[x\\]`.\n\n"
        "```text\n\\operatorname{ignored}\n\\[ignored\\]\n```\n",
        encoding="utf-8",
    )

    assert find_violations((document,)) == ()


def test_incompatible_delimiters_environments_macros_and_fences_are_rejected(
    tmp_path: Path,
) -> None:
    document = tmp_path / "unsupported.md"
    document.write_text(
        "\\(x\\)\n"
        "\\[y\\]\n"
        "$\\operatorname{curl} E$\n"
        "\\begin{align}a &= b\\end{align}\n"
        "\\newcommand{\\R}{R}\n"
        "```math\nz = 1\n```\n",
        encoding="utf-8",
    )

    violations = find_violations((document,))

    assert {violation.kind for violation in violations} == {
        "backslash math delimiter",
        "display environment",
        "macro definition",
        "math code fence",
        "operatorname macro",
    }


def test_repository_markdown_uses_the_github_math_contract() -> None:
    assert find_violations(markdown_files(REPOSITORY_ROOT)) == ()
