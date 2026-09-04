#!/usr/bin/env python3
"""Reject Markdown math syntax that does not render in femx's GitHub target."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
    }
)
MATH_FENCE_LANGUAGES = frozenset({"latex", "math", "tex"})
FENCE_LINE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "backslash math delimiter",
        re.compile(r"\\[()\[\]]"),
        "use $...$ inline or $$...$$ on separate lines",
    ),
    (
        "operatorname macro",
        re.compile(r"\\operatorname\b"),
        "spell the label with supported symbols or \\mathrm{...}",
    ),
    (
        "display environment",
        re.compile(
            r"\\(?:begin|end)\s*\{"
            r"(?:align\*?|aligned|displaymath|equation\*?|gather\*?|multline\*?)\}"
        ),
        "use a $$...$$ display block without an equation environment",
    ),
    (
        "macro definition",
        re.compile(r"\\(?:DeclareMathOperator|def|newcommand|providecommand|renewcommand)\b"),
        "write the expression without document-local TeX macro definitions",
    ),
)


@dataclass(frozen=True, slots=True)
class MarkdownMathViolation:
    """One GitHub Markdown math compatibility violation."""

    path: Path
    line: int
    column: int
    kind: str
    reason: str

    def format(self, *, relative_to: Path) -> str:
        """Format a deterministic compiler-style diagnostic."""

        try:
            path = self.path.relative_to(relative_to)
        except ValueError:
            path = self.path
        return f"{path}:{self.line}:{self.column}: {self.kind}: {self.reason}"


def markdown_files(root: Path = REPOSITORY_ROOT) -> tuple[Path, ...]:
    """Return project Markdown files while pruning generated and dependency trees."""

    discovered: list[Path] = []
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in EXCLUDED_DIRECTORIES
        )
        directory_path = Path(directory)
        discovered.extend(
            directory_path / name for name in sorted(file_names) if name.endswith(".md")
        )
    return tuple(discovered)


def _mask_inline_code(line: str) -> str:
    masked = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        run_end = index + 1
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        delimiter = line[index:run_end]
        closing = line.find(delimiter, run_end)
        if closing < 0:
            index = run_end
            continue
        closing_end = closing + len(delimiter)
        masked[index:closing_end] = " " * (closing_end - index)
        index = closing_end
    return "".join(masked)


def find_violations(paths: Iterable[Path]) -> tuple[MarkdownMathViolation, ...]:
    """Find forbidden math delimiters, environments, and macros outside code spans."""

    violations: list[MarkdownMathViolation] = []
    for path in sorted(paths):
        fence_character: str | None = None
        fence_length = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fence_match = FENCE_LINE.match(line)
            if fence_character is not None:
                if (
                    fence_match is not None
                    and fence_match.group(1)[0] == fence_character
                    and len(fence_match.group(1)) >= fence_length
                    and not fence_match.group(2).strip()
                ):
                    fence_character = None
                    fence_length = 0
                continue
            if fence_match is not None:
                delimiter = fence_match.group(1)
                info = fence_match.group(2).strip().lower().split(maxsplit=1)
                if info and info[0] in MATH_FENCE_LANGUAGES:
                    violations.append(
                        MarkdownMathViolation(
                            path=path,
                            line=line_number,
                            column=line.index(delimiter) + 1,
                            kind="math code fence",
                            reason="use a $$...$$ display block instead",
                        )
                    )
                fence_character = delimiter[0]
                fence_length = len(delimiter)
                continue

            searchable = _mask_inline_code(line)
            for kind, pattern, reason in FORBIDDEN_PATTERNS:
                violations.extend(
                    MarkdownMathViolation(
                        path=path,
                        line=line_number,
                        column=match.start() + 1,
                        kind=kind,
                        reason=reason,
                    )
                    for match in pattern.finditer(searchable)
                )
    return tuple(violations)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository GitHub Markdown math gate."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    paths = tuple(Path(argument).resolve() for argument in arguments)
    if not paths:
        paths = markdown_files()
    violations = find_violations(paths)
    if violations:
        for violation in violations:
            print(violation.format(relative_to=REPOSITORY_ROOT), file=sys.stderr)
        print(
            f"markdown math check failed with {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1
    print(f"markdown math check passed ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
