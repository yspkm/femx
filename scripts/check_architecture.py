#!/usr/bin/env python3
"""Enforce femx package boundaries using only the Python standard library."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "femx"


@dataclass(frozen=True, slots=True)
class LayerRule:
    """Imports forbidden from a module prefix."""

    source_prefix: str
    forbidden_prefixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Violation:
    """One deterministic architecture-policy violation."""

    path: Path
    line: int
    module: str
    imported: str
    reason: str

    def format(self, *, relative_to: Path) -> str:
        """Format a stable, compiler-style diagnostic."""

        try:
            path = self.path.relative_to(relative_to)
        except ValueError:
            path = self.path
        return f"{path}:{self.line}: {self.module} imports {self.imported}: {self.reason}"


SUBSTRATE = (
    "femx.core",
    "femx.mesh",
    "femx.forms",
    "femx.materials",
    "femx.artifacts",
    "femx.validation",
    "femx.physics",
)

RULES = (
    *(
        LayerRule(
            source_prefix=prefix,
            forbidden_prefixes=(
                "femx.backends",
                "femx.interop",
                "femx.workflows",
                "fdtdx",
                "gmsh",
                "h5py",
                "jax",
                "meshio",
                "numpy",
                "tidy3d",
            ),
        )
        for prefix in SUBSTRATE
    ),
    LayerRule(
        source_prefix="femx.backends.elmer",
        forbidden_prefixes=("femx.backends.jax", "fdtdx", "jax", "tidy3d"),
    ),
    LayerRule(
        source_prefix="femx.backends.jax",
        forbidden_prefixes=("femx.backends.elmer", "fdtdx", "tidy3d"),
    ),
    LayerRule(
        source_prefix="femx.meshing",
        forbidden_prefixes=(
            "femx.backends",
            "femx.interop",
            "femx.workflows",
            "fdtdx",
            "gmsh",
            "jax",
            "meshio",
            "tidy3d",
        ),
    ),
    LayerRule(
        source_prefix="femx.interop.fdtdx",
        forbidden_prefixes=("femx.backends",),
    ),
)


def _module_name(path: Path, source_root: Path) -> tuple[str, bool]:
    relative = path.relative_to(source_root)
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(("femx", *parts)), is_package


def _resolve_relative_import(
    *, current_module: str, is_package: bool, level: int, imported_module: str | None
) -> str:
    if level == 0:
        return imported_module or ""
    current_parts = current_module.split(".")
    package_parts = current_parts if is_package else current_parts[:-1]
    drop = level - 1
    if drop > len(package_parts):
        return imported_module or ""
    base = package_parts[: len(package_parts) - drop]
    if imported_module:
        base.extend(imported_module.split("."))
    return ".".join(base)


def _imports(path: Path, source_root: Path) -> tuple[tuple[int, str], ...]:
    module, is_package = _module_name(path, source_root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_import(
                current_module=module,
                is_package=is_package,
                level=node.level,
                imported_module=node.module,
            )
            if resolved:
                imported.append((node.lineno, resolved))
    return tuple(imported)


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def find_violations(source_root: Path = DEFAULT_SOURCE_ROOT) -> tuple[Violation, ...]:
    """Scan a femx package tree and return every boundary violation."""

    violations: list[Violation] = []
    for path in sorted(source_root.rglob("*.py")):
        module, _ = _module_name(path, source_root)
        for line, imported in _imports(path, source_root):
            for rule in RULES:
                if not _matches(module, rule.source_prefix):
                    continue
                forbidden = next(
                    (prefix for prefix in rule.forbidden_prefixes if _matches(imported, prefix)),
                    None,
                )
                if forbidden is not None:
                    violations.append(
                        Violation(
                            path=path,
                            line=line,
                            module=module,
                            imported=imported,
                            reason=f"layer forbids dependency on {forbidden}",
                        )
                    )
            subprocess_boundaries = {
                "femx.backends.elmer.runner",
                "femx.meshing.gmsh.runner",
            }
            if imported == "subprocess" and module not in subprocess_boundaries:
                violations.append(
                    Violation(
                        path=path,
                        line=line,
                        module=module,
                        imported=imported,
                        reason="subprocess is restricted to guarded external-tool runners",
                    )
                )
    return tuple(violations)


def main() -> int:
    """Run the repository architecture gate."""

    violations = find_violations()
    if violations:
        for violation in violations:
            print(violation.format(relative_to=REPOSITORY_ROOT), file=sys.stderr)
        print(f"architecture check failed with {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("architecture check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
