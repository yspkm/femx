from pathlib import Path

import pytest
from scripts.check_architecture import DEFAULT_SOURCE_ROOT, find_violations

pytestmark = pytest.mark.architecture


def test_repository_obeys_declared_dependency_boundaries() -> None:
    assert find_violations(DEFAULT_SOURCE_ROOT) == ()


def test_boundary_checker_detects_forbidden_and_subprocess_imports(tmp_path: Path) -> None:
    source_root = tmp_path / "femx"
    core = source_root / "core"
    core.mkdir(parents=True)
    (core / "bad.py").write_text("import jax\nimport subprocess\n", encoding="utf-8")

    violations = find_violations(source_root)

    assert [(violation.imported, violation.line) for violation in violations] == [
        ("jax", 1),
        ("subprocess", 2),
    ]


def test_material_catalog_is_a_solver_neutral_substrate(tmp_path: Path) -> None:
    source_root = tmp_path / "femx"
    materials = source_root / "materials"
    materials.mkdir(parents=True)
    (materials / "bad.py").write_text("import numpy\n", encoding="utf-8")

    violations = find_violations(source_root)

    assert [(violation.module, violation.imported) for violation in violations] == [
        ("femx.materials.bad", "numpy")
    ]


def test_meshing_adapter_cannot_reach_solver_or_gmsh_python_apis(tmp_path: Path) -> None:
    source_root = tmp_path / "femx"
    meshing = source_root / "meshing"
    meshing.mkdir(parents=True)
    (meshing / "bad.py").write_text(
        "import gmsh\nfrom femx.backends import protocol\n",
        encoding="utf-8",
    )

    violations = find_violations(source_root)

    assert [(violation.module, violation.imported) for violation in violations] == [
        ("femx.meshing.bad", "gmsh"),
        ("femx.meshing.bad", "femx.backends"),
    ]
