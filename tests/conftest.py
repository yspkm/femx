"""Global pytest policy for explicit test-layer ownership."""

import os
import shutil
from pathlib import Path

import pytest

PRIMARY_MARKERS = frozenset({"unit", "architecture", "contract", "integration", "scientific"})


def _configured_elmer_executable() -> Path:
    configured = os.environ.get("FEMX_ELMER_EXECUTABLE")
    if configured is None:
        discovered = shutil.which("ElmerSolver")
        if discovered is None:
            pytest.skip("ElmerSolver is not available on PATH")
        return Path(discovered).resolve()
    executable = Path(configured)
    if not executable.is_absolute():
        pytest.fail("FEMX_ELMER_EXECUTABLE must be an absolute path")
    return executable.resolve()


def _configured_gmsh_executable() -> Path:
    configured = os.environ.get("FEMX_GMSH_EXECUTABLE")
    if configured is None:
        discovered = shutil.which("gmsh")
        if discovered is None:
            pytest.skip("Gmsh is not available on PATH")
        return Path(discovered).resolve()
    executable = Path(configured)
    if not executable.is_absolute():
        pytest.fail("FEMX_GMSH_EXECUTABLE must be an absolute path")
    return executable.resolve()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Require exactly one primary layer marker on every collected test."""

    errors: list[str] = []
    for item in items:
        present = sorted(
            marker.name for marker in item.iter_markers() if marker.name in PRIMARY_MARKERS
        )
        if len(present) != 1:
            errors.append(f"{item.nodeid}: expected one primary marker, got {present}")
    if errors:
        raise pytest.UsageError("test layer marker policy failed:\n" + "\n".join(errors))


@pytest.fixture(scope="session")
def locked_gmsh_runner():
    """Return the explicitly opted-in Gmsh process adapter or skip honestly."""

    if os.environ.get("FEMX_RUN_GMSH_TESTS") != "1":
        pytest.skip("set FEMX_RUN_GMSH_TESTS=1 to authorize real Gmsh tests")
    from femx.meshing.gmsh import GmshInstallation, GmshRunner

    return GmshRunner(GmshInstallation(_configured_gmsh_executable()))


@pytest.fixture(scope="session")
def locked_elmer_backend():
    """Return the explicitly opted-in locked Elmer oracle or skip honestly."""

    if os.environ.get("FEMX_RUN_ELMER_TESTS") != "1":
        pytest.skip("set FEMX_RUN_ELMER_TESTS=1 to authorize real Elmer tests")
    executable = _configured_elmer_executable()

    from femx.backends.elmer.runner import ElmerInstallation
    from femx.backends.elmer.steady_heat import (
        ElmerSteadyHeatBackend,
        ElmerSteadyHeatIdentity,
    )

    identity = ElmerSteadyHeatIdentity(
        version=os.environ.get("FEMX_ELMER_VERSION", "26.2-devel"),
        revision=os.environ.get("FEMX_ELMER_REVISION", "4f2d7e4b9"),
        executable_sha256=os.environ.get(
            "FEMX_ELMER_EXECUTABLE_SHA256",
            "1862fc1234c98ccb03d6e2c1fb3bdf0c9166331655fcab534366c2d8b73ece99",
        ),
        heat_solve_sha256=os.environ.get(
            "FEMX_ELMER_HEAT_SOLVE_SHA256",
            "779d9dd4f2b7b29ef8845533bb0a793f3484c8831d596411e6dcafc07ea65e12",
        ),
        source_commit=os.environ.get(
            "FEMX_ELMER_SOURCE_COMMIT",
            "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
        ),
        source_digest=os.environ.get(
            "FEMX_ELMER_SOURCE_DIGEST",
            "5c4b0e9f6e29ad646e3d378f14a8b882cbfe647fdf8452451b60459d070622bf",
        ),
        source_worktree_state=os.environ.get("FEMX_ELMER_SOURCE_WORKTREE_STATE", "not_checked"),
    )

    return ElmerSteadyHeatBackend(
        ElmerInstallation(executable),
        identity,
    )


@pytest.fixture(scope="session")
def locked_elmer_current_backend():
    """Return the opted-in locked StatCurrentSolve oracle or skip honestly."""

    if os.environ.get("FEMX_RUN_ELMER_TESTS") != "1":
        pytest.skip("set FEMX_RUN_ELMER_TESTS=1 to authorize real Elmer tests")
    executable = _configured_elmer_executable()

    from femx.backends.elmer.runner import ElmerInstallation
    from femx.backends.elmer.steady_current import (
        ElmerSteadyCurrentBackend,
        ElmerSteadyCurrentIdentity,
    )

    identity = ElmerSteadyCurrentIdentity(
        version=os.environ.get("FEMX_ELMER_VERSION", "26.2-devel"),
        revision=os.environ.get("FEMX_ELMER_REVISION", "4f2d7e4b9"),
        executable_sha256=os.environ.get(
            "FEMX_ELMER_EXECUTABLE_SHA256",
            "1862fc1234c98ccb03d6e2c1fb3bdf0c9166331655fcab534366c2d8b73ece99",
        ),
        stat_current_solve_sha256=os.environ.get(
            "FEMX_ELMER_STAT_CURRENT_SOLVE_SHA256",
            "3e2c2b567fba6965e062047b125961abd6decf1f21aa8abb043b92c5daf0ba08",
        ),
        source_commit=os.environ.get(
            "FEMX_ELMER_SOURCE_COMMIT",
            "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
        ),
        source_digest=os.environ.get(
            "FEMX_ELMER_SOURCE_DIGEST",
            "5c4b0e9f6e29ad646e3d378f14a8b882cbfe647fdf8452451b60459d070622bf",
        ),
        source_worktree_state=os.environ.get("FEMX_ELMER_SOURCE_WORKTREE_STATE", "not_checked"),
    )
    return ElmerSteadyCurrentBackend(ElmerInstallation(executable), identity)


@pytest.fixture(scope="session")
def locked_elmer_electrothermal_backend():
    """Return the opted-in locked two-module Elmer electrothermal oracle."""

    if os.environ.get("FEMX_RUN_ELMER_TESTS") != "1":
        pytest.skip("set FEMX_RUN_ELMER_TESTS=1 to authorize real Elmer tests")
    executable = _configured_elmer_executable()

    from femx.backends.elmer.runner import ElmerInstallation
    from femx.backends.elmer.self_consistent import (
        ElmerSelfConsistentElectrothermalBackend,
    )
    from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity
    from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity

    common = {
        "version": os.environ.get("FEMX_ELMER_VERSION", "26.2-devel"),
        "revision": os.environ.get("FEMX_ELMER_REVISION", "4f2d7e4b9"),
        "executable_sha256": os.environ.get(
            "FEMX_ELMER_EXECUTABLE_SHA256",
            "1862fc1234c98ccb03d6e2c1fb3bdf0c9166331655fcab534366c2d8b73ece99",
        ),
        "source_commit": os.environ.get(
            "FEMX_ELMER_SOURCE_COMMIT",
            "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
        ),
        "source_digest": os.environ.get(
            "FEMX_ELMER_SOURCE_DIGEST",
            "5c4b0e9f6e29ad646e3d378f14a8b882cbfe647fdf8452451b60459d070622bf",
        ),
        "source_worktree_state": os.environ.get(
            "FEMX_ELMER_SOURCE_WORKTREE_STATE",
            "not_checked",
        ),
    }
    current_identity = ElmerSteadyCurrentIdentity(
        **common,
        stat_current_solve_sha256=os.environ.get(
            "FEMX_ELMER_STAT_CURRENT_SOLVE_SHA256",
            "3e2c2b567fba6965e062047b125961abd6decf1f21aa8abb043b92c5daf0ba08",
        ),
    )
    heat_identity = ElmerSteadyHeatIdentity(
        **common,
        heat_solve_sha256=os.environ.get(
            "FEMX_ELMER_HEAT_SOLVE_SHA256",
            "779d9dd4f2b7b29ef8845533bb0a793f3484c8831d596411e6dcafc07ea65e12",
        ),
    )
    return ElmerSelfConsistentElectrothermalBackend(
        ElmerInstallation(executable),
        current_identity,
        heat_identity,
    )


@pytest.fixture(scope="session")
def locked_elmer_tet4_electrothermal_oracle():
    """Return the opted-in locked distinct-space 3D Tet4 Elmer oracle."""

    if os.environ.get("FEMX_RUN_ELMER_TESTS") != "1":
        pytest.skip("set FEMX_RUN_ELMER_TESTS=1 to authorize real Elmer tests")
    executable = _configured_elmer_executable()

    from femx.backends.elmer.runner import ElmerInstallation
    from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity
    from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity
    from femx.backends.elmer.tet4_electrothermal import (
        ElmerTet4ElectrothermalOracle,
    )

    common = {
        "version": os.environ.get("FEMX_ELMER_VERSION", "26.2-devel"),
        "revision": os.environ.get("FEMX_ELMER_REVISION", "4f2d7e4b9"),
        "executable_sha256": os.environ.get(
            "FEMX_ELMER_EXECUTABLE_SHA256",
            "1862fc1234c98ccb03d6e2c1fb3bdf0c9166331655fcab534366c2d8b73ece99",
        ),
        "source_commit": os.environ.get(
            "FEMX_ELMER_SOURCE_COMMIT",
            "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
        ),
        "source_digest": os.environ.get(
            "FEMX_ELMER_SOURCE_DIGEST",
            "5c4b0e9f6e29ad646e3d378f14a8b882cbfe647fdf8452451b60459d070622bf",
        ),
        "source_worktree_state": os.environ.get(
            "FEMX_ELMER_SOURCE_WORKTREE_STATE",
            "not_checked",
        ),
    }
    current_identity = ElmerSteadyCurrentIdentity(
        **common,
        stat_current_solve_sha256=os.environ.get(
            "FEMX_ELMER_STAT_CURRENT_SOLVE_SHA256",
            "3e2c2b567fba6965e062047b125961abd6decf1f21aa8abb043b92c5daf0ba08",
        ),
    )
    heat_identity = ElmerSteadyHeatIdentity(
        **common,
        heat_solve_sha256=os.environ.get(
            "FEMX_ELMER_HEAT_SOLVE_SHA256",
            "779d9dd4f2b7b29ef8845533bb0a793f3484c8831d596411e6dcafc07ea65e12",
        ),
    )
    return ElmerTet4ElectrothermalOracle(
        ElmerInstallation(executable),
        current_identity,
        heat_identity,
    )


@pytest.fixture(scope="session")
def locked_elmer_port_backend():
    """Return the opted-in locked EMPort oracle or skip honestly."""

    if os.environ.get("FEMX_RUN_ELMER_TESTS") != "1":
        pytest.skip("set FEMX_RUN_ELMER_TESTS=1 to authorize real Elmer tests")
    executable = _configured_elmer_executable()

    from femx.backends.elmer.port_eigenmode import (
        ElmerPortEigenmodeBackend,
        ElmerPortEigenmodeIdentity,
    )
    from femx.backends.elmer.runner import ElmerInstallation

    identity = ElmerPortEigenmodeIdentity(
        version=os.environ.get("FEMX_ELMER_VERSION", "26.2-devel"),
        revision=os.environ.get("FEMX_ELMER_REVISION", "4f2d7e4b9"),
        executable_sha256=os.environ.get(
            "FEMX_ELMER_EXECUTABLE_SHA256",
            "1862fc1234c98ccb03d6e2c1fb3bdf0c9166331655fcab534366c2d8b73ece99",
        ),
        em_port_sha256=os.environ.get(
            "FEMX_ELMER_EM_PORT_SHA256",
            "712ce2fa4b23e60ddf47973ed16fbfad79d3e5d3cf2ae93a7a36e73eb00ae53f",
        ),
        result_output_sha256=os.environ.get(
            "FEMX_ELMER_RESULT_OUTPUT_SHA256",
            "d1855a622ccc65a8e445531e4be2ded1456880ae32b2eb2a985a69d758bf5519",
        ),
        save_data_sha256=os.environ.get(
            "FEMX_ELMER_SAVE_DATA_SHA256",
            "44b4b020befe0328e26b3125be8d5ba03a41a5eead8095b5258cd5efc0aecf7e",
        ),
        source_commit=os.environ.get(
            "FEMX_ELMER_SOURCE_COMMIT",
            "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
        ),
        source_digest=os.environ.get(
            "FEMX_ELMER_SOURCE_DIGEST",
            "5c4b0e9f6e29ad646e3d378f14a8b882cbfe647fdf8452451b60459d070622bf",
        ),
        source_worktree_state=os.environ.get("FEMX_ELMER_SOURCE_WORKTREE_STATE", "not_checked"),
    )
    return ElmerPortEigenmodeBackend(ElmerInstallation(executable), identity)
