from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from scripts.run_tpu_scalar_h1_collective_evidence import (  # noqa: E402
    ACTION_TOLERANCE,
    CG_MAX_ITERATIONS,
    CG_RELATIVE_TOLERANCE,
    EVIDENCE_SCHEMA,
    EXECUTION_SAMPLES,
    HOST_PRECISION_TOLERANCE,
    MULTILEVEL_EXTENSION_SCHEMA,
    MULTILEVEL_MAXIMUM_RELATIVE_SYMMETRY_ERROR,
    MULTILEVEL_MAXIMUM_REPLICATED_DOFS,
    REAL_SCALAR_CONTRACT,
    RHS_TOLERANCE,
    VJP_TOLERANCE,
    WORKER_ENTRY_CLAIM_SCHEMA,
    _atomic_json,
    _boundary_incidence,
    _build_explicit_scalar_kernels,
    _build_host_case,
    _cell_load,
    _claim_worker_entry,
    _manifest_provenance,
    _memory_report,
    _nonnegative_environment_integer,
    _numpy_assemble_cell_vector,
    _numpy_cg,
    _numpy_matvec,
    _numpy_relative_difference,
    _pack_cells,
    _pack_owned,
    _pack_owner_mask,
    _physical_coefficients,
    _physical_multilevel_hierarchy,
    _positive_environment_integer,
    _require_expected_count,
    _runtime,
    _slab_cell_owners,
    _solver_mode,
    _structured_rectangle,
    _tpu_index_array,
    _triangle_cell_stiffness,
    _write_process_evidence,
)

from femx.backends.jax.scalar_collective import (  # noqa: E402
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _topology(partition_count: int = 2):
    coordinates, cells, facets = _structured_rectangle(4, 2)
    coefficients = _physical_coefficients("heat", coordinates, cells, facets)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        _slab_cell_owners(coordinates, cells, partition_count),
        node_count=coordinates.shape[0],
        free_nodes=coefficients["free_nodes"],
        partition_count=partition_count,
    )
    return coordinates, cells, facets, topology


def test_runner_contract_constants_are_explicit() -> None:
    assert EVIDENCE_SCHEMA == "femx.jax.scalar_h1_collective.tpu_evidence/v1"
    assert MULTILEVEL_EXTENSION_SCHEMA.endswith("multilevel_extension/v1")
    assert MULTILEVEL_MAXIMUM_REPLICATED_DOFS == 2048
    assert MULTILEVEL_MAXIMUM_RELATIVE_SYMMETRY_ERROR == 2.0e-6
    assert WORKER_ENTRY_CLAIM_SCHEMA.endswith("worker_entry_claim/v1")
    assert (ACTION_TOLERANCE, RHS_TOLERANCE, VJP_TOLERANCE, HOST_PRECISION_TOLERANCE) == (
        4.0e-4,
        2.0e-6,
        2.0e-3,
        2.0e-3,
    )
    assert CG_RELATIVE_TOLERANCE == 2.0e-5
    assert CG_MAX_ITERATIONS == 4000
    assert EXECUTION_SAMPLES == 5
    assert REAL_SCALAR_CONTRACT["precision_fallback"] is False


def test_environment_counts_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEMX_COUNT", raising=False)
    assert _positive_environment_integer("FEMX_COUNT") is None
    monkeypatch.setenv("FEMX_COUNT", "4")
    assert _positive_environment_integer("FEMX_COUNT") == 4
    _require_expected_count("FEMX_COUNT", 4)
    with pytest.raises(RuntimeError, match="requires 4, observed 2"):
        _require_expected_count("FEMX_COUNT", 2)
    for invalid in ("0", "-1", "one"):
        monkeypatch.setenv("FEMX_COUNT", invalid)
        with pytest.raises(RuntimeError, match="positive integer"):
            _positive_environment_integer("FEMX_COUNT")
    monkeypatch.delenv("FEMX_INDEX", raising=False)
    assert _nonnegative_environment_integer("FEMX_INDEX") is None
    monkeypatch.setenv("FEMX_INDEX", "0")
    assert _nonnegative_environment_integer("FEMX_INDEX") == 0
    for invalid in ("-1", "one"):
        monkeypatch.setenv("FEMX_INDEX", invalid)
        with pytest.raises(RuntimeError, match="nonnegative integer"):
            _nonnegative_environment_integer("FEMX_INDEX")


def test_solver_mode_is_explicit_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEMX_SCALAR_SOLVER_MODE", raising=False)
    assert _solver_mode() == "cg"
    monkeypatch.setenv("FEMX_SCALAR_SOLVER_MODE", "multilevel_pcg")
    assert _solver_mode() == "multilevel_pcg"
    monkeypatch.setenv("FEMX_SCALAR_SOLVER_MODE", "auto")
    with pytest.raises(RuntimeError, match="must be 'cg' or 'multilevel_pcg'"):
        _solver_mode()


def test_runtime_rejects_missing_explicit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(RuntimeError, match="must be set before Python starts"):
        _runtime()
    monkeypatch.setenv("JAX_PLATFORMS", "tpu,cpu")
    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    with pytest.raises(RuntimeError, match="JAX_DEFAULT_MATMUL_PRECISION=highest"):
        _runtime()


def test_structured_mesh_and_host_cases_are_finite_and_convergent() -> None:
    coordinates, cells, facets, topology = _topology()
    assert coordinates.shape == (15, 2)
    assert cells.shape == (16, 3)
    assert facets.shape == (12, 2)
    for name in ("heat", "current"):
        host = _build_host_case(name, coordinates, cells, facets, topology)
        assert np.all(np.isfinite(host["input_solution"]))
        assert host["input_residual_norm"] >= 0.0
        assert host["host_precision_relative_difference"] < HOST_PRECISION_TOLERANCE
        assert host["expected_matrix_vjp"].shape == (cells.shape[0], 3, 3)
        assert host["expected_cell_rhs_vjp"].shape == (cells.shape[0], 3)
    with pytest.raises(ValueError, match="unknown scalar physical case"):
        _physical_coefficients("unknown", coordinates, cells, facets)


def test_physical_multilevel_hierarchy_matches_the_exact_rectangular_mesh() -> None:
    _, _, _, topology = _topology()
    layout = prepare_collective_scalar_h1_layout(topology)
    hierarchy = _physical_multilevel_hierarchy(layout, 4, 2)
    assert hierarchy.layout_sha256 == layout.digest()
    assert hierarchy.level_dof_counts == (9, 2)
    assert hierarchy.maximum_replicated_dofs == MULTILEVEL_MAXIMUM_REPLICATED_DOFS
    with pytest.raises(RuntimeError, match="even x interval"):
        _physical_multilevel_hierarchy(layout, 3, 2)
    with pytest.raises(RuntimeError, match="positive y interval"):
        _physical_multilevel_hierarchy(layout, 4, 0)


def test_numpy_element_load_action_and_cg_match_dense_authority() -> None:
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    facets = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    stiffness = _triangle_cell_stiffness(
        coordinates,
        cells,
        np.asarray((2.0,)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        stiffness[0],
        np.asarray(((2.0, -1.0, -1.0), (-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0))),
    )
    load = _cell_load(
        coordinates,
        cells,
        facets,
        np.asarray((6.0,)),
        np.asarray((0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(load, np.asarray(((1.0, 1.0, 1.0),)))
    matrix = np.asarray((((2.0, -1.0), (-1.0, 2.0)),))
    mapping = np.asarray(((0, 1),), dtype=np.int64)
    vector = np.asarray((1.0, 2.0))
    np.testing.assert_allclose(_numpy_matvec(matrix, mapping, vector), (0.0, 3.0))
    rhs = _numpy_assemble_cell_vector(np.asarray(((1.0, 2.0),)), mapping, 2)
    solution, _, residual = _numpy_cg(
        matrix,
        mapping,
        rhs,
        relative_tolerance=1.0e-12,
        max_iterations=10,
    )
    np.testing.assert_allclose(solution, np.linalg.solve(matrix[0], rhs))
    assert residual < 1.0e-12
    assert _numpy_relative_difference(solution, solution) == 0.0
    assert np.isinf(_numpy_relative_difference(np.ones(2), np.zeros(2)))


def test_numpy_helpers_reject_invalid_geometry_and_non_spd_or_unconverged_systems() -> None:
    bad_coordinates = np.asarray(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    with pytest.raises(RuntimeError, match="positive finite triangles"):
        _triangle_cell_stiffness(
            bad_coordinates,
            cells,
            np.ones(1),
            dtype=np.float64,
        )
    with pytest.raises(RuntimeError, match="exact exterior edge set"):
        _boundary_incidence(cells, np.asarray(((0, 3),), dtype=np.int64))
    mapping = np.asarray(((0,),), dtype=np.int64)
    with pytest.raises(RuntimeError, match="nonpositive curvature"):
        _numpy_cg(
            np.asarray((([-1.0],),)),
            mapping,
            np.ones(1),
            relative_tolerance=1.0e-8,
            max_iterations=2,
        )
    with pytest.raises(RuntimeError, match="did not satisfy"):
        _numpy_cg(
            np.asarray((((2.0, -1.0), (-1.0, 2.0)),)),
            np.asarray(((0, 1),), dtype=np.int64),
            np.asarray((1.0, 0.0)),
            relative_tolerance=1.0e-30,
            max_iterations=1,
        )


def test_transport_packing_and_index_contracts() -> None:
    _, cells, _, topology = _topology()
    from femx.backends.jax.scalar_collective import prepare_collective_scalar_h1_layout

    layout = prepare_collective_scalar_h1_layout(topology)
    packed_cells = _pack_cells(layout, np.ones((cells.shape[0], 3), dtype=np.float32))
    packed_owned = _pack_owned(layout, np.arange(topology.free_dof_count, dtype=np.float32))
    mask = _pack_owner_mask(layout)
    assert packed_cells.shape[:2] == layout.transport.cell_ids.shape
    assert packed_owned.shape == layout.transport.owned_dof_ids.shape == mask.shape
    assert mask.dtype == np.bool_
    assert _tpu_index_array(np.asarray(((0, 1),), dtype=np.int64)).dtype == np.int32
    with pytest.raises(RuntimeError, match="must be integers"):
        _tpu_index_array(np.asarray((0.0,)))
    with pytest.raises(RuntimeError, match="exceed"):
        _tpu_index_array(np.asarray((np.iinfo(np.int32).max + 1,), dtype=np.int64))


def test_explicit_kernels_keep_mapping_and_mask_at_jit_boundary() -> None:
    def assemble(cell_rhs: jax.Array, mapping: jax.Array) -> jax.Array:
        return jnp.sum(cell_rhs, axis=1) + jnp.sum(mapping, axis=1)

    class _Result(NamedTuple):
        solution: jax.Array

    def solve(
        matrix: jax.Array,
        mapping: jax.Array,
        mask: jax.Array,
        rhs: jax.Array,
        *strategy: jax.Array,
    ) -> _Result:
        del mapping
        strategy_value = sum(strategy, start=jnp.asarray(0.0))
        return _Result(jnp.sum(matrix, axis=(1, 2)) + rhs + mask.astype(rhs.dtype) + strategy_value)

    forward, vjp = _build_explicit_scalar_kernels(jax, assemble, solve)
    matrix = jnp.ones((2, 1, 1))
    rhs = jnp.ones((2, 1))
    mapping = jnp.zeros((2, 1), dtype=jnp.int32)
    mask = jnp.ones((2,), dtype=jnp.bool_)
    result = forward(matrix, rhs, mapping, mask)
    matrix_vjp, rhs_vjp = vjp(matrix, rhs, mapping, mask, jnp.ones(2))
    np.testing.assert_allclose(result.solution, np.full(2, 3.0))
    np.testing.assert_allclose(matrix_vjp, np.ones_like(matrix))
    np.testing.assert_allclose(rhs_vjp, np.ones_like(rhs))
    strategy = jnp.asarray(2.0)
    strategy_result = jax.jit(forward)(matrix, rhs, mapping, mask, strategy)
    strategy_matrix_vjp, strategy_rhs_vjp = jax.jit(vjp)(
        matrix,
        rhs,
        mapping,
        mask,
        jnp.ones(2),
        strategy,
    )
    np.testing.assert_allclose(strategy_result.solution, np.full(2, 5.0))
    np.testing.assert_allclose(strategy_matrix_vjp, np.ones_like(matrix))
    np.testing.assert_allclose(strategy_rhs_vjp, np.ones_like(rhs))


def test_atomic_manifest_claim_and_memory_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.json"
    _atomic_json(destination, {"value": 1})
    assert json.loads(destination.read_text()) == {"value": 1}
    remote = tmp_path / "run"
    manifest = remote / ".phoxla" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "profile": "profile",
                "source": {"digest": "a" * 64},
                "config": {"digest": "b" * 64},
            }
        )
    )
    provenance = _manifest_provenance(remote)
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "0")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "0")
    monkeypatch.setenv("PHOXLA_RUN_ID", "run-1")
    claim = _claim_worker_entry(remote.resolve(), provenance)
    assert claim["schema_version"] == WORKER_ENTRY_CLAIM_SCHEMA
    with pytest.raises(RuntimeError, match="duplicate femx scalar entry"):
        _claim_worker_entry(remote.resolve(), provenance)
    manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="invalid deployed Phoxla manifest"):
        _manifest_provenance(remote)

    class Analysis:
        generated_code_size_in_bytes = 1
        argument_size_in_bytes = 2
        output_size_in_bytes = 3
        alias_size_in_bytes = 1
        temp_size_in_bytes = 4

    class Compiled:
        def memory_analysis(self) -> object:
            return Analysis()

    report = _memory_report(Compiled(), 100)
    assert report.compiler_peak_bytes == 8


def test_process_zero_publishes_sync_compatibility_record(tmp_path: Path) -> None:
    remote_run = tmp_path / "remote"
    process_zero = remote_run / "raw" / "process-0"
    _write_process_evidence(process_zero, remote_run, 0, {"process": 0})
    assert json.loads((process_zero / "results" / "process-metrics.json").read_text()) == {
        "process": 0
    }
    assert json.loads((process_zero / "results" / "metrics.json").read_text()) == {"process": 0}
    assert json.loads((remote_run / "results" / "metrics.json").read_text()) == {"process": 0}

    process_one = remote_run / "raw" / "process-1"
    _write_process_evidence(process_one, remote_run, 1, {"process": 1})
    assert json.loads((process_one / "results" / "process-metrics.json").read_text()) == {
        "process": 1
    }
    assert not (process_one / "results" / "metrics.json").exists()
