from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from scripts.run_tpu_collective_port_evidence import (  # noqa: E402
    CHECKPOINT_ID,
    CHECKPOINT_STEP,
    COMPLEX_SCALAR_CONTRACT,
    EVIDENCE_SCHEMA,
    WORKER_ENTRY_CLAIM_SCHEMA,
    _atomic_json,
    _build_explicit_packed_kernels,
    _claim_worker_entry,
    _manifest_provenance,
    _memory_report,
    _nonnegative_environment_integer,
    _numpy_matrix_free_matvec,
    _numpy_matrix_free_vjp,
    _numpy_relative_difference,
    _positive_environment_integer,
    _require_expected_count,
    _runtime,
    _structured_rectangle,
    _tpu_index_array,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_evidence_and_checkpoint_schemas_are_explicitly_versioned() -> None:
    assert EVIDENCE_SCHEMA == "femx.jax.port_collective.tpu_evidence/v4"
    assert WORKER_ENTRY_CLAIM_SCHEMA == "femx.jax.port_collective.worker_entry_claim/v1"
    assert CHECKPOINT_ID == "port-collective-step-000000"
    assert CHECKPOINT_STEP == 0
    assert COMPLEX_SCALAR_CONTRACT == {
        "logical_dtype": "complex64",
        "matrix_dtype": "float32",
        "index_dtype": "int32",
        "execution_representation": "native complex64",
        "matmul_precision": "highest",
        "host_reference_dtype": "complex128",
        "precision_fallback": False,
    }


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
    for raw, expected in (("0", 0), ("7", 7)):
        monkeypatch.setenv("FEMX_INDEX", raw)
        assert _nonnegative_environment_integer("FEMX_INDEX") == expected
    for invalid in ("-1", "one"):
        monkeypatch.setenv("FEMX_INDEX", invalid)
        with pytest.raises(RuntimeError, match="nonnegative integer"):
            _nonnegative_environment_integer("FEMX_INDEX")


def test_runtime_rejects_missing_explicit_platform_before_jax_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(RuntimeError, match="must be set before Python starts"):
        _runtime()
    monkeypatch.setenv("JAX_PLATFORMS", "tpu,cpu")
    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    with pytest.raises(RuntimeError, match="JAX_DEFAULT_MATMUL_PRECISION=highest"):
        _runtime()


def test_structured_rectangle_has_positive_triangles_and_complete_boundary() -> None:
    coordinates, cells, facets = _structured_rectangle(3, 2)
    assert coordinates.shape == (12, 2)
    assert cells.shape == (12, 3)
    assert facets.shape == (10, 2)
    p0 = coordinates[cells[:, 0]]
    p1 = coordinates[cells[:, 1]]
    p2 = coordinates[cells[:, 2]]
    twice_area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    assert np.all(twice_area > 0.0)
    assert np.unique(np.sort(facets, axis=1), axis=0).shape == facets.shape


@pytest.mark.parametrize("complex_values", [False, True])
def test_numpy_matrix_free_action_and_vjp_match_jax(complex_values: bool) -> None:
    rng = np.random.default_rng(9)
    mapping = np.asarray(((0, 1, 2, 3, 4, 5), (2, 4, 6, 0, 3, 5)), dtype=np.int64)
    matrix = rng.normal(size=(2, 6, 6))
    vector = rng.normal(size=6)
    cotangent = rng.normal(size=6)
    if complex_values:
        vector = vector + 1j * rng.normal(size=6)
        cotangent = cotangent + 1j * rng.normal(size=6)

    expected_action = _numpy_matrix_free_matvec(matrix, mapping, vector)

    def operator(cell_matrix: jax.Array, coefficients: jax.Array) -> jax.Array:
        extended = jnp.concatenate((coefficients, jnp.zeros((1,), dtype=coefficients.dtype)))
        local_input = extended[jnp.asarray(mapping)]
        local_output = jnp.einsum("cij,cj->ci", cell_matrix, local_input)
        return (
            jnp.zeros((7,), dtype=local_output.dtype)
            .at[jnp.asarray(mapping).reshape(-1)]
            .add(local_output.reshape(-1))
        )[:6]

    observed_action, pullback = jax.vjp(operator, jnp.asarray(matrix), jnp.asarray(vector))
    observed_matrix_vjp, observed_vector_vjp = pullback(jnp.asarray(cotangent))
    expected_matrix_vjp, expected_vector_vjp = _numpy_matrix_free_vjp(
        matrix,
        mapping,
        vector,
        cotangent,
    )
    np.testing.assert_allclose(np.asarray(observed_action), expected_action, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(observed_matrix_vjp),
        expected_matrix_vjp,
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(observed_vector_vjp),
        expected_vector_vjp,
        rtol=1e-14,
        atol=1e-14,
    )


def test_packed_kernels_keep_distributed_map_at_jit_boundary() -> None:
    matrix = jnp.asarray(
        (
            ((2.0, -1.0), (-1.0, 2.0)),
            ((3.0, 0.5), (0.5, 4.0)),
        )
    )
    first_map = jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int64)
    second_map = jnp.asarray(((0, 2), (2, 1)), dtype=jnp.int64)
    vector = jnp.asarray((1.0, 2.0, 3.0))
    cotangent = jnp.asarray((0.5, -1.0, 2.0))

    def packed_operator(
        cell_matrix: jax.Array,
        cell_dof_map: jax.Array,
        owned_vector: jax.Array,
    ) -> jax.Array:
        local_output = jnp.einsum(
            "cij,cj->ci",
            cell_matrix,
            owned_vector[cell_dof_map],
        )
        return (
            jnp.zeros_like(owned_vector).at[cell_dof_map.reshape(-1)].add(local_output.reshape(-1))
        )

    apply, vjp = _build_explicit_packed_kernels(jax, packed_operator)
    compiled_apply = jax.jit(apply)
    compiled_vjp = jax.jit(vjp)

    first = compiled_apply(matrix, first_map, vector)
    second = compiled_apply(matrix, second_map, vector)
    assert not np.array_equal(np.asarray(first), np.asarray(second))
    observed_matrix_vjp, observed_vector_vjp = compiled_vjp(
        matrix,
        first_map,
        vector,
        cotangent,
    )
    _, expected_pullback = jax.vjp(
        lambda cell_matrix, owned_vector: packed_operator(
            cell_matrix,
            first_map,
            owned_vector,
        ),
        matrix,
        vector,
    )
    expected_matrix_vjp, expected_vector_vjp = expected_pullback(cotangent)
    np.testing.assert_allclose(observed_matrix_vjp, expected_matrix_vjp)
    np.testing.assert_allclose(observed_vector_vjp, expected_vector_vjp)
    assert all(not isinstance(cell.cell_contents, jax.Array) for cell in apply.__closure__ or ())


def test_numpy_relative_difference_is_scale_aware_and_handles_zero_reference() -> None:
    assert _numpy_relative_difference(np.asarray((1.0, 2.0)), np.asarray((1.0, 2.0))) == 0.0
    assert _numpy_relative_difference(np.asarray((2.0, 4.0)), np.asarray((1.0, 2.0))) == 1.0
    assert _numpy_relative_difference(np.zeros(2), np.zeros(2)) == 0.0
    assert np.isinf(_numpy_relative_difference(np.ones(2), np.zeros(2)))


def test_tpu_transport_indices_are_explicit_int32_and_range_checked() -> None:
    observed = _tpu_index_array(np.asarray(((0, 1), (2, 3)), dtype=np.int64))
    assert observed.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(observed, np.asarray(((0, 1), (2, 3)), dtype=np.int32))
    with pytest.raises(RuntimeError, match="must be integers"):
        _tpu_index_array(np.asarray((0.0, 1.0), dtype=np.float32))
    with pytest.raises(RuntimeError, match="exceed the explicit int32 contract"):
        _tpu_index_array(np.asarray((0, np.iinfo(np.int32).max + 1), dtype=np.int64))


def test_atomic_json_and_deployed_manifest_provenance(tmp_path: Path) -> None:
    destination = tmp_path / "results" / "metrics.json"
    _atomic_json(destination, {"finite": 1.0})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"finite": 1.0}
    assert not destination.with_suffix(".json.tmp").exists()
    with pytest.raises(ValueError, match="Out of range float values"):
        _atomic_json(destination, {"invalid": float("nan")})

    remote = tmp_path / "remote"
    manifest_path = remote / ".phoxla" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "profile": "test-profile",
                "source": {"digest": "a" * 64},
                "config": {"digest": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    assert _manifest_provenance(remote) == {
        "run_id": "run-1",
        "profile": "test-profile",
        "source_digest": "a" * 64,
        "config_digest": "b" * 64,
    }
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid deployed Phoxla manifest"):
        _manifest_provenance(remote)


def test_worker_entry_claim_is_atomic_immutable_and_provenance_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    remote.mkdir()
    provenance: dict[str, object] = {
        "run_id": "run-1",
        "profile": "test-profile",
        "source_digest": "a" * 64,
        "config_digest": "b" * 64,
    }
    monkeypatch.setenv("PHOXLA_RUN_ID", "run-1")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "3")
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "5")
    claim = _claim_worker_entry(remote.resolve(), provenance)
    assert claim["worker_index"] == 3
    assert claim["process_index"] == 5
    claim_path = remote / "logs" / "femx-entry.claim" / "identity.json"
    assert json.loads(claim_path.read_text(encoding="utf-8")) == claim
    with pytest.raises(RuntimeError, match="duplicate femx entry refused"):
        _claim_worker_entry(remote.resolve(), provenance)

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("PHOXLA_RUN_ID", "wrong")
    with pytest.raises(RuntimeError, match="disagrees with the deployed manifest"):
        _claim_worker_entry(other.resolve(), provenance)


class _MemoryAnalysis:
    generated_code_size_in_bytes = 7
    argument_size_in_bytes = 11
    output_size_in_bytes = 13
    alias_size_in_bytes = 3
    temp_size_in_bytes = 17


class _Compiled:
    def __init__(self, analysis: object) -> None:
        self._analysis = analysis

    def memory_analysis(self) -> object:
        return self._analysis


def test_compiler_memory_adapter_requires_complete_nonnegative_statistics() -> None:
    report = _memory_report(_Compiled(_MemoryAnalysis()), 100)
    assert report.compiler_peak_bytes == 38
    assert report.hbm_fraction == pytest.approx(0.38)
    with pytest.raises(RuntimeError, match="did not expose"):
        _memory_report(_Compiled(None), None)
    invalid = _MemoryAnalysis()
    invalid.temp_size_in_bytes = -1
    with pytest.raises(RuntimeError, match="invalid JAX compiler memory statistic"):
        _memory_report(_Compiled(invalid), None)
