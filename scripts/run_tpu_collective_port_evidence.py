#!/usr/bin/env python3
"""Run the bounded physical-TPU witness for femx's collective port operator.

The Phoxla bootstrap initializes JAX distribution before this file is evaluated.  Standalone TPU
execution is also supported, but only when ``JAX_PLATFORMS=tpu,cpu`` was set before Python starts.
There is deliberately no CPU fallback and no device discovery at module import time.
"""

from __future__ import annotations

import json
import math
import os
import time
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast

EVIDENCE_SCHEMA = "femx.jax.port_collective.tpu_evidence/v4"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.jax.port_collective.worker_entry_claim/v1"
CHECKPOINT_ID = "port-collective-step-000000"
CHECKPOINT_STEP = 0
ACTION_TOLERANCE = 2.0e-5
VJP_TOLERANCE = 2.0e-5
HOST_PRECISION_TOLERANCE = 2.0e-6
EXECUTION_SAMPLES = 5
COMPLEX_SCALAR_CONTRACT = {
    "logical_dtype": "complex64",
    "matrix_dtype": "float32",
    "index_dtype": "int32",
    "execution_representation": "native complex64",
    "matmul_precision": "highest",
    "host_reference_dtype": "complex128",
    "precision_fallback": False,
}


def _positive_environment_integer(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _nonnegative_environment_integer(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a nonnegative integer") from error
    if value < 0:
        raise RuntimeError(f"{name} must be a nonnegative integer")
    return value


def _require_expected_count(name: str, observed: int) -> None:
    expected = _positive_environment_integer(name)
    if expected is not None and expected != observed:
        raise RuntimeError(f"{name} requires {expected}, observed {observed}")


def _runtime() -> Any:
    if os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("JAX_PLATFORMS=tpu,cpu must be set before Python starts")
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise RuntimeError("JAX_DEFAULT_MATMUL_PRECISION=highest must be set before Python starts")

    import jax

    if "PHOXLA_PROCESS_INDEX" not in os.environ:
        jax.distributed.initialize()
    if jax.default_backend() != "tpu" or any(device.platform != "tpu" for device in jax.devices()):
        raise RuntimeError(f"physical TPU backend required, observed {jax.default_backend()!r}")
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("physical TPU witness requires the explicit float32/complex64 path")
    if str(getattr(jax.config, "jax_default_matmul_precision", None)) != "highest":
        raise RuntimeError("JAX default matmul precision must resolve to 'highest'")
    if jax.process_count() < 2:
        raise RuntimeError("physical witness requires at least two JAX processes")
    if jax.device_count() < 2:
        raise RuntimeError("physical witness requires at least two global TPU devices")
    if jax.process_count() * jax.local_device_count() != jax.device_count():
        raise RuntimeError("uniform one-process-per-worker device accounting is required")
    _require_expected_count("FEMX_EXPECTED_PROCESS_COUNT", jax.process_count())
    _require_expected_count("FEMX_EXPECTED_GLOBAL_DEVICE_COUNT", jax.device_count())
    _require_expected_count("FEMX_EXPECTED_LOCAL_DEVICE_COUNT", jax.local_device_count())
    return jax


def _structured_rectangle(
    x_intervals: int,
    y_intervals: int,
) -> tuple[Any, Any, Any]:
    import numpy as np

    width = x_intervals + 1
    coordinates = np.asarray(
        [
            (2.0e-6 * i / x_intervals, 1.0e-6 * j / y_intervals)
            for j in range(y_intervals + 1)
            for i in range(x_intervals + 1)
        ],
        dtype=np.float64,
    )

    def node(i: int, j: int) -> int:
        return j * width + i

    cells: list[tuple[int, int, int]] = []
    for j in range(y_intervals):
        for i in range(x_intervals):
            lower_left = node(i, j)
            lower_right = node(i + 1, j)
            upper_left = node(i, j + 1)
            upper_right = node(i + 1, j + 1)
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    facets: list[tuple[int, int]] = []
    facets.extend((node(i, 0), node(i + 1, 0)) for i in range(x_intervals))
    facets.extend((node(x_intervals, j), node(x_intervals, j + 1)) for j in range(y_intervals))
    facets.extend((node(i + 1, y_intervals), node(i, y_intervals)) for i in range(x_intervals))
    facets.extend((node(0, j + 1), node(0, j)) for j in range(y_intervals))
    return (
        coordinates,
        np.asarray(cells, dtype=np.int64),
        np.asarray(facets, dtype=np.int64),
    )


def _numpy_matrix_free_matvec(cell_matrix: Any, cell_map: Any, vector: Any) -> Any:
    import numpy as np

    free_dof_count = int(vector.shape[0])
    extended = np.concatenate((vector, np.zeros((1,), dtype=vector.dtype)))
    local_input = extended[cell_map]
    local_output = np.einsum("cij,cj->ci", cell_matrix, local_input)
    assembled = np.zeros((free_dof_count + 1,), dtype=local_output.dtype)
    np.add.at(assembled, cell_map.reshape(-1), local_output.reshape(-1))
    return assembled[:free_dof_count]


def _numpy_matrix_free_vjp(
    cell_matrix: Any,
    cell_map: Any,
    vector: Any,
    cotangent: Any,
) -> tuple[Any, Any]:
    import numpy as np

    free_dof_count = int(vector.shape[0])
    extended_vector = np.concatenate((vector, np.zeros((1,), dtype=vector.dtype)))
    extended_cotangent = np.concatenate((cotangent, np.zeros((1,), dtype=cotangent.dtype)))
    local_vector = extended_vector[cell_map]
    local_cotangent = extended_cotangent[cell_map]
    matrix_vjp = local_cotangent[:, :, None] * local_vector[:, None, :]
    if np.asarray(cell_matrix).dtype.kind != "c":
        matrix_vjp = np.real(matrix_vjp)
    local_vector_vjp = np.einsum("cij,ci->cj", cell_matrix, local_cotangent)
    vector_vjp = np.zeros((free_dof_count + 1,), dtype=local_vector_vjp.dtype)
    np.add.at(vector_vjp, cell_map.reshape(-1), local_vector_vjp.reshape(-1))
    return np.asarray(matrix_vjp, dtype=cell_matrix.dtype), vector_vjp[:free_dof_count]


def _numpy_relative_difference(observed: Any, expected: Any) -> float:
    import numpy as np

    numerator = float(np.linalg.norm(np.asarray(observed) - np.asarray(expected)))
    denominator = float(np.linalg.norm(np.asarray(expected)))
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else math.inf


def _tpu_index_array(values: Any) -> Any:
    """Return an explicitly bounded int32 transport index array."""

    import numpy as np

    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise RuntimeError("TPU collective indices must be integers")
    limits = np.iinfo(np.int32)
    if raw.size and (np.min(raw) < limits.min or np.max(raw) > limits.max):
        raise RuntimeError("TPU collective indices exceed the explicit int32 contract")
    return raw.astype(np.int32, copy=False)


def _pack_cells(layout: Any, values: Any) -> Any:
    import numpy as np

    extended = np.concatenate(
        (values, np.zeros((1, 6, 6), dtype=values.dtype)),
        axis=0,
    )
    return np.ascontiguousarray(extended[layout.cell_ids])


def _pack_owned(layout: Any, values: Any) -> Any:
    import numpy as np

    extended = np.concatenate((values, np.zeros((1,), dtype=values.dtype)))
    return np.ascontiguousarray(extended[layout.owned_dof_ids])


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_provenance(remote_run: Path) -> dict[str, object]:
    manifest_path = remote_run / ".phoxla" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "run_id": manifest["run_id"],
            "profile": manifest["profile"],
            "source_digest": manifest["source"]["digest"],
            "config_digest": manifest["config"]["digest"],
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid deployed Phoxla manifest: {manifest_path}") from error


def _claim_worker_entry(remote_run: Path, provenance: dict[str, object]) -> dict[str, object]:
    """Create an immutable worker-local fence before femx scientific execution."""

    if not remote_run.is_absolute() or not remote_run.is_dir() or remote_run.is_symlink():
        raise RuntimeError("PHOXLA_REMOTE_RUN_DIR must be an absolute non-symlink directory")
    process_index = _nonnegative_environment_integer("PHOXLA_PROCESS_INDEX")
    worker_index = _nonnegative_environment_integer("PHOXLA_GCLOUD_WORKER_INDEX")
    if process_index is None or worker_index is None:
        raise RuntimeError("Phoxla process and worker indexes are required before entry execution")
    run_id = os.environ.get("PHOXLA_RUN_ID")
    if run_id != provenance.get("run_id"):
        raise RuntimeError("PHOXLA_RUN_ID disagrees with the deployed manifest")

    claim_path = remote_run / "logs" / "femx-entry.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "duplicate femx entry refused for this immutable worker-local run"
        ) from error

    claim = {
        "schema_version": WORKER_ENTRY_CLAIM_SCHEMA,
        "run_id": run_id,
        "worker_index": worker_index,
        "process_index": process_index,
        "source_sha256": provenance.get("source_digest"),
        "config_sha256": provenance.get("config_digest"),
        "scope": (
            "worker-local femx entry fence after Phoxla bootstrap; prevents duplicate "
            "scientific execution but does not claim controller-level launch ownership"
        ),
    }
    _atomic_json(claim_path / "identity.json", claim)
    return claim


def _memory_report(compiled: Any, hbm_capacity_bytes: int | None) -> Any:
    from femx.backends.jax.port_collective_runtime import PortCollectiveCompilerMemoryReport

    analysis = compiled.memory_analysis()
    if analysis is None:
        raise RuntimeError("JAX executable did not expose compiler memory analysis")

    def measured(name: str) -> int:
        value = getattr(analysis, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid JAX compiler memory statistic {name!r}: {value!r}")
        return value

    return PortCollectiveCompilerMemoryReport(
        generated_code_bytes=measured("generated_code_size_in_bytes"),
        argument_bytes=measured("argument_size_in_bytes"),
        output_bytes=measured("output_size_in_bytes"),
        alias_bytes=measured("alias_size_in_bytes"),
        temporary_bytes=measured("temp_size_in_bytes"),
        hbm_capacity_bytes_per_device=hbm_capacity_bytes,
    )


def _compile_and_time(
    jax: Any,
    function: Any,
    arguments: tuple[Any, ...],
) -> tuple[Any, Any, str]:
    from femx.backends.jax.port_collective_runtime import PortCollectiveTimingReport

    started = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir("stablehlo"))

    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warmup_result = compiled(*arguments)
    jax.block_until_ready(warmup_result)
    warmup_seconds = time.perf_counter() - started

    samples: list[float] = []
    for _ in range(EXECUTION_SAMPLES):
        started = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    timing = PortCollectiveTimingReport(
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        warmup_seconds=warmup_seconds,
        execution_seconds=tuple(samples),
    )
    return compiled, timing, stablehlo


def _build_explicit_packed_kernels(jax: Any, packed_operator: Any) -> tuple[Any, Any]:
    """Keep every distributed array explicit at the outer JIT boundary."""

    def apply(cell_matrix: Any, cell_dof_map: Any, owned_vector: Any) -> Any:
        return packed_operator(cell_matrix, cell_dof_map, owned_vector)

    def vjp(
        cell_matrix: Any,
        cell_dof_map: Any,
        owned_vector: Any,
        cotangent: Any,
    ) -> tuple[Any, Any]:
        def differentiable_apply(matrix: Any, vector: Any) -> Any:
            return packed_operator(matrix, cell_dof_map, vector)

        _, pullback = jax.vjp(differentiable_apply, cell_matrix, owned_vector)
        return cast(tuple[Any, Any], pullback(cotangent))

    return apply, vjp


def main() -> int:
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    launch_claim = _claim_worker_entry(remote_run, provenance)
    jax = _runtime()
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh

    from femx.backends._hcurl import (
        canonical_mixed_port_dof_partition,
        canonical_triangle_edge_map,
    )
    from femx.backends.jax.port_collective import (
        build_packed_collective_port_matvec,
        describe_collective_port_mesh,
        prepare_collective_port_layout,
    )
    from femx.backends.jax.port_collective_checkpoint import (
        port_collective_checkpoint_fragment_path,
        restore_port_collective_checkpoint_fragment,
        write_port_collective_checkpoint_fragment,
    )
    from femx.backends.jax.port_collective_runtime import (
        make_collective_port_array_from_process_local_data,
    )
    from femx.backends.jax.port_matrix_free import (
        build_lossless_matrix_free_port_pencil,
        prepare_port_matrix_free_topology,
    )
    from femx.backends.jax.port_owned_ghost import prepare_owned_ghost_port_topology
    from femx.physics import (
        VACUUM_PERMEABILITY_H_PER_M,
        VACUUM_PERMITTIVITY_F_PER_M,
    )

    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    global_device_count = int(jax.device_count())
    local_device_count = int(jax.local_device_count())
    output_root = Path(os.environ["PHOXLA_OUTPUT_DIR"])
    if cast(int, launch_claim["process_index"]) != process_index:
        raise RuntimeError("worker entry claim disagrees with initialized JAX process identity")

    x_intervals = 2 * global_device_count
    y_intervals = max(8, global_device_count)
    coordinates, cells, boundary_facets = _structured_rectangle(x_intervals, y_intervals)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)
    pec = canonical_mixed_port_dof_partition(
        boundary_facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    serial_topology = prepare_port_matrix_free_topology(
        cells,
        edge_map.cell_edge_dofs,
        pec.free_dofs,
        node_count=coordinates.shape[0],
        edge_dof_count=edge_map.dof_count,
    )
    centroids = np.mean(coordinates[cells], axis=1)
    silicon = (np.abs(centroids[:, 0] - 1.0e-6) <= 0.25e-6) & (
        np.abs(centroids[:, 1] - 0.5e-6) <= 0.11e-6
    )
    relative_permittivity = np.where(silicon, 3.48**2, 1.444**2)
    frequency_hz = 193.414e12
    pencil = build_lossless_matrix_free_port_pencil(
        jnp.asarray(coordinates, dtype=jnp.float32),
        jnp.asarray(cells, dtype=jnp.int32),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(serial_topology.cell_reduced_dofs, dtype=jnp.int32),
        jnp.asarray(relative_permittivity, dtype=jnp.float32),
        jnp.ones(cells.shape[0], dtype=jnp.float32),
        jnp.asarray(frequency_hz, dtype=jnp.float32),
        free_dof_count=serial_topology.free_dof_count,
    )
    stiffness = np.asarray(jax.device_get(pencil.stiffness), dtype=np.float32)
    mass = np.asarray(jax.device_get(pencil.mass), dtype=np.float32)
    maximum_permittivity = VACUUM_PERMITTIVITY_F_PER_M * float(np.max(relative_permittivity))
    beta_limit = (
        2.0 * math.pi * frequency_hz * math.sqrt(maximum_permittivity * VACUUM_PERMEABILITY_H_PER_M)
    )
    shift_per_m2 = -(beta_limit * beta_limit)
    shifted = stiffness - np.float32(shift_per_m2) * mass

    normalized_x = centroids[:, 0] / np.max(coordinates[:, 0])
    cell_owners = np.minimum(
        (global_device_count * normalized_x).astype(np.int64),
        global_device_count - 1,
    )
    if not np.array_equal(np.unique(cell_owners), np.arange(global_device_count)):
        raise RuntimeError("physical mesh does not assign at least one cell to every TPU device")
    topology = prepare_owned_ghost_port_topology(
        serial_topology.cell_reduced_dofs,
        cell_owners,
        free_dof_count=serial_topology.free_dof_count,
        partition_count=global_device_count,
    )
    layout = prepare_collective_port_layout(topology)
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("partition",))
    mesh_report = describe_collective_port_mesh(layout, mesh)
    packed_operator = build_packed_collective_port_matvec(layout, mesh)

    rng = np.random.default_rng(20260901)
    real_vector_reference = rng.normal(size=topology.global_dof_count)
    imaginary_vector_reference = rng.normal(size=topology.global_dof_count)
    complex_vector_reference = real_vector_reference + 1j * imaginary_vector_reference
    real_cotangent_reference = rng.normal(size=topology.global_dof_count)
    imaginary_cotangent_reference = rng.normal(size=topology.global_dof_count)
    complex_cotangent_reference = real_cotangent_reference + 1j * imaginary_cotangent_reference
    real_vector = real_vector_reference.astype(np.float32)
    complex_vector = complex_vector_reference.astype(np.complex64)
    real_cotangent = real_cotangent_reference.astype(np.float32)
    complex_cotangent = complex_cotangent_reference.astype(np.complex64)

    packed_map, map_report = make_collective_port_array_from_process_local_data(
        "cell-local-dof-map",
        _tpu_index_array(layout.cell_local_dofs),
        mesh,
    )

    def load(name: str, value: Any) -> tuple[Any, Any]:
        return make_collective_port_array_from_process_local_data(name, value, mesh)

    real_owned, real_owned_report = load("real-owned-vector", _pack_owned(layout, real_vector))
    complex_owned, complex_owned_report = load(
        "complex-owned-vector",
        _pack_owned(layout, complex_vector),
    )
    real_cotangent_array, real_cotangent_report = load(
        "real-owned-cotangent",
        _pack_owned(layout, real_cotangent),
    )
    complex_cotangent_array, complex_cotangent_report = load(
        "complex-owned-cotangent",
        _pack_owned(layout, complex_cotangent),
    )

    packed_matrices: dict[str, tuple[Any, Any]] = {}
    for name, matrix in (("stiffness", stiffness), ("mass", mass), ("shifted", shifted)):
        packed_matrices[name] = load(f"{name}-cell-blocks", _pack_cells(layout, matrix))

    checkpoint_arrays = {
        "cell-local-dof-map": packed_map,
        "complex-owned-cotangent": complex_cotangent_array,
        "complex-owned-vector": complex_owned,
        "mass-cell-blocks": packed_matrices["mass"][0],
        "real-owned-cotangent": real_cotangent_array,
        "real-owned-vector": real_owned,
        "shifted-cell-blocks": packed_matrices["shifted"][0],
        "stiffness-cell-blocks": packed_matrices["stiffness"][0],
    }
    resume_root_raw = os.environ.get("FEMX_TPU_RESUME_CHECKPOINT_ROOT")
    if resume_root_raw is None:
        checkpoint_mode = "fresh-process-roundtrip"
        checkpoint_fragment = write_port_collective_checkpoint_fragment(
            output_root.resolve() / "checkpoints",
            checkpoint_id=CHECKPOINT_ID,
            step=CHECKPOINT_STEP,
            source_sha256=cast(str, provenance["source_digest"]),
            config_sha256=cast(str, provenance["config_digest"]),
            layout=layout,
            mesh=mesh,
            arrays=checkpoint_arrays,
        )
        restored_arrays, verified_fragment = restore_port_collective_checkpoint_fragment(
            checkpoint_fragment.path,
            expected_checkpoint_id=CHECKPOINT_ID,
            expected_step=CHECKPOINT_STEP,
            expected_source_sha256=cast(str, provenance["source_digest"]),
            expected_config_sha256=cast(str, provenance["config_digest"]),
            layout=layout,
            mesh=mesh,
            templates=checkpoint_arrays,
        )
        if verified_fragment != checkpoint_fragment:
            raise RuntimeError("fresh checkpoint identity changed during immediate verification")
    else:
        checkpoint_mode = "restored-external-fragment"
        resume_root = Path(resume_root_raw)
        resume_fragment = port_collective_checkpoint_fragment_path(
            resume_root,
            CHECKPOINT_ID,
            process_index,
        )
        restored_arrays, checkpoint_fragment = restore_port_collective_checkpoint_fragment(
            resume_fragment,
            expected_checkpoint_id=CHECKPOINT_ID,
            expected_step=CHECKPOINT_STEP,
            expected_source_sha256=cast(str, provenance["source_digest"]),
            expected_config_sha256=cast(str, provenance["config_digest"]),
            layout=layout,
            mesh=mesh,
            templates=checkpoint_arrays,
        )

    multihost_utils.sync_global_devices(f"femx-checkpoint-restored-{provenance['run_id']}")
    packed_map = restored_arrays["cell-local-dof-map"]
    complex_cotangent_array = restored_arrays["complex-owned-cotangent"]
    complex_owned = restored_arrays["complex-owned-vector"]
    real_cotangent_array = restored_arrays["real-owned-cotangent"]
    real_owned = restored_arrays["real-owned-vector"]
    for name in ("mass", "shifted", "stiffness"):
        packed_matrices[name] = (
            restored_arrays[f"{name}-cell-blocks"],
            packed_matrices[name][1],
        )

    packed_apply, packed_vjp = _build_explicit_packed_kernels(jax, packed_operator)

    shifted_array = packed_matrices["shifted"][0]
    real_forward, real_forward_timing, real_forward_hlo = _compile_and_time(
        jax,
        packed_apply,
        (shifted_array, packed_map, real_owned),
    )
    complex_forward, complex_forward_timing, complex_forward_hlo = _compile_and_time(
        jax,
        packed_apply,
        (shifted_array, packed_map, complex_owned),
    )
    compiled_real_vjp, real_vjp_timing, real_vjp_hlo = _compile_and_time(
        jax,
        packed_vjp,
        (shifted_array, packed_map, real_owned, real_cotangent_array),
    )
    compiled_complex_vjp, complex_vjp_timing, complex_vjp_hlo = _compile_and_time(
        jax,
        packed_vjp,
        (shifted_array, packed_map, complex_owned, complex_cotangent_array),
    )

    def relative_difference_unjitted(observed: Any, expected: Any) -> Any:
        numerator = jnp.linalg.norm(observed - expected)
        denominator = jnp.linalg.norm(expected)
        return jnp.where(
            denominator > 0.0,
            numerator / denominator,
            jnp.where(numerator == 0.0, 0.0, jnp.inf),
        )

    def all_finite_unjitted(value: Any) -> Any:
        return jnp.all(jnp.isfinite(value))

    relative_difference = jax.jit(relative_difference_unjitted)
    all_finite = jax.jit(all_finite_unjitted)

    action_differences: dict[str, dict[str, float]] = {}
    action_finite: dict[str, dict[str, bool]] = {}
    host_precision_action_differences: dict[str, float] = {}
    for name, matrix in (("stiffness", stiffness), ("mass", mass), ("shifted", shifted)):
        matrix_array = packed_matrices[name][0]
        actual_real = real_forward(matrix_array, packed_map, real_owned)
        actual_complex = complex_forward(matrix_array, packed_map, complex_owned)
        expected_real_host = _numpy_matrix_free_matvec(
            matrix,
            serial_topology.cell_reduced_dofs,
            real_vector,
        )
        expected_complex_host = _numpy_matrix_free_matvec(
            matrix,
            serial_topology.cell_reduced_dofs,
            complex_vector,
        )
        reference_complex_host = _numpy_matrix_free_matvec(
            matrix.astype(np.float64),
            serial_topology.cell_reduced_dofs,
            complex_vector_reference,
        )
        expected_real, _ = load(
            f"expected-{name}-real-action",
            _pack_owned(layout, expected_real_host),
        )
        expected_complex, _ = load(
            f"expected-{name}-complex-action",
            _pack_owned(layout, expected_complex_host),
        )
        action_differences[name] = {
            "real": float(
                np.asarray(jax.device_get(relative_difference(actual_real, expected_real)))
            ),
            "complex": float(
                np.asarray(jax.device_get(relative_difference(actual_complex, expected_complex)))
            ),
        }
        action_finite[name] = {
            "real": bool(np.asarray(jax.device_get(all_finite(actual_real)))),
            "complex": bool(np.asarray(jax.device_get(all_finite(actual_complex)))),
        }
        host_precision_action_differences[name] = _numpy_relative_difference(
            expected_complex_host,
            reference_complex_host,
        )

    real_matrix_vjp, real_vector_vjp = compiled_real_vjp(
        shifted_array,
        packed_map,
        real_owned,
        real_cotangent_array,
    )
    complex_matrix_vjp, complex_vector_vjp = compiled_complex_vjp(
        shifted_array,
        packed_map,
        complex_owned,
        complex_cotangent_array,
    )
    jax.block_until_ready(
        (real_matrix_vjp, real_vector_vjp, complex_matrix_vjp, complex_vector_vjp)
    )
    expected_real_matrix, expected_real_vector = _numpy_matrix_free_vjp(
        shifted,
        serial_topology.cell_reduced_dofs,
        real_vector,
        real_cotangent,
    )
    expected_complex_matrix, expected_complex_vector = _numpy_matrix_free_vjp(
        shifted,
        serial_topology.cell_reduced_dofs,
        complex_vector,
        complex_cotangent,
    )
    reference_complex_matrix, reference_complex_vector = _numpy_matrix_free_vjp(
        shifted.astype(np.float64),
        serial_topology.cell_reduced_dofs,
        complex_vector_reference,
        complex_cotangent_reference,
    )
    host_precision_vjp_differences = {
        "complex_cell_matrix": _numpy_relative_difference(
            expected_complex_matrix,
            reference_complex_matrix,
        ),
        "complex_owned_vector": _numpy_relative_difference(
            expected_complex_vector,
            reference_complex_vector,
        ),
    }
    expected_real_matrix_array, _ = load(
        "expected-real-cell-vjp",
        _pack_cells(layout, expected_real_matrix),
    )
    expected_real_vector_array, _ = load(
        "expected-real-vector-vjp",
        _pack_owned(layout, expected_real_vector),
    )
    expected_complex_matrix_array, _ = load(
        "expected-complex-cell-vjp",
        _pack_cells(layout, expected_complex_matrix),
    )
    expected_complex_vector_array, _ = load(
        "expected-complex-vector-vjp",
        _pack_owned(layout, expected_complex_vector),
    )
    vjp_differences = {
        "real_cell_matrix": float(
            np.asarray(
                jax.device_get(relative_difference(real_matrix_vjp, expected_real_matrix_array))
            )
        ),
        "real_owned_vector": float(
            np.asarray(
                jax.device_get(relative_difference(real_vector_vjp, expected_real_vector_array))
            )
        ),
        "complex_cell_matrix": float(
            np.asarray(
                jax.device_get(
                    relative_difference(complex_matrix_vjp, expected_complex_matrix_array)
                )
            )
        ),
        "complex_owned_vector": float(
            np.asarray(
                jax.device_get(
                    relative_difference(complex_vector_vjp, expected_complex_vector_array)
                )
            )
        ),
    }
    vjp_finite = {
        "real_cell_matrix": bool(np.asarray(jax.device_get(all_finite(real_matrix_vjp)))),
        "real_owned_vector": bool(np.asarray(jax.device_get(all_finite(real_vector_vjp)))),
        "complex_cell_matrix": bool(np.asarray(jax.device_get(all_finite(complex_matrix_vjp)))),
        "complex_owned_vector": bool(np.asarray(jax.device_get(all_finite(complex_vector_vjp)))),
    }

    local_partition_mask = np.zeros((global_device_count,), dtype=np.int32)
    for shard in map_report.addressable_shards:
        local_partition_mask[shard.partition_index] = 1
    gathered_partition_masks = np.asarray(
        multihost_utils.process_allgather(local_partition_mask, tiled=False)
    ).reshape(process_count, global_device_count)
    partition_addressability_counts = np.sum(gathered_partition_masks, axis=0)

    hbm_capacity_bytes = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    executables = {
        "real_forward": (real_forward, real_forward_timing, real_forward_hlo),
        "complex_forward": (complex_forward, complex_forward_timing, complex_forward_hlo),
        "real_vjp": (compiled_real_vjp, real_vjp_timing, real_vjp_hlo),
        "complex_vjp": (compiled_complex_vjp, complex_vjp_timing, complex_vjp_hlo),
    }
    executable_evidence: dict[str, dict[str, object]] = {}
    for name, (compiled, timing, stablehlo) in executables.items():
        executable_evidence[name] = {
            "timing": timing.canonical_data(),
            "memory": _memory_report(compiled, hbm_capacity_bytes).canonical_data(),
            "stablehlo_collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
            "stablehlo_contains_all_gather": "all_gather" in stablehlo.lower(),
        }

    expected_forward_permutations = 2 * len(layout.halo_links)
    maximum_action_difference = max(
        difference
        for scalar_kind in action_differences.values()
        for difference in scalar_kind.values()
    )
    maximum_vjp_difference = max(vjp_differences.values())
    maximum_host_precision_difference = max(
        *host_precision_action_differences.values(),
        *host_precision_vjp_differences.values(),
    )
    real_forward_permutations = cast(
        int,
        executable_evidence["real_forward"]["stablehlo_collective_permute_count"],
    )
    complex_forward_permutations = cast(
        int,
        executable_evidence["complex_forward"]["stablehlo_collective_permute_count"],
    )
    all_gathers_absent = all(
        not bool(record["stablehlo_contains_all_gather"]) for record in executable_evidence.values()
    )
    passed = (
        maximum_action_difference <= ACTION_TOLERANCE
        and maximum_vjp_difference <= VJP_TOLERANCE
        and maximum_host_precision_difference <= HOST_PRECISION_TOLERANCE
        and all(value for values in action_finite.values() for value in values.values())
        and all(vjp_finite.values())
        and np.array_equal(partition_addressability_counts, np.ones(global_device_count))
        and real_forward_permutations == expected_forward_permutations
        and complex_forward_permutations == expected_forward_permutations
        and all_gathers_absent
    )

    storage = layout.storage_report
    process_payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": distribution_version("jaxlib"),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "default_matmul_precision": str(
                getattr(jax.config, "jax_default_matmul_precision", None)
            ),
            "process_index": process_index,
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
            "complex_scalar_contract": COMPLEX_SCALAR_CONTRACT,
        },
        "launch_claim": launch_claim,
        "physics": {
            "model": "2D lossless Si/SiO2 mixed H(curl)/H1 port operator",
            "frequency_hz": frequency_hz,
            "silicon_refractive_index": 3.48,
            "silica_refractive_index": 1.444,
            "core_width_m": 0.5e-6,
            "core_height_m": 0.22e-6,
            "cross_section_width_m": 2.0e-6,
            "cross_section_height_m": 1.0e-6,
            "shift_per_m2": shift_per_m2,
        },
        "problem": {
            "x_intervals": x_intervals,
            "y_intervals": y_intervals,
            "node_count": int(coordinates.shape[0]),
            "triangle_count": int(cells.shape[0]),
            "free_mixed_dof_count": topology.global_dof_count,
            "partition_count": layout.partition_count,
            "layout_sha256": layout.digest(),
            "halo_link_count": len(layout.halo_links),
            "halo_value_count": storage.halo_value_count,
            "cell_padding_fraction": storage.cell_padding_fraction,
            "owned_dof_padding_fraction": storage.owned_dof_padding_fraction,
            "ghost_dof_padding_fraction": storage.ghost_dof_padding_fraction,
        },
        "mesh_report": mesh_report.canonical_data(),
        "addressability": {
            "process_local_partition_mask": local_partition_mask.tolist(),
            "partition_addressability_counts": partition_addressability_counts.tolist(),
            "every_partition_addressable_once": bool(
                np.array_equal(partition_addressability_counts, np.ones(global_device_count))
            ),
        },
        "array_reports": {
            "cell_local_dofs": map_report.canonical_data(),
            "shifted_cell_blocks": packed_matrices["shifted"][1].canonical_data(),
            "real_owned_vector": real_owned_report.canonical_data(),
            "complex_owned_vector": complex_owned_report.canonical_data(),
            "real_owned_cotangent": real_cotangent_report.canonical_data(),
            "complex_owned_cotangent": complex_cotangent_report.canonical_data(),
        },
        "checkpoint": {
            "mode": checkpoint_mode,
            "fragment": checkpoint_fragment.canonical_data(),
            "restored_state_consumed_by_operator": True,
            "deterministic_next_action_authority": (
                "independent NumPy action and analytic VJP in this evidence run"
            ),
            "actual_preemption_event": False,
            "cross_topology_restore": False,
            "control_plane_scope": (
                "external durable upload, Spot recreation, and process-fragment placement"
            ),
        },
        "numerics": {
            "authority": (
                "independent NumPy float32/complex64 cell gather/einsum/scatter and analytic VJP; "
                "host float64/complex128 arithmetic reference recorded separately"
            ),
            "action_relative_differences": action_differences,
            "maximum_action_relative_difference": maximum_action_difference,
            "action_finite": action_finite,
            "vjp_relative_differences": vjp_differences,
            "maximum_vjp_relative_difference": maximum_vjp_difference,
            "vjp_finite": vjp_finite,
            "action_tolerance": ACTION_TOLERANCE,
            "vjp_tolerance": VJP_TOLERANCE,
            "host_c64_vs_c128_action_relative_differences": (host_precision_action_differences),
            "host_c64_vs_c128_vjp_relative_differences": host_precision_vjp_differences,
            "maximum_host_c64_vs_c128_relative_difference": (maximum_host_precision_difference),
            "host_precision_tolerance": HOST_PRECISION_TOLERANCE,
            "host_precision_scope": (
                "operator arithmetic and vector/cotangent quantization with float32 cell "
                "coefficients held fixed; not float64 FEM assembly parity"
            ),
        },
        "executables": executable_evidence,
        "claim_scope": (
            "physical multi-process TPU action/VJP/compile evidence for one bounded lossless "
            "Si/SiO2 port operator in float32/complex64 with a host complex128 arithmetic "
            "reference; not eigensolve scaling, Elmer parity, FDTDX integration, "
            "actual preemption recovery, cross-topology restore, or device prediction"
        ),
    }

    for name, (_, _, stablehlo) in executables.items():
        hlo_path = output_root / "hlo" / f"{name}.stablehlo.mlir"
        hlo_path.parent.mkdir(parents=True, exist_ok=True)
        hlo_path.write_text(stablehlo, encoding="utf-8")
    _atomic_json(output_root / "results" / "process-metrics.json", process_payload)
    if process_index == 0:
        _atomic_json(output_root / "results" / "metrics.json", process_payload)

    multihost_utils.sync_global_devices(f"femx-tpu-evidence-written-{provenance['run_id']}")
    print(json.dumps({"status": process_payload["status"], "run_id": provenance["run_id"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
