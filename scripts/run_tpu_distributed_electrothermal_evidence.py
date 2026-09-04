#!/usr/bin/env python3
"""Run one process-complete physical-TPU coupled electrothermal forward and VJP witness."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    import_path = str(import_root)
    while import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

from femx.validation.tpu_distributed_electrothermal_evidence import (
    COUPLED_ADJOINT_POLICY,
    PROCESS_EVIDENCE_SCHEMA,
    REPLICATION_INTENT,
    SCALAR_CG_POLICY,
    TOLERANCES,
    WORKER_ENTRY_CLAIM_SCHEMA,
)
from femx.validation.tpu_distributed_electrothermal_evidence import (
    REAL_SCALAR_CONTRACT as ADMISSION_REAL_SCALAR_CONTRACT,
)

EVIDENCE_SCHEMA = PROCESS_EVIDENCE_SCHEMA
NUMERICAL_DIAGNOSTIC_SCHEMA = "femx.jax.distributed_electrothermal.tpu_numerical_diagnostic/v1"
EXECUTION_SAMPLES = 5
CG_RELATIVE_TOLERANCE = cast(float, SCALAR_CG_POLICY["relative_tolerance"])
CG_ABSOLUTE_TOLERANCE = cast(float, SCALAR_CG_POLICY["absolute_tolerance"])
CG_MAX_ITERATIONS = cast(int, SCALAR_CG_POLICY["max_iterations"])
CG_BACKWARD_ERROR_TOLERANCE = cast(float, SCALAR_CG_POLICY["backward_error_tolerance"])
JACOBI_MINIMUM_RELATIVE_DIAGONAL = cast(
    float,
    cast(Mapping[str, object], SCALAR_CG_POLICY["preconditioner"])["minimum_relative_diagonal"],
)
ADJOINT_RELATIVE_TOLERANCE = cast(float, COUPLED_ADJOINT_POLICY["relative_tolerance"])
ADJOINT_ABSOLUTE_TOLERANCE = cast(float, COUPLED_ADJOINT_POLICY["absolute_tolerance"])
ADJOINT_RESTART = cast(int, COUPLED_ADJOINT_POLICY["restart"])
ADJOINT_MAX_RESTARTS = cast(int, COUPLED_ADJOINT_POLICY["max_restarts"])
POTENTIAL_TOLERANCE = TOLERANCES["potential_relative_difference"]
TEMPERATURE_TOLERANCE = TOLERANCES["temperature_relative_difference"]
GRADIENT_TOLERANCE = TOLERANCES["gradient_relative_difference"]
NATIVE_GRADIENT_TOLERANCE = TOLERANCES["native_explicit_gradient_relative_difference"]
OBJECTIVE_TOLERANCE = TOLERANCES["objective_relative_difference"]
TRANSFER_TOLERANCE = TOLERANCES["transfer_relative_error"]
REAL_SCALAR_CONTRACT = dict(ADMISSION_REAL_SCALAR_CONTRACT)
_PARTITIONED_INPUTS = {
    "cell_local_dofs",
    "owner_mask",
    "cell_mask",
    "unit_stiffness",
    "basis_gradients",
    "cell_areas",
    "current_conductivity_base",
    "current_conductivity_weights",
    "current_cell_load_base",
    "current_cell_load_weights",
    "current_dirichlet_base",
    "current_dirichlet_weights",
    "thermal_conductivity_base",
    "thermal_conductivity_weights",
    "thermal_cell_load_base",
    "thermal_cell_load_weights",
    "thermal_dirichlet_base",
    "thermal_dirichlet_weights",
    "feedback_reference_base",
    "feedback_reference_weights",
    "feedback_coefficient_base",
    "feedback_coefficient_weights",
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
        raise RuntimeError("physical coupled witness requires the explicit float32 path")
    if str(getattr(jax.config, "jax_default_matmul_precision", None)) != "highest":
        raise RuntimeError("JAX default matmul precision must resolve to 'highest'")
    if jax.process_count() < 2 or jax.device_count() < 2:
        raise RuntimeError("physical coupled witness requires multiple processes and TPU devices")
    if jax.process_count() * jax.local_device_count() != jax.device_count():
        raise RuntimeError("uniform one-process-per-worker device accounting is required")
    _require_expected_count("FEMX_EXPECTED_PROCESS_COUNT", jax.process_count())
    _require_expected_count("FEMX_EXPECTED_GLOBAL_DEVICE_COUNT", jax.device_count())
    _require_expected_count("FEMX_EXPECTED_LOCAL_DEVICE_COUNT", jax.local_device_count())
    return jax


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _array_finiteness(jax: Any, value: object) -> dict[str, object]:
    import jax.numpy as jnp
    import numpy as np

    array = jnp.asarray(value)
    counts = np.asarray(
        jax.device_get(
            jnp.stack(
                (
                    jnp.count_nonzero(jnp.isfinite(array)),
                    jnp.count_nonzero(jnp.isnan(array)),
                    jnp.count_nonzero(jnp.isinf(array)),
                )
            )
        )
    )
    finite_count, nan_count, inf_count = (int(value) for value in counts)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_finite": finite_count == int(array.size),
    }


def _scalar_finiteness(value: float) -> dict[str, object]:
    converted = float(value)
    if math.isnan(converted):
        classification = "nan"
    elif math.isinf(converted):
        classification = "positive_infinity" if converted > 0.0 else "negative_infinity"
    else:
        return {"finite": True, "classification": "finite", "value": converted}
    return {"finite": False, "classification": classification, "value": None}


def _numerical_diagnostic(
    jax: Any,
    arrays: Mapping[str, object],
    scalars: Mapping[str, float],
) -> dict[str, object]:
    array_reports = {name: _array_finiteness(jax, value) for name, value in arrays.items()}
    scalar_reports = {name: _scalar_finiteness(value) for name, value in scalars.items()}
    nonfinite_names = sorted(
        [
            *(name for name, report in array_reports.items() if not report["all_finite"]),
            *(name for name, report in scalar_reports.items() if not report["finite"]),
        ]
    )
    return {
        "all_finite": not nonfinite_names,
        "nonfinite_names": nonfinite_names,
        "arrays": array_reports,
        "scalars": scalar_reports,
    }


def _json_nonfinite_paths(value: object, *, path: str = "$") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_json_nonfinite_paths(item, path=f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_json_nonfinite_paths(item, path=f"{path}[{index}]"))
        return paths
    return []


def _manifest_provenance(remote_run: Path) -> dict[str, object]:
    manifest_path = remote_run / ".phoxla" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        return {
            "run_id": manifest["run_id"],
            "profile": manifest["profile"],
            "source_digest": source["digest"],
            "source_commit": source["commit"],
            "config_digest": manifest["config"]["digest"],
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid deployed Phoxla manifest: {manifest_path}") from error


def _claim_worker_entry(remote_run: Path, provenance: dict[str, object]) -> dict[str, object]:
    if not remote_run.is_absolute() or not remote_run.is_dir() or remote_run.is_symlink():
        raise RuntimeError("PHOXLA_REMOTE_RUN_DIR must be an absolute non-symlink directory")
    process_index = _nonnegative_environment_integer("PHOXLA_PROCESS_INDEX")
    worker_index = _nonnegative_environment_integer("PHOXLA_GCLOUD_WORKER_INDEX")
    if process_index is None or worker_index is None:
        raise RuntimeError("Phoxla process and worker indexes are required before entry execution")
    run_id = os.environ.get("PHOXLA_RUN_ID")
    if run_id != provenance.get("run_id"):
        raise RuntimeError("PHOXLA_RUN_ID disagrees with the deployed manifest")
    claim_path = remote_run / "logs" / "femx-coupled-entry.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError("duplicate femx coupled entry refused for this immutable run") from error
    claim = {
        "schema_version": WORKER_ENTRY_CLAIM_SCHEMA,
        "run_id": run_id,
        "worker_index": worker_index,
        "process_index": process_index,
        "source_sha256": provenance.get("source_digest"),
        "config_sha256": provenance.get("config_digest"),
        "scope": "worker-local coupled electrothermal entry fence after distributed bootstrap",
    }
    _atomic_json(claim_path / "identity.json", claim)
    return claim


def _memory_report(compiled: Any, hbm_capacity_bytes: int) -> dict[str, object]:
    from femx.backends.jax.collective_runtime import CollectiveCompilerMemoryReport

    analysis = compiled.memory_analysis()
    if analysis is None:
        raise RuntimeError("JAX executable did not expose compiler memory analysis")

    def measured(name: str) -> int:
        value = getattr(analysis, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid JAX compiler memory statistic {name!r}: {value!r}")
        return value

    return CollectiveCompilerMemoryReport(
        generated_code_bytes=measured("generated_code_size_in_bytes"),
        argument_bytes=measured("argument_size_in_bytes"),
        output_bytes=measured("output_size_in_bytes"),
        alias_bytes=measured("alias_size_in_bytes"),
        temporary_bytes=measured("temp_size_in_bytes"),
        hbm_capacity_bytes_per_device=hbm_capacity_bytes,
    ).canonical_data()


def _compile_and_time(
    jax: Any,
    function: Any,
    arguments: tuple[Any, ...],
) -> tuple[Any, dict[str, object], str]:
    from femx.backends.jax.collective_runtime import CollectiveTimingReport

    started = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir("stablehlo"))
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warmup = compiled(*arguments)
    jax.block_until_ready(warmup)
    warmup_seconds = time.perf_counter() - started
    samples: list[float] = []
    for _ in range(EXECUTION_SAMPLES):
        started = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    report = CollectiveTimingReport(
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        warmup_seconds=warmup_seconds,
        execution_seconds=tuple(samples),
    )
    return compiled, report.canonical_data(), stablehlo


def _relative_difference(jax: Any, observed: Any, expected: Any) -> float:
    import jax.numpy as jnp
    import numpy as np

    numerator = jnp.linalg.norm(observed - expected)
    denominator = jnp.linalg.norm(expected)
    result = jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )
    return float(np.asarray(jax.device_get(result)))


def _host_pack_owned(layout: Any, full_node_values: Any) -> Any:
    import numpy as np

    full = np.asarray(full_node_values)
    free = full[np.asarray(layout.topology.free_nodes)]
    extended = np.concatenate((free, np.zeros((1,), dtype=free.dtype)))
    return np.ascontiguousarray(extended[layout.transport.owned_dof_ids])


def _stablehlo_report(text: str) -> dict[str, object]:
    lowered = text.lower()
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "collective_permute_count": lowered.count("stablehlo.collective_permute"),
        "all_reduce_count": lowered.count("stablehlo.all_reduce"),
        "contains_all_gather": "all_gather" in lowered,
    }


def _write_process_evidence(
    output_root: Path,
    remote_run: Path,
    process_index: int,
    process_payload: object,
) -> None:
    _atomic_json(output_root / "results" / "process-metrics.json", process_payload)
    if process_index == 0:
        _atomic_json(output_root / "results" / "metrics.json", process_payload)
        _atomic_json(remote_run / "results" / "metrics.json", process_payload)


def _write_numerical_diagnostic(
    output_root: Path,
    remote_run: Path,
    process_index: int,
    diagnostic_payload: object,
) -> None:
    _atomic_json(
        output_root / "results" / "numerical-diagnostic.json",
        diagnostic_payload,
    )
    if process_index == 0:
        _atomic_json(
            remote_run / "results" / "numerical-diagnostic.json",
            diagnostic_payload,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    launch_claim = _claim_worker_entry(remote_run, provenance)
    jax = _runtime()
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh

    from femx.backends.jax.collective_runtime import (
        describe_collective_mesh,
        make_collective_array_from_process_local_data,
        make_replicated_array_from_process_local_data,
    )
    from femx.backends.jax.distributed_electrothermal import (
        ElectrothermalAdjointPolicy,
        PackedDistributedElectrothermalInputs,
        PackedElectrothermalVector,
        build_distributed_electrothermal_runtime,
        pack_distributed_electrothermal_inputs_host,
    )
    from femx.backends.jax.scalar_cg import (
        ScalarH1CGPolicy,
        ScalarH1JacobiPolicy,
        build_packed_scalar_h1_jacobi_preconditioner_factory,
    )
    from scripts._tpu_distributed_electrothermal_plan import (
        read_distributed_electrothermal_artifact,
    )

    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    global_device_count = int(jax.device_count())
    local_device_count = int(jax.local_device_count())
    if cast(int, launch_claim["process_index"]) != process_index:
        raise RuntimeError("worker entry claim disagrees with initialized JAX process identity")
    input_root = (remote_run / arguments.input).resolve(strict=True)
    if remote_run.resolve() not in input_root.parents:
        raise RuntimeError("coupled electrothermal input must remain inside the deployed run")
    loaded = read_distributed_electrothermal_artifact(input_root)
    plan = loaded.plan
    authority = loaded.authority
    if loaded.manifest.get("source_commit") != provenance.get("source_commit"):
        raise RuntimeError("controller authority source commit disagrees with deployed source")
    if not authority.forward_converged or not authority.adjoint_converged:
        raise RuntimeError("controller authority did not converge before TPU admission")
    if plan.layout.partition_count != global_device_count:
        raise RuntimeError("immutable plan partition count must equal global TPU device count")

    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("partition",))
    mesh_report = describe_collective_mesh(
        plan.layout.transport,
        mesh,
        layout_sha256=plan.layout.digest(),
    )
    host_inputs = pack_distributed_electrothermal_inputs_host(plan, value_dtype=np.float32)
    partitioned_reports: dict[str, object] = {}
    replicated_reports: dict[str, object] = {}

    def load_partitioned(name: str, value: Any) -> Any:
        array, report = make_collective_array_from_process_local_data(name, value, mesh)
        partitioned_reports[name] = report.canonical_data()
        return array

    def load_replicated(name: str, value: Any) -> Any:
        raw = np.asarray(value)
        transport = raw.reshape((1,)) if raw.ndim == 0 else raw
        array, report = make_replicated_array_from_process_local_data(
            name,
            transport,
            mesh,
            replication_intent=REPLICATION_INTENT,
        )
        replicated_reports[name] = report.canonical_data()
        return array[0] if raw.ndim == 0 else array

    input_values = []
    for name, value in zip(host_inputs._fields, host_inputs, strict=True):
        loader = load_partitioned if name in _PARTITIONED_INPUTS else load_replicated
        input_values.append(loader(f"input-{name.replace('_', '-')}", value))
    inputs = PackedDistributedElectrothermalInputs(*input_values)
    current = load_replicated("current-parameters", plan.current_initial.astype(np.float32))
    thermal = load_replicated("thermal-parameters", plan.thermal_initial.astype(np.float32))
    feedback = load_replicated("feedback-parameters", plan.feedback_initial.astype(np.float32))
    owner_weights = load_partitioned(
        "temperature-cotangent",
        _host_pack_owned(plan.layout, authority.temperature_cotangent).astype(np.float32),
    )
    expected_potential = load_partitioned(
        "authority-potential",
        _host_pack_owned(plan.layout, authority.potential).astype(np.float32),
    )
    expected_temperature = load_partitioned(
        "authority-temperature",
        _host_pack_owned(plan.layout, authority.temperature).astype(np.float32),
    )
    expected_gradients = (
        load_replicated(
            "authority-current-gradient",
            authority.current_parameter_gradient.astype(np.float32),
        ),
        load_replicated(
            "authority-thermal-gradient",
            authority.thermal_parameter_gradient.astype(np.float32),
        ),
        load_replicated(
            "authority-feedback-gradient",
            authority.feedback_parameter_gradient.astype(np.float32),
        ),
    )

    jacobi_policy = ScalarH1JacobiPolicy(JACOBI_MINIMUM_RELATIVE_DIAGONAL)
    linear_preconditioner_factory = build_packed_scalar_h1_jacobi_preconditioner_factory(
        plan.layout,
        mesh,
        jacobi_policy,
    )
    runtime = build_distributed_electrothermal_runtime(
        plan,
        mesh,
        ScalarH1CGPolicy(
            CG_RELATIVE_TOLERANCE,
            CG_ABSOLUTE_TOLERANCE,
            CG_MAX_ITERATIONS,
            backward_error_tolerance=CG_BACKWARD_ERROR_TOLERANCE,
        ),
        ElectrothermalAdjointPolicy(
            ADJOINT_RELATIVE_TOLERANCE,
            ADJOINT_ABSOLUTE_TOLERANCE,
            ADJOINT_RESTART,
            ADJOINT_MAX_RESTARTS,
        ),
        linear_preconditioner_factory=linear_preconditioner_factory,
    )
    cotangent = PackedElectrothermalVector(jnp.zeros_like(owner_weights), owner_weights)

    def native_reverse(
        packed_inputs: PackedDistributedElectrothermalInputs,
        current_values: Any,
        thermal_values: Any,
        feedback_values: Any,
        weights: Any,
    ) -> Any:
        def objective(
            current_candidate: Any,
            thermal_candidate: Any,
            feedback_candidate: Any,
        ) -> Any:
            state = runtime.state(
                packed_inputs,
                current_candidate,
                thermal_candidate,
                feedback_candidate,
            )
            reference = packed_inputs.thermal_reference_base + jnp.vdot(
                packed_inputs.thermal_reference_weights,
                thermal_candidate,
            )
            return jnp.sum((state.temperature - reference) * weights)

        return jax.value_and_grad(objective, argnums=(0, 1, 2))(
            current_values,
            thermal_values,
            feedback_values,
        )

    forward_arguments = (inputs, current, thermal, feedback)
    compiled_forward, forward_timing, forward_hlo = _compile_and_time(
        jax,
        runtime.solve,
        forward_arguments,
    )
    explicit_arguments = (*forward_arguments, cotangent)
    compiled_explicit, explicit_timing, explicit_hlo = _compile_and_time(
        jax,
        runtime.vjp,
        explicit_arguments,
    )
    native_arguments = (*forward_arguments, owner_weights)
    compiled_native, native_timing, native_hlo = _compile_and_time(
        jax,
        native_reverse,
        native_arguments,
    )
    forward = compiled_forward(*forward_arguments)
    explicit = compiled_explicit(*explicit_arguments)
    objective, native_gradients = compiled_native(*native_arguments)
    jax.block_until_ready((forward, explicit, objective, native_gradients))

    potential_difference = _relative_difference(
        jax,
        forward.state.potential,
        expected_potential,
    )
    temperature_difference = _relative_difference(
        jax,
        forward.state.temperature,
        expected_temperature,
    )
    explicit_gradients = (
        explicit.current_parameter_gradient,
        explicit.thermal_parameter_gradient,
        explicit.feedback_parameter_gradient,
    )
    explicit_differences = [
        _relative_difference(jax, observed, expected)
        for observed, expected in zip(explicit_gradients, expected_gradients, strict=True)
    ]
    native_authority_differences = [
        _relative_difference(jax, observed, expected)
        for observed, expected in zip(native_gradients, expected_gradients, strict=True)
    ]
    native_explicit_differences = [
        _relative_difference(jax, observed, expected)
        for observed, expected in zip(native_gradients, explicit_gradients, strict=True)
    ]
    objective_value = float(np.asarray(jax.device_get(objective)))
    objective_difference = abs(objective_value - authority.objective) / max(
        abs(authority.objective),
        float(np.finfo(np.float32).tiny),
    )
    finite = bool(
        np.asarray(
            jax.device_get(
                jnp.all(
                    jnp.stack(
                        (
                            jnp.all(jnp.isfinite(forward.state.potential)),
                            jnp.all(jnp.isfinite(forward.state.temperature)),
                            jnp.all(jnp.isfinite(explicit.current_parameter_gradient)),
                            jnp.all(jnp.isfinite(explicit.thermal_parameter_gradient)),
                            jnp.all(jnp.isfinite(explicit.feedback_parameter_gradient)),
                            *(jnp.all(jnp.isfinite(value)) for value in native_gradients),
                        )
                    )
                )
            )
        )
    )
    hlo_by_name = {
        "forward": forward_hlo,
        "explicit_vjp": explicit_hlo,
        "native_reverse": native_hlo,
    }
    hlo_reports = {name: _stablehlo_report(value) for name, value in hlo_by_name.items()}
    hbm_capacity_bytes = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    if hbm_capacity_bytes is None:
        raise RuntimeError("FEMX_HBM_BYTES_PER_DEVICE is required for physical TPU evidence")
    executables = {
        "forward": {
            "timing": forward_timing,
            "memory": _memory_report(compiled_forward, hbm_capacity_bytes),
            "stablehlo": hlo_reports["forward"],
        },
        "explicit_vjp": {
            "timing": explicit_timing,
            "memory": _memory_report(compiled_explicit, hbm_capacity_bytes),
            "stablehlo": hlo_reports["explicit_vjp"],
        },
        "native_reverse": {
            "timing": native_timing,
            "memory": _memory_report(compiled_native, hbm_capacity_bytes),
            "stablehlo": hlo_reports["native_reverse"],
        },
    }
    collectives_valid = all(
        not bool(report["contains_all_gather"])
        and cast(int, report["collective_permute_count"]) > 0
        and cast(int, report["all_reduce_count"]) > 0
        for report in hlo_reports.values()
    )

    map_report = partitioned_reports["input-cell-local-dofs"]
    assert isinstance(map_report, dict)
    local_partition_mask = np.zeros((global_device_count,), dtype=np.int32)
    for shard in cast(list[dict[str, object]], map_report["addressable_shards"]):
        local_partition_mask[cast(int, shard["partition_index"])] = 1
    gathered_masks = np.asarray(
        multihost_utils.process_allgather(local_partition_mask, tiled=False)
    ).reshape(process_count, global_device_count)
    addressability_counts = np.sum(gathered_masks, axis=0)
    exact_addressability = bool(
        np.array_equal(addressability_counts, np.ones(global_device_count, dtype=np.int64))
    )
    current_residual = float(np.asarray(jax.device_get(forward.current_residual_error)))
    heat_residual = float(np.asarray(jax.device_get(forward.heat_residual_error)))
    transfer_error = float(np.asarray(jax.device_get(forward.transfer_relative_error)))
    adjoint_error = float(np.asarray(jax.device_get(explicit.adjoint_backward_error)))
    current_linear_backward_error = float(
        np.asarray(jax.device_get(forward.current_linear_backward_error))
    )
    heat_linear_backward_error = float(
        np.asarray(jax.device_get(forward.heat_linear_backward_error))
    )
    electrical_power = float(np.asarray(jax.device_get(forward.electrical_joule_power)))
    thermal_power = float(np.asarray(jax.device_get(forward.thermal_joule_load)))
    diagnostic_arrays = {
        "forward_potential": forward.state.potential,
        "forward_temperature": forward.state.temperature,
        "explicit_current_parameter_gradient": explicit.current_parameter_gradient,
        "explicit_thermal_parameter_gradient": explicit.thermal_parameter_gradient,
        "explicit_feedback_parameter_gradient": explicit.feedback_parameter_gradient,
        "native_current_parameter_gradient": native_gradients[0],
        "native_thermal_parameter_gradient": native_gradients[1],
        "native_feedback_parameter_gradient": native_gradients[2],
    }
    diagnostic_scalars = {
        "objective": objective_value,
        "objective_relative_difference": objective_difference,
        "current_residual_error": current_residual,
        "heat_residual_error": heat_residual,
        "adjoint_backward_error": adjoint_error,
        "current_linear_backward_error": current_linear_backward_error,
        "heat_linear_backward_error": heat_linear_backward_error,
        "electrical_joule_power_W_per_m": electrical_power,
        "thermal_joule_load_W_per_m": thermal_power,
        "transfer_relative_error": transfer_error,
        "potential_relative_difference": potential_difference,
        "temperature_relative_difference": temperature_difference,
        **{
            f"explicit_gradient_relative_difference_{index}": value
            for index, value in enumerate(explicit_differences)
        },
        **{
            f"native_gradient_authority_relative_difference_{index}": value
            for index, value in enumerate(native_authority_differences)
        },
        **{
            f"native_gradient_explicit_relative_difference_{index}": value
            for index, value in enumerate(native_explicit_differences)
        },
    }
    passed = (
        bool(np.asarray(jax.device_get(forward.converged)))
        and bool(np.asarray(jax.device_get(explicit.adjoint_converged)))
        and finite
        and exact_addressability
        and collectives_valid
        and potential_difference <= POTENTIAL_TOLERANCE
        and temperature_difference <= TEMPERATURE_TOLERANCE
        and max(explicit_differences) <= GRADIENT_TOLERANCE
        and max(native_authority_differences) <= GRADIENT_TOLERANCE
        and max(native_explicit_differences) <= NATIVE_GRADIENT_TOLERANCE
        and objective_difference <= OBJECTIVE_TOLERANCE
        and current_linear_backward_error <= CG_BACKWARD_ERROR_TOLERANCE
        and heat_linear_backward_error <= CG_BACKWARD_ERROR_TOLERANCE
        and current_residual <= plan.iteration_policy.residual_tolerance
        and heat_residual <= plan.iteration_policy.residual_tolerance
        and transfer_error <= TRANSFER_TOLERANCE
        and adjoint_error <= ADJOINT_RELATIVE_TOLERANCE
    )
    process_payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": package_version("jaxlib"),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "default_matmul_precision": str(
                getattr(jax.config, "jax_default_matmul_precision", None)
            ),
            "process_index": process_index,
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
            "real_scalar_contract": REAL_SCALAR_CONTRACT,
        },
        "launch_claim": launch_claim,
        "plan": {
            "schema_version": plan.schema_version,
            "sha256": plan.digest(),
            "layout_sha256": plan.layout.digest(),
            "arrays_sha256": loaded.arrays_sha256,
            "partition_count": plan.layout.partition_count,
            "node_count": plan.layout.topology.node_count,
            "triangle_count": plan.layout.topology.cell_count,
            "free_dof_count": plan.layout.topology.free_dof_count,
            "host_input_replication": (
                "bounded complete plan file exists on every worker; only addressable partition "
                "slices are transferred for partition-leading device arrays"
            ),
        },
        "mesh_report": mesh_report.canonical_data(),
        "addressability": {
            "process_local_partition_mask": local_partition_mask.tolist(),
            "partition_addressability_counts": addressability_counts.tolist(),
            "every_partition_addressable_once": exact_addressability,
        },
        "partitioned_array_reports": partitioned_reports,
        "replicated_array_reports": replicated_reports,
        "policies": {
            "coupled_iteration": dict(plan.iteration_policy.canonical_data()),
            "scalar_cg": {
                "relative_tolerance": CG_RELATIVE_TOLERANCE,
                "absolute_tolerance": CG_ABSOLUTE_TOLERANCE,
                "max_iterations": CG_MAX_ITERATIONS,
                "admission_metric": "componentwise_normwise_backward_error",
                "backward_error_tolerance": CG_BACKWARD_ERROR_TOLERANCE,
                "preconditioner": {
                    "name": "stopped_positive_diagonal_jacobi",
                    "minimum_relative_diagonal": JACOBI_MINIMUM_RELATIVE_DIAGONAL,
                },
            },
            "coupled_adjoint": {
                "relative_tolerance": ADJOINT_RELATIVE_TOLERANCE,
                "absolute_tolerance": ADJOINT_ABSOLUTE_TOLERANCE,
                "restart": ADJOINT_RESTART,
                "max_restarts": ADJOINT_MAX_RESTARTS,
                "preconditioner": "stopped uncoupled current/heat right block inverse",
                "preconditioning_side": "right",
            },
        },
        "numerics": {
            "finite": finite,
            "forward_converged": bool(np.asarray(jax.device_get(forward.converged))),
            "adjoint_converged": bool(np.asarray(jax.device_get(explicit.adjoint_converged))),
            "iterations": int(np.asarray(jax.device_get(forward.iterations))),
            "current_linear_iterations": int(
                np.asarray(jax.device_get(forward.current_linear_iterations))
            ),
            "heat_linear_iterations": int(
                np.asarray(jax.device_get(forward.heat_linear_iterations))
            ),
            "current_linear_recursive_residual": float(
                np.asarray(jax.device_get(forward.current_linear_recursive_residual))
            ),
            "heat_linear_recursive_residual": float(
                np.asarray(jax.device_get(forward.heat_linear_recursive_residual))
            ),
            "current_linear_recomputed_residual": float(
                np.asarray(jax.device_get(forward.current_linear_recomputed_residual))
            ),
            "heat_linear_recomputed_residual": float(
                np.asarray(jax.device_get(forward.heat_linear_recomputed_residual))
            ),
            "current_linear_relative_residual": float(
                np.asarray(jax.device_get(forward.current_linear_relative_residual))
            ),
            "heat_linear_relative_residual": float(
                np.asarray(jax.device_get(forward.heat_linear_relative_residual))
            ),
            "current_linear_backward_error": current_linear_backward_error,
            "heat_linear_backward_error": heat_linear_backward_error,
            "current_linear_converged": bool(
                np.asarray(jax.device_get(forward.current_linear_converged))
            ),
            "heat_linear_converged": bool(
                np.asarray(jax.device_get(forward.heat_linear_converged))
            ),
            "current_linear_breakdown": bool(
                np.asarray(jax.device_get(forward.current_linear_breakdown))
            ),
            "heat_linear_breakdown": bool(
                np.asarray(jax.device_get(forward.heat_linear_breakdown))
            ),
            "current_residual_error": current_residual,
            "heat_residual_error": heat_residual,
            "adjoint_backward_error": adjoint_error,
            "electrical_joule_power_W_per_m": electrical_power,
            "thermal_joule_load_W_per_m": thermal_power,
            "transfer_relative_error": transfer_error,
            "potential_relative_difference": potential_difference,
            "temperature_relative_difference": temperature_difference,
            "explicit_gradient_relative_differences": explicit_differences,
            "native_gradient_authority_relative_differences": native_authority_differences,
            "native_gradient_explicit_relative_differences": native_explicit_differences,
            "objective": objective_value,
            "authority_objective": authority.objective,
            "objective_relative_difference": objective_difference,
            "authority": (
                "controller-generated dense float64 same-discretization forward and coupled "
                "residual VJP from the immutable input artifact"
            ),
        },
        "tolerances": {
            "potential_relative_difference": POTENTIAL_TOLERANCE,
            "temperature_relative_difference": TEMPERATURE_TOLERANCE,
            "gradient_relative_difference": GRADIENT_TOLERANCE,
            "native_explicit_gradient_relative_difference": NATIVE_GRADIENT_TOLERANCE,
            "objective_relative_difference": OBJECTIVE_TOLERANCE,
            "transfer_relative_error": TRANSFER_TOLERANCE,
        },
        "executables": executables,
        "claim_scope": (
            "bounded process-local physical multi-host TPU current-to-Joule-to-heat forward, "
            "coupled residual adjoint, and native JAX reverse correctness witness against an "
            "immutable dense float64 authority; not Elmer re-execution, scaling, live HBM, "
            "3D production FEM, measured-device, foundry, FDTDX, or recovery evidence"
        ),
    }
    output_root = Path(os.environ["PHOXLA_OUTPUT_DIR"])
    for name, stablehlo in hlo_by_name.items():
        hlo_path = output_root / "hlo" / f"{name}.stablehlo.mlir"
        hlo_path.parent.mkdir(parents=True, exist_ok=True)
        hlo_path.write_text(stablehlo, encoding="utf-8")
    nonfinite_paths = _json_nonfinite_paths(process_payload)
    if nonfinite_paths or not finite:
        numerical_diagnostic = _numerical_diagnostic(
            jax,
            diagnostic_arrays,
            diagnostic_scalars,
        )
        diagnostic_payload = {
            "schema_version": NUMERICAL_DIAGNOSTIC_SCHEMA,
            "status": "failed",
            "provenance": provenance,
            "runtime": {
                "process_index": process_index,
                "process_count": process_count,
                "local_device_count": local_device_count,
                "global_device_count": global_device_count,
            },
            "plan_sha256": plan.digest(),
            "numerical": numerical_diagnostic,
            "json_nonfinite_paths": nonfinite_paths,
            "failure": "non-finite physical TPU coupled electrothermal result",
        }
        _write_numerical_diagnostic(
            output_root,
            remote_run,
            process_index,
            diagnostic_payload,
        )
        multihost_utils.sync_global_devices(f"femx-coupled-tpu-nonfinite-{provenance['run_id']}")
        print(
            json.dumps(
                {
                    "status": "failed",
                    "run_id": provenance["run_id"],
                    "nonfinite_names": numerical_diagnostic["nonfinite_names"],
                    "json_nonfinite_paths": nonfinite_paths,
                },
                sort_keys=True,
            )
        )
        return 2
    _write_process_evidence(output_root, remote_run, process_index, process_payload)
    multihost_utils.sync_global_devices(f"femx-coupled-tpu-written-{provenance['run_id']}")
    print(json.dumps({"status": process_payload["status"], "run_id": provenance["run_id"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
