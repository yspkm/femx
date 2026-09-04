"""Process-set admission for Elmer/JAX silicon-waveguide modes on physical TPU FDTDX.

This is deliberately stricter than the homogeneous infrastructure witness.  Every initialized
JAX process must consume both hash-bound FEM artifacts in the same Si/SiO2 scene, expose one
addressable source shard per local device, and report the same reduced complex-field comparison.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from femx.validation.tpu_fdtdx_mode_source_evidence import (
    _FDTDX_PACKAGE_VERSION,
    _FDTDX_SOURCE_DIGEST,
    _FDTDX_SOURCE_REVISION,
    _SCALAR_CONTRACT,
    _WORKER_ENTRY_CLAIM_SCHEMA,
    _binding,
    _boolean,
    _canonical_digest,
    _fail,
    _integer,
    _mapping,
    _number,
    _sequence,
    _sha256,
    _text,
)

PROCESS_EVIDENCE_SCHEMA = "femx.fdtdx.waveguide_source.tpu_process/v1"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.fdtdx_waveguide_source.tpu_process_set/v1"
INPUT_MANIFEST_SCHEMA = "femx.fdtdx.waveguide_source.inputs/v1"
SOLVERS = ("elmer", "jax")
GRID_SHAPE = (64, 52, 36)
SOURCE_Z_INDEX = 6
DETECTOR_Z_INDEX = 24
BOUNDARIES = ("pec", "pec", "pec", "pec", "pml", "pml")
MAXIMUM_SOURCE_RELATIVE_L2_ERROR = 2.0e-5
MAXIMUM_DOWNSTREAM_RELATIVE_L2_ERROR = 2.0e-5
_FDTDX_MODULE_SHA256 = {
    "fdtdx.core.grid": "d24739b9229ad8c61a57e4f688e6224eae63a680ff6554ddd7a5ef765edab6dd",
    "fdtdx.fdtd.wrapper": "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384",
    "fdtdx.objects.object": "24c986b9fa73bf474bce9fefc2145436654be4758e83dbcaf6fb955b7eb8557f",
    "fdtdx.objects.sources.custom_mode": (
        "0c5925a784da33f8d8236a874d4759d4ebe6df29317dcc1ce68877b4a4036df5"
    ),
    "fdtdx.objects.sources.tfsf": (
        "bd270995bffd174c7014adf9a02c7648134547c3bab7a294570e0a179326e611"
    ),
}


@dataclass(frozen=True, slots=True)
class _WaveguideProcess:
    process_index: int
    worker_index: int
    process_count: int
    local_device_count: int
    global_device_count: int
    digest: str
    common_identity: tuple[object, ...]
    binding_digests: tuple[str, str]
    ranges: tuple[tuple[str, tuple[int, int]], ...]
    lowering_seconds: float
    compilation_seconds: float
    warmup_seconds: tuple[float, float]
    execution_seconds: tuple[float, float]
    compiler_peak_bytes: int
    hbm_capacity_bytes: int
    all_gather_count: int


def _source(
    value: object,
    *,
    solver: str,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> tuple[tuple[object, ...], str, tuple[tuple[int, int], ...]]:
    source = _mapping(value, label=f"sources.{solver}")
    binding_record = _mapping(source.get("binding"), label=f"sources.{solver}.binding")
    binding_identity, ranges = _binding(
        binding_record,
        process_index=process_index,
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
    )
    digest = _sha256(source.get("binding_sha256"), label=f"sources.{solver}.binding_sha256")
    if digest != _canonical_digest(binding_record):
        _fail(f"has an inconsistent {solver} distributed binding digest")
    if binding_identity[0] != "femx-waveguide-port":
        _fail("requires the canonical shared waveguide source name")
    identity = (
        binding_identity,
        _sha256(
            source.get("canonical_bundle_sha256"),
            label=f"sources.{solver}.canonical_bundle_sha256",
        ),
        _sha256(
            source.get("runtime_bundle_sha256"),
            label=f"sources.{solver}.runtime_bundle_sha256",
        ),
        _sha256(
            source.get("precision_report_sha256"),
            label=f"sources.{solver}.precision_report_sha256",
        ),
    )
    return identity, digest, ranges


def _artifact(value: object, *, solver: str) -> tuple[object, ...]:
    artifact = _mapping(value, label=f"inputs.artifacts.{solver}")
    reference = _mapping(artifact.get("reference"), label=f"inputs.artifacts.{solver}.reference")
    path = _text(reference.get("path"), label=f"inputs.artifacts.{solver}.reference.path")
    if path != f"modes/{solver}-mode.h5":
        _fail(f"requires the canonical {solver} artifact path")
    return (
        path,
        _sha256(reference.get("sha256"), label=f"inputs.artifacts.{solver}.reference.sha256"),
        _sha256(
            artifact.get("content_sha256"),
            label=f"inputs.artifacts.{solver}.content_sha256",
        ),
        _sha256(artifact.get("bundle_sha256"), label=f"inputs.artifacts.{solver}.bundle_sha256"),
    )


def _process(record: Mapping[str, object]) -> _WaveguideProcess:
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA or record.get("status") != "passed":
        _fail("contains an unsupported or non-passing waveguide process record")
    provenance = _mapping(record.get("provenance"), label="provenance")
    runtime = _mapping(record.get("runtime"), label="runtime")
    claim = _mapping(record.get("launch_claim"), label="launch_claim")
    inputs = _mapping(record.get("inputs"), label="inputs")
    sources = _mapping(record.get("sources"), label="sources")
    simulation = _mapping(record.get("simulation"), label="simulation")
    numerics = _mapping(record.get("numerics"), label="numerics")
    execution = _mapping(record.get("execution"), label="execution")

    process_index = _integer(runtime.get("process_index"), label="runtime.process_index")
    process_count = _integer(
        runtime.get("process_count"), label="runtime.process_count", positive=True
    )
    local_device_count = _integer(
        runtime.get("local_device_count"), label="runtime.local_device_count", positive=True
    )
    global_device_count = _integer(
        runtime.get("global_device_count"), label="runtime.global_device_count", positive=True
    )
    device_kinds = tuple(
        _text(item, label="runtime.device_kinds[]")
        for item in _sequence(runtime.get("device_kinds"), label="runtime.device_kinds")
    )
    if (
        process_count < 2
        or process_index >= process_count
        or global_device_count != process_count * local_device_count
        or runtime.get("backend") != "tpu"
        or runtime.get("x64_enabled") is not False
        or not device_kinds
        or dict(_mapping(runtime.get("scalar_contract"), label="runtime.scalar_contract"))
        != _SCALAR_CONTRACT
    ):
        _fail("requires the exact physical multi-process TPU scalar contract")
    versions = (
        _text(runtime.get("jax_version"), label="runtime.jax_version"),
        _text(runtime.get("jaxlib_version"), label="runtime.jaxlib_version"),
        _text(runtime.get("fdtdx_version"), label="runtime.fdtdx_version"),
        device_kinds,
    )
    if versions[2] != _FDTDX_PACKAGE_VERSION:
        _fail("was not produced by the locked FDTDX package version")

    provenance_identity = (
        _text(provenance.get("run_id"), label="provenance.run_id"),
        _text(provenance.get("profile"), label="provenance.profile"),
        _sha256(provenance.get("source_digest"), label="provenance.source_digest"),
        _sha256(provenance.get("config_digest"), label="provenance.config_digest"),
    )
    if claim.get("schema_version") != _WORKER_ENTRY_CLAIM_SCHEMA:
        _fail("has an unsupported worker-entry claim schema")
    worker_index = _integer(claim.get("worker_index"), label="launch_claim.worker_index")
    if (
        worker_index >= process_count
        or claim.get("run_id") != provenance_identity[0]
        or claim.get("source_sha256") != provenance_identity[2]
        or claim.get("config_sha256") != provenance_identity[3]
        or _integer(claim.get("process_index"), label="launch_claim.process_index") != process_index
    ):
        _fail("has a worker-entry claim inconsistent with provenance")

    if inputs.get("schema_version") != INPUT_MANIFEST_SCHEMA:
        _fail("has an unsupported waveguide input manifest schema")
    input_manifest_sha256 = _sha256(inputs.get("manifest_sha256"), label="inputs.manifest_sha256")
    artifacts = _mapping(inputs.get("artifacts"), label="inputs.artifacts")
    artifact_identities = tuple(_artifact(artifacts.get(name), solver=name) for name in SOLVERS)
    source_identities: list[tuple[object, ...]] = []
    binding_digests: list[str] = []
    labeled_ranges: list[tuple[str, tuple[int, int]]] = []
    for solver in SOLVERS:
        identity, digest, ranges = _source(
            sources.get(solver),
            solver=solver,
            process_index=process_index,
            process_count=process_count,
            local_device_count=local_device_count,
            global_device_count=global_device_count,
        )
        source_identities.append(identity)
        binding_digests.append(digest)
        labeled_ranges.extend((solver, item) for item in ranges)
    for artifact, source in zip(artifact_identities, source_identities, strict=True):
        if artifact[3] != source[1]:
            _fail("has a canonical bundle identity inconsistent with its HDF5 input")

    fingerprint = _mapping(inputs.get("fdtdx_fingerprint"), label="inputs.fdtdx_fingerprint")
    if (
        fingerprint.get("package_version") != _FDTDX_PACKAGE_VERSION
        or fingerprint.get("source_revision") != _FDTDX_SOURCE_REVISION
        or fingerprint.get("source_digest") != _FDTDX_SOURCE_DIGEST
    ):
        _fail("does not match the locked FDTDX source fingerprint")
    runtime_module_hashes = dict(
        _mapping(inputs.get("runtime_module_sha256"), label="inputs.runtime_module_sha256")
    )
    if runtime_module_hashes != _FDTDX_MODULE_SHA256:
        _fail("does not match the locked FDTDX runtime module hashes")

    grid_shape = tuple(
        _integer(item, label="simulation.grid_shape_xyz[]", positive=True)
        for item in _sequence(simulation.get("grid_shape_xyz"), label="simulation.grid_shape_xyz")
    )
    boundaries = tuple(
        _text(item, label="simulation.boundaries[]")
        for item in _sequence(simulation.get("boundaries"), label="simulation.boundaries")
    )
    if (
        grid_shape != GRID_SHAPE
        or _integer(simulation.get("source_z_index"), label="simulation.source_z_index")
        != SOURCE_Z_INDEX
        or _integer(simulation.get("detector_z_index"), label="simulation.detector_z_index")
        != DETECTOR_Z_INDEX
        or boundaries != BOUNDARIES
    ):
        _fail("requires the canonical 64x52 Si/SiO2 PEC/PML waveguide scene")
    time_steps = _integer(
        simulation.get("time_steps"), label="simulation.time_steps", positive=True
    )
    simulation_identity = (
        grid_shape,
        _number(simulation.get("frequency_hz"), label="simulation.frequency_hz", positive=True),
        _number(
            simulation.get("simulation_time_s"), label="simulation.simulation_time_s", positive=True
        ),
        time_steps,
        boundaries,
        _integer(
            simulation.get("core_cell_count"), label="simulation.core_cell_count", positive=True
        ),
        _number(
            simulation.get("cladding_relative_permittivity"),
            label="simulation.cladding_relative_permittivity",
            positive=True,
        ),
        _number(
            simulation.get("core_relative_permittivity"),
            label="simulation.core_relative_permittivity",
            positive=True,
        ),
    )
    if simulation_identity[5] != 32:
        _fail("requires the exact 8x4 silicon core on the source plane")

    completed = _mapping(numerics.get("completed_step"), label="numerics.completed_step")
    finite = _mapping(numerics.get("all_fields_finite"), label="numerics.all_fields_finite")
    final_e = _mapping(numerics.get("final_e_l2"), label="numerics.final_e_l2")
    final_h = _mapping(numerics.get("final_h_l2"), label="numerics.final_h_l2")
    downstream = _mapping(
        numerics.get("downstream_phasor_l2"), label="numerics.downstream_phasor_l2"
    )
    for solver in SOLVERS:
        if _integer(completed.get(solver), label=f"numerics.completed_step.{solver}") != time_steps:
            _fail(f"did not complete the declared {solver} FDTD time steps")
        if not _boolean(finite.get(solver), label=f"numerics.all_fields_finite.{solver}"):
            _fail(f"contains non-finite {solver} final fields")
        _number(final_e.get(solver), label=f"numerics.final_e_l2.{solver}", positive=True)
        _number(final_h.get(solver), label=f"numerics.final_h_l2.{solver}", positive=True)
        _number(
            downstream.get(solver), label=f"numerics.downstream_phasor_l2.{solver}", positive=True
        )
    source_e_error = _number(
        numerics.get("source_electric_relative_l2"),
        label="numerics.source_electric_relative_l2",
    )
    source_h_error = _number(
        numerics.get("source_magnetic_relative_l2"),
        label="numerics.source_magnetic_relative_l2",
    )
    detector_error = _number(
        numerics.get("downstream_phasor_relative_l2"),
        label="numerics.downstream_phasor_relative_l2",
    )
    if max(source_e_error, source_h_error) > MAXIMUM_SOURCE_RELATIVE_L2_ERROR:
        _fail("exceeds the admitted float32 FEM source parity threshold")
    if detector_error > MAXIMUM_DOWNSTREAM_RELATIVE_L2_ERROR:
        _fail("exceeds the admitted downstream complex-field parity threshold")
    numerical_identity = (
        tuple((solver, completed[solver]) for solver in SOLVERS),
        tuple((solver, final_e[solver]) for solver in SOLVERS),
        tuple((solver, final_h[solver]) for solver in SOLVERS),
        tuple((solver, downstream[solver]) for solver in SOLVERS),
        source_e_error,
        source_h_error,
        detector_error,
    )

    memory = _mapping(execution.get("compiler_memory"), label="execution.compiler_memory")
    if not _boolean(
        execution.get("shared_compiled_pytree"), label="execution.shared_compiled_pytree"
    ):
        _fail("did not use one shared static FDTDX pytree for both runtime mode sources")
    compiler_peak = _integer(
        memory.get("compiler_peak_bytes"), label="execution.compiler_memory.compiler_peak_bytes"
    )
    hbm_capacity = _integer(
        memory.get("hbm_capacity_bytes_per_device"),
        label="execution.compiler_memory.hbm_capacity_bytes_per_device",
        positive=True,
    )
    if compiler_peak > hbm_capacity:
        _fail("reports a compiler peak larger than the admitted per-device HBM capacity")
    warmup = _mapping(execution.get("warmup_seconds"), label="execution.warmup_seconds")
    run_time = _mapping(execution.get("execution_seconds"), label="execution.execution_seconds")
    common_identity = (
        provenance_identity,
        versions,
        input_manifest_sha256,
        artifact_identities,
        tuple(source_identities),
        tuple(sorted(runtime_module_hashes.items())),
        simulation_identity,
        numerical_identity,
    )
    return _WaveguideProcess(
        process_index=process_index,
        worker_index=worker_index,
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        digest=_canonical_digest(record),
        common_identity=common_identity,
        binding_digests=cast(tuple[str, str], tuple(binding_digests)),
        ranges=tuple(labeled_ranges),
        lowering_seconds=_number(
            execution.get("lowering_seconds"), label="execution.lowering_seconds"
        ),
        compilation_seconds=_number(
            execution.get("compilation_seconds"), label="execution.compilation_seconds"
        ),
        warmup_seconds=cast(
            tuple[float, float],
            tuple(
                _number(warmup.get(solver), label=f"execution.warmup_seconds.{solver}")
                for solver in SOLVERS
            ),
        ),
        execution_seconds=cast(
            tuple[float, float],
            tuple(
                _number(run_time.get(solver), label=f"execution.execution_seconds.{solver}")
                for solver in SOLVERS
            ),
        ),
        compiler_peak_bytes=compiler_peak,
        hbm_capacity_bytes=hbm_capacity,
        all_gather_count=_integer(
            execution.get("stablehlo_all_gather_count"),
            label="execution.stablehlo_all_gather_count",
        ),
    )


def _range(values: Sequence[float]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


@dataclass(frozen=True, slots=True)
class TpuFdtdxWaveguideSourceProcessSetEvidence:
    """Canonical summary of a complete physical Elmer/JAX-to-FDTDX process set."""

    payload: Mapping[str, object]

    def canonical_data(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.canonical_json()))

    def canonical_json(self) -> str:
        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def aggregate_tpu_fdtdx_waveguide_source_process_evidence(
    records: Sequence[Mapping[str, object]],
) -> TpuFdtdxWaveguideSourceProcessSetEvidence:
    """Admit exactly one physical waveguide record per initialized JAX process."""

    if not records:
        _fail("requires at least one waveguide process record")
    ordered = tuple(
        sorted(
            (_process(_mapping(record, label="waveguide process record")) for record in records),
            key=lambda item: item.process_index,
        )
    )
    baseline = ordered[0]
    if len(ordered) != baseline.process_count or tuple(
        item.process_index for item in ordered
    ) != tuple(range(baseline.process_count)):
        _fail("requires exactly one waveguide record for every declared process index")
    if tuple(sorted(item.worker_index for item in ordered)) != tuple(range(baseline.process_count)):
        _fail("requires exactly one immutable worker-entry claim per TPU worker")
    if any(item.common_identity != baseline.common_identity for item in ordered[1:]):
        _fail("has inconsistent per-process waveguide identities or reduced numerics")

    combined: dict[str, list[tuple[int, int]]] = {solver: [] for solver in SOLVERS}
    for item in ordered:
        for solver, interval in item.ranges:
            combined[solver].append(interval)
    global_x = GRID_SHAPE[0]
    for solver, ranges in combined.items():
        canonical = sorted(ranges)
        shard_width = global_x // baseline.global_device_count
        expected = [
            (index * shard_width, (index + 1) * shard_width)
            for index in range(baseline.global_device_count)
        ]
        if canonical != expected:
            _fail(f"does not cover the {solver} source x extent exactly once per TPU device")

    provenance, versions, manifest_sha, artifacts, sources, module_hashes, simulation, numerics = (
        baseline.common_identity
    )
    run_id, profile, source_digest, config_digest = cast(tuple[object, ...], provenance)
    jax_version, jaxlib_version, fdtdx_version, device_kinds = cast(tuple[object, ...], versions)
    artifact_values = cast(tuple[tuple[object, ...], tuple[object, ...]], artifacts)
    source_values = cast(tuple[tuple[object, ...], tuple[object, ...]], sources)
    simulation_values = cast(tuple[object, ...], simulation)
    numerical_values = cast(tuple[object, ...], numerics)
    grid_shape = cast(tuple[int, int, int], simulation_values[0])
    payload: dict[str, object] = {
        "schema_version": PROCESS_SET_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": run_id,
            "profile": profile,
            "source_digest": source_digest,
            "config_digest": config_digest,
            "input_manifest_sha256": manifest_sha,
            "process_records": [
                {"process_index": item.process_index, "sha256": item.digest} for item in ordered
            ],
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": jax_version,
            "jaxlib_version": jaxlib_version,
            "fdtdx_version": fdtdx_version,
            "x64_enabled": False,
            "process_count": baseline.process_count,
            "local_device_count": baseline.local_device_count,
            "global_device_count": baseline.global_device_count,
            "device_kinds": list(cast(tuple[str, ...], device_kinds)),
            "scalar_contract": _SCALAR_CONTRACT,
        },
        "inputs": {
            "artifacts": {
                solver: {
                    "path": artifact_values[index][0],
                    "file_sha256": artifact_values[index][1],
                    "content_sha256": artifact_values[index][2],
                    "bundle_sha256": artifact_values[index][3],
                }
                for index, solver in enumerate(SOLVERS)
            },
            "fdtdx_fingerprint": {
                "package_version": _FDTDX_PACKAGE_VERSION,
                "source_revision": _FDTDX_SOURCE_REVISION,
                "source_digest": _FDTDX_SOURCE_DIGEST,
                "runtime_module_sha256": dict(cast(tuple[tuple[str, str], ...], module_hashes)),
            },
        },
        "sources": {
            solver: {
                "canonical_bundle_sha256": source_values[index][1],
                "runtime_bundle_sha256": source_values[index][2],
                "precision_report_sha256": source_values[index][3],
                "combined_addressable_x_ranges": [list(item) for item in sorted(combined[solver])],
                "every_global_source_shard_addressable_once": True,
                "process_bindings": [
                    {"process_index": item.process_index, "sha256": item.binding_digests[index]}
                    for item in ordered
                ],
            }
            for index, solver in enumerate(SOLVERS)
        },
        "simulation": {
            "grid_shape_xyz": list(grid_shape),
            "frequency_hz": simulation_values[1],
            "simulation_time_s": simulation_values[2],
            "time_steps": simulation_values[3],
            "source_z_index": SOURCE_Z_INDEX,
            "detector_z_index": DETECTOR_Z_INDEX,
            "boundaries": list(BOUNDARIES),
            "core_cell_count": simulation_values[5],
        },
        "numerics": {
            "all_processes_completed_same_step": True,
            "all_process_fields_finite": True,
            "source_electric_relative_l2": numerical_values[4],
            "source_magnetic_relative_l2": numerical_values[5],
            "downstream_phasor_relative_l2": numerical_values[6],
            "thresholds": {
                "maximum_source_relative_l2": MAXIMUM_SOURCE_RELATIVE_L2_ERROR,
                "maximum_downstream_relative_l2": MAXIMUM_DOWNSTREAM_RELATIVE_L2_ERROR,
            },
        },
        "execution": {
            "shared_compiled_pytree": True,
            "lowering_seconds_across_processes": _range(
                tuple(item.lowering_seconds for item in ordered)
            ),
            "compilation_seconds_across_processes": _range(
                tuple(item.compilation_seconds for item in ordered)
            ),
            "warmup_seconds_across_processes": {
                solver: _range(tuple(item.warmup_seconds[index] for item in ordered))
                for index, solver in enumerate(SOLVERS)
            },
            "execution_seconds_across_processes": {
                solver: _range(tuple(item.execution_seconds[index] for item in ordered))
                for index, solver in enumerate(SOLVERS)
            },
            "maximum_compiler_peak_bytes": max(item.compiler_peak_bytes for item in ordered),
            "hbm_capacity_bytes_per_device": min(item.hbm_capacity_bytes for item in ordered),
            "stablehlo_all_gather_counts": [item.all_gather_count for item in ordered],
        },
        "claim_scope": (
            "process-complete physical multi-host TPU float32/complex64 execution of independently "
            "generated Elmer and JAX same-mesh silicon-waveguide modes through one locked FDTDX "
            "Si/SiO2 time-domain scene, with reduced downstream complex-field parity; this is not "
            "spatial or temporal convergence, S-parameters, eigen-adjoint, performance scaling, "
            "fabricated-device agreement, or Spot-preemption recovery"
        ),
    }
    return TpuFdtdxWaveguideSourceProcessSetEvidence(payload)


__all__ = [
    "INPUT_MANIFEST_SCHEMA",
    "PROCESS_EVIDENCE_SCHEMA",
    "PROCESS_SET_EVIDENCE_SCHEMA",
    "TpuFdtdxWaveguideSourceProcessSetEvidence",
    "aggregate_tpu_fdtdx_waveguide_source_process_evidence",
]
