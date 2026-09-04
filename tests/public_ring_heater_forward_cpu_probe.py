from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version as package_version
from pathlib import Path

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("public ring-heater CPU probe requires explicit JAX_PLATFORMS=cpu")

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.applications import (  # noqa: E402
    PublicRingHeaterReferenceParameters,
    calibrate_public_ring_heater_current,
    prepare_public_ring_heater_forward_plan,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    ScalarH1JacobiPolicy,
    build_packed_scalar_h1_jacobi_preconditioner_factory,
)
from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    Tet4ElectrothermalAdmissionPolicy,
    Tet4ElectrothermalParameters,
    build_tet4_electrothermal_runtime,
    pack_tet4_electrothermal_inputs,
    reconstruct_tet4_electrothermal_state,
)
from femx.meshing.gmsh import (  # noqa: E402
    PublicRingHeater3D,
    read_gmsh_msh_3d,
    ring_heater_mesh_profile,
)


def _field_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _scalar(value: object) -> float:
    return float(np.asarray(jax.device_get(value)))


def _integer(value: object) -> int:
    return int(np.asarray(jax.device_get(value)))


def _atomic_json(path: Path, value: object) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise RuntimeError("public ring-heater record output must be a new absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_state(path: Path, potential: np.ndarray, temperature: np.ndarray) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise RuntimeError("public ring-heater state output must be a new absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("xb") as stream:
        np.savez(stream, potential=potential, temperature=temperature)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _region_temperature(
    temperature: np.ndarray,
    cells: np.ndarray,
    volumes: np.ndarray,
    cell_ids: np.ndarray,
) -> tuple[float, float, float]:
    nodal = temperature[cells[cell_ids]]
    cell_mean = np.mean(nodal, axis=1)
    weights = volumes[cell_ids]
    volume_mean = float(np.sum(cell_mean * weights) / np.sum(weights))
    return volume_mean, float(np.min(nodal)), float(np.max(nodal))


def run(
    mesh_path: Path,
    profile_name: str,
    *,
    state_output: Path | None = None,
) -> dict[str, object]:
    """Run one single-device float64 public-ring forward witness."""

    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 1:
        raise RuntimeError("public ring-heater CPU probe requires exactly one CPU device")
    if not bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("public ring-heater CPU probe requires JAX float64")

    recipe = PublicRingHeater3D(ring_heater_mesh_profile(profile_name))
    imported = read_gmsh_msh_3d(mesh_path, coordinate_scale_to_m=recipe.coordinate_scale_to_m)
    cell_count = imported.mesh.topology.cell_count
    prepare_started = time.perf_counter()
    forward = prepare_public_ring_heater_forward_plan(
        imported,
        recipe,
        np.zeros((cell_count,), dtype=np.int64),
        partition_count=1,
    )
    prepare_seconds = time.perf_counter() - prepare_started

    jax_mesh = Mesh(np.asarray((devices[0],), dtype=object), ("partition",))
    jacobi = ScalarH1JacobiPolicy(1.0e-15)
    current_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        forward.tet4.current_layout,
        jax_mesh,
        jacobi,
    )
    thermal_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        forward.tet4.thermal_layout,
        jax_mesh,
        jacobi,
    )
    cg = ScalarH1CGPolicy(
        relative_tolerance=1.0e-10,
        absolute_tolerance=0.0,
        max_iterations=10_000,
        backward_error_tolerance=1.0e-9,
    )
    runtime = build_tet4_electrothermal_runtime(
        forward.tet4,
        jax_mesh,
        cg,
        cg,
        Tet4ElectrothermalAdmissionPolicy(1.0e-7, 1.0e-7, 1.0e-12, 1.0e-7),
        current_preconditioner_factory=current_preconditioner,
        thermal_preconditioner_factory=thermal_preconditioner,
    )
    inputs = pack_tet4_electrothermal_inputs(forward.tet4, value_dtype=np.float64)
    solve = jax.jit(runtime.solve)

    unit_parameters = Tet4ElectrothermalParameters(
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    unit_started = time.perf_counter()
    unit = solve(inputs, unit_parameters)
    unit.numerically_admitted.block_until_ready()
    unit_seconds_including_compile = time.perf_counter() - unit_started
    if not bool(np.asarray(jax.device_get(unit.numerically_admitted))):
        raise RuntimeError("unit-voltage public ring-heater solve was not numerically admitted")

    reference = PublicRingHeaterReferenceParameters()
    calibration = calibrate_public_ring_heater_current(
        _scalar(unit.electrical_joule_power),
        reference=reference,
    )
    target_parameters = Tet4ElectrothermalParameters(
        jnp.asarray(calibration.target_voltage_v, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    target_started = time.perf_counter()
    target = solve(inputs, target_parameters)
    target.numerically_admitted.block_until_ready()
    target_seconds = time.perf_counter() - target_started
    if not bool(np.asarray(jax.device_get(target.numerically_admitted))):
        raise RuntimeError("target-current public ring-heater solve was not numerically admitted")

    potential_device, temperature_device = reconstruct_tet4_electrothermal_state(
        forward.tet4,
        target.state,
        target_parameters,
    )
    potential, temperature = (
        np.asarray(jax.device_get(potential_device), dtype=np.float64),
        np.asarray(jax.device_get(temperature_device), dtype=np.float64),
    )
    if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(temperature)):
        raise RuntimeError("public ring-heater reconstructed fields must be finite")
    if float(np.min(temperature)) < reference.ambient_temperature_k - 1.0e-7:
        raise RuntimeError("public ring-heater temperature fell below the passive ambient bound")
    if state_output is not None:
        _atomic_state(state_output, potential, temperature)

    joule_power = _scalar(target.electrical_joule_power)
    inferred_current = joule_power / calibration.target_voltage_v
    current_error = abs(inferred_current - reference.target_current_a) / reference.target_current_a
    power_error = abs(joule_power - calibration.predicted_joule_power_w) / joule_power
    if current_error > 1.0e-10 or power_error > 1.0e-10:
        raise RuntimeError("public ring-heater target-current scaling identity failed")

    cells = np.asarray(imported.mesh.topology.connectivity, dtype=np.int64)
    volumes = np.asarray(forward.tet4.thermal_cell_volumes, dtype=np.float64)
    region_temperatures: dict[str, object] = {}
    for name in ("silicon_ring", "tin_heater", "al_contact_negative", "al_contact_positive"):
        cell_ids = np.asarray(imported.mesh.tag(name).entity_ids, dtype=np.int64)
        volume_mean, minimum, maximum = _region_temperature(
            temperature,
            cells,
            volumes,
            cell_ids,
        )
        region_temperatures[name] = {
            "volume_weighted_cell_mean_K": volume_mean,
            "minimum_nodal_K": minimum,
            "maximum_nodal_K": maximum,
        }

    return {
        "schema_version": "femx.public-ring-heater-forward.cpu-witness/v1",
        "status": "passed",
        "profile": profile_name,
        "runtime": {
            "backend": jax.default_backend(),
            "device_kind": devices[0].device_kind,
            "device_count": len(devices),
            "jax_version": jax.__version__,
            "jaxlib_version": package_version("jaxlib"),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "partition_count": 1,
        },
        "provenance": {
            "source_msh_sha256": imported.record.source_sha256,
            "import_record_sha256": imported.record.digest(),
            "canonical_mesh_sha256": imported.record.canonical_mesh_sha256,
            "mesh_report_sha256": forward.mesh_report.digest(),
            "reference_sha256": reference.digest(),
            "tet4_plan_sha256": forward.tet4.digest(),
            "forward_plan_sha256": forward.digest(),
        },
        "mesh": {
            "node_count": forward.mesh_report.node_count,
            "tetrahedron_count": forward.mesh_report.tetrahedron_count,
            "conductor_node_count": forward.tet4.current_layout.topology.node_count,
            "conductor_tetrahedron_count": forward.tet4.current_layout.topology.cell_count,
            "minimum_mean_ratio": forward.mesh_report.minimum_mean_ratio,
            "maximum_region_volume_relative_error": (
                forward.mesh_report.maximum_region_volume_relative_error
            ),
        },
        "excitation": calibration.canonical_data(),
        "numerics": {
            "unit_voltage_current_iterations": _integer(unit.current_linear.iterations),
            "unit_voltage_thermal_iterations": _integer(unit.thermal_linear.iterations),
            "target_current_iterations": _integer(target.current_linear.iterations),
            "target_thermal_iterations": _integer(target.thermal_linear.iterations),
            "current_backward_error": _scalar(target.current_linear.backward_error),
            "thermal_backward_error": _scalar(target.thermal_linear.backward_error),
            "charge_balance_relative_error": _scalar(target.charge_balance_relative_error),
            "electrical_energy_relative_error": _scalar(target.electrical_energy_relative_error),
            "joule_transfer_relative_error": _scalar(target.joule_transfer_relative_error),
            "thermal_balance_relative_error": _scalar(target.thermal_balance_relative_error),
            "inferred_current_A": inferred_current,
            "target_current_relative_error": current_error,
            "target_power_relative_error": power_error,
            "electrical_joule_power_W": joule_power,
            "convection_outward_power_W": _scalar(target.convection_outward_power),
            "bottom_outward_power_W": _scalar(target.dirichlet_outward_power),
            "minimum_temperature_K": float(np.min(temperature)),
            "maximum_temperature_K": float(np.max(temperature)),
            "region_temperature": region_temperatures,
            "potential_sha256_float64": _field_sha256(potential),
            "temperature_sha256_float64": _field_sha256(temperature),
        },
        "timing": {
            "host_prepare_seconds": prepare_seconds,
            "unit_solve_seconds_including_compile": unit_seconds_including_compile,
            "target_solve_seconds": target_seconds,
        },
        "claim_scope": (
            "single-device CPU float64 forward solution of the source-pinned public 3D ring "
            "current/Joule/heat model; not Elmer parity, formal mesh convergence, TPU, FDTDX, "
            "foundry calibration, or fabricated-device validation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("coarse", "medium", "fine"))
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--record-output", type=Path)
    arguments = parser.parse_args()
    if (arguments.state_output is None) != (arguments.record_output is None):
        parser.error("--state-output and --record-output must be provided together")
    state_output = None if arguments.state_output is None else arguments.state_output.absolute()
    payload = run(
        arguments.mesh.resolve(),
        arguments.profile,
        state_output=state_output,
    )
    if arguments.record_output is not None:
        _atomic_json(arguments.record_output.absolute(), payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
