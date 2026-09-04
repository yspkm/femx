"""Four-device CPU probe for distinct-space Tet4 current/Joule/heat and VJP."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.scalar_cg import ScalarH1CGPolicy  # noqa: E402
from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    Tet4ElectrothermalAdmissionPolicy,
    Tet4ElectrothermalParameters,
    build_tet4_electrothermal_runtime,
    pack_tet4_electrothermal_inputs,
    reconstruct_tet4_electrothermal_state,
)
from tests.unit.test_tet4_electrothermal import _plan  # noqa: E402


def _parameters(voltage: jax.Array) -> Tet4ElectrothermalParameters:
    one = jnp.asarray(1.0, dtype=jnp.float64)
    return Tet4ElectrothermalParameters(voltage, one, one)


def _relative_difference(observed: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.linalg.norm(observed - expected))
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return numerator / denominator


def _runtime(plan, mesh: Mesh):
    policy = ScalarH1CGPolicy(
        2.0e-12,
        1.0e-14,
        600,
        backward_error_tolerance=2.0e-12,
    )
    admission = Tet4ElectrothermalAdmissionPolicy(2.0e-10, 2.0e-10, 2.0e-14, 2.0e-10)
    return build_tet4_electrothermal_runtime(plan, mesh, policy, policy, admission)


def main() -> int:
    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("Tet4 electrothermal probe requires exactly four forced CPU devices")

    parameters = _parameters(jnp.asarray(0.8, dtype=jnp.float64))
    results: dict[int, tuple[object, np.ndarray, np.ndarray]] = {}
    compiled = None
    four_inputs = None
    four_plan = None
    four_runtime = None
    partition_reports: dict[str, object] = {}
    for partition_count in (1, 2, 4):
        plan = _plan(
            intervals=4,
            partition_count=partition_count,
            embedded_current=True,
        )
        inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
        mesh = Mesh(np.asarray(devices[:partition_count], dtype=object), ("partition",))
        runtime = _runtime(plan, mesh)
        compiled_solve = jax.jit(runtime.solve)
        result = compiled_solve(inputs, parameters)
        result.state.temperature_rise.block_until_ready()
        potential, temperature = reconstruct_tet4_electrothermal_state(
            plan,
            result.state,
            parameters,
        )
        if not bool(result.numerically_admitted):
            raise RuntimeError(f"partition count {partition_count} failed numerical admission")
        results[partition_count] = (result, np.asarray(potential), np.asarray(temperature))
        partition_reports[str(partition_count)] = {
            "current_iterations": int(result.current_linear.iterations),
            "thermal_iterations": int(result.thermal_linear.iterations),
            "current_backward_error": float(result.current_linear.backward_error),
            "thermal_backward_error": float(result.thermal_linear.backward_error),
            "charge_balance_relative_error": float(result.charge_balance_relative_error),
            "electrical_energy_relative_error": float(result.electrical_energy_relative_error),
            "joule_transfer_relative_error": float(result.joule_transfer_relative_error),
            "thermal_balance_relative_error": float(result.thermal_balance_relative_error),
            "current_layout_sha256": plan.current_layout.digest(),
            "thermal_layout_sha256": plan.thermal_layout.digest(),
            "plan_sha256": plan.digest(),
        }
        if partition_count == 4:
            compiled = compiled_solve
            four_inputs = inputs
            four_plan = plan
            four_runtime = runtime

    if compiled is None or four_inputs is None or four_plan is None or four_runtime is None:
        raise RuntimeError("four-device Tet4 electrothermal witness was not constructed")
    _reference_result, reference_potential, reference_temperature = results[1]
    maximum_potential_difference = 0.0
    maximum_temperature_difference = 0.0
    for partition_count in (2, 4):
        _result, potential, temperature = results[partition_count]
        maximum_potential_difference = max(
            maximum_potential_difference,
            _relative_difference(potential, reference_potential),
        )
        maximum_temperature_difference = max(
            maximum_temperature_difference,
            _relative_difference(temperature, reference_temperature),
        )

    owner_weights = jnp.where(
        four_inputs.thermal_owner_mask,
        jnp.asarray(
            1.0 / four_plan.thermal_layout.topology.free_dof_count,
            dtype=jnp.float64,
        ),
        0.0,
    )

    def objective(voltage: jax.Array) -> jax.Array:
        result = four_runtime.solve(four_inputs, _parameters(voltage))
        return jnp.sum(result.state.temperature_rise * owner_weights)

    value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.asarray(0.8, dtype=jnp.float64))
    step = 2.0e-5
    finite_difference = (
        objective(jnp.asarray(0.8 + step, dtype=jnp.float64))
        - objective(jnp.asarray(0.8 - step, dtype=jnp.float64))
    ) / (2.0 * step)
    gradient_relative_error = abs(float(derivative - finite_difference)) / max(
        abs(float(finite_difference)),
        np.finfo(np.float64).tiny,
    )
    stablehlo = str(compiled.lower(four_inputs, parameters).compiler_ir("stablehlo")).lower()
    payload = {
        "schema_version": "femx.jax.tet4_electrothermal.cpu_portability/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "global_device_count": jax.device_count(),
        "thermal_node_count": four_plan.thermal_layout.topology.node_count,
        "thermal_tet4_cell_count": four_plan.thermal_layout.topology.cell_count,
        "current_node_count": four_plan.current_layout.topology.node_count,
        "current_tet4_cell_count": four_plan.current_layout.topology.cell_count,
        "partition_reports": partition_reports,
        "maximum_potential_relative_difference": maximum_potential_difference,
        "maximum_temperature_relative_difference": maximum_temperature_difference,
        "objective": float(value),
        "gradient": float(derivative),
        "gradient_finite_difference": float(finite_difference),
        "gradient_relative_error": gradient_relative_error,
        "stablehlo_collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
        "stablehlo_all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
        "stablehlo_contains_all_gather": "all_gather" in stablehlo,
        "claim_scope": (
            "forced single-process four-CPU-device Tet4 current/Joule/heat/VJP portability with "
            "an exact parent-cell transfer; not TPU, multi-host, public-ring, or Elmer evidence"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
