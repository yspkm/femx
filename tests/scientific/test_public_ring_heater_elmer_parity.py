from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.applications import (  # noqa: E402
    calibrate_public_ring_heater_current,
    prepare_public_ring_heater_elmer_plan,
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
from femx.core.execution import ExecutionPolicy  # noqa: E402
from femx.meshing.gmsh import (  # noqa: E402
    GmshMeshingRequest,
    PublicRingHeater3D,
    read_gmsh_msh_3d,
    ring_heater_mesh_profile,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.slow,
    pytest.mark.requires_elmer,
    pytest.mark.requires_gmsh,
    pytest.mark.requires_jax,
]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def test_public_coarse_ring_heater_matches_locked_elmer_full_fields(
    locked_gmsh_runner,
    locked_elmer_tet4_electrothermal_oracle,
    tmp_path: Path,
) -> None:
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    meshing = tmp_path / "gmsh"
    meshing.mkdir()
    geometry_path = meshing / "public_ring_heater.geo"
    geometry_path.write_text(recipe.render_geo(), encoding="utf-8", newline="\n")
    gmsh = locked_gmsh_runner.run(
        GmshMeshingRequest(geometry_path.name, dimension=3, timeout_seconds=300.0),
        working_directory=meshing,
        policy=_AUTHORIZED,
    )
    assert gmsh.process_succeeded, gmsh.stderr
    imported = read_gmsh_msh_3d(
        meshing / "mesh.msh",
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )
    cell_count = imported.mesh.topology.cell_count
    forward = prepare_public_ring_heater_forward_plan(
        imported,
        recipe,
        np.zeros((cell_count,), dtype=np.int64),
        partition_count=1,
    )

    device_mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    jacobi = ScalarH1JacobiPolicy(1.0e-15)
    current_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        forward.tet4.current_layout,
        device_mesh,
        jacobi,
    )
    thermal_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        forward.tet4.thermal_layout,
        device_mesh,
        jacobi,
    )
    cg = ScalarH1CGPolicy(1.0e-10, 0.0, 10_000, backward_error_tolerance=1.0e-9)
    runtime = build_tet4_electrothermal_runtime(
        forward.tet4,
        device_mesh,
        cg,
        cg,
        Tet4ElectrothermalAdmissionPolicy(1.0e-7, 1.0e-7, 1.0e-12, 1.0e-7),
        current_preconditioner_factory=current_preconditioner,
        thermal_preconditioner_factory=thermal_preconditioner,
    )
    inputs = pack_tet4_electrothermal_inputs(forward.tet4, value_dtype=np.float64)

    def parameters(voltage_v: float) -> Tet4ElectrothermalParameters:
        return Tet4ElectrothermalParameters(
            jnp.asarray(voltage_v, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )

    solve = jax.jit(runtime.solve)
    unit = solve(inputs, parameters(1.0))
    unit.numerically_admitted.block_until_ready()
    assert bool(unit.numerically_admitted)
    calibration = calibrate_public_ring_heater_current(
        float(unit.electrical_joule_power),
        reference=forward.reference,
    )
    target_parameters = parameters(calibration.target_voltage_v)
    jax_result = solve(inputs, target_parameters)
    jax_result.numerically_admitted.block_until_ready()
    assert bool(jax_result.numerically_admitted)
    jax_potential, jax_temperature = reconstruct_tet4_electrothermal_state(
        forward.tet4,
        jax_result.state,
        target_parameters,
    )
    jax_potential = np.asarray(jax.device_get(jax_potential), dtype=np.float64)
    jax_temperature = np.asarray(jax.device_get(jax_temperature), dtype=np.float64)

    elmer_plan = prepare_public_ring_heater_elmer_plan(
        imported,
        recipe,
        forward,
        applied_voltage_v=calibration.target_voltage_v,
    )
    elmer = locked_elmer_tet4_electrothermal_oracle.run(
        elmer_plan.case,
        run_directory=tmp_path / "elmer",
        policy=_AUTHORIZED,
    )
    assert elmer.process.process_succeeded
    assert elmer.numerical_convergence_evaluated
    assert elmer.numerically_converged

    np.testing.assert_array_equal(elmer.potential_node_ids, forward.tet4.current_parent_node_ids)
    potential_difference = elmer.potential_v - jax_potential
    temperature_difference = elmer.temperature_k - jax_temperature
    temperature_rise = jax_temperature - forward.reference.ambient_temperature_k
    assert np.max(np.abs(potential_difference)) < 2.0e-8
    assert np.linalg.norm(potential_difference) / np.linalg.norm(jax_potential) < 1.0e-8
    assert np.max(np.abs(temperature_difference)) < 2.0e-5
    assert np.linalg.norm(temperature_difference) / np.linalg.norm(temperature_rise) < 1.0e-8
    assert 160.0 < float(np.max(temperature_rise)) < 170.0
    assert float(jax_result.electrical_energy_relative_error) < 1.0e-10
    assert float(jax_result.charge_balance_relative_error) < 1.0e-7
    assert float(jax_result.joule_transfer_relative_error) < 1.0e-12
    assert float(jax_result.thermal_balance_relative_error) < 1.0e-10
    inferred_current = float(jax_result.electrical_joule_power) / calibration.target_voltage_v
    assert inferred_current == pytest.approx(forward.reference.target_current_a, rel=1.0e-10)
