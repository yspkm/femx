from __future__ import annotations

import numpy as np
import pytest
from tests.elmer_tet4_support import (
    structured_distinct_space_case,
    structured_distinct_space_jax_plan,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.scalar_cg import ScalarH1CGPolicy  # noqa: E402
from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    Tet4ElectrothermalAdmissionPolicy,
    Tet4ElectrothermalParameters,
    build_tet4_electrothermal_runtime,
    pack_tet4_electrothermal_inputs,
    reconstruct_tet4_electrothermal_state,
)
from femx.core.execution import ExecutionPolicy  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_elmer, pytest.mark.requires_jax]


def test_locked_elmer_solves_partial_current_and_full_3d_heat(
    locked_elmer_tet4_electrothermal_oracle,
    tmp_path,
) -> None:
    mesh, case = structured_distinct_space_case()
    result = locked_elmer_tet4_electrothermal_oracle.run(
        case,
        run_directory=tmp_path / "tet4-electrothermal",
        policy=ExecutionPolicy(
            execution_authorized=True,
            allow_external_process=True,
        ),
    )

    assert result.process.process_succeeded
    assert result.numerical_convergence_evaluated
    assert result.numerically_converged

    plan = structured_distinct_space_jax_plan(mesh)
    jax_mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    cg = ScalarH1CGPolicy(2.0e-12, 1.0e-14, 500, backward_error_tolerance=2.0e-12)
    runtime = build_tet4_electrothermal_runtime(
        plan,
        jax_mesh,
        cg,
        cg,
        Tet4ElectrothermalAdmissionPolicy(2.0e-10, 2.0e-10, 2.0e-14, 2.0e-10),
    )
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    parameters = Tet4ElectrothermalParameters(
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    jax_result = jax.jit(runtime.solve)(inputs, parameters)
    jax_potential, jax_temperature = reconstruct_tet4_electrothermal_state(
        plan,
        jax_result.state,
        parameters,
    )

    np.testing.assert_array_equal(result.potential_node_ids, plan.current_parent_node_ids)
    np.testing.assert_allclose(result.potential_v, jax_potential, rtol=0.0, atol=3.0e-13)
    np.testing.assert_allclose(result.temperature_k, jax_temperature, rtol=0.0, atol=4.0e-11)
    np.testing.assert_allclose(
        jax_result.current_joule_density[0, : plan.current_layout.topology.cell_count],
        2.0,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert bool(jax_result.numerically_admitted)
    assert float(jax_result.electrical_joule_power) == pytest.approx(2.0, abs=3.0e-13)
    assert float(jax_result.thermal_balance_relative_error) < 2.0e-13
    assert np.min(result.temperature_k) == pytest.approx(300.0, abs=2.0e-11)
    assert result.provenance["elmer_source_commit"] == ("4f2d7e4b99f8f0dcf2f7ac579e056969373bf594")
