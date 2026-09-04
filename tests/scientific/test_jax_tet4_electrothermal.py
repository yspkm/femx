from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from tests.unit.test_tet4_electrothermal import (  # noqa: E402
    _parameters,
    _plan,
    _runtime,
    _structured_tet4_mesh,
)

from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    pack_tet4_electrothermal_inputs,
    reconstruct_tet4_electrothermal_state,
)

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_manufactured_heat_solution_has_second_order_nodal_rms_convergence() -> None:
    errors: list[float] = []
    sizes: list[float] = []
    for intervals in (2, 3, 4):
        plan = _plan(intervals=intervals)
        inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
        parameters = _parameters()
        result = jax.jit(_runtime(plan).solve)(inputs, parameters)
        _potential, temperature = reconstruct_tet4_electrothermal_state(
            plan,
            result.state,
            parameters,
        )
        coordinates = _structured_tet4_mesh(intervals, intervals, intervals)[0]
        total_source = 5.0
        slope = total_source * (1.0 + 6.0 / 8.0) / 10.0
        exact = 300.0 + slope * coordinates[:, 2] - total_source * coordinates[:, 2] ** 2 / 8.0
        errors.append(float(np.sqrt(np.mean((np.asarray(temperature) - exact) ** 2))))
        sizes.append(1.0 / intervals)
        assert bool(result.numerically_admitted)

    observed_orders = tuple(
        math.log(errors[index] / errors[index + 1]) / math.log(sizes[index] / sizes[index + 1])
        for index in range(len(errors) - 1)
    )
    assert errors[0] > errors[1] > errors[2]
    assert min(observed_orders) > 1.9
