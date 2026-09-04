"""Four-device CPU smoke test for the physical scalar-TPU runner data boundary."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from scripts.run_tpu_scalar_h1_collective_evidence import (  # noqa: E402
    ACTION_TOLERANCE,
    CG_ABSOLUTE_TOLERANCE,
    CG_MAX_ITERATIONS,
    CG_RELATIVE_TOLERANCE,
    HOST_PRECISION_TOLERANCE,
    RHS_TOLERANCE,
    VJP_TOLERANCE,
    _build_explicit_scalar_kernels,
    _build_host_case,
    _pack_cells,
    _pack_owned,
    _pack_owner_mask,
    _physical_coefficients,
    _slab_cell_owners,
    _structured_rectangle,
    _tpu_index_array,
)

from femx.backends.jax.collective_runtime import (  # noqa: E402
    make_collective_array_from_process_local_data,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    build_packed_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    build_packed_collective_scalar_h1_rhs_assembly,
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    numerator = jnp.linalg.norm(observed - expected)
    denominator = jnp.linalg.norm(expected)
    value = jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )
    return float(np.asarray(jax.device_get(value)))


def main() -> int:
    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("scalar TPU-harness probe requires exactly four forced CPU devices")
    coordinates, cells, facets = _structured_rectangle(8, 8)
    coefficients = _physical_coefficients("heat", coordinates, cells, facets)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        _slab_cell_owners(coordinates, cells, 4),
        node_count=coordinates.shape[0],
        free_nodes=coefficients["free_nodes"],
        partition_count=4,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    mesh = Mesh(np.asarray(devices, dtype=object), ("partition",))
    policy = ScalarH1CGPolicy(
        CG_RELATIVE_TOLERANCE,
        CG_ABSOLUTE_TOLERANCE,
        CG_MAX_ITERATIONS,
    )
    assemble = build_packed_collective_scalar_h1_rhs_assembly(layout, mesh)
    solve = build_packed_collective_scalar_h1_cg(layout, mesh, policy)
    forward, vjp = _build_explicit_scalar_kernels(jax, assemble, solve)

    def load(name: str, value: np.ndarray) -> jax.Array:
        return make_collective_array_from_process_local_data(name, value, mesh)[0]

    mapping = load("cell-map", _tpu_index_array(layout.transport.cell_local_dofs))
    mask = load("owner-mask", _pack_owner_mask(layout))
    case_reports: dict[str, object] = {}
    stablehlo_counts: dict[str, object] = {}
    for name in ("heat", "current"):
        host = _build_host_case(name, coordinates, cells, facets, topology)
        stiffness = load(
            f"{name}-stiffness",
            _pack_cells(layout, host["float32"]["stiffness"]),
        )
        cell_rhs = load(
            f"{name}-cell-rhs",
            _pack_cells(layout, host["float32"]["cell_rhs"]),
        )
        cotangent = load(
            f"{name}-cotangent",
            _pack_owned(layout, np.asarray(host["cotangent"], dtype=np.float32)),
        )
        compiled_forward = jax.jit(forward)
        compiled_vjp = jax.jit(vjp)
        result = compiled_forward(stiffness, cell_rhs, mapping, mask)
        matrix_vjp, cell_rhs_vjp = compiled_vjp(
            stiffness,
            cell_rhs,
            mapping,
            mask,
            cotangent,
        )
        jax.block_until_ready((result, matrix_vjp, cell_rhs_vjp))
        expected_solution = load(
            f"expected-{name}-solution",
            _pack_owned(layout, np.asarray(host["input_solution"], dtype=np.float32)),
        )
        expected_rhs = load(
            f"expected-{name}-rhs",
            _pack_owned(layout, np.asarray(host["float32"]["rhs"], dtype=np.float32)),
        )
        expected_matrix_vjp = load(
            f"expected-{name}-matrix-vjp",
            _pack_cells(layout, np.asarray(host["expected_matrix_vjp"], dtype=np.float32)),
        )
        expected_cell_rhs_vjp = load(
            f"expected-{name}-cell-rhs-vjp",
            _pack_cells(layout, np.asarray(host["expected_cell_rhs_vjp"], dtype=np.float32)),
        )
        case_reports[name] = {
            "iterations": int(np.asarray(jax.device_get(result.iterations))),
            "relative_residual": float(np.asarray(jax.device_get(result.relative_residual))),
            "converged": bool(np.asarray(jax.device_get(result.converged))),
            "breakdown": bool(np.asarray(jax.device_get(result.breakdown))),
            "solution_relative_difference": _relative_difference(
                result.solution,
                expected_solution,
            ),
            "rhs_relative_difference": _relative_difference(
                result.right_hand_side,
                expected_rhs,
            ),
            "matrix_vjp_relative_difference": _relative_difference(
                matrix_vjp,
                expected_matrix_vjp,
            ),
            "cell_rhs_vjp_relative_difference": _relative_difference(
                cell_rhs_vjp,
                expected_cell_rhs_vjp,
            ),
            "host_precision_relative_difference": host["host_precision_relative_difference"],
        }
        for executable_name, function, arguments in (
            (f"{name}_forward", forward, (stiffness, cell_rhs, mapping, mask)),
            (f"{name}_vjp", vjp, (stiffness, cell_rhs, mapping, mask, cotangent)),
        ):
            stablehlo = str(jax.jit(function).lower(*arguments).compiler_ir("stablehlo")).lower()
            stablehlo_counts[executable_name] = {
                "collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
                "all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
                "contains_all_gather": "all_gather" in stablehlo,
            }

    payload = {
        "schema_version": "femx.jax.scalar_h1_collective.tpu_harness_cpu_smoke/v1",
        "backend": jax.default_backend(),
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
        "process_count": jax.process_count(),
        "global_device_count": jax.device_count(),
        "layout_sha256": layout.digest(),
        "tolerances": {
            "solution": ACTION_TOLERANCE,
            "rhs": RHS_TOLERANCE,
            "vjp": VJP_TOLERANCE,
            "host_precision": HOST_PRECISION_TOLERANCE,
        },
        "cases": case_reports,
        "stablehlo": stablehlo_counts,
        "claim_scope": (
            "forced multi-CPU smoke test of physical-runner inputs; not accelerator or multi-host "
            "evidence"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
