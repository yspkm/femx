"""Four-device CPU smoke test for explicit multilevel TPU-runner inputs."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]
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
    make_replicated_array_from_process_local_data,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    build_packed_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    build_packed_collective_scalar_h1_rhs_assembly,
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_multilevel import (  # noqa: E402
    PackedScalarH1MultilevelTransfer,
    ScalarH1MultilevelHierarchy,
    ScalarH1MultilevelPolicy,
    build_packed_scalar_h1_multilevel_runtime,
    pack_scalar_h1_multilevel_transfer_host,
    prepare_scalar_h1_multilevel_hierarchy,
    prepare_scalar_h1_nested_prolongation,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)


def _free_nodes(x_intervals: int, y_intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates, cells, facets = _structured_rectangle(x_intervals, y_intervals)
    coefficients = _physical_coefficients("heat", coordinates, cells, facets)
    return coordinates, cells, np.asarray(coefficients["free_nodes"])


def _hierarchy(layout: object) -> ScalarH1MultilevelHierarchy:
    prolongations = []
    fine_x = 8
    fine_y = 8
    while fine_x > 2 and fine_y > 2:
        coarse_x = fine_x // 2
        coarse_y = fine_y // 2
        fine_coordinates, _, fine_free = _free_nodes(fine_x, fine_y)
        coarse_coordinates, coarse_cells, coarse_free = _free_nodes(coarse_x, coarse_y)
        prolongations.append(
            prepare_scalar_h1_nested_prolongation(
                fine_coordinates,
                fine_free,
                coarse_coordinates,
                coarse_cells,
                coarse_free,
            )
        )
        fine_x = coarse_x
        fine_y = coarse_y
    return prepare_scalar_h1_multilevel_hierarchy(
        layout,  # type: ignore[arg-type]
        prolongations,
        maximum_replicated_dofs=64,
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
        raise RuntimeError("explicit multilevel probe requires exactly four forced CPU devices")
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
    hierarchy = _hierarchy(layout)
    mesh = Mesh(np.asarray(devices, dtype=object), ("partition",))
    host_transfer = pack_scalar_h1_multilevel_transfer_host(
        layout,
        hierarchy,
        value_dtype=np.float32,
    )

    partitioned_reports: dict[str, object] = {}

    def partitioned(name: str, value: np.ndarray) -> jax.Array:
        array, report = make_collective_array_from_process_local_data(name, value, mesh)
        partitioned_reports[name] = report.canonical_data()
        return array

    replicated_reports: dict[str, object] = {}

    def replicated(name: str, value: np.ndarray) -> jax.Array:
        array, report = make_replicated_array_from_process_local_data(
            name,
            value,
            mesh,
            replication_intent="bounded multilevel coarse interpolation",
        )
        replicated_reports[name] = report.canonical_data()
        return array

    transfer = PackedScalarH1MultilevelTransfer(
        owner_columns=partitioned("transfer-owner-columns", host_transfer.owner_columns),
        owner_weights=partitioned("transfer-owner-weights", host_transfer.owner_weights),
        cell_columns=partitioned("transfer-cell-columns", host_transfer.cell_columns),
        cell_weights=partitioned("transfer-cell-weights", host_transfer.cell_weights),
        coarse_columns=tuple(
            replicated(f"transfer-coarse-{index}-columns", values)
            for index, values in enumerate(host_transfer.coarse_columns, start=1)
        ),
        coarse_weights=tuple(
            replicated(f"transfer-coarse-{index}-weights", values)
            for index, values in enumerate(host_transfer.coarse_weights, start=1)
        ),
    )
    mapping = partitioned(
        "cell-map",
        _tpu_index_array(layout.transport.cell_local_dofs),
    )
    mask = partitioned("owner-mask", _pack_owner_mask(layout))
    policy = ScalarH1CGPolicy(
        CG_RELATIVE_TOLERANCE,
        CG_ABSOLUTE_TOLERANCE,
        CG_MAX_ITERATIONS,
    )
    multilevel_policy = ScalarH1MultilevelPolicy(
        maximum_relative_symmetry_error=2.0e-6,
        maximum_coarse_condition_number=1.0e12,
    )
    runtime = build_packed_scalar_h1_multilevel_runtime(
        layout,
        mesh,
        hierarchy,
        multilevel_policy,
    )
    solve = build_packed_collective_scalar_h1_cg(
        layout,
        mesh,
        policy,
        preconditioner_factory=runtime.factory,
    )
    baseline_solve = build_packed_collective_scalar_h1_cg(layout, mesh, policy)
    assemble = build_packed_collective_scalar_h1_rhs_assembly(layout, mesh)
    forward, vjp = _build_explicit_scalar_kernels(jax, assemble, solve)
    baseline_forward, _ = _build_explicit_scalar_kernels(jax, assemble, baseline_solve)

    cases: dict[str, object] = {}
    stablehlo_reports: dict[str, object] = {}
    for name in ("heat", "current"):
        host = _build_host_case(name, coordinates, cells, facets, topology)
        stiffness = partitioned(
            f"{name}-stiffness",
            _pack_cells(layout, host["float32"]["stiffness"]),
        )
        cell_rhs = partitioned(
            f"{name}-cell-rhs",
            _pack_cells(layout, host["float32"]["cell_rhs"]),
        )
        cotangent = partitioned(
            f"{name}-cotangent",
            _pack_owned(layout, np.asarray(host["cotangent"], dtype=np.float32)),
        )
        forward_arguments = (stiffness, cell_rhs, mapping, mask, transfer)
        result = jax.jit(forward)(*forward_arguments)
        baseline_result = jax.jit(baseline_forward)(stiffness, cell_rhs, mapping, mask)
        matrix_vjp, cell_rhs_vjp = jax.jit(vjp)(
            stiffness,
            cell_rhs,
            mapping,
            mask,
            cotangent,
            transfer,
        )
        state = jax.jit(runtime.setup)(stiffness, mapping, mask, transfer)
        jax.block_until_ready(  # type: ignore[no-untyped-call]
            (result, baseline_result, matrix_vjp, cell_rhs_vjp, state)
        )
        expected_solution = partitioned(
            f"expected-{name}-solution",
            _pack_owned(layout, np.asarray(host["input_solution"], dtype=np.float32)),
        )
        expected_matrix_vjp = partitioned(
            f"expected-{name}-matrix-vjp",
            _pack_cells(layout, np.asarray(host["expected_matrix_vjp"], dtype=np.float32)),
        )
        expected_cell_rhs_vjp = partitioned(
            f"expected-{name}-cell-rhs-vjp",
            _pack_cells(layout, np.asarray(host["expected_cell_rhs_vjp"], dtype=np.float32)),
        )
        cases[name] = {
            "iterations": int(np.asarray(jax.device_get(result.iterations))),
            "unpreconditioned_iterations": int(
                np.asarray(jax.device_get(baseline_result.iterations))
            ),
            "relative_residual": float(np.asarray(jax.device_get(result.relative_residual))),
            "converged": bool(np.asarray(jax.device_get(result.converged))),
            "breakdown": bool(np.asarray(jax.device_get(result.breakdown))),
            "setup_valid": bool(np.asarray(jax.device_get(state.valid))),
            "minimum_relative_diagonal": float(
                np.asarray(jax.device_get(state.minimum_relative_diagonal))
            ),
            "maximum_relative_symmetry_error": float(
                np.asarray(jax.device_get(state.maximum_relative_symmetry_error))
            ),
            "maximum_coarse_condition_number": float(
                np.asarray(jax.device_get(state.maximum_coarse_condition_number))
            ),
            "solution_relative_difference": _relative_difference(
                result.solution,
                expected_solution,
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
            (
                f"{name}_setup",
                runtime.setup,
                (stiffness, mapping, mask, transfer),
            ),
            (f"{name}_forward", forward, forward_arguments),
            (
                f"{name}_vjp",
                vjp,
                (stiffness, cell_rhs, mapping, mask, cotangent, transfer),
            ),
        ):
            stablehlo = str(jax.jit(function).lower(*arguments).compiler_ir("stablehlo")).lower()
            stablehlo_reports[executable_name] = {
                "collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
                "all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
                "contains_all_gather": "all_gather" in stablehlo,
            }

    payload = {
        "schema_version": "femx.jax.scalar_h1_multilevel.explicit_cpu_smoke/v1",
        "backend": jax.default_backend(),
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
        "process_count": jax.process_count(),
        "global_device_count": jax.device_count(),
        "layout_sha256": layout.digest(),
        "hierarchy_sha256": hierarchy.digest(),
        "level_dof_counts": list(hierarchy.level_dof_counts),
        "partitioned_transfer_reports": partitioned_reports,
        "replicated_transfer_reports": replicated_reports,
        "tolerances": {
            "solution": ACTION_TOLERANCE,
            "vjp": VJP_TOLERANCE,
            "host_precision": HOST_PRECISION_TOLERANCE,
            "rhs": RHS_TOLERANCE,
        },
        "cases": cases,
        "stablehlo": stablehlo_reports,
        "claim_scope": (
            "forced multi-CPU smoke test of explicit physical multilevel-runner inputs; "
            "not accelerator or multi-host evidence"
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
