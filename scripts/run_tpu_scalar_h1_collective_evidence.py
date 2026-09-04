#!/usr/bin/env python3
"""Run the bounded physical-TPU witness for scalar H1 RHS, CG, and implicit VJP.

The Phoxla bootstrap initializes JAX distribution before this file is evaluated. Standalone TPU
execution is also supported when ``JAX_PLATFORMS=tpu,cpu`` was set before Python starts. There is
no CPU fallback and no JAX device discovery at module import time.
"""

from __future__ import annotations

import json
import math
import os
import time
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast

EVIDENCE_SCHEMA = "femx.jax.scalar_h1_collective.tpu_evidence/v1"
MULTILEVEL_EXTENSION_SCHEMA = "femx.jax.scalar_h1_collective.multilevel_extension/v1"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.jax.scalar_h1_collective.worker_entry_claim/v1"
ACTION_TOLERANCE = 4.0e-4
RHS_TOLERANCE = 2.0e-6
VJP_TOLERANCE = 2.0e-3
HOST_PRECISION_TOLERANCE = 2.0e-3
EXECUTION_SAMPLES = 5
CG_RELATIVE_TOLERANCE = 2.0e-5
CG_ABSOLUTE_TOLERANCE = 0.0
CG_MAX_ITERATIONS = 4000
MULTILEVEL_MAXIMUM_REPLICATED_DOFS = 2048
MULTILEVEL_MINIMUM_RELATIVE_DIAGONAL = 1.0e-14
MULTILEVEL_MAXIMUM_RELATIVE_SYMMETRY_ERROR = 2.0e-6
MULTILEVEL_MAXIMUM_COARSE_CONDITION_NUMBER = 1.0e8
MULTILEVEL_REPLICATION_INTENT = "bounded multilevel coarse interpolation"
REAL_SCALAR_CONTRACT = {
    "logical_dtype": "float32",
    "matrix_dtype": "float32",
    "load_dtype": "float32",
    "index_dtype": "int32",
    "mask_dtype": "bool",
    "matmul_precision": "highest",
    "host_reference_dtype": "float64",
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


def _solver_mode() -> str:
    value = os.environ.get("FEMX_SCALAR_SOLVER_MODE", "cg")
    if value not in {"cg", "multilevel_pcg"}:
        raise RuntimeError("FEMX_SCALAR_SOLVER_MODE must be 'cg' or 'multilevel_pcg'")
    return value


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
        raise RuntimeError("physical TPU scalar witness requires the explicit float32 path")
    if str(getattr(jax.config, "jax_default_matmul_precision", None)) != "highest":
        raise RuntimeError("JAX default matmul precision must resolve to 'highest'")
    if jax.process_count() < 2:
        raise RuntimeError("physical scalar witness requires at least two JAX processes")
    if jax.device_count() < 2:
        raise RuntimeError("physical scalar witness requires at least two global TPU devices")
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
            (2.0e-6 * i / x_intervals, 0.8e-6 * j / y_intervals)
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
    return coordinates, np.asarray(cells, dtype=np.int64), np.asarray(facets, dtype=np.int64)


def _slab_cell_owners(coordinates: Any, cells: Any, partition_count: int) -> Any:
    import numpy as np

    normalized_x = np.mean(coordinates[cells, 0], axis=1) / np.max(coordinates[:, 0])
    return np.minimum(
        (partition_count * normalized_x).astype(np.int64),
        partition_count - 1,
    )


def _physical_multilevel_hierarchy(
    layout: Any,
    x_intervals: int,
    y_intervals: int,
) -> Any:
    from femx.backends.jax.scalar_multilevel import (
        prepare_scalar_h1_multilevel_hierarchy,
        prepare_scalar_h1_nested_prolongation,
    )

    if x_intervals <= 2 or x_intervals % 2 != 0:
        raise RuntimeError(
            "physical multilevel hierarchy requires an even x interval count above 2"
        )
    if y_intervals <= 0:
        raise RuntimeError("physical multilevel hierarchy requires a positive y interval count")
    prolongations = []
    fine_x = x_intervals
    fine_y = y_intervals
    while fine_x > 2:
        if fine_x % 2 != 0 or (fine_y > 1 and fine_y % 2 != 0):
            raise RuntimeError(
                "physical multilevel hierarchy requires nested power-of-two intervals"
            )
        coarse_x = fine_x // 2
        coarse_y = max(1, fine_y // 2)
        fine_coordinates, fine_cells, fine_facets = _structured_rectangle(fine_x, fine_y)
        coarse_coordinates, coarse_cells, coarse_facets = _structured_rectangle(
            coarse_x,
            coarse_y,
        )
        fine_free = _physical_coefficients(
            "heat",
            fine_coordinates,
            fine_cells,
            fine_facets,
        )["free_nodes"]
        coarse_free = _physical_coefficients(
            "heat",
            coarse_coordinates,
            coarse_cells,
            coarse_facets,
        )["free_nodes"]
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
        layout,
        prolongations,
        maximum_replicated_dofs=MULTILEVEL_MAXIMUM_REPLICATED_DOFS,
    )


def _triangle_cell_stiffness(
    coordinates: Any,
    cells: Any,
    coefficient: Any,
    *,
    dtype: Any,
) -> Any:
    import numpy as np

    points = np.asarray(coordinates, dtype=dtype)[cells]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
        raise RuntimeError("scalar TPU evidence mesh requires positive finite triangles")
    twice_area = determinant
    gradients = (
        np.stack(
            (
                np.stack(
                    (points[:, 1, 1] - points[:, 2, 1], points[:, 2, 0] - points[:, 1, 0]), axis=1
                ),
                np.stack(
                    (points[:, 2, 1] - points[:, 0, 1], points[:, 0, 0] - points[:, 2, 0]), axis=1
                ),
                np.stack(
                    (points[:, 0, 1] - points[:, 1, 1], points[:, 1, 0] - points[:, 0, 0]), axis=1
                ),
            ),
            axis=1,
        )
        / twice_area[:, None, None]
    )
    area = twice_area / np.asarray(2.0, dtype=dtype)
    return (
        np.asarray(coefficient, dtype=dtype)[:, None, None]
        * area[:, None, None]
        * np.einsum(
            "cik,cjk->cij",
            gradients,
            gradients,
        )
    )


def _boundary_incidence(cells: Any, boundary_facets: Any) -> tuple[Any, Any]:
    import numpy as np

    occurrences: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for cell_index, cell in enumerate(cells):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            first_node = int(cell[first])
            second_node = int(cell[second])
            edge = (min(first_node, second_node), max(first_node, second_node))
            occurrences.setdefault(edge, []).append((cell_index, (first, second)))
    facet_cells = np.empty((boundary_facets.shape[0],), dtype=np.int64)
    facet_local = np.empty((boundary_facets.shape[0], 2), dtype=np.int64)
    for facet_index, facet in enumerate(boundary_facets):
        first_node = int(facet[0])
        second_node = int(facet[1])
        edge = (min(first_node, second_node), max(first_node, second_node))
        matches = occurrences.get(edge, ())
        if len(matches) != 1:
            raise RuntimeError("scalar TPU evidence boundary must be an exact exterior edge set")
        cell_index, local_pair = matches[0]
        local_by_node = {int(cells[cell_index, local]): local for local in local_pair}
        facet_cells[facet_index] = cell_index
        facet_local[facet_index] = (local_by_node[int(facet[0])], local_by_node[int(facet[1])])
    return facet_cells, facet_local


def _cell_load(
    coordinates: Any,
    cells: Any,
    boundary_facets: Any,
    source: Any,
    facet_load: Any,
    *,
    dtype: Any,
) -> Any:
    import numpy as np

    points = np.asarray(coordinates, dtype=dtype)[cells]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    area = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / np.asarray(
        2.0,
        dtype=dtype,
    )
    local = np.broadcast_to(
        np.asarray(source, dtype=dtype)[:, None] * area[:, None] / np.asarray(3.0, dtype=dtype),
        (cells.shape[0], 3),
    ).copy()
    facet_cells, facet_local = _boundary_incidence(cells, boundary_facets)
    facet_points = np.asarray(coordinates, dtype=dtype)[boundary_facets]
    lengths = np.linalg.norm(facet_points[:, 1] - facet_points[:, 0], axis=1)
    contributions = np.asarray(facet_load, dtype=dtype) * lengths / np.asarray(2.0, dtype=dtype)
    np.add.at(local, (facet_cells, facet_local[:, 0]), contributions)
    np.add.at(local, (facet_cells, facet_local[:, 1]), contributions)
    return local


def _numpy_matvec(cell_matrix: Any, cell_map: Any, vector: Any) -> Any:
    import numpy as np

    extended = np.concatenate((vector, np.zeros((1,), dtype=vector.dtype)))
    local_input = extended[cell_map]
    local_output = np.einsum("cij,cj->ci", cell_matrix, local_input)
    assembled = np.zeros((vector.shape[0] + 1,), dtype=local_output.dtype)
    np.add.at(assembled, cell_map.reshape(-1), local_output.reshape(-1))
    return assembled[:-1]


def _numpy_assemble_cell_vector(cell_vector: Any, cell_map: Any, free_dof_count: int) -> Any:
    import numpy as np

    assembled = np.zeros((free_dof_count + 1,), dtype=cell_vector.dtype)
    np.add.at(assembled, cell_map.reshape(-1), cell_vector.reshape(-1))
    return assembled[:-1]


def _numpy_cg(
    cell_matrix: Any,
    cell_map: Any,
    right_hand_side: Any,
    *,
    relative_tolerance: float,
    max_iterations: int,
) -> tuple[Any, int, float]:
    import numpy as np

    rhs = np.asarray(right_hand_side)
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    residual_squared = float(np.vdot(residual, residual).real)
    target = relative_tolerance * float(np.linalg.norm(rhs))
    iteration = 0
    while math.sqrt(max(residual_squared, 0.0)) > target and iteration < max_iterations:
        action = _numpy_matvec(cell_matrix, cell_map, direction)
        curvature = float(np.vdot(direction, action).real)
        if not math.isfinite(curvature) or curvature <= 0.0:
            raise RuntimeError("independent NumPy CG encountered nonpositive curvature")
        alpha = residual_squared / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * action
        candidate = float(np.vdot(residual, residual).real)
        if not math.isfinite(candidate) or candidate < 0.0:
            raise RuntimeError("independent NumPy CG produced an invalid residual")
        beta = candidate / residual_squared
        direction = residual + beta * direction
        residual_squared = candidate
        iteration += 1
    residual_norm = float(np.linalg.norm(_numpy_matvec(cell_matrix, cell_map, solution) - rhs))
    if residual_norm > target:
        raise RuntimeError("independent NumPy CG did not satisfy its recomputed residual")
    return solution, iteration, residual_norm


def _numpy_relative_difference(observed: Any, expected: Any) -> float:
    import numpy as np

    numerator = float(np.linalg.norm(np.asarray(observed) - np.asarray(expected)))
    denominator = float(np.linalg.norm(np.asarray(expected)))
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else math.inf


def _physical_coefficients(
    name: str,
    coordinates: Any,
    cells: Any,
    boundary_facets: Any,
) -> dict[str, Any]:
    import numpy as np

    centroids = np.mean(coordinates[cells], axis=1)
    facet_centers = np.mean(coordinates[boundary_facets], axis=1)
    core = np.abs(centroids[:, 1] - 0.4e-6) <= 0.11e-6
    constrained = np.isclose(coordinates[:, 0], 0.0) | np.isclose(
        coordinates[:, 0],
        2.0e-6,
    )
    constrained_nodes = np.flatnonzero(constrained).astype(np.int64)
    free_nodes = np.flatnonzero(~constrained).astype(np.int64)
    if name == "heat":
        coefficient = np.where(core, 130.0, 1.4)
        heater = (np.abs(centroids[:, 0] - 1.0e-6) <= 0.25e-6) & (
            np.abs(centroids[:, 1] - 0.58e-6) <= 0.08e-6
        )
        source = np.where(heater, 4.0e13, 0.0)
        top = np.isclose(facet_centers[:, 1], 0.8e-6)
        facet_load = np.where(top, 2.0e4, 0.0)
        dirichlet = np.where(
            np.isclose(coordinates[constrained_nodes, 0], 0.0),
            300.0,
            305.0,
        )
        model = "2D per-unit-depth representative Si/SiO2 steady heat diffusion"
        units = {
            "state": "K",
            "coefficient": "W/(m*K)",
            "source": "W/m^3",
            "facet_load": "W/m^2",
        }
    elif name == "current":
        coefficient = np.where(core, 2.0e5, 5.0e4)
        source = np.zeros(cells.shape[0], dtype=np.float64)
        facet_load = np.zeros(boundary_facets.shape[0], dtype=np.float64)
        dirichlet = np.where(
            np.isclose(coordinates[constrained_nodes, 0], 0.0),
            0.0,
            1.0,
        )
        model = "2D per-unit-depth representative steady electrical conduction"
        units = {
            "state": "V",
            "coefficient": "S/m",
            "source": "A/m^3",
            "facet_load": "A/m^2",
        }
    else:
        raise ValueError(f"unknown scalar physical case {name!r}")
    return {
        "name": name,
        "model": model,
        "units": units,
        "coefficient": coefficient,
        "source": source,
        "facet_load": facet_load,
        "dirichlet": dirichlet,
        "free_nodes": free_nodes,
        "constrained_nodes": constrained_nodes,
    }


def _build_host_case(
    name: str,
    coordinates: Any,
    cells: Any,
    boundary_facets: Any,
    topology: Any,
) -> dict[str, Any]:
    import numpy as np

    coefficients = _physical_coefficients(name, coordinates, cells, boundary_facets)
    if not np.array_equal(coefficients["free_nodes"], topology.free_nodes):
        raise RuntimeError("scalar host case free-node identity disagrees with its topology")
    cell_map = np.asarray(topology.cell_reduced_dofs, dtype=np.int64)
    full_dirichlet = np.zeros((coordinates.shape[0],), dtype=np.float64)
    full_dirichlet[coefficients["constrained_nodes"]] = coefficients["dirichlet"]

    assembled: dict[str, dict[str, Any]] = {}
    for precision, dtype in (("float32", np.float32), ("float64", np.float64)):
        stiffness = _triangle_cell_stiffness(
            coordinates,
            cells,
            coefficients["coefficient"],
            dtype=dtype,
        )
        load = _cell_load(
            coordinates,
            cells,
            boundary_facets,
            coefficients["source"],
            coefficients["facet_load"],
            dtype=dtype,
        )
        local_boundary_values = np.asarray(full_dirichlet, dtype=dtype)[cells]
        cell_rhs = load - np.einsum("cij,cj->ci", stiffness, local_boundary_values)
        cell_rhs = np.where(cell_map < topology.free_dof_count, cell_rhs, 0.0)
        rhs = _numpy_assemble_cell_vector(cell_rhs, cell_map, topology.free_dof_count)
        assembled[precision] = {
            "stiffness": np.asarray(stiffness, dtype=dtype),
            "cell_rhs": np.asarray(cell_rhs, dtype=dtype),
            "rhs": np.asarray(rhs, dtype=dtype),
        }

    input_matrix = assembled["float32"]["stiffness"].astype(np.float64)
    input_rhs = assembled["float32"]["rhs"].astype(np.float64)
    input_solution, input_iterations, input_residual = _numpy_cg(
        input_matrix,
        cell_map,
        input_rhs,
        relative_tolerance=2.0e-12,
        max_iterations=20_000,
    )
    high_solution, high_iterations, high_residual = _numpy_cg(
        assembled["float64"]["stiffness"],
        cell_map,
        assembled["float64"]["rhs"],
        relative_tolerance=2.0e-12,
        max_iterations=20_000,
    )
    cotangent = np.cos(np.arange(topology.free_dof_count, dtype=np.float64) * 0.37 + 0.2) / max(
        topology.free_dof_count, 1
    )
    adjoint, adjoint_iterations, adjoint_residual = _numpy_cg(
        input_matrix,
        cell_map,
        cotangent,
        relative_tolerance=2.0e-12,
        max_iterations=20_000,
    )
    extended_solution = np.concatenate((input_solution, np.zeros((1,), dtype=np.float64)))
    extended_adjoint = np.concatenate((adjoint, np.zeros((1,), dtype=np.float64)))
    local_solution = extended_solution[cell_map]
    local_adjoint = extended_adjoint[cell_map]
    expected_matrix_vjp = -local_adjoint[:, :, None] * local_solution[:, None, :]
    expected_cell_rhs_vjp = local_adjoint
    return {
        **coefficients,
        "float32": assembled["float32"],
        "float64": assembled["float64"],
        "input_solution": input_solution,
        "input_iterations": input_iterations,
        "input_residual_norm": input_residual,
        "high_solution": high_solution,
        "high_iterations": high_iterations,
        "high_residual_norm": high_residual,
        "cotangent": cotangent,
        "adjoint_iterations": adjoint_iterations,
        "adjoint_residual_norm": adjoint_residual,
        "expected_matrix_vjp": expected_matrix_vjp,
        "expected_cell_rhs_vjp": expected_cell_rhs_vjp,
        "host_precision_relative_difference": _numpy_relative_difference(
            input_solution,
            high_solution,
        ),
    }


def _tpu_index_array(values: Any) -> Any:
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

    tail = tuple(values.shape[1:])
    extended = np.concatenate((values, np.zeros((1, *tail), dtype=values.dtype)), axis=0)
    return np.ascontiguousarray(extended[layout.transport.cell_ids])


def _pack_owned(layout: Any, values: Any) -> Any:
    import numpy as np

    extended = np.concatenate((values, np.zeros((1,), dtype=values.dtype)))
    return np.ascontiguousarray(extended[layout.transport.owned_dof_ids])


def _pack_owner_mask(layout: Any) -> Any:
    import numpy as np

    return np.ascontiguousarray(
        layout.transport.owned_dof_ids < layout.topology.free_dof_count,
        dtype=np.bool_,
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
    if not remote_run.is_absolute() or not remote_run.is_dir() or remote_run.is_symlink():
        raise RuntimeError("PHOXLA_REMOTE_RUN_DIR must be an absolute non-symlink directory")
    process_index = _nonnegative_environment_integer("PHOXLA_PROCESS_INDEX")
    worker_index = _nonnegative_environment_integer("PHOXLA_GCLOUD_WORKER_INDEX")
    if process_index is None or worker_index is None:
        raise RuntimeError("Phoxla process and worker indexes are required before entry execution")
    run_id = os.environ.get("PHOXLA_RUN_ID")
    if run_id != provenance.get("run_id"):
        raise RuntimeError("PHOXLA_RUN_ID disagrees with the deployed manifest")
    claim_path = remote_run / "logs" / "femx-scalar-entry.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "duplicate femx scalar entry refused for this immutable worker-local run"
        ) from error
    claim = {
        "schema_version": WORKER_ENTRY_CLAIM_SCHEMA,
        "run_id": run_id,
        "worker_index": worker_index,
        "process_index": process_index,
        "source_sha256": provenance.get("source_digest"),
        "config_sha256": provenance.get("config_digest"),
        "scope": (
            "worker-local scalar-H1 entry fence after Phoxla bootstrap; prevents duplicate "
            "scientific execution but does not claim controller-level launch ownership"
        ),
    }
    _atomic_json(claim_path / "identity.json", claim)
    return claim


def _memory_report(compiled: Any, hbm_capacity_bytes: int | None) -> Any:
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
    )


def _compile_and_time(
    jax: Any,
    function: Any,
    arguments: tuple[Any, ...],
) -> tuple[Any, Any, str]:
    from femx.backends.jax.collective_runtime import CollectiveTimingReport

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
    timing = CollectiveTimingReport(
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        warmup_seconds=warmup_seconds,
        execution_seconds=tuple(samples),
    )
    return compiled, timing, stablehlo


def _build_explicit_scalar_kernels(
    jax: Any,
    assemble_rhs: Any,
    solve: Any,
) -> tuple[Any, Any]:
    def forward(
        cell_matrix: Any,
        cell_rhs: Any,
        cell_dof_map: Any,
        owner_mask: Any,
        *solver_arguments: Any,
    ) -> Any:
        right_hand_side = assemble_rhs(cell_rhs, cell_dof_map)
        return solve(
            cell_matrix,
            cell_dof_map,
            owner_mask,
            right_hand_side,
            *solver_arguments,
        )

    def vjp(
        cell_matrix: Any,
        cell_rhs: Any,
        cell_dof_map: Any,
        owner_mask: Any,
        cotangent: Any,
        *solver_arguments: Any,
    ) -> tuple[Any, Any]:
        def differentiable(matrix: Any, local_rhs: Any) -> Any:
            assembled = assemble_rhs(local_rhs, cell_dof_map)
            return solve(
                matrix,
                cell_dof_map,
                owner_mask,
                assembled,
                *solver_arguments,
            ).solution

        _, pullback = jax.vjp(differentiable, cell_matrix, cell_rhs)
        return cast(tuple[Any, Any], pullback(cotangent))

    return forward, vjp


def main() -> int:
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    launch_claim = _claim_worker_entry(remote_run, provenance)
    solver_mode = _solver_mode()
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
    from femx.backends.jax.scalar_cg import ScalarH1CGPolicy, build_packed_collective_scalar_h1_cg
    from femx.backends.jax.scalar_collective import (
        build_packed_collective_scalar_h1_rhs_assembly,
        prepare_collective_scalar_h1_layout,
    )
    from femx.backends.jax.scalar_owned_ghost import prepare_scalar_h1_owned_ghost_topology

    if solver_mode == "multilevel_pcg":
        from femx.backends.jax.scalar_multilevel import (
            PackedScalarH1MultilevelTransfer,
            ScalarH1MultilevelPolicy,
            build_packed_scalar_h1_multilevel_runtime,
            pack_scalar_h1_multilevel_transfer_host,
        )

    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    global_device_count = int(jax.device_count())
    local_device_count = int(jax.local_device_count())
    output_root = Path(os.environ["PHOXLA_OUTPUT_DIR"])
    if cast(int, launch_claim["process_index"]) != process_index:
        raise RuntimeError("worker entry claim disagrees with initialized JAX process identity")

    x_intervals = 2 * global_device_count
    y_intervals = min(32, max(8, global_device_count // 2))
    coordinates, cells, boundary_facets = _structured_rectangle(x_intervals, y_intervals)
    heat_coefficients = _physical_coefficients("heat", coordinates, cells, boundary_facets)
    free_nodes = heat_coefficients["free_nodes"]
    cell_owners = _slab_cell_owners(coordinates, cells, global_device_count)
    if not np.array_equal(np.unique(cell_owners), np.arange(global_device_count)):
        raise RuntimeError("physical scalar mesh must assign cells to every global TPU device")
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        cell_owners,
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=global_device_count,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("partition",))
    mesh_report = describe_collective_mesh(
        layout.transport,
        mesh,
        layout_sha256=layout.digest(),
    )
    policy = ScalarH1CGPolicy(
        relative_tolerance=CG_RELATIVE_TOLERANCE,
        absolute_tolerance=CG_ABSOLUTE_TOLERANCE,
        max_iterations=CG_MAX_ITERATIONS,
    )
    assemble_rhs = build_packed_collective_scalar_h1_rhs_assembly(layout, mesh)
    solve = build_packed_collective_scalar_h1_cg(layout, mesh, policy)
    forward, vjp = _build_explicit_scalar_kernels(jax, assemble_rhs, solve)

    def load(name: str, value: Any) -> tuple[Any, Any]:
        return make_collective_array_from_process_local_data(name, value, mesh)

    multilevel_hierarchy = None
    multilevel_policy = None
    multilevel_runtime = None
    multilevel_forward = None
    multilevel_vjp = None
    multilevel_transfer = None
    multilevel_partitioned_reports: dict[str, object] = {}
    multilevel_replicated_reports: dict[str, object] = {}
    if solver_mode == "multilevel_pcg":
        multilevel_hierarchy = _physical_multilevel_hierarchy(
            layout,
            x_intervals,
            y_intervals,
        )
        host_transfer = pack_scalar_h1_multilevel_transfer_host(
            layout,
            multilevel_hierarchy,
            value_dtype=np.float32,
        )

        def load_partitioned_transfer(name: str, value: Any) -> Any:
            array, report = load(name, value)
            multilevel_partitioned_reports[name] = report.canonical_data()
            return array

        def load_replicated_transfer(name: str, value: Any) -> Any:
            array, report = make_replicated_array_from_process_local_data(
                name,
                value,
                mesh,
                replication_intent=MULTILEVEL_REPLICATION_INTENT,
            )
            multilevel_replicated_reports[name] = report.canonical_data()
            return array

        multilevel_transfer = PackedScalarH1MultilevelTransfer(
            owner_columns=load_partitioned_transfer(
                "multilevel-owner-columns",
                host_transfer.owner_columns,
            ),
            owner_weights=load_partitioned_transfer(
                "multilevel-owner-weights",
                host_transfer.owner_weights,
            ),
            cell_columns=load_partitioned_transfer(
                "multilevel-cell-columns",
                host_transfer.cell_columns,
            ),
            cell_weights=load_partitioned_transfer(
                "multilevel-cell-weights",
                host_transfer.cell_weights,
            ),
            coarse_columns=tuple(
                load_replicated_transfer(
                    f"multilevel-coarse-{index}-columns",
                    values,
                )
                for index, values in enumerate(host_transfer.coarse_columns, start=1)
            ),
            coarse_weights=tuple(
                load_replicated_transfer(
                    f"multilevel-coarse-{index}-weights",
                    values,
                )
                for index, values in enumerate(host_transfer.coarse_weights, start=1)
            ),
        )
        multilevel_policy = ScalarH1MultilevelPolicy(
            minimum_relative_diagonal=MULTILEVEL_MINIMUM_RELATIVE_DIAGONAL,
            maximum_relative_symmetry_error=MULTILEVEL_MAXIMUM_RELATIVE_SYMMETRY_ERROR,
            maximum_coarse_condition_number=MULTILEVEL_MAXIMUM_COARSE_CONDITION_NUMBER,
        )
        multilevel_runtime = build_packed_scalar_h1_multilevel_runtime(
            layout,
            mesh,
            multilevel_hierarchy,
            multilevel_policy,
        )
        multilevel_solve = build_packed_collective_scalar_h1_cg(
            layout,
            mesh,
            policy,
            preconditioner_factory=multilevel_runtime.factory,
        )
        multilevel_forward, multilevel_vjp = _build_explicit_scalar_kernels(
            jax,
            assemble_rhs,
            multilevel_solve,
        )

    packed_map, map_report = load(
        "cell-local-dof-map",
        _tpu_index_array(layout.transport.cell_local_dofs),
    )
    packed_mask, mask_report = load("owner-mask", _pack_owner_mask(layout))

    def relative_difference_unjitted(observed: Any, expected: Any) -> Any:
        numerator = jnp.linalg.norm(observed - expected)
        denominator = jnp.linalg.norm(expected)
        return jnp.where(
            denominator > 0.0,
            numerator / denominator,
            jnp.where(numerator == 0.0, 0.0, jnp.inf),
        )

    relative_difference = jax.jit(relative_difference_unjitted)
    all_finite = jax.jit(lambda value: jnp.all(jnp.isfinite(value)))
    hbm_capacity_bytes = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    if hbm_capacity_bytes is None:
        raise RuntimeError("FEMX_HBM_BYTES_PER_DEVICE is required for physical TPU evidence")

    cases_payload: dict[str, dict[str, object]] = {}
    array_reports: dict[str, object] = {
        "cell_local_dofs": map_report.canonical_data(),
        "owner_mask": mask_report.canonical_data(),
    }
    executable_evidence: dict[str, dict[str, object]] = {}
    multilevel_cases_payload: dict[str, dict[str, object]] = {}
    multilevel_executable_evidence: dict[str, dict[str, object]] = {}
    stablehlo_by_name: dict[str, str] = {}
    all_cases_passed = True
    all_multilevel_cases_passed = True
    for name in ("heat", "current"):
        host = _build_host_case(name, coordinates, cells, boundary_facets, topology)
        packed_stiffness_host = _pack_cells(layout, host["float32"]["stiffness"])
        packed_cell_rhs_host = _pack_cells(layout, host["float32"]["cell_rhs"])
        packed_cotangent_host = _pack_owned(
            layout,
            np.asarray(host["cotangent"], dtype=np.float32),
        )
        packed_stiffness, stiffness_report = load(
            f"{name}-cell-stiffness",
            packed_stiffness_host,
        )
        packed_cell_rhs, cell_rhs_report = load(
            f"{name}-cell-rhs",
            packed_cell_rhs_host,
        )
        packed_cotangent, cotangent_report = load(
            f"{name}-owned-cotangent",
            packed_cotangent_host,
        )
        array_reports[f"{name}_cell_stiffness"] = stiffness_report.canonical_data()
        array_reports[f"{name}_cell_rhs"] = cell_rhs_report.canonical_data()
        array_reports[f"{name}_owned_cotangent"] = cotangent_report.canonical_data()

        forward_arguments = (packed_stiffness, packed_cell_rhs, packed_map, packed_mask)
        compiled_forward, forward_timing, forward_hlo = _compile_and_time(
            jax,
            forward,
            forward_arguments,
        )
        vjp_arguments = (*forward_arguments, packed_cotangent)
        compiled_vjp, vjp_timing, vjp_hlo = _compile_and_time(jax, vjp, vjp_arguments)
        result = compiled_forward(*forward_arguments)
        matrix_vjp, cell_rhs_vjp = compiled_vjp(*vjp_arguments)
        jax.block_until_ready((result, matrix_vjp, cell_rhs_vjp))

        expected_solution, _ = load(
            f"expected-{name}-solution",
            _pack_owned(layout, np.asarray(host["input_solution"], dtype=np.float32)),
        )
        expected_rhs, _ = load(
            f"expected-{name}-rhs",
            _pack_owned(layout, np.asarray(host["float32"]["rhs"], dtype=np.float32)),
        )
        expected_matrix_vjp, _ = load(
            f"expected-{name}-matrix-vjp",
            _pack_cells(layout, np.asarray(host["expected_matrix_vjp"], dtype=np.float32)),
        )
        expected_cell_rhs_vjp, _ = load(
            f"expected-{name}-cell-rhs-vjp",
            _pack_cells(layout, np.asarray(host["expected_cell_rhs_vjp"], dtype=np.float32)),
        )
        solution_difference = float(
            np.asarray(jax.device_get(relative_difference(result.solution, expected_solution)))
        )
        rhs_difference = float(
            np.asarray(jax.device_get(relative_difference(result.right_hand_side, expected_rhs)))
        )
        matrix_vjp_difference = float(
            np.asarray(jax.device_get(relative_difference(matrix_vjp, expected_matrix_vjp)))
        )
        cell_rhs_vjp_difference = float(
            np.asarray(jax.device_get(relative_difference(cell_rhs_vjp, expected_cell_rhs_vjp)))
        )
        finite = {
            "solution": bool(np.asarray(jax.device_get(all_finite(result.solution)))),
            "right_hand_side": bool(np.asarray(jax.device_get(all_finite(result.right_hand_side)))),
            "matrix_vjp": bool(np.asarray(jax.device_get(all_finite(matrix_vjp)))),
            "cell_rhs_vjp": bool(np.asarray(jax.device_get(all_finite(cell_rhs_vjp)))),
        }
        converged = bool(np.asarray(jax.device_get(result.converged)))
        breakdown = bool(np.asarray(jax.device_get(result.breakdown)))
        case_passed = (
            converged
            and not breakdown
            and all(finite.values())
            and solution_difference <= ACTION_TOLERANCE
            and rhs_difference <= RHS_TOLERANCE
            and max(matrix_vjp_difference, cell_rhs_vjp_difference) <= VJP_TOLERANCE
            and host["host_precision_relative_difference"] <= HOST_PRECISION_TOLERANCE
        )
        all_cases_passed = all_cases_passed and case_passed
        cases_payload[name] = {
            "status": "passed" if case_passed else "failed",
            "physics": {
                "model": host["model"],
                "units": host["units"],
                "coefficient_minimum": float(np.min(host["coefficient"])),
                "coefficient_maximum": float(np.max(host["coefficient"])),
                "source_maximum": float(np.max(host["source"])),
                "dirichlet_minimum": float(np.min(host["dirichlet"])),
                "dirichlet_maximum": float(np.max(host["dirichlet"])),
                "material_scope": "representative values; not foundry-calibrated material data",
            },
            "cg": {
                "relative_tolerance": policy.relative_tolerance,
                "absolute_tolerance": policy.absolute_tolerance,
                "max_iterations": policy.max_iterations,
                "iterations": int(np.asarray(jax.device_get(result.iterations))),
                "rhs_norm": float(np.asarray(jax.device_get(result.rhs_norm))),
                "recursive_residual_norm": float(
                    np.asarray(jax.device_get(result.recursive_residual_norm))
                ),
                "recomputed_residual_norm": float(
                    np.asarray(jax.device_get(result.recomputed_residual_norm))
                ),
                "relative_residual": float(np.asarray(jax.device_get(result.relative_residual))),
                "converged": converged,
                "breakdown": breakdown,
            },
            "numerics": {
                "solution_relative_difference": solution_difference,
                "rhs_relative_difference": rhs_difference,
                "matrix_vjp_relative_difference": matrix_vjp_difference,
                "cell_rhs_vjp_relative_difference": cell_rhs_vjp_difference,
                "host_float32_input_vs_float64_assembly_solution_relative_difference": host[
                    "host_precision_relative_difference"
                ],
                "finite": finite,
                "numpy_input_authority_iterations": host["input_iterations"],
                "numpy_input_authority_residual_norm": host["input_residual_norm"],
                "numpy_float64_authority_iterations": host["high_iterations"],
                "numpy_float64_authority_residual_norm": host["high_residual_norm"],
                "numpy_adjoint_iterations": host["adjoint_iterations"],
                "numpy_adjoint_residual_norm": host["adjoint_residual_norm"],
                "authority": (
                    "independent NumPy float64 matrix-free CG and analytic residual adjoint "
                    "applied to the explicit float32 FEM inputs"
                ),
            },
        }

        if solver_mode == "multilevel_pcg":
            if (
                multilevel_forward is None
                or multilevel_vjp is None
                or multilevel_runtime is None
                or multilevel_transfer is None
                or multilevel_policy is None
            ):
                raise RuntimeError("multilevel solver mode was not fully initialized")
            multilevel_forward_arguments = (
                packed_stiffness,
                packed_cell_rhs,
                packed_map,
                packed_mask,
                multilevel_transfer,
            )
            compiled_multilevel_forward, multilevel_forward_timing, multilevel_forward_hlo = (
                _compile_and_time(
                    jax,
                    multilevel_forward,
                    multilevel_forward_arguments,
                )
            )
            multilevel_vjp_arguments = (
                packed_stiffness,
                packed_cell_rhs,
                packed_map,
                packed_mask,
                packed_cotangent,
                multilevel_transfer,
            )
            compiled_multilevel_vjp, multilevel_vjp_timing, multilevel_vjp_hlo = _compile_and_time(
                jax,
                multilevel_vjp,
                multilevel_vjp_arguments,
            )
            multilevel_setup_arguments = (
                packed_stiffness,
                packed_map,
                packed_mask,
                multilevel_transfer,
            )
            compiled_multilevel_setup, multilevel_setup_timing, multilevel_setup_hlo = (
                _compile_and_time(
                    jax,
                    multilevel_runtime.setup,
                    multilevel_setup_arguments,
                )
            )
            multilevel_result = compiled_multilevel_forward(*multilevel_forward_arguments)
            multilevel_matrix_vjp, multilevel_cell_rhs_vjp = compiled_multilevel_vjp(
                *multilevel_vjp_arguments
            )
            multilevel_state = compiled_multilevel_setup(*multilevel_setup_arguments)
            jax.block_until_ready(
                (
                    multilevel_result,
                    multilevel_matrix_vjp,
                    multilevel_cell_rhs_vjp,
                    multilevel_state,
                )
            )
            multilevel_solution_difference = float(
                np.asarray(
                    jax.device_get(
                        relative_difference(multilevel_result.solution, expected_solution)
                    )
                )
            )
            multilevel_baseline_difference = float(
                np.asarray(
                    jax.device_get(relative_difference(multilevel_result.solution, result.solution))
                )
            )
            multilevel_rhs_difference = float(
                np.asarray(
                    jax.device_get(
                        relative_difference(multilevel_result.right_hand_side, expected_rhs)
                    )
                )
            )
            multilevel_matrix_vjp_difference = float(
                np.asarray(
                    jax.device_get(relative_difference(multilevel_matrix_vjp, expected_matrix_vjp))
                )
            )
            multilevel_cell_rhs_vjp_difference = float(
                np.asarray(
                    jax.device_get(
                        relative_difference(multilevel_cell_rhs_vjp, expected_cell_rhs_vjp)
                    )
                )
            )
            multilevel_finite = {
                "solution": bool(
                    np.asarray(jax.device_get(all_finite(multilevel_result.solution)))
                ),
                "right_hand_side": bool(
                    np.asarray(jax.device_get(all_finite(multilevel_result.right_hand_side)))
                ),
                "matrix_vjp": bool(np.asarray(jax.device_get(all_finite(multilevel_matrix_vjp)))),
                "cell_rhs_vjp": bool(
                    np.asarray(jax.device_get(all_finite(multilevel_cell_rhs_vjp)))
                ),
            }
            multilevel_converged = bool(np.asarray(jax.device_get(multilevel_result.converged)))
            multilevel_breakdown = bool(np.asarray(jax.device_get(multilevel_result.breakdown)))
            setup_valid = bool(np.asarray(jax.device_get(multilevel_state.valid)))
            baseline_iterations = int(np.asarray(jax.device_get(result.iterations)))
            multilevel_iterations = int(np.asarray(jax.device_get(multilevel_result.iterations)))
            iteration_improved = multilevel_iterations < baseline_iterations
            multilevel_case_passed = (
                setup_valid
                and multilevel_converged
                and not multilevel_breakdown
                and iteration_improved
                and all(multilevel_finite.values())
                and multilevel_solution_difference <= ACTION_TOLERANCE
                and multilevel_baseline_difference <= ACTION_TOLERANCE
                and multilevel_rhs_difference <= RHS_TOLERANCE
                and max(
                    multilevel_matrix_vjp_difference,
                    multilevel_cell_rhs_vjp_difference,
                )
                <= VJP_TOLERANCE
            )
            all_multilevel_cases_passed = all_multilevel_cases_passed and multilevel_case_passed
            multilevel_cases_payload[name] = {
                "status": "passed" if multilevel_case_passed else "failed",
                "baseline_cg_iterations": baseline_iterations,
                "iteration_improved": iteration_improved,
                "pcg": {
                    "relative_tolerance": policy.relative_tolerance,
                    "absolute_tolerance": policy.absolute_tolerance,
                    "max_iterations": policy.max_iterations,
                    "iterations": multilevel_iterations,
                    "rhs_norm": float(np.asarray(jax.device_get(multilevel_result.rhs_norm))),
                    "recursive_residual_norm": float(
                        np.asarray(jax.device_get(multilevel_result.recursive_residual_norm))
                    ),
                    "recomputed_residual_norm": float(
                        np.asarray(jax.device_get(multilevel_result.recomputed_residual_norm))
                    ),
                    "relative_residual": float(
                        np.asarray(jax.device_get(multilevel_result.relative_residual))
                    ),
                    "converged": multilevel_converged,
                    "breakdown": multilevel_breakdown,
                },
                "setup": {
                    "valid": setup_valid,
                    "minimum_relative_diagonal": float(
                        np.asarray(jax.device_get(multilevel_state.minimum_relative_diagonal))
                    ),
                    "maximum_relative_symmetry_error": float(
                        np.asarray(jax.device_get(multilevel_state.maximum_relative_symmetry_error))
                    ),
                    "maximum_coarse_condition_number": float(
                        np.asarray(jax.device_get(multilevel_state.maximum_coarse_condition_number))
                    ),
                },
                "numerics": {
                    "solution_relative_difference": multilevel_solution_difference,
                    "solution_vs_unpreconditioned_relative_difference": (
                        multilevel_baseline_difference
                    ),
                    "rhs_relative_difference": multilevel_rhs_difference,
                    "matrix_vjp_relative_difference": multilevel_matrix_vjp_difference,
                    "cell_rhs_vjp_relative_difference": (multilevel_cell_rhs_vjp_difference),
                    "finite": multilevel_finite,
                    "authority": (
                        "the same independent NumPy float64 matrix-free CG and analytic "
                        "residual adjoint used by the unpreconditioned physical witness"
                    ),
                },
            }

            for executable_name, compiled, timing, stablehlo in (
                (
                    f"{name}_setup",
                    compiled_multilevel_setup,
                    multilevel_setup_timing,
                    multilevel_setup_hlo,
                ),
                (
                    f"{name}_forward",
                    compiled_multilevel_forward,
                    multilevel_forward_timing,
                    multilevel_forward_hlo,
                ),
                (
                    f"{name}_vjp",
                    compiled_multilevel_vjp,
                    multilevel_vjp_timing,
                    multilevel_vjp_hlo,
                ),
            ):
                lowered = stablehlo.lower()
                multilevel_executable_evidence[executable_name] = {
                    "timing": timing.canonical_data(),
                    "memory": _memory_report(compiled, hbm_capacity_bytes).canonical_data(),
                    "stablehlo_collective_permute_count": lowered.count(
                        "stablehlo.collective_permute"
                    ),
                    "stablehlo_all_reduce_count": lowered.count("stablehlo.all_reduce"),
                    "stablehlo_contains_all_gather": "all_gather" in lowered,
                }
                stablehlo_by_name[f"multilevel_{executable_name}"] = stablehlo

        for executable_name, compiled, timing, stablehlo in (
            (f"{name}_forward", compiled_forward, forward_timing, forward_hlo),
            (f"{name}_vjp", compiled_vjp, vjp_timing, vjp_hlo),
        ):
            lowered = stablehlo.lower()
            executable_evidence[executable_name] = {
                "timing": timing.canonical_data(),
                "memory": _memory_report(compiled, hbm_capacity_bytes).canonical_data(),
                "stablehlo_collective_permute_count": lowered.count("stablehlo.collective_permute"),
                "stablehlo_all_reduce_count": lowered.count("stablehlo.all_reduce"),
                "stablehlo_contains_all_gather": "all_gather" in lowered,
            }
            stablehlo_by_name[executable_name] = stablehlo

    local_partition_mask = np.zeros((global_device_count,), dtype=np.int32)
    for shard in map_report.addressable_shards:
        local_partition_mask[shard.partition_index] = 1
    gathered_partition_masks = np.asarray(
        multihost_utils.process_allgather(local_partition_mask, tiled=False)
    ).reshape(process_count, global_device_count)
    partition_addressability_counts = np.sum(gathered_partition_masks, axis=0)
    exact_addressability = bool(
        np.array_equal(partition_addressability_counts, np.ones(global_device_count))
    )
    collectives_valid = all(
        not bool(record["stablehlo_contains_all_gather"])
        and cast(int, record["stablehlo_collective_permute_count"]) > 0
        and cast(int, record["stablehlo_all_reduce_count"]) > 0
        for record in executable_evidence.values()
    )
    multilevel_collectives_valid = True
    if solver_mode == "multilevel_pcg":
        multilevel_collectives_valid = all(
            not bool(record["stablehlo_contains_all_gather"])
            and cast(int, record["stablehlo_all_reduce_count"]) > 0
            and cast(int, record["stablehlo_collective_permute_count"]) > 0
            for record in multilevel_executable_evidence.values()
        )
    passed = (
        all_cases_passed
        and exact_addressability
        and collectives_valid
        and (solver_mode != "multilevel_pcg" or all_multilevel_cases_passed)
        and multilevel_collectives_valid
    )
    storage = layout.transport.storage_report
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
            "real_scalar_contract": REAL_SCALAR_CONTRACT,
        },
        "launch_claim": launch_claim,
        "problem": {
            "model": "two bounded scalar H1/P1 diffusion systems on one exact triangular mesh",
            "x_intervals": x_intervals,
            "y_intervals": y_intervals,
            "node_count": int(coordinates.shape[0]),
            "triangle_count": int(cells.shape[0]),
            "free_dof_count": topology.free_dof_count,
            "partition_count": layout.partition_count,
            "layout_sha256": layout.digest(),
            "halo_link_count": len(layout.transport.halo_links),
            "halo_value_count": storage.halo_value_count,
            "cell_padding_fraction": storage.cell_padding_fraction,
            "owned_dof_padding_fraction": storage.owned_dof_padding_fraction,
            "ghost_dof_padding_fraction": storage.ghost_dof_padding_fraction,
        },
        "mesh_report": mesh_report.canonical_data(),
        "addressability": {
            "process_local_partition_mask": local_partition_mask.tolist(),
            "partition_addressability_counts": partition_addressability_counts.tolist(),
            "every_partition_addressable_once": exact_addressability,
        },
        "array_reports": array_reports,
        "tolerances": {
            "solution_relative_difference": ACTION_TOLERANCE,
            "rhs_relative_difference": RHS_TOLERANCE,
            "vjp_relative_difference": VJP_TOLERANCE,
            "host_precision_relative_difference": HOST_PRECISION_TOLERANCE,
        },
        "cases": cases_payload,
        "executables": executable_evidence,
        "claim_scope": (
            "physical process-complete multi-process TPU scalar H1/P1 RHS, unpreconditioned CG, "
            "and residual-defined implicit-VJP correctness evidence for bounded representative "
            "heat and current systems; not Elmer parity, coupled electrothermal execution, "
            "preconditioned scaling, or live HBM; not a foundry prediction; and not "
            "Spot-preemption recovery"
        ),
    }
    if solver_mode == "multilevel_pcg":
        if multilevel_hierarchy is None or multilevel_policy is None:
            raise RuntimeError("multilevel evidence metadata was not initialized")
        process_payload["multilevel"] = {
            "schema_version": MULTILEVEL_EXTENSION_SCHEMA,
            "status": "passed"
            if all_multilevel_cases_passed and multilevel_collectives_valid
            else "failed",
            "hierarchy": {
                "schema_version": multilevel_hierarchy.schema_version,
                "sha256": multilevel_hierarchy.digest(),
                "layout_sha256": multilevel_hierarchy.layout_sha256,
                "level_dof_counts": list(multilevel_hierarchy.level_dof_counts),
                "maximum_replicated_dofs": (multilevel_hierarchy.maximum_replicated_dofs),
                "prolongation_sha256": [
                    level.digest() for level in multilevel_hierarchy.prolongations
                ],
            },
            "policy": {
                "diagonal_weight": multilevel_policy.diagonal_weight,
                "minimum_relative_diagonal": (multilevel_policy.minimum_relative_diagonal),
                "maximum_relative_symmetry_error": (
                    multilevel_policy.maximum_relative_symmetry_error
                ),
                "maximum_coarse_condition_number": (
                    multilevel_policy.maximum_coarse_condition_number
                ),
                "iteration_admission": (
                    "PCG iterations must be strictly below same-run unpreconditioned CG "
                    "for both heat and current"
                ),
            },
            "partitioned_transfer_reports": multilevel_partitioned_reports,
            "replicated_transfer_reports": multilevel_replicated_reports,
            "cases": multilevel_cases_payload,
            "executables": multilevel_executable_evidence,
            "collectives_valid": multilevel_collectives_valid,
            "claim_scope": (
                "physical process-local record for explicit multilevel-PCG setup, forward, "
                "and residual-defined implicit VJP on the same bounded heat/current systems; "
                "coarse interpolation replication is bounded and declared; timing is not a "
                "scaling result, compiler memory is not live HBM, and this is not Elmer parity, "
                "coupled electrothermal execution, a foundry prediction, or Spot-preemption "
                "recovery"
            ),
        }
    for name, stablehlo in stablehlo_by_name.items():
        hlo_path = output_root / "hlo" / f"{name}.stablehlo.mlir"
        hlo_path.parent.mkdir(parents=True, exist_ok=True)
        hlo_path.write_text(stablehlo, encoding="utf-8")
    _write_process_evidence(output_root, remote_run, process_index, process_payload)
    multihost_utils.sync_global_devices(f"femx-scalar-tpu-evidence-written-{provenance['run_id']}")
    print(json.dumps({"status": process_payload["status"], "run_id": provenance["run_id"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
