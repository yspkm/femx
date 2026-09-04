"""Hashed JAX transfer from mixed port FEM coefficients to exact FDTDX Yee samples."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple, cast

from femx.core.arrays import ArrayLike, shape_of
from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.capabilities import FunctionSpaceFamily
from femx.core.errors import ContractError
from femx.core.solution import Solution
from femx.interop.fdtdx.mode_bundle import (
    YEE_SPATIAL_OFFSETS,
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeGrid,
    YeeVectorField,
)
from femx.interop.fdtdx.thermo_optic import FDTDXFingerprint
from femx.mesh import FunctionSpace
from femx.physics.port_eigenmode import (
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
)

_GRID_SCHEMA = "femx.fdtdx.yee_edges_xyz/v1"
_TRANSFER_SCHEMA = "femx.transfer.mixed_port_to_fdtdx_yee/v1"
_LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))
_REFERENCE_GRADIENTS = ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))
_FDTDX_VACUUM_IMPEDANCE_OHM = 4.0e-7 * math.pi * 299_792_458.0


class SamplingAmbiguityPolicy(StrEnum):
    """Policy for target points that lie on more than one source triangle."""

    REJECT = "reject"
    LOWEST_CELL_ID = "lowest_cell_id"


@dataclass(frozen=True, slots=True)
class YeePortSamplingPlan:
    """Static FEM-to-Yee point-location operator for the lossless positive-z port slice."""

    source_coordinates: ArrayLike
    source_cells: ArrayLike
    edge_nodes: ArrayLike
    cell_edge_dofs: ArrayLike
    cell_edge_signs: ArrayLike
    target_grid: YeeGrid
    electric_cell_indices: ArrayLike
    electric_barycentric_weights: ArrayLike
    magnetic_cell_indices: ArrayLike
    magnetic_barycentric_weights: ArrayLike
    source_mesh_sha256: str
    operator_sha256: str
    containment_tolerance: float
    ambiguity_policy: SamplingAmbiguityPolicy
    ambiguous_target_point_count: int
    maximum_partition_error: float
    minimum_barycentric_weight: float
    plane_axes: tuple[int, int] = (0, 1)
    schema_version: str = _TRANSFER_SCHEMA

    def __post_init__(self) -> None:
        coordinate_shape = shape_of(self.source_coordinates)
        cell_shape = shape_of(self.source_cells)
        if len(coordinate_shape) != 2 or coordinate_shape[1] != 2:
            raise ContractError("Yee transfer source coordinates must have shape (nodes, 2)")
        if len(cell_shape) != 2 or cell_shape[1] != 3 or cell_shape[0] == 0:
            raise ContractError("Yee transfer source cells must have non-empty shape (cells, 3)")
        if shape_of(self.edge_nodes) != (int(self.edge_nodes.shape[0]), 2):
            raise ContractError("Yee transfer canonical edges must have shape (edges, 2)")
        if (
            shape_of(self.cell_edge_dofs) != cell_shape
            or shape_of(self.cell_edge_signs) != cell_shape
        ):
            raise ContractError("Yee transfer edge topology must match source triangles")
        target_shape = self.target_grid.shape
        expected_indices = (3, *target_shape)
        expected_weights = (*expected_indices, 3)
        for label, values in (
            ("electric cell indices", self.electric_cell_indices),
            ("magnetic cell indices", self.magnetic_cell_indices),
        ):
            if shape_of(values) != expected_indices:
                raise ContractError(f"Yee transfer {label} must have shape {expected_indices}")
        for label, values in (
            ("electric barycentric weights", self.electric_barycentric_weights),
            ("magnetic barycentric weights", self.magnetic_barycentric_weights),
        ):
            if shape_of(values) != expected_weights:
                raise ContractError(f"Yee transfer {label} must have shape {expected_weights}")
        for label, digest in (
            ("source mesh", self.source_mesh_sha256),
            ("operator", self.operator_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError(f"Yee transfer {label} digest must be a lowercase SHA-256")
        if self.plane_axes != (0, 1) or self.target_grid.shape[2] != 1:
            raise ContractError("Yee port transfer v1 requires an x-y plane and one z cell")
        if not math.isfinite(self.containment_tolerance) or self.containment_tolerance <= 0.0:
            raise ContractError("Yee transfer containment tolerance must be finite and positive")
        if self.ambiguous_target_point_count < 0:
            raise ContractError("Yee transfer ambiguous-point count cannot be negative")
        if (
            self.ambiguity_policy is SamplingAmbiguityPolicy.REJECT
            and self.ambiguous_target_point_count
        ):
            raise ContractError("a rejecting Yee transfer plan cannot retain ambiguous points")
        if not math.isfinite(self.maximum_partition_error) or self.maximum_partition_error < 0.0:
            raise ContractError("Yee transfer partition error must be finite and non-negative")
        if not math.isfinite(self.minimum_barycentric_weight):
            raise ContractError("Yee transfer minimum barycentric weight must be finite")
        if self.minimum_barycentric_weight < -self.containment_tolerance:
            raise ContractError("Yee transfer plan contains a point outside its assigned triangle")
        if self.schema_version != _TRANSFER_SCHEMA:
            raise ContractError(f"unsupported Yee transfer schema {self.schema_version!r}")


class YeeModeSamples(NamedTuple):
    """JAX-compatible transferred fields and signed-power evidence."""

    electric_v_per_m: ArrayLike
    magnetic_a_per_m: ArrayLike
    magnetic_eta0_v_per_m: ArrayLike
    pre_correction_power_watts: ArrayLike
    transferred_power_watts: ArrayLike
    power_correction_scale: ArrayLike


def _readonly(values: Any, *, dtype: str) -> Any:
    import numpy as np

    result = np.ascontiguousarray(values, dtype=dtype)
    result.flags.writeable = False
    return result


def _hash_array(hasher: Any, label: str, values: Any, *, dtype: str) -> None:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
    hasher.update(label.encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes(order="C"))


def build_yee_grid(edge_coordinates: Sequence[Any]) -> YeeGrid:
    """Validate and hash three physical FDTDX edge-coordinate axes."""

    import numpy as np

    if len(edge_coordinates) != 3:
        raise ContractError("a Yee grid requires exactly three edge-coordinate axes")
    axes: list[Any] = []
    hasher = hashlib.sha256()
    hasher.update(_GRID_SCHEMA.encode("ascii"))
    for axis, raw_values in enumerate(edge_coordinates):
        values = np.asarray(raw_values)
        if values.ndim != 1 or values.size < 2:
            raise ContractError(f"Yee edge axis {axis} must contain at least two coordinates")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ContractError(f"Yee edge axis {axis} must contain finite real coordinates")
        values = _readonly(values, dtype="<f8")
        if np.any(np.diff(values) <= 0.0):
            raise ContractError(f"Yee edge axis {axis} must be strictly increasing")
        _hash_array(hasher, f"axis_{axis}_edges_m", values, dtype="<f8")
        axes.append(values)
    return YeeGrid(
        edge_coordinates=cast(tuple[ArrayLike, ArrayLike, ArrayLike], tuple(axes)),
        coordinate_sha256=hasher.hexdigest(),
    )


def _source_mesh_digest(coordinates: Any, cells: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"femx.triangle_mixed_port_mesh/v1")
    _hash_array(hasher, "coordinates_m", coordinates, dtype="<f8")
    _hash_array(hasher, "cells", cells, dtype="<i8")
    return hasher.hexdigest()


def _canonical_edge_topology(cells: Any, edge_signs: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    local_edges = cells[:, _LOCAL_EDGES]
    expected_signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1)
    if edge_signs is None:
        raise ContractError("Yee port transfer requires explicit cell-local edge orientations")
    raw_signs = np.asarray(edge_signs)
    if raw_signs.dtype.kind not in "iu" or raw_signs.shape != cells.shape:
        raise ContractError("Yee port transfer edge orientations must be integer (cells, 3)")
    if not np.array_equal(raw_signs, expected_signs):
        raise ContractError("Yee port transfer orientations disagree with canonical edge order")
    edge_nodes, inverse = np.unique(
        np.sort(local_edges, axis=2).reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    return (
        _readonly(edge_nodes, dtype="<i8"),
        _readonly(inverse.reshape(cells.shape), dtype="<i8"),
        _readonly(expected_signs, dtype="i1"),
    )


def _component_axis_coordinates(edges: Any, offset: float) -> Any:
    if offset == 0.0:
        return edges[:-1]
    if offset == 0.5:
        return 0.5 * (edges[:-1] + edges[1:])
    raise ContractError(f"unsupported Yee spatial offset {offset}")


def _component_points(grid: YeeGrid, offsets: tuple[float, float, float]) -> Any:
    import numpy as np

    axes = [
        _component_axis_coordinates(np.asarray(edges), offset)
        for edges, offset in zip(grid.edge_coordinates, offsets, strict=True)
    ]
    coordinates = np.meshgrid(*axes, indexing="ij")
    return np.stack((coordinates[0].reshape(-1), coordinates[1].reshape(-1)), axis=1)


def _locate_points(
    source_points: Any,
    target_points: Any,
    *,
    containment_tolerance: float,
) -> tuple[Any, Any, Any]:
    import numpy as np

    assigned_cells = np.full(target_points.shape[0], -1, dtype=np.int64)
    assigned_weights = np.zeros((target_points.shape[0], 3), dtype=np.float64)
    match_counts = np.zeros(target_points.shape[0], dtype=np.int64)
    for cell_id, points in enumerate(source_points):
        basis = np.column_stack((points[1] - points[0], points[2] - points[0]))
        local = (target_points - points[0]) @ np.linalg.inv(basis).T
        weights = np.column_stack((1.0 - local[:, 0] - local[:, 1], local[:, 0], local[:, 1]))
        inside = np.all(weights >= -containment_tolerance, axis=1)
        inside &= np.all(weights <= 1.0 + containment_tolerance, axis=1)
        first = inside & (assigned_cells < 0)
        assigned_cells[first] = cell_id
        assigned_weights[first] = weights[first]
        match_counts += inside.astype(np.int64)
    return assigned_cells, assigned_weights, match_counts


def build_yee_port_sampling_plan(
    source_coordinates: Any,
    source_cells: Any,
    cell_edge_signs: Any,
    target_grid: YeeGrid,
    *,
    ambiguity_policy: SamplingAmbiguityPolicy = SamplingAmbiguityPolicy.REJECT,
    containment_tolerance: float = 1.0e-12,
) -> YeePortSamplingPlan:
    """Build six deterministic point-location maps for the FDTDX Yee staggering."""

    import numpy as np

    coordinates = np.asarray(source_coordinates)
    cells = np.asarray(source_cells)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ContractError("Yee transfer source coordinates must have shape (nodes, 2)")
    if coordinates.dtype.kind not in "fiu" or not np.isfinite(coordinates).all():
        raise ContractError("Yee transfer source coordinates must be finite real values")
    coordinates = _readonly(coordinates, dtype="<f8")
    if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] == 0:
        raise ContractError("Yee transfer source cells must have non-empty shape (cells, 3)")
    if cells.dtype.kind not in "iu":
        raise ContractError("Yee transfer source cells must use an integer dtype")
    cells = _readonly(cells, dtype="<i8")
    if np.any(cells < 0) or np.any(cells >= coordinates.shape[0]):
        raise ContractError("Yee transfer source connectivity is outside the node range")
    if coordinates.shape[0] > np.iinfo(np.int32).max or cells.shape[0] > np.iinfo(np.int32).max:
        raise ContractError("Yee transfer reference operator requires int32-addressable meshes")
    if not math.isfinite(containment_tolerance) or containment_tolerance <= 0.0:
        raise ContractError("Yee transfer containment tolerance must be finite and positive")
    if target_grid.shape[2] != 1:
        raise ContractError("Yee port transfer v1 requires exactly one cell along z")

    source_points = coordinates[cells]
    for cell_id, points in enumerate(source_points):
        edge_a = points[1] - points[0]
        edge_b = points[2] - points[0]
        determinant = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
        scale = max(
            float(np.linalg.norm(edge_a)),
            float(np.linalg.norm(edge_b)),
            float(np.linalg.norm(points[2] - points[1])),
        )
        threshold = (
            64.0
            * np.finfo(np.float64).eps
            * max(
                scale * scale,
                np.finfo(float).tiny,
            )
        )
        if abs(determinant) <= threshold:
            raise ContractError(f"Yee transfer source triangle {cell_id} is degenerate")

    edge_nodes, cell_edge_dofs, canonical_signs = _canonical_edge_topology(
        cells,
        cell_edge_signs,
    )
    target_shape = target_grid.shape
    located: dict[YeeFieldKind, tuple[Any, Any]] = {}
    ambiguous_count = 0
    partition_error = 0.0
    minimum_weight = math.inf
    for field_kind in (YeeFieldKind.ELECTRIC, YeeFieldKind.MAGNETIC):
        component_cells = []
        component_weights = []
        for component, offsets in enumerate(YEE_SPATIAL_OFFSETS[field_kind]):
            points = _component_points(target_grid, offsets)
            assigned_cells, assigned_weights, match_counts = _locate_points(
                source_points,
                points,
                containment_tolerance=containment_tolerance,
            )
            missing = np.flatnonzero(assigned_cells < 0)
            if missing.size:
                first = points[int(missing[0])]
                raise ContractError(
                    f"Yee {field_kind.value} component {component} has {missing.size} point(s) "
                    f"outside the source mesh; first planar coordinate={first.tolist()}"
                )
            ambiguous = np.flatnonzero(match_counts > 1)
            if ambiguous.size and ambiguity_policy is SamplingAmbiguityPolicy.REJECT:
                first = points[int(ambiguous[0])]
                raise ContractError(
                    f"Yee {field_kind.value} component {component} has {ambiguous.size} "
                    "point(s) on multiple source triangles; choose a non-coincident grid or "
                    "explicitly request lowest_cell_id; "
                    f"first planar coordinate={first.tolist()}"
                )
            ambiguous_count += int(ambiguous.size)
            reshaped_weights = assigned_weights.reshape((*target_shape, 3))
            component_cells.append(assigned_cells.reshape(target_shape))
            component_weights.append(reshaped_weights)
            partition_error = max(
                partition_error,
                float(np.max(np.abs(np.sum(reshaped_weights, axis=-1) - 1.0))),
            )
            minimum_weight = min(minimum_weight, float(np.min(reshaped_weights)))
        located[field_kind] = (
            _readonly(np.stack(component_cells), dtype="<i8"),
            _readonly(np.stack(component_weights), dtype="<f8"),
        )

    source_digest = _source_mesh_digest(coordinates, cells)
    electric_cells, electric_weights = located[YeeFieldKind.ELECTRIC]
    magnetic_cells, magnetic_weights = located[YeeFieldKind.MAGNETIC]
    operator_hasher = hashlib.sha256()
    operator_hasher.update(_TRANSFER_SCHEMA.encode("ascii"))
    operator_hasher.update(source_digest.encode("ascii"))
    operator_hasher.update(target_grid.coordinate_sha256.encode("ascii"))
    operator_hasher.update(ambiguity_policy.value.encode("ascii"))
    _hash_array(operator_hasher, "edge_nodes", edge_nodes, dtype="<i8")
    _hash_array(operator_hasher, "cell_edge_dofs", cell_edge_dofs, dtype="<i8")
    _hash_array(operator_hasher, "cell_edge_signs", canonical_signs, dtype="i1")
    _hash_array(operator_hasher, "electric_cells", electric_cells, dtype="<i8")
    _hash_array(operator_hasher, "electric_weights", electric_weights, dtype="<f8")
    _hash_array(operator_hasher, "magnetic_cells", magnetic_cells, dtype="<i8")
    _hash_array(operator_hasher, "magnetic_weights", magnetic_weights, dtype="<f8")
    return YeePortSamplingPlan(
        source_coordinates=coordinates,
        source_cells=cells,
        edge_nodes=edge_nodes,
        cell_edge_dofs=cell_edge_dofs,
        cell_edge_signs=canonical_signs,
        target_grid=target_grid,
        electric_cell_indices=electric_cells,
        electric_barycentric_weights=electric_weights,
        magnetic_cell_indices=magnetic_cells,
        magnetic_barycentric_weights=magnetic_weights,
        source_mesh_sha256=source_digest,
        operator_sha256=operator_hasher.hexdigest(),
        containment_tolerance=containment_tolerance,
        ambiguity_policy=ambiguity_policy,
        ambiguous_target_point_count=ambiguous_count,
        maximum_partition_error=partition_error,
        minimum_barycentric_weight=minimum_weight,
    )


def _validate_mode_coefficients(
    plan: YeePortSamplingPlan,
    scalar_coefficients: ArrayLike,
    edge_coefficients: ArrayLike,
    cell_reluctivity_per_henry_m: ArrayLike,
) -> None:
    expected_scalar = (int(plan.source_coordinates.shape[0]),)
    expected_edge = (int(plan.edge_nodes.shape[0]),)
    expected_cells = (int(plan.source_cells.shape[0]),)
    if shape_of(scalar_coefficients) != expected_scalar:
        raise ContractError(
            f"port scalar coefficients must have shape {expected_scalar}, "
            f"got {shape_of(scalar_coefficients)}"
        )
    if shape_of(edge_coefficients) != expected_edge:
        raise ContractError(
            f"port edge coefficients must have shape {expected_edge}, "
            f"got {shape_of(edge_coefficients)}"
        )
    if shape_of(cell_reluctivity_per_henry_m) != expected_cells:
        raise ContractError(
            f"port cell reluctivity must have shape {expected_cells}, "
            f"got {shape_of(cell_reluctivity_per_henry_m)}"
        )
    for label, values in (
        ("scalar", scalar_coefficients),
        ("edge", edge_coefficients),
    ):
        dtype_kind = getattr(values.dtype, "kind", None)
        if dtype_kind is not None and dtype_kind != "c":
            raise ContractError(f"port {label} coefficients must use a complex dtype")
    reluctivity_dtype_kind = getattr(cell_reluctivity_per_henry_m.dtype, "kind", None)
    if reluctivity_dtype_kind is not None and reluctivity_dtype_kind not in "fiu":
        raise ContractError("port cell reluctivity must use a real dtype")


def sample_port_mode_to_yee(
    plan: YeePortSamplingPlan,
    scalar_coefficients: ArrayLike,
    edge_coefficients: ArrayLike,
    propagation_constant_per_m: complex | ArrayLike,
    cell_reluctivity_per_henry_m: ArrayLike,
    angular_frequency_rad_per_s: float | ArrayLike,
    target_power_watts: float | ArrayLike,
) -> YeeModeSamples:
    """Evaluate native mixed fields at six Yee locations and conserve signed power."""

    import jax.numpy as jnp

    _validate_mode_coefficients(
        plan,
        scalar_coefficients,
        edge_coefficients,
        cell_reluctivity_per_henry_m,
    )
    scalar = jnp.asarray(scalar_coefficients)
    edge = jnp.asarray(edge_coefficients)
    beta = jnp.asarray(propagation_constant_per_m)
    omega = jnp.asarray(angular_frequency_rad_per_s)
    target_power = jnp.asarray(target_power_watts)
    if beta.ndim != 0 or omega.ndim != 0 or target_power.ndim != 0:
        raise ContractError("port beta, angular frequency, and target power must be scalars")

    coordinates = jnp.asarray(plan.source_coordinates)
    cells = jnp.asarray(plan.source_cells, dtype=jnp.int32)
    cell_edge_dofs = jnp.asarray(plan.cell_edge_dofs, dtype=jnp.int32)
    cell_edge_signs = jnp.asarray(plan.cell_edge_signs)
    cell_reluctivity = jnp.asarray(cell_reluctivity_per_henry_m)
    electric_cell_indices = jnp.asarray(plan.electric_cell_indices, dtype=jnp.int32)
    electric_barycentric_weights = jnp.asarray(plan.electric_barycentric_weights)
    magnetic_cell_indices = jnp.asarray(plan.magnetic_cell_indices, dtype=jnp.int32)
    magnetic_barycentric_weights = jnp.asarray(plan.magnetic_barycentric_weights)
    points = coordinates[cells]
    jacobians = jnp.stack((points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=2)
    determinants = jacobians[:, 0, 0] * jacobians[:, 1, 1] - jacobians[:, 0, 1] * jacobians[:, 1, 0]
    inverse_transpose = (
        jnp.stack(
            (
                jacobians[:, 1, 1],
                -jacobians[:, 1, 0],
                -jacobians[:, 0, 1],
                jacobians[:, 0, 0],
            ),
            axis=1,
        ).reshape((-1, 2, 2))
        / determinants[:, None, None]
    )
    reference_gradients = jnp.asarray(_REFERENCE_GRADIENTS, dtype=coordinates.dtype)
    scalar_gradients = jnp.einsum("cij,nj->cni", inverse_transpose, reference_gradients)
    curl_basis = 2.0 * cell_edge_signs / determinants[:, None]
    starts = jnp.asarray((0, 1, 2), dtype=jnp.int32)
    ends = jnp.asarray((1, 2, 0), dtype=jnp.int32)

    def transverse_at(cell_indices: ArrayLike, weights: ArrayLike) -> tuple[Any, Any, Any]:
        selected = jnp.asarray(cell_indices, dtype=jnp.int32)
        barycentric = jnp.asarray(weights)
        gradients = scalar_gradients[selected]
        basis = (
            barycentric[..., starts, None] * gradients[..., ends, :]
            - barycentric[..., ends, None] * gradients[..., starts, :]
        )
        basis = basis * cell_edge_signs[selected][..., :, None]
        local_coefficients = edge[cell_edge_dofs[selected]]
        transverse = jnp.einsum("...e,...ed->...d", local_coefficients, basis)
        local_scalar = scalar[cells[selected]]
        scalar_value = jnp.einsum("...n,...n->...", local_scalar, barycentric)
        scalar_gradient = jnp.einsum("...n,...nd->...d", local_scalar, gradients)
        transverse_curl = jnp.einsum(
            "...e,...e->...",
            local_coefficients,
            curl_basis[selected],
        )
        return (
            transverse,
            scalar_value,
            jnp.concatenate(
                (scalar_gradient, transverse_curl[..., None]),
                axis=-1,
            ),
        )

    electric_components = []
    for component in range(3):
        transverse, scalar_value, _derivatives = transverse_at(
            electric_cell_indices[component],
            electric_barycentric_weights[component],
        )
        value = transverse[..., component] if component < 2 else scalar_value / (1j * beta)
        electric_components.append(value)
    electric = jnp.stack(electric_components, axis=0)

    magnetic_components = []
    for component in range(3):
        selected = magnetic_cell_indices[component]
        transverse, _scalar_value, derivatives = transverse_at(
            magnetic_cell_indices[component],
            magnetic_barycentric_weights[component],
        )
        longitudinal_gradient = derivatives[..., :2] / (1j * beta)
        prefactor = cell_reluctivity[selected] / (1j * omega)
        if component == 0:
            value = prefactor * (longitudinal_gradient[..., 1] - 1j * beta * transverse[..., 1])
        elif component == 1:
            value = prefactor * (1j * beta * transverse[..., 0] - longitudinal_gradient[..., 0])
        else:
            value = prefactor * derivatives[..., 2]
        magnetic_components.append(value)
    magnetic = jnp.stack(magnetic_components, axis=0)

    edge_axes = [jnp.asarray(axis) for axis in plan.target_grid.edge_coordinates]
    widths_x = edge_axes[0][1:] - edge_axes[0][:-1]
    widths_y = edge_axes[1][1:] - edge_axes[1][:-1]
    face_area = widths_x[:, None, None] * widths_y[None, :, None]
    signed_flux = 0.5 * jnp.real(
        electric[0] * jnp.conj(magnetic[1]) - electric[1] * jnp.conj(magnetic[0])
    )
    pre_correction_power = jnp.sum(face_area * signed_flux)
    correction_scale = jnp.sqrt(target_power / pre_correction_power)
    electric = electric * correction_scale
    magnetic = magnetic * correction_scale
    transferred_power = pre_correction_power * correction_scale * correction_scale
    return YeeModeSamples(
        electric_v_per_m=electric,
        magnetic_a_per_m=magnetic,
        magnetic_eta0_v_per_m=magnetic * _FDTDX_VACUUM_IMPEDANCE_OHM,
        pre_correction_power_watts=pre_correction_power,
        transferred_power_watts=transferred_power,
        power_correction_scale=correction_scale,
    )


def port_mode_solution_to_bundle(
    solution: Solution,
    plan: YeePortSamplingPlan,
    cell_reluctivity_per_henry_m: ArrayLike,
    *,
    frequency_hz: float,
    solver: SolverFingerprint,
    fdtdx: FDTDXFingerprint,
    relative_interpolation_error: float | None = None,
) -> ModeBundle:
    """Construct ``femx.mode/v1`` from backend-neutral normalized mixed fields."""

    import numpy as np

    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ContractError("port mode bundle frequency must be finite and positive")
    if solver.mesh_sha256 != plan.source_mesh_sha256:
        raise ContractError("mode solver mesh digest does not match the Yee transfer source mesh")
    try:
        scalar_field = solution.fields[PORT_LONGITUDINAL_POTENTIAL_FIELD]
        edge_field = solution.fields[PORT_TRANSVERSE_ELECTRIC_FIELD]
        beta = complex(solution.observables["propagation_constant_rad_per_m"])
        effective_index = complex(solution.observables["effective_index"])
        target_power_value = solution.observables["target_forward_power_W"]
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("solution lacks normalized mixed port fields or observables") from error
    if isinstance(target_power_value, complex):
        if target_power_value.imag != 0.0:
            raise ContractError("port target power observable must be real")
        target_power = float(target_power_value.real)
    else:
        target_power = float(target_power_value)
    if not math.isfinite(target_power) or target_power <= 0.0:
        raise ContractError("port target power observable must be finite and positive")
    if not all(
        math.isfinite(value)
        for value in (beta.real, beta.imag, effective_index.real, effective_index.imag)
    ):
        raise ContractError("port propagation constant and effective index must be finite")
    if beta.real <= 0.0:
        raise ContractError("positive-z port transfer requires positive real beta")
    expected_scalar_space = FunctionSpace(FunctionSpaceFamily.H1, order=1)
    expected_edge_space = FunctionSpace(FunctionSpaceFamily.HCURL, order=1, value_shape=(2,))
    if (
        scalar_field.function_space != expected_scalar_space
        or scalar_field.unit != PORT_LONGITUDINAL_POTENTIAL_UNIT
    ):
        raise ContractError("port longitudinal-potential coefficients have incompatible semantics")
    if (
        edge_field.function_space != expected_edge_space
        or edge_field.unit != PORT_TRANSVERSE_ELECTRIC_DOF_UNIT
    ):
        raise ContractError("port transverse edge coefficients have incompatible semantics")
    samples = sample_port_mode_to_yee(
        plan,
        scalar_field.values,
        edge_field.values,
        beta,
        cell_reluctivity_per_henry_m,
        2.0 * math.pi * frequency_hz,
        target_power,
    )
    electric = np.asarray(samples.electric_v_per_m)
    magnetic_eta0 = np.asarray(samples.magnetic_eta0_v_per_m)
    pre_power = float(np.asarray(samples.pre_correction_power_watts))
    transferred_power = float(np.asarray(samples.transferred_power_watts))
    correction_scale = float(np.asarray(samples.power_correction_scale))
    if not (
        np.isfinite(electric).all()
        and np.isfinite(magnetic_eta0).all()
        and math.isfinite(pre_power)
        and pre_power > 0.0
        and math.isfinite(transferred_power)
        and transferred_power > 0.0
        and math.isfinite(correction_scale)
        and correction_scale > 0.0
    ):
        raise ContractError("FEM-to-Yee transfer produced non-finite or backward fields")
    relative_power_error = abs(transferred_power - target_power) / target_power
    transfer = TransferReport(
        source_representation=FieldRepresentation.FEM_DOFS,
        target_representation=FieldRepresentation.CARTESIAN_YEE_SAMPLES,
        operator_sha256=plan.operator_sha256,
        relative_power_error=relative_power_error,
        relative_interpolation_error=relative_interpolation_error,
        source_power_watts=target_power,
        pre_correction_power_watts=pre_power,
        relative_pre_correction_power_error=abs(pre_power - target_power) / target_power,
        transferred_power_watts=transferred_power,
        power_correction_scale=correction_scale,
        target_runtime_name="fdtdx",
        target_runtime_version=fdtdx.package_version,
        target_source_revision=fdtdx.source_revision,
        target_source_digest=fdtdx.source_digest,
    )
    return ModeBundle(
        frequency_hz=frequency_hz,
        effective_index=effective_index,
        beta_per_m=beta,
        electric=YeeVectorField(
            values=electric,
            grid=plan.target_grid,
            field_kind=YeeFieldKind.ELECTRIC,
            unit="V/m",
        ),
        magnetic=YeeVectorField(
            values=magnetic_eta0,
            grid=plan.target_grid,
            field_kind=YeeFieldKind.MAGNETIC,
            unit="V/m",
        ),
        propagation=AxisDirection(Axis.Z, Direction.POSITIVE),
        magnetic_convention=MagneticFieldConvention.ETA0_H,
        normalization=ModeNormalization(target_power_watts=target_power),
        solver=solver,
        transfer=transfer,
    )


def make_fdtdx_mode_function(
    bundle: ModeBundle,
) -> Callable[..., tuple[ArrayLike, ArrayLike]]:
    """Return the callable consumed by FDTDX ``CustomModeOverlapDetector``."""

    import numpy as np

    expected_centers = []
    for edges in bundle.electric.grid.edge_coordinates:
        values = np.asarray(edges)
        expected_centers.append(0.5 * (values[:-1] + values[1:]))

    def mode_function(
        *,
        coordinates: tuple[ArrayLike, ArrayLike, ArrayLike],
        frequency: float,
        propagation_axis: int,
        inv_permittivity: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        import jax.numpy as jnp
        import numpy as np

        if propagation_axis != 2:
            raise ContractError("FDTDX mode callback propagation axis differs from ModeBundle")
        if not math.isclose(
            float(frequency),
            bundle.frequency_hz,
            rel_tol=32.0 * np.finfo(np.float64).eps,
            abs_tol=0.0,
        ):
            raise ContractError("FDTDX mode callback frequency differs from ModeBundle")
        expected_shape = bundle.electric.grid.shape
        inverse_permittivity_shape = shape_of(inv_permittivity)
        if (
            len(inverse_permittivity_shape) != 4
            or inverse_permittivity_shape[0] not in (1, 3, 9)
            or inverse_permittivity_shape[1:] != expected_shape
        ):
            raise ContractError(
                "FDTDX mode callback inverse permittivity must have 1, 3, or 9 components "
                f"and spatial shape {expected_shape}"
            )
        if len(coordinates) != 3 or any(
            shape_of(values) != expected_shape for values in coordinates
        ):
            raise ContractError("FDTDX mode callback coordinates do not match the Yee grid shape")
        expected_mesh = np.meshgrid(*expected_centers, indexing="ij")
        for axis, (actual, expected) in enumerate(zip(coordinates, expected_mesh, strict=True)):
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise ContractError(f"FDTDX mode callback center coordinates differ on axis {axis}")
        return jnp.asarray(bundle.electric.values), jnp.asarray(bundle.magnetic.values)

    return mode_function


__all__ = [
    "SamplingAmbiguityPolicy",
    "YeeModeSamples",
    "YeePortSamplingPlan",
    "build_yee_grid",
    "build_yee_port_sampling_plan",
    "make_fdtdx_mode_function",
    "port_mode_solution_to_bundle",
    "sample_port_mode_to_yee",
]
