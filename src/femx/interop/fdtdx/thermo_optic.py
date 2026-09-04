"""Differentiable temperature-to-material handoff for the locked FDTDX boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, NamedTuple, Protocol, cast

from femx.core.arrays import ArrayLike, shape_of
from femx.core.errors import ContractError

_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAMPLING_SCHEMA = "femx.transfer.triangle_p1_to_fdtdx_cell_centers/v1"
_THERMO_OPTIC_SCHEMA = "femx.fdtdx.thermo_optic_parameter/v1"


def _require_trimmed(value: str, *, label: str) -> None:
    if not value or value.strip() != value:
        raise ContractError(f"{label} must be non-empty and trimmed")


def _require_sha256(value: str, *, label: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FDTDXFingerprint:
    """Exact FDTDX package and source identity attested by the source harness."""

    package_version: str
    source_revision: str
    source_digest: str

    def __post_init__(self) -> None:
        _require_trimmed(self.package_version, label="FDTDX package version")
        if not _GIT_REVISION_PATTERN.fullmatch(self.source_revision):
            raise ContractError("FDTDX source revision must be a lowercase 40-character Git id")
        _require_sha256(self.source_digest, label="FDTDX source digest")


@dataclass(frozen=True, slots=True)
class ThermoOpticLaw:
    r"""Lossless isotropic law ``n(T)=n_ref + dn_dT*(T-T_ref)`` at one wavelength."""

    material_region: str
    reference_temperature_k: float
    reference_refractive_index: float
    thermo_optic_coefficient_per_k: float
    vacuum_wavelength_m: float
    schema_version: str = "femx.thermo_optic.linear_index/v1"

    def __post_init__(self) -> None:
        _require_trimmed(self.material_region, label="thermo-optic material region")
        for label, value in (
            ("reference temperature", self.reference_temperature_k),
            ("reference refractive index", self.reference_refractive_index),
            ("thermo-optic coefficient", self.thermo_optic_coefficient_per_k),
            ("vacuum wavelength", self.vacuum_wavelength_m),
        ):
            if not math.isfinite(value):
                raise ContractError(f"thermo-optic {label} must be finite")
        if self.reference_temperature_k <= 0.0:
            raise ContractError("thermo-optic reference temperature must be positive")
        if self.reference_refractive_index <= 0.0:
            raise ContractError("thermo-optic reference refractive index must be positive")
        if self.vacuum_wavelength_m <= 0.0:
            raise ContractError("thermo-optic vacuum wavelength must be positive")
        if self.schema_version != "femx.thermo_optic.linear_index/v1":
            raise ContractError(f"unsupported thermo-optic law schema {self.schema_version!r}")

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic physical-law metadata."""

        return {
            "schema_version": self.schema_version,
            "material_region": self.material_region,
            "formula": "epsilon_r=(n_ref+dn_dT*(temperature-reference_temperature))**2",
            "reference_temperature_K": self.reference_temperature_k,
            "reference_refractive_index": self.reference_refractive_index,
            "thermo_optic_coefficient_per_K": self.thermo_optic_coefficient_per_k,
            "vacuum_wavelength_m": self.vacuum_wavelength_m,
            "loss_model": "none",
        }

    @property
    def sha256(self) -> str:
        """Hash the complete wavelength-specific physical law."""

        payload = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TriangleP1SamplingPlan:
    """A hashed P1 sampling operator from a 2D triangle mesh to a 3D FDTDX grid."""

    source_coordinates: ArrayLike
    source_cells: ArrayLike
    target_coordinates: tuple[ArrayLike, ArrayLike, ArrayLike]
    plane_axes: tuple[int, int]
    target_cell_indices: ArrayLike
    barycentric_weights: ArrayLike
    source_mesh_sha256: str
    target_coordinate_sha256: str
    operator_sha256: str
    containment_tolerance: float
    maximum_partition_error: float
    minimum_barycentric_weight: float
    schema_version: str = _SAMPLING_SCHEMA

    def __post_init__(self) -> None:
        coordinate_shape = shape_of(self.source_coordinates)
        cell_shape = shape_of(self.source_cells)
        if len(coordinate_shape) != 2 or coordinate_shape[1] != 2:
            raise ContractError(
                f"P1 sampling source coordinates must have shape (nodes, 2), got {coordinate_shape}"
            )
        if len(cell_shape) != 2 or cell_shape[1] != 3:
            raise ContractError(
                f"P1 sampling source cells must have shape (cells, 3), got {cell_shape}"
            )
        if cell_shape[0] <= 0:
            raise ContractError("P1 sampling requires at least one source triangle")
        if len(self.plane_axes) != 2 or len(set(self.plane_axes)) != 2:
            raise ContractError("P1 sampling plane_axes must contain two distinct axes")
        if any(axis < 0 or axis > 2 for axis in self.plane_axes):
            raise ContractError("P1 sampling plane axes must lie in the FDTDX x/y/z range")
        target_shape = tuple(shape_of(axis)[0] for axis in self.target_coordinates)
        if any(
            shape_of(axis) != (length,)
            for axis, length in zip(self.target_coordinates, target_shape, strict=True)
        ):
            raise ContractError("FDTDX target coordinates must be one-dimensional axes")
        if any(length <= 0 for length in target_shape):
            raise ContractError("FDTDX target coordinate axes cannot be empty")
        if shape_of(self.target_cell_indices) != target_shape:
            raise ContractError("P1 sampling cell-index array does not match the target grid")
        if shape_of(self.barycentric_weights) != (*target_shape, 3):
            raise ContractError("P1 sampling barycentric weights do not match the target grid")
        if getattr(self.source_cells.dtype, "kind", None) not in (None, "i", "u"):
            raise ContractError("P1 sampling source cells must use an integer dtype")
        if getattr(self.target_cell_indices.dtype, "kind", None) not in (None, "i", "u"):
            raise ContractError("P1 sampling cell indices must use an integer dtype")
        for digest, label in (
            (self.source_mesh_sha256, "P1 sampling source mesh digest"),
            (self.target_coordinate_sha256, "P1 sampling target-coordinate digest"),
            (self.operator_sha256, "P1 sampling operator digest"),
        ):
            _require_sha256(digest, label=label)
        for label, value in (
            ("containment tolerance", self.containment_tolerance),
            ("maximum partition error", self.maximum_partition_error),
            ("minimum barycentric weight", self.minimum_barycentric_weight),
        ):
            if not math.isfinite(value):
                raise ContractError(f"P1 sampling {label} must be finite")
        if self.containment_tolerance <= 0.0:
            raise ContractError("P1 sampling containment tolerance must be positive")
        if self.maximum_partition_error < 0.0:
            raise ContractError("P1 sampling partition error cannot be negative")
        if self.minimum_barycentric_weight < -self.containment_tolerance:
            raise ContractError("P1 sampling plan contains a point outside its assigned triangle")
        if self.schema_version != _SAMPLING_SCHEMA:
            raise ContractError(f"unsupported P1 sampling schema {self.schema_version!r}")

    @property
    def target_shape(self) -> tuple[int, int, int]:
        """Return FDTDX x/y/z cell counts."""

        return tuple(int(axis.shape[0]) for axis in self.target_coordinates)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FDTDXDeviceParameterContract:
    """Static contract for one raw continuous FDTDX thermo-optic device."""

    device_name: str
    target_shape: tuple[int, int, int]
    plane_axes: tuple[int, int]
    lower_relative_permittivity: float
    upper_relative_permittivity: float
    parameter_dtype: str
    thermo_optic_law_sha256: str
    target_coordinate_sha256: str
    transfer_operator_sha256: str
    fdtdx: FDTDXFingerprint
    schema_version: str = _THERMO_OPTIC_SCHEMA

    def __post_init__(self) -> None:
        _require_trimmed(self.device_name, label="FDTDX thermo-optic device name")
        if len(self.target_shape) != 3 or any(size <= 0 for size in self.target_shape):
            raise ContractError("FDTDX thermo-optic target shape must contain three positive sizes")
        if len(self.plane_axes) != 2 or len(set(self.plane_axes)) != 2:
            raise ContractError("FDTDX thermo-optic plane_axes must contain two distinct axes")
        if any(axis < 0 or axis > 2 for axis in self.plane_axes):
            raise ContractError("FDTDX thermo-optic plane axes must lie in the x/y/z range")
        for label, value in (
            ("lower relative permittivity", self.lower_relative_permittivity),
            ("upper relative permittivity", self.upper_relative_permittivity),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"FDTDX thermo-optic {label} must be finite and positive")
        if self.lower_relative_permittivity >= self.upper_relative_permittivity:
            raise ContractError("FDTDX thermo-optic permittivity bracket must be strictly ordered")
        if self.parameter_dtype not in {"float32", "float64"}:
            raise ContractError("FDTDX thermo-optic parameter dtype must be float32 or float64")
        _require_sha256(
            self.thermo_optic_law_sha256,
            label="FDTDX thermo-optic physical-law digest",
        )
        _require_sha256(
            self.target_coordinate_sha256,
            label="FDTDX thermo-optic target-coordinate digest",
        )
        _require_sha256(
            self.transfer_operator_sha256,
            label="FDTDX thermo-optic transfer-operator digest",
        )
        if self.schema_version != _THERMO_OPTIC_SCHEMA:
            raise ContractError(
                f"unsupported FDTDX thermo-optic parameter schema {self.schema_version!r}"
            )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic FDTDX handoff metadata."""

        return {
            "schema_version": self.schema_version,
            "device_name": self.device_name,
            "target_shape_xyz": list(self.target_shape),
            "plane_axes": list(self.plane_axes),
            "parameter_semantics": "linear_relative_permittivity_fraction",
            "lower_relative_permittivity": self.lower_relative_permittivity,
            "upper_relative_permittivity": self.upper_relative_permittivity,
            "parameter_dtype": self.parameter_dtype,
            "thermo_optic_law_sha256": self.thermo_optic_law_sha256,
            "target_coordinate_sha256": self.target_coordinate_sha256,
            "transfer_operator_sha256": self.transfer_operator_sha256,
            "fdtdx_package_version": self.fdtdx.package_version,
            "fdtdx_source_revision": self.fdtdx.source_revision,
            "fdtdx_source_digest": self.fdtdx.source_digest,
            "out_of_range_policy": "nan_no_clipping",
            "source_overlap_policy": "forbidden",
        }


class ThermoOpticParameterState(NamedTuple):
    """JAX-compatible dynamic values at the FDTDX material boundary."""

    sampled_temperature_k: ArrayLike
    refractive_index: ArrayLike
    relative_permittivity: ArrayLike
    parameter: ArrayLike
    valid_cells: ArrayLike
    all_valid: ArrayLike


def _hash_array(hasher: Any, label: str, values: Any, *, dtype: str) -> None:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
    hasher.update(label.encode("ascii"))
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.tobytes(order="C"))


def _mesh_digest(coordinates: Any, cells: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"femx.triangle_p1_mesh/v1")
    _hash_array(hasher, "coordinates_m", coordinates, dtype="<f8")
    _hash_array(hasher, "cells", cells, dtype="<i8")
    return hasher.hexdigest()


def target_coordinate_digest(target_coordinates: Sequence[Any]) -> str:
    """Hash three FDTDX cell-center coordinate axes in SI metres."""

    if len(target_coordinates) != 3:
        raise ContractError("FDTDX target coordinates require exactly three axes")
    hasher = hashlib.sha256()
    hasher.update(b"femx.fdtdx.cell_centers_xyz/v1")
    for axis, values in enumerate(target_coordinates):
        _hash_array(hasher, f"axis_{axis}_m", values, dtype="<f8")
    return hasher.hexdigest()


def build_triangle_p1_sampling_plan(
    source_coordinates: Any,
    source_cells: Any,
    target_coordinates: Sequence[Any],
    *,
    plane_axes: tuple[int, int],
    containment_tolerance: float = 1.0e-12,
) -> TriangleP1SamplingPlan:
    """Build a deterministic dense reference sampler for a 2D field extruded on one FDTDX axis."""

    import numpy as np

    coordinates = np.asarray(source_coordinates)
    cells = np.asarray(source_cells)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ContractError("P1 sampling source coordinates must have shape (nodes, 2)")
    if coordinates.dtype.kind not in "fiu" or not np.isfinite(coordinates).all():
        raise ContractError("P1 sampling source coordinates must be finite real values")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] == 0:
        raise ContractError("P1 sampling source cells must have non-empty shape (cells, 3)")
    if cells.dtype.kind not in "iu":
        raise ContractError("P1 sampling source cells must use an integer dtype")
    cells = np.asarray(cells, dtype=np.int64)
    if np.any(cells < 0) or np.any(cells >= coordinates.shape[0]):
        raise ContractError("P1 sampling source connectivity is outside the node range")
    if coordinates.shape[0] > np.iinfo(np.int32).max or cells.shape[0] > np.iinfo(np.int32).max:
        raise ContractError("P1 sampling reference operator requires int32-addressable meshes")
    if (
        len(plane_axes) != 2
        or len(set(plane_axes)) != 2
        or any(axis not in (0, 1, 2) for axis in plane_axes)
    ):
        raise ContractError("P1 sampling plane_axes must contain two distinct x/y/z axes")
    if not math.isfinite(containment_tolerance) or containment_tolerance <= 0.0:
        raise ContractError("P1 sampling containment tolerance must be finite and positive")
    if len(target_coordinates) != 3:
        raise ContractError("FDTDX target coordinates require exactly three axes")
    target_axes: list[Any] = []
    for axis, raw_values in enumerate(target_coordinates):
        values = np.asarray(raw_values)
        if values.ndim != 1 or values.size == 0:
            raise ContractError(f"FDTDX target axis {axis} must be a non-empty vector")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ContractError(f"FDTDX target axis {axis} must contain finite real values")
        values = np.asarray(values, dtype=np.float64)
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ContractError(f"FDTDX target axis {axis} must be strictly increasing")
        target_axes.append(values)

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
        threshold = 64.0 * np.finfo(np.float64).eps * max(scale * scale, np.finfo(float).tiny)
        if abs(determinant) <= threshold:
            raise ContractError(f"P1 sampling source triangle {cell_id} is degenerate")

    grids = np.meshgrid(*target_axes, indexing="ij")
    target_points = np.stack(
        (grids[plane_axes[0]].reshape(-1), grids[plane_axes[1]].reshape(-1)),
        axis=1,
    )
    assigned_cells = np.full((target_points.shape[0],), -1, dtype=np.int64)
    assigned_weights = np.zeros((target_points.shape[0], 3), dtype=np.float64)
    for cell_id, points in enumerate(source_points):
        basis = np.column_stack((points[1] - points[0], points[2] - points[0]))
        local = (target_points - points[0]) @ np.linalg.inv(basis).T
        weights = np.column_stack((1.0 - local[:, 0] - local[:, 1], local[:, 0], local[:, 1]))
        inside = (assigned_cells < 0) & np.all(weights >= -containment_tolerance, axis=1)
        inside &= np.all(weights <= 1.0 + containment_tolerance, axis=1)
        assigned_cells[inside] = cell_id
        assigned_weights[inside] = weights[inside]
    missing = np.flatnonzero(assigned_cells < 0)
    if missing.size:
        first = target_points[int(missing[0])]
        raise ContractError(
            f"P1 sampling target grid has {missing.size} point(s) outside the source mesh; "
            f"first planar coordinate={first.tolist()}"
        )

    target_shape = tuple(int(axis.size) for axis in target_axes)
    assigned_cells = assigned_cells.reshape(target_shape)
    assigned_weights = assigned_weights.reshape((*target_shape, 3))
    partition_error = float(np.max(np.abs(np.sum(assigned_weights, axis=-1) - 1.0)))
    minimum_weight = float(np.min(assigned_weights))
    source_digest = _mesh_digest(coordinates, cells)
    target_digest = target_coordinate_digest(target_axes)
    operator_hasher = hashlib.sha256()
    operator_hasher.update(_SAMPLING_SCHEMA.encode("ascii"))
    operator_hasher.update(source_digest.encode("ascii"))
    operator_hasher.update(target_digest.encode("ascii"))
    operator_hasher.update(str(plane_axes).encode("ascii"))
    _hash_array(operator_hasher, "cell_indices", assigned_cells, dtype="<i8")
    _hash_array(operator_hasher, "barycentric_weights", assigned_weights, dtype="<f8")
    return TriangleP1SamplingPlan(
        source_coordinates=coordinates,
        source_cells=cells,
        target_coordinates=cast(tuple[ArrayLike, ArrayLike, ArrayLike], tuple(target_axes)),
        plane_axes=plane_axes,
        target_cell_indices=assigned_cells,
        barycentric_weights=assigned_weights,
        source_mesh_sha256=source_digest,
        target_coordinate_sha256=target_digest,
        operator_sha256=operator_hasher.hexdigest(),
        containment_tolerance=containment_tolerance,
        maximum_partition_error=partition_error,
        minimum_barycentric_weight=minimum_weight,
    )


def sample_triangle_p1(plan: TriangleP1SamplingPlan, nodal_values: ArrayLike) -> ArrayLike:
    """Apply the sampling plan with JAX so its transpose participates in reverse mode."""

    import jax.numpy as jnp

    value_shape = shape_of(nodal_values)
    expected_shape = (int(plan.source_coordinates.shape[0]),)
    if value_shape != expected_shape:
        raise ContractError(f"P1 nodal values must have shape {expected_shape}, got {value_shape}")
    if getattr(nodal_values.dtype, "kind", None) not in (None, "f"):
        raise ContractError("P1 nodal values must use a real floating dtype")
    values = jnp.asarray(nodal_values)
    cells = jnp.asarray(plan.source_cells, dtype=jnp.int32)
    cell_indices = jnp.asarray(plan.target_cell_indices, dtype=jnp.int32)
    weights = jnp.asarray(plan.barycentric_weights, dtype=values.dtype)
    local_values = values[cells[cell_indices]]
    return jnp.sum(weights * local_values, axis=-1)


def thermo_optic_parameter_state(
    plan: TriangleP1SamplingPlan,
    nodal_temperature_k: ArrayLike,
    law: ThermoOpticLaw,
    contract: FDTDXDeviceParameterContract,
) -> ThermoOpticParameterState:
    """Sample temperature and form the raw continuous parameter expected by FDTDX."""

    import jax.numpy as jnp

    if contract.target_shape != plan.target_shape:
        raise ContractError("thermo-optic contract target shape differs from the sampling plan")
    if contract.plane_axes != plan.plane_axes:
        raise ContractError("thermo-optic contract plane axes differ from the sampling plan")
    if contract.target_coordinate_sha256 != plan.target_coordinate_sha256:
        raise ContractError(
            "thermo-optic contract target coordinates differ from the sampling plan"
        )
    if contract.transfer_operator_sha256 != plan.operator_sha256:
        raise ContractError(
            "thermo-optic contract transfer operator differs from the sampling plan"
        )
    if contract.thermo_optic_law_sha256 != law.sha256:
        raise ContractError("thermo-optic physical law differs from the FDTDX contract")
    temperature = jnp.asarray(sample_triangle_p1(plan, nodal_temperature_k))
    refractive_index = law.reference_refractive_index + law.thermo_optic_coefficient_per_k * (
        temperature - law.reference_temperature_k
    )
    relative_permittivity = refractive_index**2
    bracket_width = contract.upper_relative_permittivity - contract.lower_relative_permittivity
    fraction = (relative_permittivity - contract.lower_relative_permittivity) / bracket_width
    valid_cells = (
        jnp.isfinite(temperature)
        & jnp.isfinite(refractive_index)
        & jnp.isfinite(relative_permittivity)
        & (refractive_index > 0.0)
        & (fraction >= 0.0)
        & (fraction <= 1.0)
    )
    parameter = jnp.asarray(
        jnp.where(valid_cells, fraction, jnp.nan),
        dtype=jnp.dtype(contract.parameter_dtype),
    )
    return ThermoOpticParameterState(
        sampled_temperature_k=temperature,
        refractive_index=refractive_index,
        relative_permittivity=relative_permittivity,
        parameter=parameter,
        valid_cells=valid_cells,
        all_valid=jnp.all(valid_cells),
    )


def _dtype_name(value: ArrayLike) -> str:
    name = getattr(value.dtype, "name", None)
    return str(name if name is not None else value.dtype)


def with_fdtdx_device_parameter(
    parameters: Mapping[str, object],
    state: ThermoOpticParameterState,
    contract: FDTDXDeviceParameterContract,
) -> dict[str, object]:
    """Return a copied FDTDX parameter container with exactly one device value replaced."""

    if contract.device_name not in parameters:
        raise ContractError(f"FDTDX parameter container has no device {contract.device_name!r}")
    existing = parameters[contract.device_name]
    if isinstance(existing, Mapping):
        raise ContractError(
            "raw thermo-optic FDTDX device parameter must be one array, not a mapping"
        )
    if not isinstance(existing, ArrayLike):
        raise ContractError("raw thermo-optic FDTDX device parameter is not array-like")
    if shape_of(existing) != contract.target_shape:
        raise ContractError("existing FDTDX device parameter shape differs from the contract")
    if _dtype_name(existing) != contract.parameter_dtype:
        raise ContractError("existing FDTDX device parameter dtype differs from the contract")
    if shape_of(state.parameter) != contract.target_shape:
        raise ContractError("thermo-optic FDTDX parameter shape differs from the contract")
    if _dtype_name(state.parameter) != contract.parameter_dtype:
        raise ContractError("thermo-optic FDTDX parameter dtype differs from the contract")
    updated = dict(parameters)
    updated[contract.device_name] = state.parameter
    return updated


class _ApplyParams(Protocol):
    def __call__(
        self,
        arrays: object,
        objects: object,
        params: Mapping[str, object],
        key: object | None = None,
    ) -> tuple[object, object, dict[str, Any]]: ...


def _require_isotropic_bracket_materials(
    device: object, contract: FDTDXDeviceParameterContract
) -> None:
    materials = getattr(device, "materials", None)
    if not isinstance(materials, dict) or len(materials) != 2:
        raise ContractError("thermo-optic FDTDX device must contain exactly two materials")
    permittivities: list[float] = []
    for material in materials.values():
        permittivity = tuple(float(value) for value in getattr(material, "permittivity", ()))
        permeability = tuple(float(value) for value in getattr(material, "permeability", ()))
        electric_conductivity = tuple(
            float(value) for value in getattr(material, "electric_conductivity", ())
        )
        magnetic_conductivity = tuple(
            float(value) for value in getattr(material, "magnetic_conductivity", ())
        )
        isotropic_permittivity = (
            len(permittivity) == 9
            and permittivity[0] == permittivity[4] == permittivity[8]
            and all(permittivity[index] == 0.0 for index in (1, 2, 3, 5, 6, 7))
        )
        vacuum_permeability = (
            len(permeability) == 9
            and permeability[0] == permeability[4] == permeability[8] == 1.0
            and all(permeability[index] == 0.0 for index in (1, 2, 3, 5, 6, 7))
        )
        if not isotropic_permittivity:
            raise ContractError("thermo-optic FDTDX bracket materials must be isotropic")
        if not vacuum_permeability:
            raise ContractError("thermo-optic FDTDX bracket materials must be non-magnetic")
        if len(electric_conductivity) != 9 or any(value != 0.0 for value in electric_conductivity):
            raise ContractError("thermo-optic FDTDX bracket materials cannot be conductive")
        if len(magnetic_conductivity) != 9 or any(value != 0.0 for value in magnetic_conductivity):
            raise ContractError("thermo-optic FDTDX bracket materials cannot be magnetically lossy")
        if getattr(material, "dispersion", None) is not None:
            raise ContractError("thermo-optic FDTDX v1 bracket materials cannot be dispersive")
        permittivities.append(permittivity[0])
    if tuple(sorted(permittivities)) != (
        contract.lower_relative_permittivity,
        contract.upper_relative_permittivity,
    ):
        raise ContractError("FDTDX device materials do not match the permittivity bracket")


def _validate_fdtdx_runtime_boundary(
    arrays: object,
    objects: object,
    config: object,
    contract: FDTDXDeviceParameterContract,
) -> None:
    import numpy as np

    devices = getattr(objects, "devices", None)
    if not isinstance(devices, list):
        raise ContractError("FDTDX object container does not expose its device list")
    matches = [
        device for device in devices if getattr(device, "name", None) == contract.device_name
    ]
    if len(matches) != 1:
        raise ContractError(
            f"FDTDX object container must contain exactly one device {contract.device_name!r}"
        )
    device = matches[0]
    if tuple(getattr(device, "matrix_voxel_grid_shape", ())) != contract.target_shape:
        raise ContractError("FDTDX device parameter grid shape differs from the contract")
    if tuple(getattr(device, "single_voxel_grid_shape", ())) != (1, 1, 1):
        raise ContractError("thermo-optic FDTDX v1 requires one design voxel per simulation cell")
    if getattr(device, "use_etching", None) is not False:
        raise ContractError("thermo-optic FDTDX v1 does not support etching devices")
    if tuple(getattr(device, "param_transforms", ())) != ():
        raise ContractError("thermo-optic FDTDX v1 requires a raw continuous device")
    _require_isotropic_bracket_materials(device, contract)

    grid_slice_tuple = getattr(device, "grid_slice_tuple", None)
    if not isinstance(grid_slice_tuple, tuple) or len(grid_slice_tuple) != 3:
        raise ContractError("FDTDX thermo-optic device has no resolved three-axis grid slice")
    sources = getattr(objects, "sources", None)
    if not isinstance(sources, list):
        raise ContractError("FDTDX object container does not expose its source list")
    for source in sources:
        source_slice = getattr(source, "grid_slice_tuple", None)
        if (
            not isinstance(source_slice, tuple)
            or len(source_slice) != 3
            or any(not isinstance(bounds, tuple) or len(bounds) != 2 for bounds in source_slice)
        ):
            raise ContractError("FDTDX source has no resolved three-axis grid slice")
        overlaps = all(
            max(device_bounds[0], source_bounds[0]) < min(device_bounds[1], source_bounds[1])
            for device_bounds, source_bounds in zip(
                grid_slice_tuple,
                source_slice,
                strict=True,
            )
        )
        if overlaps:
            source_name = getattr(source, "name", "<unnamed>")
            raise ContractError(
                f"FDTDX source {source_name!r} overlaps the active thermo-optic device; "
                "v1 requires sources outside the parameterized material region"
            )
    resolved_grid = getattr(config, "resolved_grid", None)
    if resolved_grid is None or not callable(getattr(resolved_grid, "centers", None)):
        raise ContractError("FDTDX thermo-optic application requires a resolved grid")
    axes: list[Any] = []
    for axis, bounds in enumerate(grid_slice_tuple):
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ContractError("FDTDX thermo-optic device grid bounds are invalid")
        lower, upper = bounds
        axes.append(np.asarray(resolved_grid.centers(axis)[lower:upper]))
    if target_coordinate_digest(axes) != contract.target_coordinate_sha256:
        raise ContractError("resolved FDTDX device coordinates differ from the transfer contract")
    inverse_permittivity = getattr(arrays, "inv_permittivities", None)
    if not isinstance(inverse_permittivity, ArrayLike):
        raise ContractError("FDTDX array container has no inverse-permittivity array")
    if len(shape_of(inverse_permittivity)) != 4 or inverse_permittivity.shape[0] != 1:
        raise ContractError("thermo-optic FDTDX v1 requires isotropic inverse permittivity")


def apply_thermo_optic_to_fdtdx(
    arrays: object,
    objects: object,
    parameters: Mapping[str, object],
    config: object,
    state: ThermoOpticParameterState,
    contract: FDTDXDeviceParameterContract,
    *,
    verified_fingerprint: FDTDXFingerprint,
    key: object | None = None,
) -> tuple[object, object, dict[str, Any]]:
    """Validate the locked runtime and invoke FDTDX's public differentiable ``apply_params``."""

    if verified_fingerprint != contract.fdtdx:
        raise ContractError("verified FDTDX source identity differs from the transfer contract")
    try:
        installed_version = package_version("fdtdx")
        module = import_module("fdtdx")
    except (ModuleNotFoundError, PackageNotFoundError) as error:
        raise ContractError("FDTDX is not installed in the active environment") from error
    if installed_version != contract.fdtdx.package_version:
        raise ContractError(
            f"FDTDX package version mismatch: expected {contract.fdtdx.package_version!r}, "
            f"got {installed_version!r}"
        )
    _validate_fdtdx_runtime_boundary(arrays, objects, config, contract)
    updated_parameters = with_fdtdx_device_parameter(parameters, state, contract)
    candidate = getattr(module, "apply_params", None)
    if not callable(candidate):
        raise ContractError("installed FDTDX does not expose callable apply_params")
    apply_params = cast(_ApplyParams, candidate)
    return apply_params(
        arrays=arrays,
        objects=objects,
        params=updated_parameters,
        key=key,
    )
