"""One-way distributed Tet4 current, cell-local Joule, and steady heat solve.

The electrical conductor is a compact submesh of the complete thermal mesh.  Preparation keeps
the exact parent cell and node identities and requires every conductor cell to retain its thermal
owner.  Joule density therefore transfers through process-local packed-cell slots; it is neither
interpolated nor globally gathered.  This module is an internal M5 numerical boundary and does not
register a public three-dimensional backend or establish Elmer, TPU, or device-physics evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from femx.backends.jax.elements.tetrahedron_h1 import (
    tetrahedron_p1_diffusion_cell_matrices,
    tetrahedron_p1_geometry,
)
from femx.backends.jax.owned_ghost import _canonical_int64_array
from femx.backends.jax.scalar_cg import (
    PackedScalarH1CGResult,
    PackedScalarH1PreconditionerFactory,
    ScalarH1CGPolicy,
    build_packed_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (
    SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA,
    ScalarH1CollectiveLayout,
    build_packed_collective_scalar_h1_cell_gather,
    build_packed_collective_scalar_h1_rhs_assembly,
    prepare_collective_scalar_h1_layout,
    prepare_scalar_h1_boundary_facet_map,
    reconstruct_scalar_h1_state,
    tetrahedron_p1_scalar_cell_load_vectors,
    tetrahedron_p1_scalar_robin_cell_terms,
    unpack_collective_scalar_h1_owned_vector,
)
from femx.backends.jax.scalar_owned_ghost import prepare_scalar_h1_owned_ghost_topology
from femx.core.errors import ContractError

TET4_ELECTROTHERMAL_SCHEMA = "femx.jax.tet4_electrothermal/v1"
TET4_ELECTROTHERMAL_RUNTIME_PLAN_SCHEMA = "femx.jax.tet4_electrothermal.runtime_plan/v1"


def _readonly(values: object, *, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _float64_array(
    values: object,
    *,
    label: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be a regular real array") from error
    if raw.dtype.kind != "f" or raw.shape != shape:
        raise ContractError(f"{label} must be a real array shaped {shape}")
    result = _readonly(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"{label} must be finite")
    return result


def _canonical_faces(
    values: object,
    *,
    label: str,
    node_count: int,
) -> np.ndarray:
    faces = _canonical_int64_array(values, label=label, rank=2, columns=3)
    if np.any(faces < 0) or np.any(faces >= node_count):
        raise ContractError(f"{label} contains an out-of-range node")
    if any(np.unique(face).shape[0] != 3 for face in faces):
        raise ContractError(f"{label} cannot repeat a node")
    return faces


def _canonical_dirichlet(
    nodes: object,
    base_values: object,
    parameter_scales: object,
    *,
    label: str,
    node_count: int,
    require_nonempty: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical_nodes = _canonical_int64_array(nodes, label=f"{label} nodes", rank=1)
    if require_nonempty and canonical_nodes.size == 0:
        raise ContractError(f"{label} requires at least one node")
    if np.any(canonical_nodes < 0) or np.any(canonical_nodes >= node_count):
        raise ContractError(f"{label} contains an out-of-range node")
    if np.unique(canonical_nodes).shape[0] != canonical_nodes.shape[0]:
        raise ContractError(f"{label} nodes must be unique")
    base = _float64_array(
        base_values,
        label=f"{label} base values",
        shape=canonical_nodes.shape,
    )
    scales = _float64_array(
        parameter_scales,
        label=f"{label} parameter scales",
        shape=canonical_nodes.shape,
    )
    order = np.argsort(canonical_nodes, kind="stable")
    return (
        _readonly(canonical_nodes[order], dtype=np.int64),
        _readonly(base[order], dtype=np.float64),
        _readonly(scales[order], dtype=np.float64),
    )


def _require_positive_tet4_geometry(coordinates: np.ndarray, cells: np.ndarray) -> None:
    points = coordinates[cells]
    jacobians = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    if not np.all(np.isfinite(determinants)) or np.any(determinants <= 0.0):
        raise ContractError("Tet4 electrothermal cells must have finite positive orientation")


def _require_anchored_components(
    cells: np.ndarray,
    constrained_nodes: np.ndarray,
    *,
    node_count: int,
    label: str,
) -> None:
    parent = np.arange(node_count, dtype=np.int64)

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for cell in cells:
        anchor = root(int(cell[0]))
        for node_value in cell[1:]:
            other = root(int(node_value))
            if anchor != other:
                parent[other] = anchor
                anchor = root(anchor)
    anchored = {root(int(node)) for node in constrained_nodes}
    if any(root(node) not in anchored for node in range(node_count)):
        raise ContractError(f"{label} has an unanchored connected component")


def _expected_transfer_slots(
    current_layout: ScalarH1CollectiveLayout,
    thermal_layout: ScalarH1CollectiveLayout,
    current_parent_cell_ids: np.ndarray,
) -> np.ndarray:
    slots = np.zeros(
        (current_layout.partition_count, current_layout.cell_capacity),
        dtype=np.int64,
    )
    for partition in range(current_layout.partition_count):
        thermal_ids = thermal_layout.transport.cell_ids[partition]
        by_cell = {
            int(cell): slot
            for slot, cell in enumerate(thermal_ids)
            if cell < thermal_layout.topology.cell_count
        }
        for slot, current_cell in enumerate(current_layout.transport.cell_ids[partition]):
            if current_cell >= current_layout.topology.cell_count:
                continue
            parent_cell = int(current_parent_cell_ids[current_cell])
            try:
                slots[partition, slot] = by_cell[parent_cell]
            except KeyError as error:
                raise ContractError(
                    "Tet4 Joule transfer requires identical current and thermal cell owners"
                ) from error
    slots.setflags(write=False)
    return slots


@dataclass(frozen=True, slots=True)
class Tet4ElectrothermalPlan:
    """Host-owned identities and exact local operators for one-way 3D heating."""

    current_layout: ScalarH1CollectiveLayout
    thermal_layout: ScalarH1CollectiveLayout
    current_parent_cell_ids: np.ndarray
    current_parent_node_ids: np.ndarray
    current_conduction_stiffness: np.ndarray
    current_basis_gradients: np.ndarray
    current_cell_volumes: np.ndarray
    current_conductivity: np.ndarray
    current_cell_load: np.ndarray
    current_cell_dirichlet_base: np.ndarray
    current_cell_dirichlet_scale: np.ndarray
    current_dirichlet_base: np.ndarray
    current_dirichlet_scale: np.ndarray
    thermal_conduction_stiffness: np.ndarray
    thermal_robin_matrix: np.ndarray
    thermal_cell_volumes: np.ndarray
    thermal_nonrobin_load: np.ndarray
    thermal_robin_ambient_load: np.ndarray
    thermal_cell_dirichlet_shifted: np.ndarray
    thermal_dirichlet_shifted: np.ndarray
    current_to_thermal_slots: np.ndarray
    thermal_reference: float
    schema_version: str = TET4_ELECTROTHERMAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TET4_ELECTROTHERMAL_SCHEMA:
            raise ContractError(
                f"Tet4 electrothermal schema must be {TET4_ELECTROTHERMAL_SCHEMA!r}"
            )
        for layout, label in (
            (self.current_layout, "current"),
            (self.thermal_layout, "thermal"),
        ):
            if not isinstance(layout, ScalarH1CollectiveLayout):
                raise ContractError(f"Tet4 electrothermal {label} layout must be scalar H1")
            if layout.schema_version != SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA:
                raise ContractError(f"Tet4 electrothermal {label} layout must use Tet4 identity")
        if self.current_layout.partition_count != self.thermal_layout.partition_count:
            raise ContractError("Tet4 electrothermal layouts must have equal partition counts")

        current_cells = self.current_layout.topology.cell_count
        current_nodes = self.current_layout.topology.node_count
        thermal_cells = self.thermal_layout.topology.cell_count
        arrays = {
            "current_parent_cell_ids": (self.current_parent_cell_ids, (current_cells,), np.int64),
            "current_parent_node_ids": (self.current_parent_node_ids, (current_nodes,), np.int64),
            "current_conduction_stiffness": (
                self.current_conduction_stiffness,
                (current_cells, 4, 4),
                np.float64,
            ),
            "current_basis_gradients": (
                self.current_basis_gradients,
                (current_cells, 4, 3),
                np.float64,
            ),
            "current_cell_volumes": (self.current_cell_volumes, (current_cells,), np.float64),
            "current_conductivity": (self.current_conductivity, (current_cells,), np.float64),
            "current_cell_load": (self.current_cell_load, (current_cells, 4), np.float64),
            "current_cell_dirichlet_base": (
                self.current_cell_dirichlet_base,
                (current_cells, 4),
                np.float64,
            ),
            "current_cell_dirichlet_scale": (
                self.current_cell_dirichlet_scale,
                (current_cells, 4),
                np.float64,
            ),
            "current_dirichlet_base": (
                self.current_dirichlet_base,
                (self.current_layout.topology.constrained_nodes.shape[0],),
                np.float64,
            ),
            "current_dirichlet_scale": (
                self.current_dirichlet_scale,
                (self.current_layout.topology.constrained_nodes.shape[0],),
                np.float64,
            ),
            "thermal_conduction_stiffness": (
                self.thermal_conduction_stiffness,
                (thermal_cells, 4, 4),
                np.float64,
            ),
            "thermal_robin_matrix": (
                self.thermal_robin_matrix,
                (thermal_cells, 4, 4),
                np.float64,
            ),
            "thermal_cell_volumes": (self.thermal_cell_volumes, (thermal_cells,), np.float64),
            "thermal_nonrobin_load": (
                self.thermal_nonrobin_load,
                (thermal_cells, 4),
                np.float64,
            ),
            "thermal_robin_ambient_load": (
                self.thermal_robin_ambient_load,
                (thermal_cells, 4),
                np.float64,
            ),
            "thermal_cell_dirichlet_shifted": (
                self.thermal_cell_dirichlet_shifted,
                (thermal_cells, 4),
                np.float64,
            ),
            "thermal_dirichlet_shifted": (
                self.thermal_dirichlet_shifted,
                (self.thermal_layout.topology.constrained_nodes.shape[0],),
                np.float64,
            ),
            "current_to_thermal_slots": (
                self.current_to_thermal_slots,
                (
                    self.current_layout.partition_count,
                    self.current_layout.cell_capacity,
                ),
                np.int64,
            ),
        }
        for name, (value, shape, dtype) in arrays.items():
            raw = np.asarray(value)
            if raw.shape != shape or raw.dtype.kind != np.dtype(dtype).kind:
                raise ContractError(f"Tet4 electrothermal {name.replace('_', ' ')} has wrong shape")
            canonical = _readonly(raw, dtype=dtype)
            if canonical.dtype.kind == "f" and not np.all(np.isfinite(canonical)):
                raise ContractError(f"Tet4 electrothermal {name.replace('_', ' ')} must be finite")
            object.__setattr__(self, name, canonical)
        if np.any(self.current_cell_volumes <= 0.0) or np.any(self.thermal_cell_volumes <= 0.0):
            raise ContractError("Tet4 electrothermal cell volumes must be positive")
        if np.any(self.current_conductivity <= 0.0):
            raise ContractError("Tet4 electrothermal electrical conductivity must be positive")
        if not math.isfinite(self.thermal_reference):
            raise ContractError("Tet4 electrothermal thermal reference must be finite")
        expected_slots = _expected_transfer_slots(
            self.current_layout,
            self.thermal_layout,
            self.current_parent_cell_ids,
        )
        if not np.array_equal(self.current_to_thermal_slots, expected_slots):
            raise ContractError("Tet4 electrothermal transfer slots disagree with cell identity")
        parent_cells = self.thermal_layout.topology.cells[self.current_parent_cell_ids]
        mapped_cells = self.current_parent_node_ids[self.current_layout.topology.cells]
        if not np.array_equal(parent_cells, mapped_cells):
            raise ContractError(
                "Tet4 electrothermal current submesh disagrees with parent identity"
            )

    def digest(self) -> str:
        """Hash topology, parent identities, operators, loads, and reference temperature."""

        metadata = {
            "schema_version": self.schema_version,
            "current_layout_sha256": self.current_layout.digest(),
            "thermal_layout_sha256": self.thermal_layout.digest(),
            "thermal_reference": self.thermal_reference,
        }
        hasher = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for name in self.__dataclass_fields__:
            if name in {"current_layout", "thermal_layout", "thermal_reference", "schema_version"}:
                continue
            array = np.asarray(getattr(self, name))
            hasher.update(name.encode("utf-8"))
            hasher.update(str(array.dtype).encode("ascii"))
            hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
            hasher.update(np.ascontiguousarray(array).tobytes())
        return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class Tet4ElectrothermalRuntimePlan:
    """Minimal immutable topology needed after numerical inputs have been packed.

    Large physical runs persist the partition-leading numerical arrays separately.  Requiring the
    complete host preparation plan on every worker would duplicate the unpadded element operators
    even though the runtime uses only the two collective layouts and the temperature reference.
    This view keeps that distinction explicit and binds the packed arrays to their source plan.
    """

    current_layout: ScalarH1CollectiveLayout
    thermal_layout: ScalarH1CollectiveLayout
    thermal_reference: float
    source_plan_sha256: str
    schema_version: str = TET4_ELECTROTHERMAL_RUNTIME_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TET4_ELECTROTHERMAL_RUNTIME_PLAN_SCHEMA:
            raise ContractError(
                "Tet4 electrothermal runtime-plan schema must be "
                f"{TET4_ELECTROTHERMAL_RUNTIME_PLAN_SCHEMA!r}"
            )
        for layout, label in (
            (self.current_layout, "current"),
            (self.thermal_layout, "thermal"),
        ):
            if not isinstance(layout, ScalarH1CollectiveLayout):
                raise ContractError(
                    f"Tet4 electrothermal runtime-plan {label} layout must be scalar H1"
                )
            if layout.schema_version != SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA:
                raise ContractError(
                    f"Tet4 electrothermal runtime-plan {label} layout must use Tet4 identity"
                )
        if self.current_layout.partition_count != self.thermal_layout.partition_count:
            raise ContractError(
                "Tet4 electrothermal runtime-plan layouts must have equal partition counts"
            )
        if isinstance(self.thermal_reference, bool) or not isinstance(
            self.thermal_reference, (int, float)
        ):
            raise ContractError("Tet4 electrothermal runtime-plan reference must be a real scalar")
        reference = float(self.thermal_reference)
        if not math.isfinite(reference):
            raise ContractError("Tet4 electrothermal runtime-plan reference must be finite")
        if (
            not isinstance(self.source_plan_sha256, str)
            or len(self.source_plan_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_plan_sha256)
        ):
            raise ContractError(
                "Tet4 electrothermal runtime-plan source digest must be a lowercase SHA-256"
            )
        object.__setattr__(self, "thermal_reference", reference)

    def digest(self) -> str:
        """Hash both transport layouts, the reference, and the source preparation identity."""

        payload = {
            "schema_version": self.schema_version,
            "source_plan_sha256": self.source_plan_sha256,
            "current_layout_sha256": self.current_layout.digest(),
            "thermal_layout_sha256": self.thermal_layout.digest(),
            "thermal_reference": self.thermal_reference,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def prepare_tet4_electrothermal_runtime_plan(
    plan: Tet4ElectrothermalPlan,
) -> Tet4ElectrothermalRuntimePlan:
    """Drop host-only element arrays after packing while preserving their source digest."""

    if not isinstance(plan, Tet4ElectrothermalPlan):
        raise ContractError("Tet4 electrothermal runtime-plan preparation requires a full plan")
    return Tet4ElectrothermalRuntimePlan(
        current_layout=plan.current_layout,
        thermal_layout=plan.thermal_layout,
        thermal_reference=plan.thermal_reference,
        source_plan_sha256=plan.digest(),
    )


def prepare_tet4_electrothermal_plan(
    coordinates: object,
    cells: object,
    cell_owners: object,
    current_parent_cell_ids: object,
    *,
    current_conductivity: object,
    current_cell_source: object,
    current_flux_facets: object,
    current_facet_flux: object,
    current_dirichlet_nodes: object,
    current_dirichlet_base: object,
    current_dirichlet_voltage_scale: object,
    thermal_conductivity: object,
    thermal_cell_source: object,
    thermal_flux_facets: object,
    thermal_facet_flux: object,
    thermal_robin_facets: object,
    thermal_robin_transfer: object,
    thermal_robin_ambient: object,
    thermal_dirichlet_nodes: object,
    thermal_dirichlet_values: object,
    thermal_reference: float,
    partition_count: int,
) -> Tet4ElectrothermalPlan:
    """Prepare distinct conductor/thermal spaces without selecting JAX devices."""

    raw_coordinates = np.asarray(coordinates)
    if (
        raw_coordinates.dtype.kind != "f"
        or raw_coordinates.ndim != 2
        or raw_coordinates.shape[1:] != (3,)
    ):
        raise ContractError(
            "Tet4 electrothermal coordinates must be a real array shaped (nodes, 3)"
        )
    canonical_coordinates = _readonly(raw_coordinates, dtype=np.float64)
    if not np.all(np.isfinite(canonical_coordinates)):
        raise ContractError("Tet4 electrothermal coordinates must be finite")
    node_count = canonical_coordinates.shape[0]
    canonical_cells = _canonical_int64_array(
        cells,
        label="Tet4 electrothermal cells",
        rank=2,
        columns=4,
    )
    if canonical_cells.shape[0] == 0:
        raise ContractError("Tet4 electrothermal mesh requires at least one cell")
    if np.any(canonical_cells < 0) or np.any(canonical_cells >= node_count):
        raise ContractError("Tet4 electrothermal cells contain an out-of-range node")
    if any(np.unique(cell).shape[0] != 4 for cell in canonical_cells):
        raise ContractError("Tet4 electrothermal cells cannot repeat a node")
    _require_positive_tet4_geometry(canonical_coordinates, canonical_cells)
    owners = _canonical_int64_array(
        cell_owners,
        label="Tet4 electrothermal cell owners",
        rank=1,
    )
    if owners.shape != (canonical_cells.shape[0],):
        raise ContractError("Tet4 electrothermal cell owners must match the thermal cell count")

    parent_cells = _canonical_int64_array(
        current_parent_cell_ids,
        label="Tet4 electrothermal current parent cells",
        rank=1,
    )
    if parent_cells.size == 0:
        raise ContractError("Tet4 electrothermal current domain requires at least one cell")
    if np.any(parent_cells < 0) or np.any(parent_cells >= canonical_cells.shape[0]):
        raise ContractError("Tet4 electrothermal current parent cell is out of range")
    if parent_cells.size > 1 and np.any(np.diff(parent_cells) <= 0):
        raise ContractError("Tet4 electrothermal current parent cells must be strictly increasing")
    current_parent_nodes = np.unique(canonical_cells[parent_cells]).astype(np.int64)
    full_to_current = np.full(node_count, current_parent_nodes.shape[0], dtype=np.int64)
    full_to_current[current_parent_nodes] = np.arange(current_parent_nodes.shape[0], dtype=np.int64)
    current_cells = full_to_current[canonical_cells[parent_cells]]
    current_coordinates = canonical_coordinates[current_parent_nodes]

    current_sigma = _float64_array(
        current_conductivity,
        label="Tet4 electrothermal current conductivity",
        shape=(parent_cells.shape[0],),
    )
    if np.any(current_sigma <= 0.0):
        raise ContractError("Tet4 electrothermal current conductivity must be positive")
    current_source = _float64_array(
        current_cell_source,
        label="Tet4 electrothermal current cell source",
        shape=(parent_cells.shape[0],),
    )
    current_faces_full = _canonical_faces(
        current_flux_facets,
        label="Tet4 electrothermal current flux facets",
        node_count=node_count,
    )
    if current_faces_full.size and np.any(
        full_to_current[current_faces_full] >= current_parent_nodes.shape[0]
    ):
        raise ContractError(
            "Tet4 electrothermal current flux facet lies outside the current domain"
        )
    current_faces = full_to_current[current_faces_full]
    current_flux = _float64_array(
        current_facet_flux,
        label="Tet4 electrothermal current facet flux",
        shape=(current_faces.shape[0],),
    )
    current_nodes_full, current_base, current_scale = _canonical_dirichlet(
        current_dirichlet_nodes,
        current_dirichlet_base,
        current_dirichlet_voltage_scale,
        label="Tet4 electrothermal current Dirichlet boundary",
        node_count=node_count,
    )
    if np.any(full_to_current[current_nodes_full] >= current_parent_nodes.shape[0]):
        raise ContractError(
            "Tet4 electrothermal current Dirichlet node lies outside the current domain"
        )
    if not np.any(current_scale != 0.0):
        raise ContractError("Tet4 electrothermal current boundary must depend on applied voltage")
    current_nodes = full_to_current[current_nodes_full]
    current_order = np.argsort(current_nodes, kind="stable")
    current_nodes = _readonly(current_nodes[current_order], dtype=np.int64)
    current_base = _readonly(current_base[current_order], dtype=np.float64)
    current_scale = _readonly(current_scale[current_order], dtype=np.float64)
    current_free = np.setdiff1d(
        np.arange(current_parent_nodes.shape[0], dtype=np.int64),
        current_nodes,
        assume_unique=True,
    )
    if current_free.size == 0:
        raise ContractError("Tet4 electrothermal current domain requires a free node")
    _require_anchored_components(
        current_cells,
        current_nodes,
        node_count=current_parent_nodes.shape[0],
        label="Tet4 electrothermal current domain",
    )

    thermal_k = _float64_array(
        thermal_conductivity,
        label="Tet4 electrothermal thermal conductivity",
        shape=(canonical_cells.shape[0],),
    )
    if np.any(thermal_k <= 0.0):
        raise ContractError("Tet4 electrothermal thermal conductivity must be positive")
    thermal_source = _float64_array(
        thermal_cell_source,
        label="Tet4 electrothermal thermal cell source",
        shape=(canonical_cells.shape[0],),
    )
    thermal_flux_faces = _canonical_faces(
        thermal_flux_facets,
        label="Tet4 electrothermal thermal flux facets",
        node_count=node_count,
    )
    thermal_flux = _float64_array(
        thermal_facet_flux,
        label="Tet4 electrothermal thermal facet flux",
        shape=(thermal_flux_faces.shape[0],),
    )
    robin_faces = _canonical_faces(
        thermal_robin_facets,
        label="Tet4 electrothermal Robin facets",
        node_count=node_count,
    )
    robin_transfer = _float64_array(
        thermal_robin_transfer,
        label="Tet4 electrothermal Robin transfer",
        shape=(robin_faces.shape[0],),
    )
    if np.any(robin_transfer < 0.0):
        raise ContractError("Tet4 electrothermal Robin transfer cannot be negative")
    robin_ambient = _float64_array(
        thermal_robin_ambient,
        label="Tet4 electrothermal Robin ambient",
        shape=(robin_faces.shape[0],),
    )
    flux_keys = {tuple(sorted(int(node) for node in face)) for face in thermal_flux_faces}
    robin_keys = {tuple(sorted(int(node) for node in face)) for face in robin_faces}
    if flux_keys & robin_keys:
        raise ContractError("Tet4 electrothermal thermal flux and Robin facets must be disjoint")
    thermal_nodes, thermal_values, thermal_scale = _canonical_dirichlet(
        thermal_dirichlet_nodes,
        thermal_dirichlet_values,
        np.zeros(np.asarray(thermal_dirichlet_nodes).shape, dtype=np.float64),
        label="Tet4 electrothermal thermal Dirichlet boundary",
        node_count=node_count,
        require_nonempty=False,
    )
    del thermal_scale
    thermal_free = np.setdiff1d(
        np.arange(node_count, dtype=np.int64),
        thermal_nodes,
        assume_unique=True,
    )
    if thermal_free.size == 0:
        raise ContractError("Tet4 electrothermal thermal domain requires a free node")
    positive_robin_nodes = np.unique(robin_faces[robin_transfer > 0.0])
    thermal_anchor_nodes = np.union1d(thermal_nodes, positive_robin_nodes).astype(np.int64)
    _require_anchored_components(
        canonical_cells,
        thermal_anchor_nodes,
        node_count=node_count,
        label="Tet4 electrothermal thermal domain",
    )
    if isinstance(thermal_reference, bool) or not isinstance(thermal_reference, (int, float)):
        raise ContractError("Tet4 electrothermal thermal reference must be a real scalar")
    reference = float(thermal_reference)
    if not math.isfinite(reference):
        raise ContractError("Tet4 electrothermal thermal reference must be finite")

    thermal_layout = prepare_collective_scalar_h1_layout(
        prepare_scalar_h1_owned_ghost_topology(
            canonical_cells,
            owners,
            node_count=node_count,
            free_nodes=thermal_free,
            partition_count=partition_count,
        )
    )
    current_layout = prepare_collective_scalar_h1_layout(
        prepare_scalar_h1_owned_ghost_topology(
            current_cells,
            owners[parent_cells],
            node_count=current_parent_nodes.shape[0],
            free_nodes=current_free,
            partition_count=partition_count,
        )
    )

    current_coordinates_jax = jnp.asarray(current_coordinates, dtype=jnp.float64)
    current_cells_jax = jnp.asarray(current_cells)
    current_geometry = tetrahedron_p1_geometry(current_coordinates_jax, current_cells_jax)
    current_stiffness = tetrahedron_p1_diffusion_cell_matrices(
        current_coordinates_jax,
        current_cells_jax,
        jnp.asarray(current_sigma),
    )
    current_boundary_map = prepare_scalar_h1_boundary_facet_map(
        current_cells,
        current_faces,
        node_count=current_parent_nodes.shape[0],
    )
    current_load = tetrahedron_p1_scalar_cell_load_vectors(
        current_coordinates_jax,
        current_cells_jax,
        jnp.asarray(current_source),
        jnp.asarray(current_faces),
        jnp.asarray(current_flux),
        current_boundary_map,
    )
    current_base_nodes = np.zeros((current_parent_nodes.shape[0],), dtype=np.float64)
    current_base_nodes[current_nodes] = current_base
    current_scale_nodes = np.zeros((current_parent_nodes.shape[0],), dtype=np.float64)
    current_scale_nodes[current_nodes] = current_scale

    thermal_coordinates_jax = jnp.asarray(canonical_coordinates, dtype=jnp.float64)
    thermal_cells_jax = jnp.asarray(canonical_cells)
    thermal_geometry = tetrahedron_p1_geometry(thermal_coordinates_jax, thermal_cells_jax)
    thermal_stiffness = tetrahedron_p1_diffusion_cell_matrices(
        thermal_coordinates_jax,
        thermal_cells_jax,
        jnp.asarray(thermal_k),
    )
    thermal_flux_map = prepare_scalar_h1_boundary_facet_map(
        canonical_cells,
        thermal_flux_faces,
        node_count=node_count,
    )
    thermal_load = tetrahedron_p1_scalar_cell_load_vectors(
        thermal_coordinates_jax,
        thermal_cells_jax,
        jnp.asarray(thermal_source),
        jnp.asarray(thermal_flux_faces),
        jnp.asarray(thermal_flux),
        thermal_flux_map,
    )
    robin_map = prepare_scalar_h1_boundary_facet_map(
        canonical_cells,
        robin_faces,
        node_count=node_count,
    )
    robin_matrix, robin_load = tetrahedron_p1_scalar_robin_cell_terms(
        thermal_coordinates_jax,
        thermal_cells_jax,
        jnp.asarray(robin_faces),
        jnp.asarray(robin_transfer),
        jnp.asarray(robin_ambient),
        robin_map,
    )
    thermal_shifted_nodes = np.zeros((node_count,), dtype=np.float64)
    thermal_shifted_nodes[thermal_nodes] = thermal_values - reference
    transfer_slots = _expected_transfer_slots(current_layout, thermal_layout, parent_cells)

    return Tet4ElectrothermalPlan(
        current_layout=current_layout,
        thermal_layout=thermal_layout,
        current_parent_cell_ids=parent_cells,
        current_parent_node_ids=current_parent_nodes,
        current_conduction_stiffness=np.asarray(current_stiffness),
        current_basis_gradients=np.asarray(current_geometry.basis_gradients),
        current_cell_volumes=np.asarray(current_geometry.volumes),
        current_conductivity=current_sigma,
        current_cell_load=np.asarray(current_load),
        current_cell_dirichlet_base=current_base_nodes[current_cells],
        current_cell_dirichlet_scale=current_scale_nodes[current_cells],
        current_dirichlet_base=current_base,
        current_dirichlet_scale=current_scale,
        thermal_conduction_stiffness=np.asarray(thermal_stiffness),
        thermal_robin_matrix=np.asarray(robin_matrix),
        thermal_cell_volumes=np.asarray(thermal_geometry.volumes),
        thermal_nonrobin_load=np.asarray(thermal_load),
        thermal_robin_ambient_load=np.asarray(robin_load),
        thermal_cell_dirichlet_shifted=thermal_shifted_nodes[canonical_cells],
        thermal_dirichlet_shifted=thermal_values - reference,
        current_to_thermal_slots=transfer_slots,
        thermal_reference=reference,
    )


class HostPackedTet4ElectrothermalInputs(NamedTuple):
    """Host arrays before a caller chooses device placement and sharding."""

    current_cell_local_dofs: np.ndarray
    current_owner_mask: np.ndarray
    current_cell_mask: np.ndarray
    current_conduction_stiffness: np.ndarray
    current_basis_gradients: np.ndarray
    current_cell_volumes: np.ndarray
    current_conductivity: np.ndarray
    current_cell_load: np.ndarray
    current_cell_dirichlet_base: np.ndarray
    current_cell_dirichlet_scale: np.ndarray
    current_to_thermal_slots: np.ndarray
    thermal_cell_local_dofs: np.ndarray
    thermal_owner_mask: np.ndarray
    thermal_cell_mask: np.ndarray
    thermal_conduction_stiffness: np.ndarray
    thermal_robin_matrix: np.ndarray
    thermal_cell_volumes: np.ndarray
    thermal_nonrobin_load: np.ndarray
    thermal_robin_ambient_load: np.ndarray
    thermal_cell_dirichlet_shifted: np.ndarray


class PackedTet4ElectrothermalInputs(NamedTuple):
    """Explicit JAX inputs with partition-leading cell and owner arrays."""

    current_cell_local_dofs: jax.Array
    current_owner_mask: jax.Array
    current_cell_mask: jax.Array
    current_conduction_stiffness: jax.Array
    current_basis_gradients: jax.Array
    current_cell_volumes: jax.Array
    current_conductivity: jax.Array
    current_cell_load: jax.Array
    current_cell_dirichlet_base: jax.Array
    current_cell_dirichlet_scale: jax.Array
    current_to_thermal_slots: jax.Array
    thermal_cell_local_dofs: jax.Array
    thermal_owner_mask: jax.Array
    thermal_cell_mask: jax.Array
    thermal_conduction_stiffness: jax.Array
    thermal_robin_matrix: jax.Array
    thermal_cell_volumes: jax.Array
    thermal_nonrobin_load: jax.Array
    thermal_robin_ambient_load: jax.Array
    thermal_cell_dirichlet_shifted: jax.Array


def _pack_cell_array(
    layout: ScalarH1CollectiveLayout,
    values: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    canonical = np.asarray(values, dtype=dtype)
    sentinel = np.zeros((1, *canonical.shape[1:]), dtype=dtype)
    return _readonly(
        np.concatenate((canonical, sentinel), axis=0)[layout.transport.cell_ids],
        dtype=dtype,
    )


def pack_tet4_electrothermal_inputs_host(
    plan: Tet4ElectrothermalPlan,
    *,
    value_dtype: np.dtype | type[np.generic],
) -> HostPackedTet4ElectrothermalInputs:
    """Pack current and thermal arrays without discovering or selecting devices."""

    if not isinstance(plan, Tet4ElectrothermalPlan):
        raise ContractError("Tet4 electrothermal packing requires a prepared plan")
    dtype = np.dtype(value_dtype)
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise ContractError("Tet4 electrothermal values require float32 or float64")
    current = plan.current_layout
    thermal = plan.thermal_layout

    def pack_current(values: np.ndarray) -> np.ndarray:
        return _pack_cell_array(current, values, dtype=dtype)

    def pack_thermal(values: np.ndarray) -> np.ndarray:
        return _pack_cell_array(thermal, values, dtype=dtype)

    return HostPackedTet4ElectrothermalInputs(
        current_cell_local_dofs=_readonly(current.transport.cell_local_dofs, dtype=np.int32),
        current_owner_mask=_readonly(
            current.transport.owned_dof_ids < current.topology.free_dof_count,
            dtype=np.bool_,
        ),
        current_cell_mask=_readonly(
            current.transport.cell_ids < current.topology.cell_count,
            dtype=np.bool_,
        ),
        current_conduction_stiffness=pack_current(plan.current_conduction_stiffness),
        current_basis_gradients=pack_current(plan.current_basis_gradients),
        current_cell_volumes=pack_current(plan.current_cell_volumes),
        current_conductivity=pack_current(plan.current_conductivity),
        current_cell_load=pack_current(plan.current_cell_load),
        current_cell_dirichlet_base=pack_current(plan.current_cell_dirichlet_base),
        current_cell_dirichlet_scale=pack_current(plan.current_cell_dirichlet_scale),
        current_to_thermal_slots=_readonly(plan.current_to_thermal_slots, dtype=np.int32),
        thermal_cell_local_dofs=_readonly(thermal.transport.cell_local_dofs, dtype=np.int32),
        thermal_owner_mask=_readonly(
            thermal.transport.owned_dof_ids < thermal.topology.free_dof_count,
            dtype=np.bool_,
        ),
        thermal_cell_mask=_readonly(
            thermal.transport.cell_ids < thermal.topology.cell_count,
            dtype=np.bool_,
        ),
        thermal_conduction_stiffness=pack_thermal(plan.thermal_conduction_stiffness),
        thermal_robin_matrix=pack_thermal(plan.thermal_robin_matrix),
        thermal_cell_volumes=pack_thermal(plan.thermal_cell_volumes),
        thermal_nonrobin_load=pack_thermal(plan.thermal_nonrobin_load),
        thermal_robin_ambient_load=pack_thermal(plan.thermal_robin_ambient_load),
        thermal_cell_dirichlet_shifted=pack_thermal(plan.thermal_cell_dirichlet_shifted),
    )


def pack_tet4_electrothermal_inputs(
    plan: Tet4ElectrothermalPlan,
    *,
    value_dtype: np.dtype | type[np.generic],
) -> PackedTet4ElectrothermalInputs:
    """Convert host-packed fields to ordinary JAX arrays without device selection."""

    host = pack_tet4_electrothermal_inputs_host(plan, value_dtype=value_dtype)
    return PackedTet4ElectrothermalInputs(*(jnp.asarray(value) for value in host))


class Tet4ElectrothermalParameters(NamedTuple):
    """Three explicit differentiable scalar controls for the M5b.2 numerical kernel."""

    applied_voltage: jax.Array
    electrical_conductivity_scale: jax.Array
    thermal_conductivity_scale: jax.Array


@dataclass(frozen=True, slots=True)
class Tet4ElectrothermalAdmissionPolicy:
    """Conservation tolerances applied after both residual-defined linear solves."""

    charge_balance_tolerance: float
    electrical_energy_tolerance: float
    joule_transfer_tolerance: float
    thermal_balance_tolerance: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(
                    f"Tet4 electrothermal {name.replace('_', ' ')} must be a real scalar"
                )
            canonical = float(value)
            if not math.isfinite(canonical) or canonical <= 0.0:
                raise ContractError(
                    f"Tet4 electrothermal {name.replace('_', ' ')} must be finite and positive"
                )
            object.__setattr__(self, name, canonical)


class PackedTet4ElectrothermalState(NamedTuple):
    """Owner-authoritative free potential and reference-relative temperature shards."""

    potential: jax.Array
    temperature_rise: jax.Array


class PackedTet4ElectrothermalResult(NamedTuple):
    """Forward state plus independent numerical and conservation diagnostics."""

    state: PackedTet4ElectrothermalState
    current_joule_density: jax.Array
    thermal_joule_density: jax.Array
    current_linear: PackedScalarH1CGResult
    thermal_linear: PackedScalarH1CGResult
    electrical_joule_power: jax.Array
    electrical_variational_power: jax.Array
    electrical_energy_relative_error: jax.Array
    charge_balance_relative_error: jax.Array
    thermal_joule_load: jax.Array
    joule_transfer_relative_error: jax.Array
    thermal_input_power: jax.Array
    convection_outward_power: jax.Array
    dirichlet_outward_power: jax.Array
    thermal_balance_relative_error: jax.Array
    numerically_admitted: jax.Array


@dataclass(frozen=True, slots=True)
class Tet4ElectrothermalRuntime:
    """Functions bound to one plan, explicit JAX mesh, and immutable policies."""

    solve: Callable[..., PackedTet4ElectrothermalResult]
    thermal_cell_temperature: Callable[..., jax.Array]


def _relative_error(numerator: jax.Array, *terms: jax.Array) -> jax.Array:
    denominator = sum(jnp.abs(term) for term in terms)
    return jnp.where(
        denominator > 0.0,
        jnp.abs(numerator) / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )


def build_tet4_electrothermal_runtime(
    plan: Tet4ElectrothermalPlan | Tet4ElectrothermalRuntimePlan,
    mesh: Mesh,
    current_cg_policy: ScalarH1CGPolicy,
    thermal_cg_policy: ScalarH1CGPolicy,
    admission_policy: Tet4ElectrothermalAdmissionPolicy,
    *,
    axis_name: str = "partition",
    current_preconditioner_factory: PackedScalarH1PreconditionerFactory | None = None,
    thermal_preconditioner_factory: PackedScalarH1PreconditionerFactory | None = None,
) -> Tet4ElectrothermalRuntime:
    """Build the one-way current/Joule/heat path on one explicit partition mesh."""

    if not isinstance(plan, (Tet4ElectrothermalPlan, Tet4ElectrothermalRuntimePlan)):
        raise ContractError("Tet4 electrothermal runtime requires a prepared plan")
    if not isinstance(current_cg_policy, ScalarH1CGPolicy) or not isinstance(
        thermal_cg_policy,
        ScalarH1CGPolicy,
    ):
        raise ContractError("Tet4 electrothermal runtime requires scalar CG policies")
    if not isinstance(admission_policy, Tet4ElectrothermalAdmissionPolicy):
        raise ContractError("Tet4 electrothermal runtime requires an admission policy")

    current = plan.current_layout
    thermal = plan.thermal_layout
    current_gather = build_packed_collective_scalar_h1_cell_gather(
        current,
        mesh,
        axis_name=axis_name,
    )
    current_assemble = build_packed_collective_scalar_h1_rhs_assembly(
        current,
        mesh,
        axis_name=axis_name,
    )
    thermal_gather = build_packed_collective_scalar_h1_cell_gather(
        thermal,
        mesh,
        axis_name=axis_name,
    )
    thermal_assemble = build_packed_collective_scalar_h1_rhs_assembly(
        thermal,
        mesh,
        axis_name=axis_name,
    )
    current_solve = build_packed_collective_scalar_h1_cg(
        current,
        mesh,
        current_cg_policy,
        axis_name=axis_name,
        preconditioner_factory=current_preconditioner_factory,
    )
    thermal_solve = build_packed_collective_scalar_h1_cg(
        thermal,
        mesh,
        thermal_cg_policy,
        axis_name=axis_name,
        preconditioner_factory=thermal_preconditioner_factory,
    )

    cell_spec = P(axis_name, None)  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(cell_spec, cell_spec),
        out_specs=P(),  # type: ignore[no-untyped-call]
        check_vma=True,
    )
    def cell_sum(values: jax.Array, active: jax.Array) -> jax.Array:
        local = jnp.sum(jnp.where(active[0], values[0], 0.0))
        return cast(jax.Array, lax.psum(local, axis_name))  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(cell_spec, cell_spec, cell_spec),
        out_specs=cell_spec,
        check_vma=True,
    )
    def transfer_joule(
        current_density: jax.Array,
        active: jax.Array,
        target_slots: jax.Array,
    ) -> jax.Array:
        result = jnp.zeros((1, thermal.cell_capacity), dtype=current_density.dtype)
        safe_slots = jnp.where(active[0], target_slots[0], 0)
        values = jnp.where(active[0], current_density[0], 0.0)
        return result.at[0, safe_slots].add(values)

    current_cell_shape = (current.partition_count, current.cell_capacity)
    thermal_cell_shape = (thermal.partition_count, thermal.cell_capacity)
    current_owner_shape = (current.partition_count, current.owned_dof_capacity)
    thermal_owner_shape = (thermal.partition_count, thermal.owned_dof_capacity)

    def validate_inputs(inputs: PackedTet4ElectrothermalInputs) -> None:
        if not isinstance(inputs, PackedTet4ElectrothermalInputs):
            raise ContractError("Tet4 electrothermal inputs must use the packed contract")
        expected = (
            (inputs.current_cell_local_dofs, (*current_cell_shape, 4)),
            (inputs.current_owner_mask, current_owner_shape),
            (inputs.current_cell_mask, current_cell_shape),
            (inputs.current_conduction_stiffness, (*current_cell_shape, 4, 4)),
            (inputs.current_basis_gradients, (*current_cell_shape, 4, 3)),
            (inputs.current_cell_volumes, current_cell_shape),
            (inputs.current_conductivity, current_cell_shape),
            (inputs.current_cell_load, (*current_cell_shape, 4)),
            (inputs.current_cell_dirichlet_base, (*current_cell_shape, 4)),
            (inputs.current_cell_dirichlet_scale, (*current_cell_shape, 4)),
            (inputs.current_to_thermal_slots, current_cell_shape),
            (inputs.thermal_cell_local_dofs, (*thermal_cell_shape, 4)),
            (inputs.thermal_owner_mask, thermal_owner_shape),
            (inputs.thermal_cell_mask, thermal_cell_shape),
            (inputs.thermal_conduction_stiffness, (*thermal_cell_shape, 4, 4)),
            (inputs.thermal_robin_matrix, (*thermal_cell_shape, 4, 4)),
            (inputs.thermal_cell_volumes, thermal_cell_shape),
            (inputs.thermal_nonrobin_load, (*thermal_cell_shape, 4)),
            (inputs.thermal_robin_ambient_load, (*thermal_cell_shape, 4)),
            (inputs.thermal_cell_dirichlet_shifted, (*thermal_cell_shape, 4)),
        )
        if any(value.shape != shape for value, shape in expected):
            raise ValueError("Tet4 electrothermal inputs disagree with the plan")
        integer_fields = (
            inputs.current_cell_local_dofs,
            inputs.current_to_thermal_slots,
            inputs.thermal_cell_local_dofs,
        )
        if not all(jnp.issubdtype(value.dtype, jnp.integer) for value in integer_fields):
            raise TypeError("Tet4 electrothermal maps must use integer dtypes")
        masks = (
            inputs.current_owner_mask,
            inputs.current_cell_mask,
            inputs.thermal_owner_mask,
            inputs.thermal_cell_mask,
        )
        if not all(value.dtype == jnp.bool_ for value in masks):
            raise TypeError("Tet4 electrothermal masks must use boolean dtypes")
        numeric = (
            inputs.current_conduction_stiffness,
            inputs.current_basis_gradients,
            inputs.current_cell_volumes,
            inputs.current_conductivity,
            inputs.current_cell_load,
            inputs.current_cell_dirichlet_base,
            inputs.current_cell_dirichlet_scale,
            inputs.thermal_conduction_stiffness,
            inputs.thermal_robin_matrix,
            inputs.thermal_cell_volumes,
            inputs.thermal_nonrobin_load,
            inputs.thermal_robin_ambient_load,
            inputs.thermal_cell_dirichlet_shifted,
        )
        if not all(jnp.issubdtype(value.dtype, jnp.floating) for value in numeric):
            raise TypeError("Tet4 electrothermal numerical inputs must be real floating")
        if not all(value.dtype == numeric[0].dtype for value in numeric):
            raise TypeError("Tet4 electrothermal numerical inputs must share one dtype")

    def validate_parameters(
        inputs: PackedTet4ElectrothermalInputs,
        parameters: Tet4ElectrothermalParameters,
    ) -> None:
        if not isinstance(parameters, Tet4ElectrothermalParameters):
            raise ContractError("Tet4 electrothermal parameters must use the typed contract")
        for value in parameters:
            if value.shape != ():
                raise ContractError("Tet4 electrothermal parameters must be scalar arrays")
            if value.dtype != inputs.current_conduction_stiffness.dtype:
                raise ContractError("Tet4 electrothermal parameters must match the input dtype")

    def reduced_cell_rhs(
        cell_load: jax.Array,
        stiffness: jax.Array,
        cell_dirichlet: jax.Array,
        cell_local_dofs: jax.Array,
        cell_mask: jax.Array,
        constrained_sentinel: int,
    ) -> jax.Array:
        local = cell_load - jnp.einsum("pcij,pcj->pci", stiffness, cell_dirichlet)
        free_rows = cell_local_dofs < constrained_sentinel
        return jnp.where(free_rows & cell_mask[:, :, None], local, 0.0)

    def thermal_cell_temperature(
        inputs: PackedTet4ElectrothermalInputs,
        state: PackedTet4ElectrothermalState,
    ) -> jax.Array:
        validate_inputs(inputs)
        if not isinstance(state, PackedTet4ElectrothermalState):
            raise ContractError("Tet4 electrothermal state must use the packed contract")
        if state.temperature_rise.shape != thermal_owner_shape:
            raise ValueError("Tet4 electrothermal state disagrees with the thermal owner layout")
        if state.temperature_rise.dtype != inputs.thermal_conduction_stiffness.dtype:
            raise TypeError("Tet4 electrothermal state must match the input dtype")
        reference = jnp.asarray(plan.thermal_reference, dtype=state.temperature_rise.dtype)
        return (
            thermal_gather(inputs.thermal_cell_local_dofs, state.temperature_rise)
            + inputs.thermal_cell_dirichlet_shifted
            + reference
        )

    def solve(
        inputs: PackedTet4ElectrothermalInputs,
        parameters: Tet4ElectrothermalParameters,
    ) -> PackedTet4ElectrothermalResult:
        validate_inputs(inputs)
        validate_parameters(inputs, parameters)
        voltage, electrical_scale, thermal_scale = parameters
        parameter_valid = (
            jnp.isfinite(voltage)
            & jnp.isfinite(electrical_scale)
            & (electrical_scale > 0.0)
            & jnp.isfinite(thermal_scale)
            & (thermal_scale > 0.0)
        )

        current_stiffness = inputs.current_conduction_stiffness * electrical_scale
        current_dirichlet = (
            inputs.current_cell_dirichlet_base + voltage * inputs.current_cell_dirichlet_scale
        )
        current_cell_rhs = reduced_cell_rhs(
            inputs.current_cell_load,
            current_stiffness,
            current_dirichlet,
            inputs.current_cell_local_dofs,
            inputs.current_cell_mask,
            current.transport.constrained_transport_sentinel,
        )
        current_rhs = current_assemble(current_cell_rhs, inputs.current_cell_local_dofs)
        current_linear = current_solve(
            current_stiffness,
            inputs.current_cell_local_dofs,
            inputs.current_owner_mask,
            current_rhs,
        )
        cell_potential = (
            current_gather(inputs.current_cell_local_dofs, current_linear.solution)
            + current_dirichlet
        )
        electric_gradient = jnp.einsum(
            "pci,pcid->pcd",
            cell_potential,
            inputs.current_basis_gradients,
        )
        current_joule = (
            electrical_scale
            * inputs.current_conductivity
            * jnp.einsum("pcd,pcd->pc", electric_gradient, electric_gradient)
        )
        current_joule = jnp.where(inputs.current_cell_mask, current_joule, 0.0)
        thermal_joule = transfer_joule(
            current_joule,
            inputs.current_cell_mask,
            inputs.current_to_thermal_slots,
        )
        joule_load = thermal_joule[:, :, None] * inputs.thermal_cell_volumes[:, :, None] / 4.0

        thermal_stiffness = (
            thermal_scale * inputs.thermal_conduction_stiffness + inputs.thermal_robin_matrix
        )
        reference = jnp.asarray(plan.thermal_reference, dtype=thermal_stiffness.dtype)
        reference_values = jnp.full(
            (*thermal_cell_shape, 4), reference, dtype=thermal_stiffness.dtype
        )
        robin_reference_load = jnp.einsum(
            "pcij,pcj->pci",
            inputs.thermal_robin_matrix,
            reference_values,
        )
        thermal_shifted_load = (
            inputs.thermal_nonrobin_load
            + inputs.thermal_robin_ambient_load
            - robin_reference_load
            + joule_load
        )
        thermal_cell_rhs = reduced_cell_rhs(
            thermal_shifted_load,
            thermal_stiffness,
            inputs.thermal_cell_dirichlet_shifted,
            inputs.thermal_cell_local_dofs,
            inputs.thermal_cell_mask,
            thermal.transport.constrained_transport_sentinel,
        )
        thermal_rhs = thermal_assemble(thermal_cell_rhs, inputs.thermal_cell_local_dofs)
        thermal_linear = thermal_solve(
            thermal_stiffness,
            inputs.thermal_cell_local_dofs,
            inputs.thermal_owner_mask,
            thermal_rhs,
        )
        state = PackedTet4ElectrothermalState(
            potential=current_linear.solution,
            temperature_rise=thermal_linear.solution,
        )
        cell_temperature = thermal_cell_temperature(inputs, state)

        current_action = jnp.einsum("pcij,pcj->pci", current_stiffness, cell_potential)
        current_residual = current_action - inputs.current_cell_load
        current_constrained = (
            inputs.current_cell_local_dofs == current.transport.constrained_transport_sentinel
        ) & inputs.current_cell_mask[:, :, None]
        current_reaction_by_cell = jnp.sum(
            jnp.where(current_constrained, current_residual, 0.0),
            axis=2,
        )
        current_reaction_magnitude_by_cell = jnp.sum(
            jnp.abs(jnp.where(current_constrained, current_residual, 0.0)),
            axis=2,
        )
        current_load_by_cell = jnp.sum(inputs.current_cell_load, axis=2)
        reaction = cell_sum(current_reaction_by_cell, inputs.current_cell_mask)
        reaction_magnitude = cell_sum(
            current_reaction_magnitude_by_cell,
            inputs.current_cell_mask,
        )
        current_load_total = cell_sum(current_load_by_cell, inputs.current_cell_mask)
        charge_error = _relative_error(
            reaction + current_load_total,
            reaction_magnitude,
            current_load_total,
        )
        joule_by_cell = current_joule * inputs.current_cell_volumes
        joule_power = cell_sum(joule_by_cell, inputs.current_cell_mask)
        variational_by_cell = jnp.sum(
            cell_potential * inputs.current_cell_load
            + jnp.where(current_constrained, cell_potential * current_residual, 0.0),
            axis=2,
        )
        variational_power = cell_sum(variational_by_cell, inputs.current_cell_mask)
        energy_error = _relative_error(
            joule_power - variational_power,
            joule_power,
            variational_power,
        )

        thermal_joule_by_cell = thermal_joule * inputs.thermal_cell_volumes
        thermal_joule_power = cell_sum(thermal_joule_by_cell, inputs.thermal_cell_mask)
        transfer_error = _relative_error(
            joule_power - thermal_joule_power,
            joule_power,
            thermal_joule_power,
        )
        original_thermal_load = (
            inputs.thermal_nonrobin_load + inputs.thermal_robin_ambient_load + joule_load
        )
        thermal_action = jnp.einsum("pcij,pcj->pci", thermal_stiffness, cell_temperature)
        thermal_residual = thermal_action - original_thermal_load
        thermal_constrained = (
            inputs.thermal_cell_local_dofs == thermal.transport.constrained_transport_sentinel
        ) & inputs.thermal_cell_mask[:, :, None]
        thermal_reaction_by_cell = jnp.sum(
            jnp.where(thermal_constrained, thermal_residual, 0.0),
            axis=2,
        )
        thermal_reaction = cell_sum(thermal_reaction_by_cell, inputs.thermal_cell_mask)
        nonrobin_by_cell = jnp.sum(inputs.thermal_nonrobin_load, axis=2)
        nonrobin_input = cell_sum(nonrobin_by_cell, inputs.thermal_cell_mask)
        thermal_input = nonrobin_input + thermal_joule_power
        convection_by_cell = jnp.sum(
            jnp.einsum("pcij,pcj->pci", inputs.thermal_robin_matrix, cell_temperature)
            - inputs.thermal_robin_ambient_load,
            axis=2,
        )
        convection_outward = cell_sum(convection_by_cell, inputs.thermal_cell_mask)
        dirichlet_outward = -thermal_reaction
        thermal_balance = _relative_error(
            thermal_input - convection_outward - dirichlet_outward,
            thermal_input,
            convection_outward,
            dirichlet_outward,
        )

        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        joule_power,
                        variational_power,
                        energy_error,
                        charge_error,
                        thermal_joule_power,
                        transfer_error,
                        thermal_input,
                        convection_outward,
                        dirichlet_outward,
                        thermal_balance,
                    )
                )
            )
        )
        admitted = (
            parameter_valid
            & current_linear.converged
            & thermal_linear.converged
            & finite
            & (charge_error <= admission_policy.charge_balance_tolerance)
            & (energy_error <= admission_policy.electrical_energy_tolerance)
            & (transfer_error <= admission_policy.joule_transfer_tolerance)
            & (thermal_balance <= admission_policy.thermal_balance_tolerance)
        )
        return PackedTet4ElectrothermalResult(
            state=state,
            current_joule_density=current_joule,
            thermal_joule_density=thermal_joule,
            current_linear=current_linear,
            thermal_linear=thermal_linear,
            electrical_joule_power=joule_power,
            electrical_variational_power=variational_power,
            electrical_energy_relative_error=energy_error,
            charge_balance_relative_error=charge_error,
            thermal_joule_load=thermal_joule_power,
            joule_transfer_relative_error=transfer_error,
            thermal_input_power=thermal_input,
            convection_outward_power=convection_outward,
            dirichlet_outward_power=dirichlet_outward,
            thermal_balance_relative_error=thermal_balance,
            numerically_admitted=admitted,
        )

    return Tet4ElectrothermalRuntime(
        solve=solve,
        thermal_cell_temperature=thermal_cell_temperature,
    )


def reconstruct_tet4_electrothermal_state(
    plan: Tet4ElectrothermalPlan,
    state: PackedTet4ElectrothermalState,
    parameters: Tet4ElectrothermalParameters,
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct compact-conductor potential and full thermal nodal temperature."""

    if not isinstance(plan, Tet4ElectrothermalPlan):
        raise ContractError("Tet4 electrothermal reconstruction requires a prepared plan")
    if not isinstance(state, PackedTet4ElectrothermalState):
        raise ContractError("Tet4 electrothermal reconstruction requires a packed state")
    if not isinstance(parameters, Tet4ElectrothermalParameters):
        raise ContractError("Tet4 electrothermal reconstruction requires typed parameters")
    voltage = parameters.applied_voltage
    if voltage.shape != () or not jnp.issubdtype(voltage.dtype, jnp.floating):
        raise ContractError("Tet4 electrothermal reconstruction voltage must be a real scalar")
    current_free = unpack_collective_scalar_h1_owned_vector(
        plan.current_layout,
        state.potential,
    )
    thermal_free = unpack_collective_scalar_h1_owned_vector(
        plan.thermal_layout,
        state.temperature_rise,
    )
    dtype = jnp.result_type(current_free.dtype, voltage.dtype)
    current_boundary = jnp.asarray(plan.current_dirichlet_base, dtype=dtype) + voltage.astype(
        dtype
    ) * jnp.asarray(plan.current_dirichlet_scale, dtype=dtype)
    potential = reconstruct_scalar_h1_state(
        plan.current_layout.topology,
        current_free.astype(dtype),
        current_boundary,
    )
    temperature_rise = reconstruct_scalar_h1_state(
        plan.thermal_layout.topology,
        thermal_free,
        jnp.asarray(plan.thermal_dirichlet_shifted, dtype=thermal_free.dtype),
    )
    return potential, temperature_rise + jnp.asarray(
        plan.thermal_reference,
        dtype=temperature_rise.dtype,
    )
