"""Array-library-neutral FEM mesh contracts."""

from dataclasses import dataclass
from enum import StrEnum

from femx.core.arrays import ArrayLike, shape_of
from femx.core.capabilities import FunctionSpaceFamily
from femx.core.errors import ContractError


class CellType(StrEnum):
    """Initial set of supported reference-cell topologies."""

    SEGMENT = "segment"
    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"

    @property
    def dimension(self) -> int:
        """Topological dimension of the reference cell."""

        return {
            CellType.SEGMENT: 1,
            CellType.TRIANGLE: 2,
            CellType.QUADRILATERAL: 2,
            CellType.TETRAHEDRON: 3,
            CellType.HEXAHEDRON: 3,
        }[self]

    @property
    def corner_count(self) -> int:
        """Number of corner nodes for a first-order cell."""

        return {
            CellType.SEGMENT: 2,
            CellType.TRIANGLE: 3,
            CellType.QUADRILATERAL: 4,
            CellType.TETRAHEDRON: 4,
            CellType.HEXAHEDRON: 8,
        }[self]


class DofLocation(StrEnum):
    """Geometric entity that owns a degree of freedom."""

    VERTEX = "vertex"
    EDGE = "edge"
    FACE = "face"
    CELL = "cell"
    QUADRATURE = "quadrature"


@dataclass(frozen=True, slots=True)
class MeshGeometry:
    """Physical node coordinates in SI metres."""

    coordinates: ArrayLike
    coordinate_unit: str = "m"

    def __post_init__(self) -> None:
        shape = shape_of(self.coordinates)
        if len(shape) != 2 or shape[1] not in (1, 2, 3):
            raise ContractError(f"mesh coordinates must have shape (nodes, 1|2|3), got {shape}")
        if self.coordinate_unit != "m":
            raise ContractError("mesh coordinates must be converted to SI metres before ingestion")

    @property
    def node_count(self) -> int:
        """Number of geometric nodes."""

        return int(self.coordinates.shape[0])

    @property
    def spatial_dimension(self) -> int:
        """Embedding-space dimension."""

        return int(self.coordinates.shape[1])


@dataclass(frozen=True, slots=True)
class MeshTopology:
    """Cell-to-node connectivity and reference cell type."""

    connectivity: ArrayLike
    cell_type: CellType
    node_count: int

    def __post_init__(self) -> None:
        shape = shape_of(self.connectivity)
        if len(shape) != 2:
            raise ContractError(f"mesh connectivity must be rank two, got {shape}")
        if shape[1] != self.cell_type.corner_count:
            raise ContractError(
                f"{self.cell_type.value} connectivity requires {self.cell_type.corner_count} "
                f"corners, got {shape[1]}"
            )
        if self.node_count <= 0:
            raise ContractError("mesh topology node_count must be positive")

    @property
    def cell_count(self) -> int:
        """Number of cells."""

        return int(self.connectivity.shape[0])


@dataclass(frozen=True, slots=True)
class EntityTag:
    """Stable semantic name for a set of mesh entities."""

    name: str
    dimension: int
    entity_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("entity tag name must be non-empty and trimmed")
        if self.dimension < 0 or self.dimension > 3:
            raise ContractError("entity tag dimension must be between zero and three")
        if any(entity_id < 0 for entity_id in self.entity_ids):
            raise ContractError(f"entity tag {self.name!r} contains a negative id")
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ContractError(f"entity tag {self.name!r} contains duplicate ids")


@dataclass(frozen=True, slots=True)
class OrientationMap:
    """Explicit edge/face orientation metadata for conforming spaces."""

    edge_signs: ArrayLike | None = None
    face_signs: ArrayLike | None = None

    def __post_init__(self) -> None:
        for name, values in (("edge_signs", self.edge_signs), ("face_signs", self.face_signs)):
            if values is not None and len(shape_of(values)) != 2:
                raise ContractError(f"{name} must be a rank-two cell-local orientation array")


@dataclass(frozen=True, slots=True)
class Mesh:
    """A mesh with stable semantic tags and optional orientation metadata."""

    geometry: MeshGeometry
    topology: MeshTopology
    tags: tuple[EntityTag, ...] = ()
    boundary_facets: MeshTopology | None = None
    orientation: OrientationMap = OrientationMap()
    schema_version: str = "femx.mesh/v1"

    def __post_init__(self) -> None:
        if self.geometry.node_count != self.topology.node_count:
            raise ContractError(
                "geometry and topology node counts differ: "
                f"{self.geometry.node_count} != {self.topology.node_count}"
            )
        if self.boundary_facets is not None:
            if self.boundary_facets.node_count != self.geometry.node_count:
                raise ContractError(
                    "boundary-facet and geometry node counts differ: "
                    f"{self.boundary_facets.node_count} != {self.geometry.node_count}"
                )
            expected_dimension = self.topology.cell_type.dimension - 1
            if self.boundary_facets.cell_type.dimension != expected_dimension:
                raise ContractError(
                    "boundary facets must have dimension one below the bulk cells: "
                    f"{self.boundary_facets.cell_type.dimension} != {expected_dimension}"
                )
        names = tuple(tag.name for tag in self.tags)
        if len(names) != len(set(names)):
            raise ContractError("mesh entity tag names must be unique")
        topological_dimension = self.topology.cell_type.dimension
        invalid = [tag.name for tag in self.tags if tag.dimension > topological_dimension]
        if invalid:
            raise ContractError(f"tags exceed mesh topological dimension: {invalid}")
        if self.schema_version != "femx.mesh/v1":
            raise ContractError(f"unsupported mesh schema {self.schema_version!r}")

    def tag(self, name: str) -> EntityTag:
        """Return a uniquely named entity tag or reject an unknown reference."""

        for tag in self.tags:
            if tag.name == name:
                return tag
        raise ContractError(f"mesh does not define entity tag {name!r}")


@dataclass(frozen=True, slots=True)
class FunctionSpace:
    """A finite-element function space independent of backend storage."""

    family: FunctionSpaceFamily
    order: int
    value_shape: tuple[int, ...] = ()
    continuity: str = "conforming"

    def __post_init__(self) -> None:
        minimum_order = 0 if self.family in {FunctionSpaceFamily.L2, FunctionSpaceFamily.DG} else 1
        if self.order < minimum_order:
            raise ContractError(f"{self.family.value} requires polynomial order >= {minimum_order}")
        if any(size <= 0 for size in self.value_shape):
            raise ContractError(f"function-space value shape is invalid: {self.value_shape}")
        if not self.continuity:
            raise ContractError("function space must declare its continuity contract")


@dataclass(frozen=True, slots=True)
class DofMap:
    """Cell-local to global DOF map."""

    cell_dofs: ArrayLike
    dof_count: int
    locations: frozenset[DofLocation]

    def __post_init__(self) -> None:
        if len(shape_of(self.cell_dofs)) != 2:
            raise ContractError("cell_dofs must be a rank-two array")
        if self.dof_count <= 0:
            raise ContractError("dof_count must be positive")
        if not self.locations:
            raise ContractError("a DOF map must declare at least one entity location")


@dataclass(frozen=True, slots=True)
class MeshPartition:
    """Process-local ownership metadata for a global mesh/DOF space."""

    process_index: int
    process_count: int
    owned_dofs: tuple[int, ...]
    ghost_dofs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.process_count <= 0:
            raise ContractError("process_count must be positive")
        if self.process_index < 0 or self.process_index >= self.process_count:
            raise ContractError("process_index must be within the global process range")
        if any(dof < 0 for dof in (*self.owned_dofs, *self.ghost_dofs)):
            raise ContractError("partition DOF ids cannot be negative")
        if len(self.owned_dofs) != len(set(self.owned_dofs)):
            raise ContractError("owned DOF ids must be unique")
        if len(self.ghost_dofs) != len(set(self.ghost_dofs)):
            raise ContractError("ghost DOF ids must be unique")
        overlap = set(self.owned_dofs) & set(self.ghost_dofs)
        if overlap:
            raise ContractError(f"owned and ghost DOFs overlap: {sorted(overlap)}")
