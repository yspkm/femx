"""Strict first-order Gmsh MSH 4.1 ingestion for photonics meshes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from femx.core.errors import ContractError, MeshingError
from femx.mesh import (
    CellType,
    EntityTag,
    Mesh,
    MeshGeometry,
    MeshTopology,
    OrientationMap,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_SECTIONS = frozenset({"MeshFormat", "PhysicalNames", "Entities", "Nodes", "Elements"})


@dataclass(frozen=True, slots=True, order=True)
class GmshPhysicalGroup:
    """One named Gmsh physical group preserved in canonical order."""

    dimension: int
    tag: int
    name: str

    def __post_init__(self) -> None:
        if self.dimension not in (1, 2, 3):
            raise ContractError("Gmsh physical groups must be curves, surfaces, or volumes")
        if self.tag <= 0:
            raise ContractError("Gmsh physical tags must be positive")
        if not self.name or self.name.strip() != self.name:
            raise ContractError("Gmsh physical names must be non-empty and trimmed")


@dataclass(frozen=True, slots=True)
class GmshImportRecord:
    """Content-addressed mapping from Gmsh ids to canonical femx ids."""

    source_sha256: str
    canonical_mesh_sha256: str
    format_version: str
    coordinate_scale_to_m: float
    physical_groups: tuple[GmshPhysicalGroup, ...]
    node_tags: tuple[int, ...]
    cell_element_tags: tuple[int, ...]
    boundary_element_tags: tuple[int, ...]
    cell_local_node_permutations: tuple[tuple[int, ...], ...]
    topological_dimension: int = 2
    boundary_local_node_permutations: tuple[tuple[int, ...], ...] = ()
    schema_version: str = "femx.gmsh-import/v1"

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise ContractError("Gmsh import source requires a lowercase SHA-256")
        if not _SHA256_PATTERN.fullmatch(self.canonical_mesh_sha256):
            raise ContractError("canonical femx mesh requires a lowercase SHA-256")
        if self.format_version != "4.1":
            raise ContractError("Gmsh import record requires MSH format 4.1")
        if not math.isfinite(self.coordinate_scale_to_m) or self.coordinate_scale_to_m <= 0.0:
            raise ContractError("Gmsh coordinate scale must be finite and positive")
        if self.topological_dimension not in (2, 3):
            raise ContractError("Gmsh import topological dimension must be 2 or 3")
        expected_schema = (
            "femx.gmsh-import/v1" if self.topological_dimension == 2 else "femx.gmsh-import/v2"
        )
        if self.schema_version != expected_schema:
            raise ContractError(
                f"Gmsh {self.topological_dimension}D import requires schema {expected_schema!r}"
            )
        group_keys = tuple((group.dimension, group.tag) for group in self.physical_groups)
        group_names = tuple(group.name for group in self.physical_groups)
        if not group_keys or len(group_keys) != len(set(group_keys)):
            raise ContractError(
                "Gmsh import physical dimension/tag pairs must be non-empty and unique"
            )
        if len(group_names) != len(set(group_names)):
            raise ContractError("Gmsh import physical names must be unique")
        allowed_group_dimensions = {
            self.topological_dimension - 1,
            self.topological_dimension,
        }
        if any(group.dimension not in allowed_group_dimensions for group in self.physical_groups):
            raise ContractError("Gmsh physical-group dimension disagrees with the imported mesh")
        for label, values in (
            ("node", self.node_tags),
            ("cell element", self.cell_element_tags),
            ("boundary element", self.boundary_element_tags),
        ):
            if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
                raise ContractError(f"Gmsh {label} tags must be non-empty unique positive integers")
        if len(self.cell_local_node_permutations) != len(self.cell_element_tags):
            raise ContractError("Gmsh cell permutations must match the canonical cell count")
        expected = tuple(range(self.topological_dimension + 1))
        if any(
            tuple(sorted(permutation)) != expected
            for permutation in self.cell_local_node_permutations
        ):
            raise ContractError("each Gmsh cell permutation must contain every local node once")
        if self.topological_dimension == 2:
            if self.boundary_local_node_permutations:
                raise ContractError("2D Gmsh imports do not normalize boundary-segment direction")
        else:
            if len(self.boundary_local_node_permutations) != len(self.boundary_element_tags):
                raise ContractError("Gmsh boundary permutations must match the boundary count")
            if any(
                tuple(sorted(permutation)) != (0, 1, 2)
                for permutation in self.boundary_local_node_permutations
            ):
                raise ContractError(
                    "each Gmsh boundary permutation must contain local nodes 0, 1, 2"
                )

    def canonical_data(self) -> dict[str, object]:
        """Return JSON-compatible provenance including the exact id permutation."""

        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "canonical_mesh_sha256": self.canonical_mesh_sha256,
            "format_version": self.format_version,
            "coordinate_scale_to_m": self.coordinate_scale_to_m,
            "physical_groups": [
                {"dimension": group.dimension, "tag": group.tag, "name": group.name}
                for group in self.physical_groups
            ],
            "node_tags": list(self.node_tags),
            "cell_element_tags": list(self.cell_element_tags),
            "boundary_element_tags": list(self.boundary_element_tags),
            "cell_local_node_permutations": [
                list(permutation) for permutation in self.cell_local_node_permutations
            ],
        }
        if self.topological_dimension == 3:
            data["topological_dimension"] = self.topological_dimension
            data["boundary_local_node_permutations"] = [
                list(permutation) for permutation in self.boundary_local_node_permutations
            ]
        return data

    def digest(self) -> str:
        """Hash the complete source-to-canonical mapping."""

        payload = json.dumps(
            self.canonical_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ImportedGmshMesh:
    """Canonical mesh paired with the exact Gmsh import record."""

    mesh: Mesh
    record: GmshImportRecord

    def __post_init__(self) -> None:
        if _canonical_mesh_sha256(self.mesh) != self.record.canonical_mesh_sha256:
            raise ContractError("Gmsh import record does not match its canonical mesh")


@dataclass(frozen=True, slots=True)
class _RawElement:
    tag: int
    dimension: int
    entity_tag: int
    node_tags: tuple[int, ...]


def read_gmsh_msh(path: Path, *, coordinate_scale_to_m: float) -> ImportedGmshMesh:
    """Read a strict ASCII MSH 4.1 file and convert coordinates to SI metres."""

    if not math.isfinite(coordinate_scale_to_m) or coordinate_scale_to_m <= 0.0:
        raise ContractError("Gmsh coordinate scale must be finite and positive")
    if not path.is_file():
        raise MeshingError(f"Gmsh mesh file does not exist: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MeshingError("Gmsh v1 ingestion requires an ASCII MSH 4.1 file") from error
    return _parse_gmsh_msh(
        text,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        coordinate_scale_to_m=float(coordinate_scale_to_m),
    )


def read_gmsh_msh_3d(path: Path, *, coordinate_scale_to_m: float) -> ImportedGmshMesh:
    """Read a strict ASCII MSH 4.1 Tet4 volume mesh and convert coordinates to SI metres."""

    if not math.isfinite(coordinate_scale_to_m) or coordinate_scale_to_m <= 0.0:
        raise ContractError("Gmsh coordinate scale must be finite and positive")
    if not path.is_file():
        raise MeshingError(f"Gmsh mesh file does not exist: {path}")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MeshingError("Gmsh 3D ingestion requires an ASCII MSH 4.1 file") from error
    return _parse_gmsh_msh_3d(
        text,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        coordinate_scale_to_m=float(coordinate_scale_to_m),
    )


def _parse_gmsh_msh(
    text: str,
    *,
    source_sha256: str,
    coordinate_scale_to_m: float,
) -> ImportedGmshMesh:
    sections = _split_sections(text)
    physical_groups = _parse_physical_names(sections["PhysicalNames"])
    if any(group.dimension not in (1, 2) for group in physical_groups):
        raise MeshingError("Gmsh 2D import accepts only curve and surface physical groups")
    names_by_key = {(group.dimension, group.tag): group.name for group in physical_groups}
    entity_groups = _parse_entities(sections["Entities"], names_by_key=names_by_key)
    nodes = _parse_nodes(sections["Nodes"])
    segments, triangles = _parse_elements(
        sections["Elements"], entity_groups=entity_groups, names_by_key=names_by_key
    )

    node_tags = tuple(sorted(nodes))
    node_index = {tag: index for index, tag in enumerate(node_tags)}
    coordinates_3d = np.asarray([nodes[tag] for tag in node_tags], dtype=np.float64)
    if np.any(coordinates_3d[:, 2] != 0.0):
        raise MeshingError("the initial Gmsh importer rejects nonzero z coordinates")
    coordinates = coordinates_3d[:, :2] * coordinate_scale_to_m
    if not np.all(np.isfinite(coordinates)):
        raise MeshingError("scaled Gmsh coordinates must remain finite")

    cells = _connectivity(triangles, node_index=node_index, width=3)
    boundary_facets = _connectivity(segments, node_index=node_index, width=2)
    determinants, cell_local_node_permutations = _orient_triangles(coordinates, cells)
    if np.any(determinants <= 0.0):
        raise MeshingError("Gmsh triangle orientation normalization failed")
    _validate_boundary_facets(cells, boundary_facets)

    tag_ids: dict[str, list[int]] = {group.name: [] for group in physical_groups}
    for canonical_id, element in enumerate(triangles):
        for physical_tag in entity_groups[(2, element.entity_tag)]:
            tag_ids[names_by_key[(2, physical_tag)]].append(canonical_id)
    for canonical_id, element in enumerate(segments):
        for physical_tag in entity_groups[(1, element.entity_tag)]:
            tag_ids[names_by_key[(1, physical_tag)]].append(canonical_id)
    empty = sorted(name for name, ids in tag_ids.items() if not ids)
    if empty:
        raise MeshingError(f"Gmsh physical groups contain no imported elements: {empty}")

    groups_by_name = {group.name: group for group in physical_groups}
    tags = tuple(
        EntityTag(name, groups_by_name[name].dimension, tuple(ids))
        for name, ids in sorted(tag_ids.items())
    )
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    edge_signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    mesh = Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cells, CellType.TRIANGLE, len(node_tags)),
        tags=tags,
        boundary_facets=MeshTopology(boundary_facets, CellType.SEGMENT, len(node_tags)),
        orientation=OrientationMap(edge_signs=edge_signs),
    )
    record = GmshImportRecord(
        source_sha256=source_sha256,
        canonical_mesh_sha256=_canonical_mesh_sha256(mesh),
        format_version="4.1",
        coordinate_scale_to_m=coordinate_scale_to_m,
        physical_groups=physical_groups,
        node_tags=node_tags,
        cell_element_tags=tuple(element.tag for element in triangles),
        boundary_element_tags=tuple(element.tag for element in segments),
        cell_local_node_permutations=cell_local_node_permutations,
    )
    return ImportedGmshMesh(mesh=mesh, record=record)


def _parse_gmsh_msh_3d(
    text: str,
    *,
    source_sha256: str,
    coordinate_scale_to_m: float,
) -> ImportedGmshMesh:
    sections = _split_sections(text)
    physical_groups = _parse_physical_names(sections["PhysicalNames"])
    if any(group.dimension not in (2, 3) for group in physical_groups):
        raise MeshingError("Gmsh 3D import accepts only surface and volume physical groups")
    if {group.dimension for group in physical_groups} != {2, 3}:
        raise MeshingError("Gmsh 3D import requires named surface and volume physical groups")
    names_by_key = {(group.dimension, group.tag): group.name for group in physical_groups}
    entity_groups = _parse_entities(sections["Entities"], names_by_key=names_by_key)
    nodes = _parse_nodes(sections["Nodes"], maximum_entity_dimension=3)
    triangles, tetrahedra = _parse_elements(
        sections["Elements"],
        entity_groups=entity_groups,
        names_by_key=names_by_key,
        topological_dimension=3,
    )

    node_tags = tuple(sorted(nodes))
    node_index = {tag: index for index, tag in enumerate(node_tags)}
    coordinates = (
        np.asarray([nodes[tag] for tag in node_tags], dtype=np.float64) * coordinate_scale_to_m
    )
    if not np.all(np.isfinite(coordinates)):
        raise MeshingError("scaled Gmsh coordinates must remain finite")

    cells = _connectivity(tetrahedra, node_index=node_index, width=4)
    boundary_facets = _connectivity(triangles, node_index=node_index, width=3)
    determinants, cell_local_node_permutations = _orient_tetrahedra(coordinates, cells)
    if np.any(determinants <= 0.0):
        raise MeshingError("Gmsh tetrahedron orientation normalization failed")
    boundary_local_node_permutations = _orient_and_validate_tetrahedron_boundary(
        coordinates,
        cells,
        boundary_facets,
    )

    tag_ids: dict[str, list[int]] = {group.name: [] for group in physical_groups}
    for canonical_id, element in enumerate(tetrahedra):
        for physical_tag in entity_groups[(3, element.entity_tag)]:
            tag_ids[names_by_key[(3, physical_tag)]].append(canonical_id)
    for canonical_id, element in enumerate(triangles):
        for physical_tag in entity_groups[(2, element.entity_tag)]:
            tag_ids[names_by_key[(2, physical_tag)]].append(canonical_id)
    empty = sorted(name for name, ids in tag_ids.items() if not ids)
    if empty:
        raise MeshingError(f"Gmsh physical groups contain no imported elements: {empty}")

    groups_by_name = {group.name: group for group in physical_groups}
    tags = tuple(
        EntityTag(name, groups_by_name[name].dimension, tuple(ids))
        for name, ids in sorted(tag_ids.items())
    )
    edge_signs, face_signs = _tetrahedron_orientation_maps(cells)
    mesh = Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cells, CellType.TETRAHEDRON, len(node_tags)),
        tags=tags,
        boundary_facets=MeshTopology(boundary_facets, CellType.TRIANGLE, len(node_tags)),
        orientation=OrientationMap(edge_signs=edge_signs, face_signs=face_signs),
    )
    record = GmshImportRecord(
        source_sha256=source_sha256,
        canonical_mesh_sha256=_canonical_mesh_sha256(mesh),
        format_version="4.1",
        coordinate_scale_to_m=coordinate_scale_to_m,
        physical_groups=physical_groups,
        node_tags=node_tags,
        cell_element_tags=tuple(element.tag for element in tetrahedra),
        boundary_element_tags=tuple(element.tag for element in triangles),
        cell_local_node_permutations=cell_local_node_permutations,
        topological_dimension=3,
        boundary_local_node_permutations=boundary_local_node_permutations,
        schema_version="femx.gmsh-import/v2",
    )
    return ImportedGmshMesh(mesh=mesh, record=record)


def _split_sections(text: str) -> dict[str, tuple[str, ...]]:
    lines = text.splitlines()
    sections: dict[str, tuple[str, ...]] = {}
    cursor = 0
    while cursor < len(lines):
        marker = lines[cursor].strip()
        if not marker:
            cursor += 1
            continue
        if not marker.startswith("$") or marker.startswith("$End"):
            raise MeshingError(f"unexpected text outside a Gmsh section: {marker!r}")
        name = marker[1:]
        end_marker = f"$End{name}"
        try:
            end = lines.index(end_marker, cursor + 1)
        except ValueError as error:
            raise MeshingError(f"Gmsh section {name!r} is not terminated") from error
        if name in sections:
            raise MeshingError(f"duplicate Gmsh section {name!r}")
        sections[name] = tuple(lines[cursor + 1 : end])
        cursor = end + 1

    missing = sorted(_SUPPORTED_SECTIONS - sections.keys())
    unsupported = sorted(sections.keys() - _SUPPORTED_SECTIONS)
    if missing:
        raise MeshingError(f"Gmsh mesh is missing required sections: {missing}")
    if unsupported:
        raise MeshingError(f"Gmsh mesh contains unsupported sections: {unsupported}")
    format_tokens = sections["MeshFormat"]
    if format_tokens != ("4.1 0 8",):
        raise MeshingError("Gmsh v1 ingestion requires exact ASCII MSH header '4.1 0 8'")
    return sections


def _parse_physical_names(lines: tuple[str, ...]) -> tuple[GmshPhysicalGroup, ...]:
    if not lines:
        raise MeshingError("Gmsh PhysicalNames section is empty")
    count = _single_int(lines[0], context="physical-group count")
    if len(lines) != count + 1:
        raise MeshingError("Gmsh PhysicalNames count does not match its records")
    groups: list[GmshPhysicalGroup] = []
    for line in lines[1:]:
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise MeshingError(f"invalid Gmsh physical-name record: {line!r}") from error
        if len(tokens) != 3:
            raise MeshingError(f"invalid Gmsh physical-name record: {line!r}")
        groups.append(
            GmshPhysicalGroup(
                dimension=_integer(tokens[0], context="physical-group dimension"),
                tag=_integer(tokens[1], context="physical-group tag"),
                name=tokens[2],
            )
        )
    keys = tuple((group.dimension, group.tag) for group in groups)
    names = tuple(group.name for group in groups)
    if len(keys) != len(set(keys)) or len(names) != len(set(names)):
        raise MeshingError("Gmsh physical groups must have unique dimension/tag pairs and names")
    return tuple(sorted(groups))


def _parse_entities(
    lines: tuple[str, ...],
    *,
    names_by_key: dict[tuple[int, int], str],
) -> dict[tuple[int, int], tuple[int, ...]]:
    if not lines:
        raise MeshingError("Gmsh Entities section is empty")
    counts = _integer_fields(lines[0], expected=4, context="entity counts")
    cursor = 1
    entities: dict[tuple[int, int], tuple[int, ...]] = {}
    for dimension, count in enumerate(counts):
        for _ in range(count):
            if cursor >= len(lines):
                raise MeshingError("Gmsh Entities section ended before its declared count")
            tokens = lines[cursor].split()
            cursor += 1
            prefix = 4 if dimension == 0 else 7
            if len(tokens) <= prefix:
                raise MeshingError("Gmsh entity record is too short")
            tag = _integer(tokens[0], context="entity tag")
            physical_count = _integer(tokens[prefix], context="entity physical-tag count")
            physical_end = prefix + 1 + physical_count
            if physical_end > len(tokens):
                raise MeshingError("Gmsh entity physical-tag count exceeds its record")
            signed_physical_tags = tuple(
                _integer(token, context="entity physical tag")
                for token in tokens[prefix + 1 : physical_end]
            )
            physical_tags = tuple(abs(tag) for tag in signed_physical_tags)
            if len(physical_tags) != len(set(physical_tags)):
                raise MeshingError("Gmsh entity repeats a physical tag after orientation removal")
            if dimension > 0:
                if physical_end >= len(tokens):
                    raise MeshingError("Gmsh entity record omits its bounding-entity count")
                bounding_count = _integer(tokens[physical_end], context="entity bounding-tag count")
                if len(tokens) != physical_end + 1 + bounding_count:
                    raise MeshingError("Gmsh entity bounding-tag count does not match its record")
            elif len(tokens) != physical_end:
                raise MeshingError("Gmsh point entity record has unexpected trailing fields")
            key = (dimension, tag)
            if key in entities:
                raise MeshingError(f"duplicate Gmsh entity {key}")
            for physical_tag in physical_tags:
                if (dimension, physical_tag) not in names_by_key:
                    raise MeshingError(
                        f"Gmsh entity {key} references unnamed physical tag {physical_tag}"
                    )
            entities[key] = physical_tags
    if cursor != len(lines):
        raise MeshingError("Gmsh Entities section contains trailing records")
    return entities


def _parse_nodes(
    lines: tuple[str, ...],
    *,
    maximum_entity_dimension: int = 2,
) -> dict[int, tuple[float, float, float]]:
    if not lines:
        raise MeshingError("Gmsh Nodes section is empty")
    block_count, node_count, minimum, maximum = _integer_fields(
        lines[0], expected=4, context="node header"
    )
    cursor = 1
    nodes: dict[int, tuple[float, float, float]] = {}
    for _ in range(block_count):
        if cursor >= len(lines):
            raise MeshingError("Gmsh Nodes section ended before its declared blocks")
        dimension, _entity_tag, parametric, block_size = _integer_fields(
            lines[cursor], expected=4, context="node block header"
        )
        cursor += 1
        if dimension < 0 or dimension > maximum_entity_dimension:
            if maximum_entity_dimension == 2:
                raise MeshingError("the initial Gmsh importer rejects 3D node blocks")
            raise MeshingError("Gmsh node block dimension lies outside the imported volume")
        if parametric != 0:
            raise MeshingError("the initial Gmsh importer rejects parametric nodes")
        block_tags: list[int] = []
        for _ in range(block_size):
            if cursor >= len(lines):
                raise MeshingError("Gmsh node-tag block ended early")
            block_tags.append(_single_int(lines[cursor], context="node tag"))
            cursor += 1
        for tag in block_tags:
            if cursor >= len(lines):
                raise MeshingError("Gmsh node-coordinate block ended early")
            coordinates = _float_fields(lines[cursor], expected=3, context="node coordinates")
            cursor += 1
            if tag in nodes:
                raise MeshingError(f"duplicate Gmsh node tag {tag}")
            nodes[tag] = coordinates
    if cursor != len(lines):
        raise MeshingError("Gmsh Nodes section contains trailing records")
    if len(nodes) != node_count or not nodes:
        raise MeshingError("Gmsh node count does not match its records")
    if min(nodes) != minimum or max(nodes) != maximum:
        raise MeshingError("Gmsh node min/max tags do not match its header")
    return nodes


def _parse_elements(
    lines: tuple[str, ...],
    *,
    entity_groups: dict[tuple[int, int], tuple[int, ...]],
    names_by_key: dict[tuple[int, int], str],
    topological_dimension: int = 2,
) -> tuple[tuple[_RawElement, ...], tuple[_RawElement, ...]]:
    if not lines:
        raise MeshingError("Gmsh Elements section is empty")
    block_count, element_count, minimum, maximum = _integer_fields(
        lines[0], expected=4, context="element header"
    )
    cursor = 1
    elements: list[_RawElement] = []
    for _ in range(block_count):
        if cursor >= len(lines):
            raise MeshingError("Gmsh Elements section ended before its declared blocks")
        dimension, entity_tag, element_type, block_size = _integer_fields(
            lines[cursor], expected=4, context="element block header"
        )
        cursor += 1
        supported = {1: (1, 2), 2: (2, 3)} if topological_dimension == 2 else {2: (2, 3), 4: (3, 4)}
        if element_type not in supported:
            expected_types = "1 or 2" if topological_dimension == 2 else "2 or 4"
            raise MeshingError(
                f"unsupported Gmsh element type {element_type}; expected {expected_types}"
            )
        expected_dimension, width = supported[element_type]
        if dimension != expected_dimension:
            raise MeshingError("Gmsh element type and entity dimension disagree")
        key = (dimension, entity_tag)
        physical_tags = entity_groups.get(key)
        if not physical_tags:
            raise MeshingError(f"Gmsh element entity {key} has no named physical group")
        for physical_tag in physical_tags:
            if (dimension, physical_tag) not in names_by_key:
                raise MeshingError(f"Gmsh element entity {key} uses an unnamed physical group")
        for _ in range(block_size):
            if cursor >= len(lines):
                raise MeshingError("Gmsh element block ended early")
            values = _integer_fields(lines[cursor], expected=width + 1, context="element record")
            cursor += 1
            elements.append(
                _RawElement(
                    tag=values[0],
                    dimension=dimension,
                    entity_tag=entity_tag,
                    node_tags=tuple(values[1:]),
                )
            )
    if cursor != len(lines):
        raise MeshingError("Gmsh Elements section contains trailing records")
    tags = tuple(element.tag for element in elements)
    if len(elements) != element_count or not elements or len(tags) != len(set(tags)):
        raise MeshingError("Gmsh element count or uniqueness does not match its records")
    if min(tags) != minimum or max(tags) != maximum:
        raise MeshingError("Gmsh element min/max tags do not match its header")
    boundary_dimension = topological_dimension - 1
    boundary = tuple(
        sorted(
            (item for item in elements if item.dimension == boundary_dimension),
            key=lambda x: x.tag,
        )
    )
    cells = tuple(
        sorted(
            (item for item in elements if item.dimension == topological_dimension),
            key=lambda x: x.tag,
        )
    )
    if not boundary or not cells:
        entity_names = (
            "boundary segments and triangles"
            if topological_dimension == 2
            else ("boundary triangles and tetrahedra")
        )
        raise MeshingError(f"Gmsh {topological_dimension}D import requires both {entity_names}")
    return boundary, cells


def _connectivity(
    elements: tuple[_RawElement, ...], *, node_index: dict[int, int], width: int
) -> np.ndarray:
    connectivity = np.empty((len(elements), width), dtype=np.int32)
    for row, element in enumerate(elements):
        for column, node_tag in enumerate(element.node_tags):
            try:
                connectivity[row, column] = node_index[node_tag]
            except KeyError as error:
                raise MeshingError(
                    f"Gmsh element {element.tag} references missing node {node_tag}"
                ) from error
    return connectivity


def _orient_triangles(
    coordinates: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    points = coordinates[cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    determinants = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if not np.all(np.isfinite(determinants)) or np.any(determinants == 0.0):
        raise MeshingError("Gmsh mesh contains a degenerate triangle")
    clockwise = determinants < 0.0
    permutations = np.tile(np.asarray((0, 1, 2), dtype=np.int8), (cells.shape[0], 1))
    permutations[clockwise] = (0, 2, 1)
    cells[clockwise, 1], cells[clockwise, 2] = (
        cells[clockwise, 2].copy(),
        cells[clockwise, 1].copy(),
    )
    return np.abs(determinants), tuple(
        (int(permutation[0]), int(permutation[1]), int(permutation[2]))
        for permutation in permutations
    )


_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_TETRAHEDRON_OUTWARD_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _orient_tetrahedra(
    coordinates: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    points = coordinates[cells]
    jacobians = np.stack(
        (
            points[:, 1, :] - points[:, 0, :],
            points[:, 2, :] - points[:, 0, :],
            points[:, 3, :] - points[:, 0, :],
        ),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    if not np.all(np.isfinite(determinants)) or np.any(determinants == 0.0):
        raise MeshingError("Gmsh mesh contains a degenerate tetrahedron")
    canonical_sets = tuple(tuple(sorted(int(node) for node in cell)) for cell in cells)
    if len(canonical_sets) != len(set(canonical_sets)):
        raise MeshingError("Gmsh volume connectivity contains duplicate tetrahedra")
    negative = determinants < 0.0
    permutations = np.tile(np.asarray((0, 1, 2, 3), dtype=np.int8), (cells.shape[0], 1))
    permutations[negative] = (0, 1, 3, 2)
    cells[negative, 2], cells[negative, 3] = (
        cells[negative, 3].copy(),
        cells[negative, 2].copy(),
    )
    return np.abs(determinants), tuple(
        (
            int(permutation[0]),
            int(permutation[1]),
            int(permutation[2]),
            int(permutation[3]),
        )
        for permutation in permutations
    )


def _orient_and_validate_tetrahedron_boundary(
    coordinates: np.ndarray,
    cells: np.ndarray,
    facets: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    face_owners: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for cell_index, cell in enumerate(cells):
        for opposite, local_face in enumerate(_TETRAHEDRON_OUTWARD_FACES):
            oriented = tuple(int(cell[local]) for local in local_face)
            ordered = sorted(oriented)
            key = (ordered[0], ordered[1], ordered[2])
            face_owners.setdefault(key, []).append((cell_index, int(cell[opposite])))
    nonmanifold = [face for face, owners in face_owners.items() if len(owners) > 2]
    if nonmanifold:
        raise MeshingError(
            "Gmsh tetrahedral connectivity contains non-manifold faces: "
            f"invalid_count={len(nonmanifold)}"
        )

    facet_keys = []
    for facet in facets:
        ordered = sorted(int(node) for node in facet)
        facet_keys.append((ordered[0], ordered[1], ordered[2]))
    if len(facet_keys) != len(set(facet_keys)):
        raise MeshingError("Gmsh boundary triangle connectivity contains duplicate faces")
    exterior = {face for face, owners in face_owners.items() if len(owners) == 1}
    if set(facet_keys) != exterior:
        missing = len(exterior - set(facet_keys))
        invalid = len(set(facet_keys) - exterior)
        raise MeshingError(
            "Gmsh 3D import requires every external boundary triangle exactly once: "
            f"missing_count={missing}, invalid_count={invalid}"
        )

    permutations = np.tile(np.asarray((0, 1, 2), dtype=np.int8), (facets.shape[0], 1))
    for facet_index, (facet, key) in enumerate(zip(facets, facet_keys, strict=True)):
        _cell_index, opposite_node = face_owners[key][0]
        points = coordinates[facet]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        inward_measure = float(np.dot(normal, coordinates[opposite_node] - points[0]))
        if not math.isfinite(inward_measure) or inward_measure == 0.0:
            raise MeshingError("Gmsh boundary triangle is degenerate")
        if inward_measure > 0.0:
            facets[facet_index, 1], facets[facet_index, 2] = (
                facets[facet_index, 2],
                facets[facet_index, 1],
            )
            permutations[facet_index] = (0, 2, 1)
    return tuple(
        (int(permutation[0]), int(permutation[1]), int(permutation[2]))
        for permutation in permutations
    )


def _permutation_sign(values: tuple[int, int, int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return 1 if inversions % 2 == 0 else -1


def _tetrahedron_orientation_maps(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_edges = cells[:, _TETRAHEDRON_EDGES]
    edge_signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    face_signs = np.empty((cells.shape[0], 4), dtype=np.int8)
    for cell_index, cell in enumerate(cells):
        for face_index, local_face in enumerate(_TETRAHEDRON_OUTWARD_FACES):
            face = tuple(int(cell[local]) for local in local_face)
            canonical = tuple(sorted(face))
            positions = (
                canonical.index(face[0]),
                canonical.index(face[1]),
                canonical.index(face[2]),
            )
            face_signs[cell_index, face_index] = _permutation_sign(positions)
    return edge_signs, face_signs


def _canonical_mesh_sha256(mesh: Mesh) -> str:
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    if mesh.boundary_facets is None:
        raise MeshingError("a canonical Gmsh mesh requires boundary facets")
    boundary_facets = np.asarray(mesh.boundary_facets.connectivity, dtype=np.int64)
    edge_signs = None
    if mesh.orientation.edge_signs is not None:
        edge_signs = np.asarray(mesh.orientation.edge_signs, dtype=np.int8).tolist()
    face_signs = None
    if mesh.orientation.face_signs is not None:
        face_signs = np.asarray(mesh.orientation.face_signs, dtype=np.int8).tolist()
    payload = {
        "schema_version": mesh.schema_version,
        "coordinate_unit": mesh.geometry.coordinate_unit,
        "coordinates_hex": [
            [float(value).hex() for value in coordinate] for coordinate in coordinates
        ],
        "cell_type": mesh.topology.cell_type.value,
        "cells": cells.tolist(),
        "boundary_cell_type": mesh.boundary_facets.cell_type.value,
        "boundary_facets": boundary_facets.tolist(),
        "tags": [
            {
                "name": tag.name,
                "dimension": tag.dimension,
                "entity_ids": list(tag.entity_ids),
            }
            for tag in mesh.tags
        ],
        "orientation": {"edge_signs": edge_signs, "face_signs": face_signs},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_boundary_facets(cells: np.ndarray, facets: np.ndarray) -> None:
    edge_counts: dict[tuple[int, int], int] = {}
    for cell in cells:
        for left, right in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            edge = (min(int(left), int(right)), max(int(left), int(right)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    facet_edges = [
        (min(int(facet[0]), int(facet[1])), max(int(facet[0]), int(facet[1]))) for facet in facets
    ]
    if len(facet_edges) != len(set(facet_edges)):
        raise MeshingError("Gmsh boundary segment connectivity contains duplicate edges")
    invalid = [edge for edge in facet_edges if edge_counts.get(edge) != 1]
    if invalid:
        raise MeshingError(
            "the initial Gmsh importer accepts external boundary facets only; "
            f"invalid_count={len(invalid)}"
        )


def _single_int(value: str, *, context: str) -> int:
    fields = _integer_fields(value, expected=1, context=context)
    return fields[0]


def _integer(value: str, *, context: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise MeshingError(f"invalid integer in Gmsh {context}: {value!r}") from error


def _integer_fields(value: str, *, expected: int, context: str) -> tuple[int, ...]:
    tokens = value.split()
    if len(tokens) != expected:
        raise MeshingError(f"Gmsh {context} requires {expected} integer fields, got {len(tokens)}")
    return tuple(_integer(token, context=context) for token in tokens)


def _float_fields(value: str, *, expected: int, context: str) -> tuple[float, float, float]:
    tokens = value.split()
    if len(tokens) != expected:
        raise MeshingError(f"Gmsh {context} requires {expected} floating-point fields")
    try:
        fields = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise MeshingError(f"invalid floating-point value in Gmsh {context}") from error
    if len(fields) != 3 or not all(math.isfinite(field) for field in fields):
        raise MeshingError(f"Gmsh {context} must contain three finite values")
    return fields
