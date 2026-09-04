"""Fail-closed parser for femx's restricted Elmer ASCII result contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from femx.core.errors import BackendError

_NODE_HEADER = re.compile(r"^\s*Number Of Nodes:\s*(\d+)\s*$", re.IGNORECASE)
_TOTAL_DOF_HEADER = re.compile(r"^\s*Total DOFs:\s*(\d+)\s*$", re.IGNORECASE)
_TIME_HEADER = re.compile(
    r"^\s*Time:\s*(\d+)\s+(\d+)\s+([+\-0-9.eEdD]+)\s*$",
    re.IGNORECASE,
)
_PERM_HEADER = re.compile(r"^\s*Perm:\s*(\d+)\s+(\d+)\s*$", re.IGNORECASE)
_PREVIOUS_PERM_HEADER = re.compile(r"^\s*Perm:\s*use previous\s*$", re.IGNORECASE)
_SCALAR_FIELD_HEADER = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s*:\s*.+$")


@dataclass(frozen=True, slots=True)
class ElmerScalarResult:
    """One full nodal scalar field in original mesh-node order."""

    values: np.ndarray
    save_count: int
    timestep: int
    field_name: str


@dataclass(frozen=True, slots=True)
class ElmerTemperatureResult:
    """One full nodal temperature field in original mesh-node order."""

    values: np.ndarray
    save_count: int
    timestep: int


@dataclass(frozen=True, slots=True)
class ElmerPotentialResult:
    """One full nodal electric-potential field in original mesh-node order."""

    values: np.ndarray
    save_count: int
    timestep: int


@dataclass(frozen=True, slots=True)
class ElmerScalarFieldsResult:
    """A closed set of full nodal scalar fields from one serial save record."""

    values: Mapping[str, np.ndarray]
    save_count: int
    timestep: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class ElmerIndexedScalarField:
    """One scalar field on an explicit subset of original mesh nodes."""

    source_node_ids: np.ndarray
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class ElmerIndexedScalarFieldsResult:
    """A closed set of full or partial nodal scalars from one serial save record."""

    fields: Mapping[str, ElmerIndexedScalarField]
    save_count: int
    timestep: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def _parse_float(token: str, *, label: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise BackendError(f"Elmer result contains an invalid {label}") from error
    if not math.isfinite(value):
        raise BackendError(f"Elmer result contains a non-finite {label}")
    return value


def _normalized_field_name(field_name: str) -> str:
    if (
        not field_name
        or field_name.strip() != field_name
        or not field_name.isascii()
        or not all(character.isalpha() or character == "_" for character in field_name)
    ):
        raise BackendError("expected Elmer scalar field name must be an ASCII identifier")
    return field_name.casefold()


def _read_ascii_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise BackendError(f"Elmer result file does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise BackendError("Elmer result is not an ASCII text file") from error
    if not lines or lines[0].strip() != "ASCII 3":
        raise BackendError("Elmer result must use the restricted ASCII 3 format")
    return lines


def _expected_source_node_ids(
    source_node_ids: Sequence[int],
    *,
    expected_node_count: int,
    field_name: str,
) -> np.ndarray:
    raw = np.asarray(source_node_ids)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise BackendError(
            f"expected Elmer {field_name} source-node ids must be a non-empty integer vector"
        )
    canonical = np.asarray(raw, dtype=np.int64)
    if (
        np.any(canonical < 0)
        or np.any(canonical >= expected_node_count)
        or not np.array_equal(canonical, np.unique(canonical))
    ):
        raise BackendError(
            f"expected Elmer {field_name} source-node ids must be sorted, unique, and in range"
        )
    return canonical


def read_indexed_scalar_fields_result(
    path: Path,
    *,
    expected_node_count: int,
    field_node_ids: Mapping[str, Sequence[int]],
) -> ElmerIndexedScalarFieldsResult:
    """Read a closed set of serial scalar fields with exact full or partial node maps.

    Elmer writes scalar values while traversing positive entries of ``Var % Perm`` in original
    source-node order.  The second column is an internal target-DOF bijection and therefore is
    validated, but it is not used to reorder values.
    """

    if expected_node_count <= 0:
        raise BackendError("expected Elmer result node count must be positive")
    expected: dict[str, np.ndarray] = {}
    for field_name, source_node_ids in field_node_ids.items():
        normalized = _normalized_field_name(field_name)
        if normalized in expected:
            raise BackendError("expected Elmer scalar field names must be unique")
        expected[normalized] = _expected_source_node_ids(
            source_node_ids,
            expected_node_count=expected_node_count,
            field_name=normalized,
        )
    if not expected:
        raise BackendError("expected Elmer scalar field map must not be empty")

    lines = _read_ascii_lines(path)
    declarations: dict[str, tuple[int, int, int]] = {}
    for line in lines:
        match = _SCALAR_FIELD_HEADER.match(line)
        if match is None:
            continue
        name = match.group(1).casefold()
        if name in declarations:
            raise BackendError(f"Elmer result declares duplicate scalar field {name!r}")
        field_size, permutation_size, dofs = (int(value) for value in match.groups()[1:])
        declarations[name] = (field_size, permutation_size, dofs)
    if set(declarations) != set(expected):
        raise BackendError(
            "Elmer result scalar declarations do not match the requested indexed field set"
        )
    for name, expected_ids_array in expected.items():
        expected_header = (expected_ids_array.size, expected_node_count, 1)
        if declarations[name] != expected_header:
            raise BackendError(f"Elmer indexed {name} field header does not match the emitted mesh")

    node_headers = [match for line in lines if (match := _NODE_HEADER.match(line))]
    total_dof_headers = [match for line in lines if (match := _TOTAL_DOF_HEADER.match(line))]
    if len(node_headers) != 1 or int(node_headers[0].group(1)) != expected_node_count:
        raise BackendError("Elmer result node header does not match the emitted mesh")
    if len(total_dof_headers) != 1 or int(total_dof_headers[0].group(1)) != len(expected):
        raise BackendError("Elmer result total scalar DOFs do not match the indexed field set")

    time_indices = [index for index, line in enumerate(lines) if _TIME_HEADER.match(line)]
    if len(time_indices) != 1:
        raise BackendError("Elmer result must contain exactly one time record")
    time_index = time_indices[0]
    time_match = _TIME_HEADER.match(lines[time_index])
    assert time_match is not None
    save_count = int(time_match.group(1))
    timestep = int(time_match.group(2))
    _parse_float(time_match.group(3), label="simulation time")

    cursor = time_index + 1
    previous_source_ids: tuple[int, ...] | None = None
    parsed: dict[str, ElmerIndexedScalarField] = {}
    for _field_index in range(len(expected)):
        if cursor >= len(lines):
            raise BackendError("Elmer result is missing a requested indexed scalar section")
        section_name = lines[cursor].strip().casefold()
        if not section_name:
            raise BackendError("Elmer result is missing a requested indexed scalar section")
        if section_name not in expected or section_name in parsed:
            raise BackendError("Elmer result contains an unexpected indexed scalar section")
        expected_ids = expected[section_name]
        expected_source_ids = tuple(int(node_id) + 1 for node_id in expected_ids)
        active_count = len(expected_source_ids)
        cursor += 1
        if cursor >= len(lines):
            raise BackendError(f"Elmer result is missing the {section_name} permutation")

        perm_match = _PERM_HEADER.match(lines[cursor])
        if perm_match is not None:
            declared_size, positive_count = (int(value) for value in perm_match.groups())
            if (declared_size, positive_count) != (expected_node_count, active_count):
                raise BackendError(
                    f"Elmer result {section_name} permutation size or positive count differs"
                )
            cursor += 1
            if cursor + active_count > len(lines):
                raise BackendError(f"Elmer result is missing the {section_name} permutation pairs")
            source_ids: list[int] = []
            target_ids: list[int] = []
            for line in lines[cursor : cursor + active_count]:
                tokens = line.split()
                if len(tokens) != 2:
                    raise BackendError("Elmer result contains a malformed permutation pair")
                try:
                    source_id, target_id = (int(token) for token in tokens)
                except ValueError as error:
                    raise BackendError(
                        "Elmer result contains a non-integer permutation pair"
                    ) from error
                source_ids.append(source_id)
                target_ids.append(target_id)
            if tuple(source_ids) != expected_source_ids:
                raise BackendError(
                    f"Elmer result {section_name} source-node map differs from the expected subset"
                )
            if sorted(target_ids) != list(range(1, active_count + 1)):
                raise BackendError(f"Elmer result {section_name} target-DOF map is not a bijection")
            previous_source_ids = tuple(source_ids)
            cursor += active_count
        elif _PREVIOUS_PERM_HEADER.match(lines[cursor]) is not None:
            if previous_source_ids is None:
                raise BackendError("Elmer result cannot reuse a missing previous permutation")
            if previous_source_ids != expected_source_ids:
                raise BackendError(
                    f"Elmer result {section_name} reused permutation differs from its node subset"
                )
            source_ids = list(previous_source_ids)
            cursor += 1
        else:
            raise BackendError(
                f"Elmer result requires an explicit or reused {section_name} permutation"
            )

        if cursor + active_count > len(lines):
            raise BackendError(f"Elmer result is missing {section_name} scalar records")
        values = np.empty((active_count,), dtype=np.float64)
        for value_index, line in enumerate(lines[cursor : cursor + active_count]):
            tokens = line.split()
            if len(tokens) != 1:
                raise BackendError(f"Elmer result {section_name} record must contain one scalar")
            values[value_index] = _parse_float(tokens[0], label=section_name)
        parsed[section_name] = ElmerIndexedScalarField(
            source_node_ids=np.asarray(source_ids, dtype=np.int64) - 1,
            values=values,
        )
        cursor += active_count
    if cursor != len(lines):
        raise BackendError("Elmer result contains unexpected trailing indexed scalar records")
    return ElmerIndexedScalarFieldsResult(
        fields=parsed,
        save_count=save_count,
        timestep=timestep,
    )


def read_scalar_fields_result(
    path: Path,
    *,
    expected_node_count: int,
    field_names: Sequence[str],
) -> ElmerScalarFieldsResult:
    """Read an exact closed set of serial nodal scalars, including shared permutations."""

    if expected_node_count <= 0:
        raise BackendError("expected Elmer result node count must be positive")
    normalized_names = tuple(_normalized_field_name(name) for name in field_names)
    if not normalized_names or len(normalized_names) != len(set(normalized_names)):
        raise BackendError("expected Elmer scalar field names must be non-empty and unique")
    expected_names = set(normalized_names)
    lines = _read_ascii_lines(path)

    declarations: dict[str, tuple[int, int, int]] = {}
    for line in lines:
        match = _SCALAR_FIELD_HEADER.match(line)
        if match is None:
            continue
        name = match.group(1).casefold()
        if name in declarations:
            raise BackendError(f"Elmer result declares duplicate scalar field {name!r}")
        field_size, permutation_size, dofs = (int(value) for value in match.groups()[1:])
        declarations[name] = (field_size, permutation_size, dofs)
    if set(declarations) != expected_names:
        raise BackendError(
            "Elmer result scalar declarations do not match the requested closed field set"
        )
    expected_header = (expected_node_count, expected_node_count, 1)
    if any(header != expected_header for header in declarations.values()):
        raise BackendError("Elmer scalar field header does not match the emitted mesh")

    node_headers = [match for line in lines if (match := _NODE_HEADER.match(line))]
    total_dof_headers = [match for line in lines if (match := _TOTAL_DOF_HEADER.match(line))]
    if len(node_headers) != 1 or int(node_headers[0].group(1)) != expected_node_count:
        raise BackendError("Elmer result node header does not match the emitted mesh")
    if len(total_dof_headers) != 1 or int(total_dof_headers[0].group(1)) != len(normalized_names):
        raise BackendError("Elmer result total scalar DOFs do not match the requested field set")

    time_indices = [index for index, line in enumerate(lines) if _TIME_HEADER.match(line)]
    if len(time_indices) != 1:
        raise BackendError("Elmer result must contain exactly one time record")
    time_index = time_indices[0]
    time_match = _TIME_HEADER.match(lines[time_index])
    assert time_match is not None
    save_count = int(time_match.group(1))
    timestep = int(time_match.group(2))
    _parse_float(time_match.group(3), label="simulation time")

    cursor = time_index + 1
    previous_source_ids: list[int] | None = None
    parsed: dict[str, np.ndarray] = {}
    expected_ids = list(range(1, expected_node_count + 1))
    for _field_index in range(len(normalized_names)):
        if cursor >= len(lines):
            raise BackendError("Elmer result is missing a requested scalar field section")
        section_name = lines[cursor].strip().casefold()
        if not section_name:
            raise BackendError("Elmer result is missing a requested scalar field section")
        if section_name not in expected_names or section_name in parsed:
            raise BackendError("Elmer result contains an unexpected scalar field section")
        cursor += 1
        if cursor >= len(lines):
            raise BackendError(f"Elmer result is missing the {section_name} permutation")

        perm_match = _PERM_HEADER.match(lines[cursor])
        if perm_match is not None:
            declared_size, positive_count = (int(value) for value in perm_match.groups())
            if (declared_size, positive_count) != (
                expected_node_count,
                expected_node_count,
            ):
                raise BackendError(
                    f"Elmer result {section_name} permutation is not a full nodal map"
                )
            cursor += 1
            source_ids: list[int] = []
            target_ids: list[int] = []
            for line in lines[cursor : cursor + expected_node_count]:
                tokens = line.split()
                if len(tokens) != 2:
                    raise BackendError("Elmer result contains a malformed permutation pair")
                try:
                    source_id, target_id = (int(token) for token in tokens)
                except ValueError as error:
                    raise BackendError(
                        "Elmer result contains a non-integer permutation pair"
                    ) from error
                source_ids.append(source_id)
                target_ids.append(target_id)
            if source_ids != expected_ids or sorted(target_ids) != expected_ids:
                raise BackendError(
                    f"Elmer result {section_name} permutation is not a nodal bijection"
                )
            previous_source_ids = source_ids
            cursor += expected_node_count
        elif _PREVIOUS_PERM_HEADER.match(lines[cursor]) is not None:
            if previous_source_ids is None:
                raise BackendError("Elmer result cannot reuse a missing previous permutation")
            source_ids = previous_source_ids
            cursor += 1
        else:
            raise BackendError(f"Elmer result requires a full or reused {section_name} permutation")

        if cursor + expected_node_count > len(lines):
            raise BackendError(f"Elmer result is missing {section_name} scalar records")
        values = np.empty((expected_node_count,), dtype=np.float64)
        for source_id, line in zip(
            source_ids,
            lines[cursor : cursor + expected_node_count],
            strict=True,
        ):
            tokens = line.split()
            if len(tokens) != 1:
                raise BackendError(f"Elmer result {section_name} record must contain one scalar")
            values[source_id - 1] = _parse_float(tokens[0], label=section_name)
        parsed[section_name] = values
        cursor += expected_node_count
    if cursor != len(lines):
        raise BackendError("Elmer result contains unexpected trailing scalar records")
    return ElmerScalarFieldsResult(
        values=parsed,
        save_count=save_count,
        timestep=timestep,
    )


def read_scalar_result(
    path: Path,
    *,
    expected_node_count: int,
    field_name: str,
) -> ElmerScalarResult:
    """Read one explicitly named serial scalar field from restricted ASCII 3 output."""

    if expected_node_count <= 0:
        raise BackendError("expected Elmer result node count must be positive")
    normalized_field = _normalized_field_name(field_name)
    field_header = re.compile(
        rf"^\s*{re.escape(field_name)}\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s*:\s*.+$",
        re.IGNORECASE,
    )
    lines = _read_ascii_lines(path)

    field_headers = [match for line in lines if (match := field_header.match(line))]
    if len(field_headers) != 1:
        raise BackendError(f"Elmer result must declare exactly one scalar {normalized_field} field")
    field_size, perm_size, dofs = (int(value) for value in field_headers[0].groups())
    if (field_size, perm_size, dofs) != (expected_node_count, expected_node_count, 1):
        raise BackendError(f"Elmer {normalized_field} field header does not match the emitted mesh")

    node_headers = [match for line in lines if (match := _NODE_HEADER.match(line))]
    total_dof_headers = [match for line in lines if (match := _TOTAL_DOF_HEADER.match(line))]
    if len(node_headers) != 1 or int(node_headers[0].group(1)) != expected_node_count:
        raise BackendError("Elmer result node header does not match the emitted mesh")
    if len(total_dof_headers) != 1 or int(total_dof_headers[0].group(1)) != 1:
        raise BackendError("Elmer result must contain exactly one scalar degree of freedom")

    time_indices = [index for index, line in enumerate(lines) if _TIME_HEADER.match(line)]
    field_indices = [
        index for index, line in enumerate(lines) if line.strip().casefold() == normalized_field
    ]
    if len(time_indices) != 1 or len(field_indices) != 1:
        raise BackendError(
            f"Elmer result must contain exactly one time and {normalized_field} section"
        )
    time_index = time_indices[0]
    field_index = field_indices[0]
    if field_index != time_index + 1:
        raise BackendError(f"Elmer {normalized_field} section is not adjacent to its time record")
    time_match = _TIME_HEADER.match(lines[time_index])
    assert time_match is not None
    save_count = int(time_match.group(1))
    timestep = int(time_match.group(2))
    _parse_float(time_match.group(3), label="simulation time")

    perm_index = field_index + 1
    if perm_index >= len(lines):
        raise BackendError(f"Elmer result is missing the {normalized_field} permutation")
    perm_match = _PERM_HEADER.match(lines[perm_index])
    if perm_match is None:
        raise BackendError(f"Elmer result requires an explicit full {normalized_field} permutation")
    declared_size, positive_count = (int(value) for value in perm_match.groups())
    if (declared_size, positive_count) != (expected_node_count, expected_node_count):
        raise BackendError(f"Elmer result {normalized_field} permutation is not a full nodal map")

    pair_start = perm_index + 1
    value_start = pair_start + expected_node_count
    value_end = value_start + expected_node_count
    if value_end != len(lines):
        raise BackendError(
            f"Elmer result has missing or unexpected trailing {normalized_field} records"
        )
    source_ids: list[int] = []
    target_ids: list[int] = []
    for line in lines[pair_start:value_start]:
        tokens = line.split()
        if len(tokens) != 2:
            raise BackendError("Elmer result contains a malformed permutation pair")
        try:
            source_id, target_id = (int(token) for token in tokens)
        except ValueError as error:
            raise BackendError("Elmer result contains a non-integer permutation pair") from error
        source_ids.append(source_id)
        target_ids.append(target_id)
    expected_ids = list(range(1, expected_node_count + 1))
    if source_ids != expected_ids or sorted(target_ids) != expected_ids:
        raise BackendError(f"Elmer result {normalized_field} permutation is not a nodal bijection")

    values = np.empty((expected_node_count,), dtype=np.float64)
    for source_id, line in zip(source_ids, lines[value_start:value_end], strict=True):
        tokens = line.split()
        if len(tokens) != 1:
            raise BackendError(f"Elmer result {normalized_field} record must contain one scalar")
        values[source_id - 1] = _parse_float(tokens[0], label=normalized_field)
    return ElmerScalarResult(
        values=values,
        save_count=save_count,
        timestep=timestep,
        field_name=normalized_field,
    )


def read_temperature_result(path: Path, *, expected_node_count: int) -> ElmerTemperatureResult:
    """Read exactly the serial scalar result emitted by the steady-heat adapter."""

    result = read_scalar_result(
        path,
        expected_node_count=expected_node_count,
        field_name="temperature",
    )
    return ElmerTemperatureResult(
        values=result.values,
        save_count=result.save_count,
        timestep=result.timestep,
    )


def read_potential_result(path: Path, *, expected_node_count: int) -> ElmerPotentialResult:
    """Read exactly the serial scalar result emitted by the steady-current adapter."""

    result = read_scalar_result(
        path,
        expected_node_count=expected_node_count,
        field_name="potential",
    )
    return ElmerPotentialResult(
        values=result.values,
        save_count=result.save_count,
        timestep=result.timestep,
    )
