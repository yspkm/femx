"""Fail-closed ingestion of Elmer EMPort logs and projected complex nodal fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from femx.backends.elmer.result import _parse_float, _read_ascii_lines
from femx.core.errors import BackendError

_TIME_HEADER = re.compile(
    r"^\s*Time:\s*(\d+)\s+(\d+)\s+([+\-0-9.eEdD]+)\s*$",
    re.IGNORECASE,
)
_PERM_HEADER = re.compile(r"^\s*Perm:\s*(\d+)\s+(\d+)\s*$", re.IGNORECASE)
_PREVIOUS_PERM_HEADER = re.compile(r"^\s*Perm:\s*use previous\s*$", re.IGNORECASE)
_NODE_HEADER = re.compile(r"^\s*Number Of Nodes:\s*(\d+)\s*$", re.IGNORECASE)
_TOTAL_DOF_HEADER = re.compile(r"^\s*Total DOFs:\s*(\d+)\s*$", re.IGNORECASE)
_COMBINED_DECLARATION = re.compile(
    r"^\s*ef2d\[ef2d re:3 ef2d im:3\]\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s*:.+$",
    re.IGNORECASE,
)
_MIXED_COMBINED_DECLARATION = re.compile(
    r"^\s*eport\[eport re:1 eport im:1\]\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s*:.+$",
    re.IGNORECASE,
)
_MIXED_COMPONENT_NAMES = ("eport re", "eport im")
_MIXED_COMPONENT_DECLARATION = re.compile(
    r"^\s*(eport\s+(?:re|im))\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s*:.+$",
    re.IGNORECASE,
)
_COMPONENT_NAMES = (
    "ef2d re 1",
    "ef2d re 2",
    "ef2d re 3",
    "ef2d im 1",
    "ef2d im 2",
    "ef2d im 3",
)
_COMPONENT_DECLARATION = re.compile(
    r"^\s*(ef2d\s+(?:re|im)\s+[123])\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s*:.+$",
    re.IGNORECASE,
)
_NUMERIC_DECLARATION = re.compile(r"^\s*.+?\s*:\s*\d+\s+\d+\s+\d+\s*:.+$")


@dataclass(frozen=True, slots=True)
class ElmerProjectedElectricFieldResult:
    """One unambiguous complex Cartesian nodal projection from repeated save records."""

    values: np.ndarray
    record_count: int
    final_save_count: int
    final_timestep: int
    permutation_size: int


@dataclass(frozen=True, slots=True)
class ElmerPortMixedEigenvectors:
    """Raw mixed eigenvectors in nodal-first, Elmer-edge-order coefficient space."""

    nodal_coefficients: np.ndarray
    edge_coefficients: np.ndarray
    target_permutation: np.ndarray
    record_count: int
    final_zero_record_verified: bool


@dataclass(frozen=True, slots=True)
class ElmerPortEigenmodeResult:
    """Closed raw mixed-spectrum and selected projected-field result contract."""

    mixed: ElmerPortMixedEigenvectors
    projected: ElmerProjectedElectricFieldResult


@dataclass(frozen=True, slots=True)
class ElmerPortEigenLog:
    """Full computed spectrum, residuals, selected beta, power, and impedance evidence."""

    eigenvalues: np.ndarray
    residuals: np.ndarray
    reported_tolerance: float
    iterations: int
    converged_count: int
    selected_beta: complex
    raw_forward_power_w: float
    port_impedance_ohm: float


def _one_match(pattern: re.Pattern[str], text: str, *, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise BackendError(f"Elmer port log must contain exactly one {label} record")
    return matches[0]


def _log_float(token: str, *, label: str) -> float:
    return _parse_float(token, label=f"port {label}")


def parse_port_eigenmode_log(
    stdout: str,
    *,
    selected_mode_index: int,
) -> ElmerPortEigenLog:
    """Parse one single-port complex eigensolve without accepting partial or ambiguous blocks."""

    if selected_mode_index < 0:
        raise BackendError("selected Elmer port mode index cannot be negative")
    tolerance_match = _one_match(
        re.compile(
            r"EigenSolveComplex:\s+Convergence criterion is:\s+(\S+)",
            re.IGNORECASE,
        ),
        stdout,
        label="eigen convergence tolerance",
    )
    iterations_match = _one_match(
        re.compile(
            r"EigenSolveComplex:\s+Number of eigensystem iterations is:\s+(\d+)",
            re.IGNORECASE,
        ),
        stdout,
        label="eigensystem iteration",
    )
    converged_match = _one_match(
        re.compile(
            r"EigenSolveComplex:\s+Number of converged Ritz values is:\s+(\d+)",
            re.IGNORECASE,
        ),
        stdout,
        label="converged Ritz count",
    )
    eigen_pattern = re.compile(
        r"^EigenSolveComplex:\s+(\d+)\s+\(\s*([^,\s]+)\s*,\s*([^\)\s]+)\s*\)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    eigen_matches = list(eigen_pattern.finditer(stdout))
    if not eigen_matches:
        raise BackendError("Elmer port log contains no computed complex eigenvalues")
    indices = [int(match.group(1)) for match in eigen_matches]
    if indices != list(range(1, len(indices) + 1)):
        raise BackendError("Elmer port eigenvalue indices must be contiguous and one-based")
    eigenvalues = np.asarray(
        [
            complex(
                _log_float(match.group(2), label="eigenvalue real part"),
                _log_float(match.group(3), label="eigenvalue imaginary part"),
            )
            for match in eigen_matches
        ],
        dtype=np.complex128,
    )
    residual_pattern = re.compile(
        r"^CheckResidualsComplex:\s+L\^2 Norm of the residual:\s+(\d+)\s+(\S+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    residual_matches = list(residual_pattern.finditer(stdout))
    residual_indices = [int(match.group(1)) for match in residual_matches]
    if residual_indices != indices:
        raise BackendError("Elmer port residual indices do not match the computed spectrum")
    residuals = np.asarray(
        [_log_float(match.group(2), label="eigen residual") for match in residual_matches],
        dtype=np.float64,
    )
    if np.any(residuals < 0.0):
        raise BackendError("Elmer port eigen residuals cannot be negative")
    if selected_mode_index >= eigenvalues.size:
        raise BackendError("selected Elmer port mode is absent from the computed spectrum")

    reported_tolerance = _log_float(
        tolerance_match.group(1),
        label="eigen convergence tolerance",
    )
    if reported_tolerance <= 0.0:
        raise BackendError("Elmer port reported a nonpositive eigen convergence tolerance")
    iterations = int(iterations_match.group(1))
    converged_count = int(converged_match.group(1))
    if converged_count < 0 or converged_count > eigenvalues.size:
        raise BackendError("Elmer port converged Ritz count is inconsistent with its spectrum")

    selected_beta = complex(np.sqrt(-eigenvalues[selected_mode_index]))
    beta_match = _one_match(
        re.compile(
            r"EMPortSolver:\s+Propagation constant beta:\s+(\S+)\s+(\S+)",
            re.IGNORECASE,
        ),
        stdout,
        label="propagation constant",
    )
    beta_log = complex(
        _log_float(beta_match.group(1), label="beta real part"),
        _log_float(beta_match.group(2), label="beta imaginary part"),
    )
    beta_scale = max(1.0, abs(selected_beta))
    if abs(beta_log - selected_beta) > 1.0e-6 * beta_scale:
        raise BackendError("Elmer port beta log disagrees with sqrt(-eigenvalue)")

    scalar_beta_match = _one_match(
        re.compile(
            r"SaveScalars:\s+\d+:\s+res:\s+port beta 1\s+(\S+)",
            re.IGNORECASE,
        ),
        stdout,
        label="high-precision beta scalar",
    )
    scalar_beta = _log_float(scalar_beta_match.group(1), label="beta scalar")
    if abs(scalar_beta - selected_beta.real) > 1.0e-11 * beta_scale:
        raise BackendError("Elmer port beta scalar disagrees with sqrt(-eigenvalue)")

    coarse_power_match = _one_match(
        re.compile(r"EMPortSolver:\s+Port power:\s+(\S+)", re.IGNORECASE),
        stdout,
        label="port power",
    )
    scalar_power_match = _one_match(
        re.compile(
            r"SaveScalars:\s+\d+:\s+res:\s+port power 1\s+(\S+)",
            re.IGNORECASE,
        ),
        stdout,
        label="high-precision port power scalar",
    )
    impedance_match = _one_match(
        re.compile(
            r"SaveScalars:\s+\d+:\s+res:\s+port impedance 1\s+(\S+)",
            re.IGNORECASE,
        ),
        stdout,
        label="port impedance scalar",
    )
    coarse_power = _log_float(coarse_power_match.group(1), label="coarse power")
    raw_power = _log_float(scalar_power_match.group(1), label="power scalar")
    impedance = _log_float(impedance_match.group(1), label="impedance scalar")
    if raw_power <= 0.0 or impedance <= 0.0:
        raise BackendError("lossless positive-z Elmer port power and impedance must be positive")
    if abs(coarse_power - raw_power) > 1.0e-6 * max(1.0, abs(raw_power)):
        raise BackendError("Elmer port coarse and high-precision power records disagree")
    return ElmerPortEigenLog(
        eigenvalues=eigenvalues,
        residuals=residuals,
        reported_tolerance=reported_tolerance,
        iterations=iterations,
        converged_count=converged_count,
        selected_beta=selected_beta,
        raw_forward_power_w=raw_power,
        port_impedance_ohm=impedance,
    )


def _result_declarations(
    lines: list[str],
    *,
    expected_node_count: int,
) -> int:
    total_indices = [index for index, line in enumerate(lines) if _TOTAL_DOF_HEADER.match(line)]
    if len(total_indices) != 1:
        raise BackendError("Elmer port result must contain exactly one total-DOF header")
    declaration_lines = [
        line for line in lines[1 : total_indices[0]] if _NUMERIC_DECLARATION.match(line)
    ]
    if len(declaration_lines) != 7:
        raise BackendError("Elmer port result must declare one vector and six scalar components")
    combined = _COMBINED_DECLARATION.match(declaration_lines[0])
    if combined is None:
        raise BackendError("Elmer port result is missing its EF2D vector declaration")
    total_values, permutation_size, dofs = (int(value) for value in combined.groups())
    if total_values != 6 * expected_node_count or dofs != 6:
        raise BackendError("Elmer port EF2D declaration does not match the emitted mesh")
    observed_names: list[str] = []
    for line in declaration_lines[1:]:
        match = _COMPONENT_DECLARATION.match(line)
        if match is None:
            raise BackendError("Elmer port result contains an invalid component declaration")
        name = " ".join(match.group(1).casefold().split())
        field_size, component_permutation_size, component_dofs = (
            int(value) for value in match.groups()[1:]
        )
        if (
            field_size != expected_node_count
            or component_permutation_size != permutation_size
            or component_dofs != 1
        ):
            raise BackendError("Elmer port component declaration does not match the emitted mesh")
        observed_names.append(name)
    if tuple(observed_names) != _COMPONENT_NAMES:
        raise BackendError("Elmer port component declarations are not in the canonical order")
    return permutation_size


def read_port_electric_field_result(
    path: Path,
    *,
    expected_node_count: int,
) -> ElmerProjectedElectricFieldResult:
    """Read six double-precision EF2D components and reject differing repeated records."""

    if expected_node_count <= 0:
        raise BackendError("expected Elmer port node count must be positive")
    lines = _read_ascii_lines(path)
    permutation_size = _result_declarations(lines, expected_node_count=expected_node_count)
    node_entries = [
        (index, match) for index, line in enumerate(lines) if (match := _NODE_HEADER.match(line))
    ]
    total_matches = [match for line in lines if (match := _TOTAL_DOF_HEADER.match(line))]
    if len(node_entries) != 1 or int(node_entries[0][1].group(1)) != expected_node_count:
        raise BackendError("Elmer port result node header does not match the emitted mesh")
    if len(total_matches) != 1 or int(total_matches[0].group(1)) != 6:
        raise BackendError("Elmer port result must contain exactly six scalar DOFs")
    time_indices = [index for index, line in enumerate(lines) if _TIME_HEADER.match(line)]
    if not time_indices:
        raise BackendError("Elmer port result contains no save records")
    if time_indices[0] != node_entries[0][0] + 1:
        raise BackendError("Elmer port save records must begin immediately after the node header")
    cursor = time_indices[0]
    records: list[np.ndarray] = []
    save_counts: list[int] = []
    timesteps: list[int] = []
    expected_ids = list(range(1, expected_node_count + 1))
    while cursor < len(lines):
        time_match = _TIME_HEADER.match(lines[cursor])
        if time_match is None:
            raise BackendError("Elmer port result contains text outside a save record")
        save_counts.append(int(time_match.group(1)))
        timesteps.append(int(time_match.group(2)))
        _parse_float(time_match.group(3), label="port simulation time")
        cursor += 1
        components: list[np.ndarray] = []
        source_ids: list[int] | None = None
        for component_index, expected_name in enumerate(_COMPONENT_NAMES):
            if (
                cursor >= len(lines)
                or " ".join(lines[cursor].strip().casefold().split()) != expected_name
            ):
                raise BackendError("Elmer port result is missing a canonical EF2D component")
            cursor += 1
            if cursor >= len(lines):
                raise BackendError("Elmer port result is missing an EF2D permutation")
            if component_index == 0:
                perm_match = _PERM_HEADER.match(lines[cursor])
                if perm_match is None:
                    raise BackendError("Elmer port result requires a full first-component map")
                declared_size, positive_count = (int(value) for value in perm_match.groups())
                if declared_size != permutation_size or positive_count != expected_node_count:
                    raise BackendError("Elmer port result permutation header is inconsistent")
                cursor += 1
                source_ids = []
                target_ids: list[int] = []
                for line in lines[cursor : cursor + expected_node_count]:
                    tokens = line.split()
                    if len(tokens) != 2:
                        raise BackendError("Elmer port result contains a malformed permutation")
                    try:
                        source_id, target_id = (int(token) for token in tokens)
                    except ValueError as error:
                        raise BackendError(
                            "Elmer port result contains a non-integer permutation"
                        ) from error
                    source_ids.append(source_id)
                    target_ids.append(target_id)
                if source_ids != expected_ids or sorted(target_ids) != expected_ids:
                    raise BackendError("Elmer port result permutation is not a nodal bijection")
                cursor += expected_node_count
            else:
                if _PREVIOUS_PERM_HEADER.match(lines[cursor]) is None:
                    raise BackendError("Elmer port result must reuse the first-component map")
                cursor += 1
            assert source_ids is not None
            if cursor + expected_node_count > len(lines):
                raise BackendError("Elmer port result is missing EF2D values")
            values = np.empty(expected_node_count, dtype=np.float64)
            for source_id, line in zip(
                source_ids,
                lines[cursor : cursor + expected_node_count],
                strict=True,
            ):
                tokens = line.split()
                if len(tokens) != 1:
                    raise BackendError("Elmer port component records must contain one scalar")
                values[source_id - 1] = _parse_float(tokens[0], label=expected_name)
            components.append(values)
            cursor += expected_node_count
        field = np.column_stack(components[:3]) + 1j * np.column_stack(components[3:])
        records.append(np.asarray(field, dtype=np.complex128))
    if save_counts != list(range(save_counts[0], save_counts[0] + len(save_counts))):
        raise BackendError("Elmer port save counts must be contiguous")
    reference = records[0]
    if any(not np.array_equal(record, reference) for record in records[1:]):
        raise BackendError("Elmer port repeated EF2D records disagree for the selected mode")
    return ElmerProjectedElectricFieldResult(
        values=reference,
        record_count=len(records),
        final_save_count=save_counts[-1],
        final_timestep=timesteps[-1],
        permutation_size=permutation_size,
    )


def _raw_port_result_declarations(
    lines: list[str],
    *,
    expected_node_count: int,
    expected_edge_count: int,
) -> int:
    mixed_count = expected_node_count + expected_edge_count
    total_indices = [index for index, line in enumerate(lines) if _TOTAL_DOF_HEADER.match(line)]
    if len(total_indices) != 1:
        raise BackendError("Elmer raw port result must contain exactly one total-DOF header")
    declaration_lines = [
        line for line in lines[1 : total_indices[0]] if _NUMERIC_DECLARATION.match(line)
    ]
    if len(declaration_lines) != 10:
        raise BackendError("Elmer raw port result must declare two vectors and eight components")

    mixed = _MIXED_COMBINED_DECLARATION.match(declaration_lines[0])
    projected = _COMBINED_DECLARATION.match(declaration_lines[1])
    if mixed is None or projected is None:
        raise BackendError("Elmer raw port vector declarations are missing or out of order")
    if tuple(int(value) for value in mixed.groups()) != (2 * mixed_count, mixed_count, 2):
        raise BackendError("Elmer raw mixed-vector declaration does not match the emitted mesh")
    if tuple(int(value) for value in projected.groups()) != (
        6 * expected_node_count,
        mixed_count,
        6,
    ):
        raise BackendError("Elmer raw projected-vector declaration does not match the emitted mesh")

    observed_mixed_names: list[str] = []
    for line in declaration_lines[2:4]:
        match = _MIXED_COMPONENT_DECLARATION.match(line)
        if match is None:
            raise BackendError("Elmer raw port result contains an invalid mixed component")
        name = " ".join(match.group(1).casefold().split())
        shape = tuple(int(value) for value in match.groups()[1:])
        if shape != (mixed_count, mixed_count, 1):
            raise BackendError("Elmer raw mixed component does not match the emitted mesh")
        observed_mixed_names.append(name)
    if tuple(observed_mixed_names) != _MIXED_COMPONENT_NAMES:
        raise BackendError("Elmer raw mixed components are not in canonical order")

    observed_projected_names: list[str] = []
    for line in declaration_lines[4:]:
        match = _COMPONENT_DECLARATION.match(line)
        if match is None:
            raise BackendError("Elmer raw port result contains an invalid projected component")
        name = " ".join(match.group(1).casefold().split())
        shape = tuple(int(value) for value in match.groups()[1:])
        if shape != (expected_node_count, mixed_count, 1):
            raise BackendError("Elmer raw projected component does not match the emitted mesh")
        observed_projected_names.append(name)
    if tuple(observed_projected_names) != _COMPONENT_NAMES:
        raise BackendError("Elmer raw projected components are not in canonical order")
    return mixed_count


def _full_permutation(
    lines: list[str],
    cursor: int,
    *,
    permutation_size: int,
    value_count: int,
    label: str,
) -> tuple[list[int], np.ndarray, int]:
    if cursor >= len(lines):
        raise BackendError(f"Elmer raw port result is missing the {label} permutation")
    match = _PERM_HEADER.match(lines[cursor])
    if match is None:
        raise BackendError(f"Elmer raw port result requires a full {label} permutation")
    if tuple(int(value) for value in match.groups()) != (permutation_size, value_count):
        raise BackendError(f"Elmer raw port {label} permutation header is inconsistent")
    cursor += 1
    if cursor + value_count > len(lines):
        raise BackendError(f"Elmer raw port result is missing the {label} permutation values")
    source_ids: list[int] = []
    target_ids: list[int] = []
    for line in lines[cursor : cursor + value_count]:
        tokens = line.split()
        if len(tokens) != 2:
            raise BackendError(f"Elmer raw port {label} permutation is malformed")
        try:
            source_id, target_id = (int(token) for token in tokens)
        except ValueError as error:
            raise BackendError(f"Elmer raw port {label} permutation is non-integer") from error
        source_ids.append(source_id)
        target_ids.append(target_id)
    expected_ids = list(range(1, value_count + 1))
    if source_ids != expected_ids or sorted(target_ids) != expected_ids:
        raise BackendError(f"Elmer raw port {label} permutation is not a bijection")
    return source_ids, np.asarray(target_ids, dtype=np.int64), cursor + value_count


def _scalar_values(
    lines: list[str],
    cursor: int,
    *,
    source_ids: list[int],
    label: str,
) -> tuple[np.ndarray, int]:
    value_count = len(source_ids)
    if cursor + value_count > len(lines):
        raise BackendError(f"Elmer raw port result is missing {label} values")
    values = np.empty(value_count, dtype=np.float64)
    for source_id, line in zip(
        source_ids,
        lines[cursor : cursor + value_count],
        strict=True,
    ):
        tokens = line.split()
        if len(tokens) != 1:
            raise BackendError(f"Elmer raw port {label} records must contain one scalar")
        values[source_id - 1] = _parse_float(tokens[0], label=label)
    return values, cursor + value_count


def _require_section(lines: list[str], cursor: int, *, name: str) -> int:
    if cursor >= len(lines) or " ".join(lines[cursor].strip().casefold().split()) != name:
        raise BackendError(f"Elmer raw port result is missing canonical {name}")
    return cursor + 1


def _require_previous_permutation(lines: list[str], cursor: int, *, label: str) -> int:
    if cursor >= len(lines) or _PREVIOUS_PERM_HEADER.match(lines[cursor]) is None:
        raise BackendError(f"Elmer raw port result must reuse the {label} permutation")
    return cursor + 1


def read_port_eigenmode_result(
    path: Path,
    *,
    expected_node_count: int,
    expected_edge_count: int,
    expected_mode_count: int,
) -> ElmerPortEigenmodeResult:
    """Read every raw eigenvector and the repeated selected nodal projection.

    The locked Elmer save path emits one raw mixed record per requested eigenvector followed by
    one exact-zero final record.  ``EF2D`` is the selected mode's L2 nodal projection and must be
    bitwise identical in every record.  Any layout or sequence drift fails closed.
    """

    if expected_node_count <= 0 or expected_edge_count <= 0 or expected_mode_count <= 0:
        raise BackendError("expected Elmer raw port counts must all be positive")
    lines = _read_ascii_lines(path)
    mixed_count = _raw_port_result_declarations(
        lines,
        expected_node_count=expected_node_count,
        expected_edge_count=expected_edge_count,
    )
    node_entries = [
        (index, match) for index, line in enumerate(lines) if (match := _NODE_HEADER.match(line))
    ]
    total_matches = [match for line in lines if (match := _TOTAL_DOF_HEADER.match(line))]
    if len(node_entries) != 1 or int(node_entries[0][1].group(1)) != expected_node_count:
        raise BackendError("Elmer raw port node header does not match the emitted mesh")
    if len(total_matches) != 1 or int(total_matches[0].group(1)) != 8:
        raise BackendError("Elmer raw port result must contain exactly eight scalar DOFs")
    time_indices = [index for index, line in enumerate(lines) if _TIME_HEADER.match(line)]
    if not time_indices:
        raise BackendError("Elmer raw port result contains no save records")
    if time_indices[0] != node_entries[0][0] + 1:
        raise BackendError("Elmer raw port save records must begin after the node header")

    cursor = time_indices[0]
    save_steps: list[tuple[int, int]] = []
    raw_records: list[np.ndarray] = []
    projected_records: list[np.ndarray] = []
    raw_target_reference: np.ndarray | None = None
    projected_target_reference: np.ndarray | None = None
    while cursor < len(lines):
        time_match = _TIME_HEADER.match(lines[cursor])
        if time_match is None:
            raise BackendError("Elmer raw port result contains text outside a save record")
        save_steps.append((int(time_match.group(1)), int(time_match.group(2))))
        _parse_float(time_match.group(3), label="raw port simulation time")
        cursor += 1

        cursor = _require_section(lines, cursor, name=_MIXED_COMPONENT_NAMES[0])
        raw_sources, raw_targets, cursor = _full_permutation(
            lines,
            cursor,
            permutation_size=mixed_count,
            value_count=mixed_count,
            label="mixed",
        )
        if raw_target_reference is None:
            raw_target_reference = raw_targets
        elif not np.array_equal(raw_targets, raw_target_reference):
            raise BackendError("Elmer raw port mixed permutation changed between records")
        raw_real, cursor = _scalar_values(
            lines,
            cursor,
            source_ids=raw_sources,
            label=_MIXED_COMPONENT_NAMES[0],
        )
        cursor = _require_section(lines, cursor, name=_MIXED_COMPONENT_NAMES[1])
        cursor = _require_previous_permutation(lines, cursor, label="mixed")
        raw_imaginary, cursor = _scalar_values(
            lines,
            cursor,
            source_ids=raw_sources,
            label=_MIXED_COMPONENT_NAMES[1],
        )
        raw_records.append(np.asarray(raw_real + 1j * raw_imaginary, dtype=np.complex128))

        components: list[np.ndarray] = []
        projected_sources: list[int] | None = None
        for component_index, name in enumerate(_COMPONENT_NAMES):
            cursor = _require_section(lines, cursor, name=name)
            if component_index == 0:
                projected_sources, projected_targets, cursor = _full_permutation(
                    lines,
                    cursor,
                    permutation_size=mixed_count,
                    value_count=expected_node_count,
                    label="projected",
                )
                if projected_target_reference is None:
                    projected_target_reference = projected_targets
                elif not np.array_equal(projected_targets, projected_target_reference):
                    raise BackendError(
                        "Elmer raw port projected permutation changed between records"
                    )
            else:
                cursor = _require_previous_permutation(lines, cursor, label="projected")
            assert projected_sources is not None
            component, cursor = _scalar_values(
                lines,
                cursor,
                source_ids=projected_sources,
                label=name,
            )
            components.append(component)
        projected_records.append(
            np.column_stack(components[:3]) + 1j * np.column_stack(components[3:])
        )

    expected_steps = [
        *((index, index) for index in range(1, expected_mode_count + 1)),
        (expected_mode_count + 1, 1),
    ]
    if save_steps != expected_steps:
        raise BackendError("Elmer raw port save/timestep sequence does not match the eigenspectrum")
    raw_matrix = np.column_stack(raw_records)
    if any(not np.any(raw_matrix[:, index]) for index in range(expected_mode_count)):
        raise BackendError("Elmer raw port contains an identically zero requested eigenvector")
    if not np.array_equal(raw_matrix[:, -1], np.zeros(mixed_count, dtype=np.complex128)):
        raise BackendError("Elmer raw port final save record is not exactly zero")
    projected_reference = projected_records[0]
    if any(not np.array_equal(record, projected_reference) for record in projected_records[1:]):
        raise BackendError("Elmer raw port repeated EF2D records disagree")
    assert raw_target_reference is not None
    mixed = ElmerPortMixedEigenvectors(
        nodal_coefficients=raw_matrix[:expected_node_count, :expected_mode_count],
        edge_coefficients=raw_matrix[expected_node_count:, :expected_mode_count],
        target_permutation=raw_target_reference,
        record_count=raw_matrix.shape[1],
        final_zero_record_verified=True,
    )
    projected = ElmerProjectedElectricFieldResult(
        values=np.asarray(projected_reference, dtype=np.complex128),
        record_count=len(projected_records),
        final_save_count=save_steps[-1][0],
        final_timestep=save_steps[-1][1],
        permutation_size=mixed_count,
    )
    return ElmerPortEigenmodeResult(mixed=mixed, projected=projected)


def reorder_elmer_edge_coefficients(
    coefficients: np.ndarray,
    *,
    elmer_edge_nodes: tuple[tuple[int, int], ...],
    canonical_edge_nodes: np.ndarray,
) -> np.ndarray:
    """Map Elmer first-encounter edge rows into femx lexicographic edge order."""

    values = np.asarray(coefficients)
    canonical = np.asarray(canonical_edge_nodes)
    if values.ndim != 2 or values.dtype.kind != "c":
        raise BackendError("Elmer port edge coefficients must be a complex rank-two array")
    if canonical.dtype.kind not in "iu" or canonical.ndim != 2 or canonical.shape[1] != 2:
        raise BackendError("canonical port edges must be an integer (edges, 2) array")
    source = tuple(tuple(int(node) for node in edge) for edge in elmer_edge_nodes)
    target = tuple(tuple(int(node) for node in edge) for edge in canonical.tolist())
    if values.shape[0] != len(source) or len(source) != len(target):
        raise BackendError("Elmer and canonical port edge counts do not match")
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise BackendError("Elmer and canonical port edge lists must be unique")
    if set(source) != set(target):
        raise BackendError("Elmer and canonical port edge sets do not match")
    target_rows = {edge: index for index, edge in enumerate(target)}
    reordered = np.empty_like(values)
    for source_row, edge in enumerate(source):
        reordered[target_rows[edge]] = values[source_row]
    return reordered
