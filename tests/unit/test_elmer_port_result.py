from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from femx.backends.elmer.port_result import (
    parse_port_eigenmode_log,
    read_port_eigenmode_result,
    read_port_electric_field_result,
    reorder_elmer_edge_coefficients,
)
from femx.core.errors import BackendError

pytestmark = pytest.mark.unit


def _log() -> str:
    return "\n".join(
        (
            "EigenSolveComplex: Convergence criterion is: 1.000E-10",
            "EigenSolveComplex: Number of eigensystem iterations is: 17",
            "EigenSolveComplex: Number of converged Ritz values is: 2",
            "EigenSolveComplex: 1 ( -1.6000000000000000E+01, 0.0 )",
            "EigenSolveComplex: 2 ( -9.0000000000000000E+00, 0.0 )",
            "CheckResidualsComplex: L^2 Norm of the residual: 1 1.0E-12",
            "CheckResidualsComplex: L^2 Norm of the residual: 2 2.0E-12",
            "EMPortSolver: Propagation constant beta: 4.000000E+00 0.000000E+00",
            "SaveScalars: 1: res: port beta 1 4.000000000000E+00",
            "EMPortSolver: Port power: 2.000000E+00",
            "SaveScalars: 2: res: port power 1 2.000000000000E+00",
            "SaveScalars: 3: res: port impedance 1 5.000000000000E-01",
        )
    )


def test_port_log_parser_preserves_complete_spectrum_and_scalar_evidence() -> None:
    parsed = parse_port_eigenmode_log(_log(), selected_mode_index=0)

    np.testing.assert_array_equal(parsed.eigenvalues, (-16.0 + 0.0j, -9.0 + 0.0j))
    np.testing.assert_array_equal(parsed.residuals, (1.0e-12, 2.0e-12))
    assert parsed.reported_tolerance == 1.0e-10
    assert parsed.iterations == 17
    assert parsed.converged_count == 2
    assert parsed.selected_beta == 4.0 + 0.0j
    assert parsed.raw_forward_power_w == 2.0
    assert parsed.port_impedance_ohm == 0.5


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "EigenSolveComplex: Convergence criterion is: 1.000E-10",
            "",
            "exactly one eigen convergence",
        ),
        (
            "EigenSolveComplex: Convergence criterion is: 1.000E-10",
            "EigenSolveComplex: Convergence criterion is: 1.000E-10\n"
            "EigenSolveComplex: Convergence criterion is: 1.000E-10",
            "exactly one eigen convergence",
        ),
        (
            "EigenSolveComplex: 1 ( -1.6000000000000000E+01, 0.0 )",
            "",
            "contiguous and one-based",
        ),
        (
            "EigenSolveComplex: 2 ( -9.0000000000000000E+00, 0.0 )",
            "EigenSolveComplex: 3 ( -9.0000000000000000E+00, 0.0 )",
            "contiguous and one-based",
        ),
        (
            "CheckResidualsComplex: L^2 Norm of the residual: 2 2.0E-12",
            "",
            "residual indices",
        ),
        (
            "CheckResidualsComplex: L^2 Norm of the residual: 2 2.0E-12",
            "CheckResidualsComplex: L^2 Norm of the residual: 2 -2.0E-12",
            "cannot be negative",
        ),
        (
            "EigenSolveComplex: Convergence criterion is: 1.000E-10",
            "EigenSolveComplex: Convergence criterion is: 0.0",
            "nonpositive",
        ),
        (
            "EigenSolveComplex: Number of converged Ritz values is: 2",
            "EigenSolveComplex: Number of converged Ritz values is: 3",
            "inconsistent",
        ),
        (
            "EMPortSolver: Propagation constant beta: 4.000000E+00 0.000000E+00",
            "EMPortSolver: Propagation constant beta: 5.000000E+00 0.000000E+00",
            "beta log disagrees",
        ),
        (
            "SaveScalars: 1: res: port beta 1 4.000000000000E+00",
            "SaveScalars: 1: res: port beta 1 4.100000000000E+00",
            "beta scalar disagrees",
        ),
        (
            "SaveScalars: 2: res: port power 1 2.000000000000E+00",
            "SaveScalars: 2: res: port power 1 0.0",
            "must be positive",
        ),
        (
            "SaveScalars: 3: res: port impedance 1 5.000000000000E-01",
            "SaveScalars: 3: res: port impedance 1 -1.0",
            "must be positive",
        ),
        (
            "EMPortSolver: Port power: 2.000000E+00",
            "EMPortSolver: Port power: 3.000000E+00",
            "power records disagree",
        ),
        (
            "EigenSolveComplex: Convergence criterion is: 1.000E-10",
            "EigenSolveComplex: Convergence criterion is: NaN",
            "non-finite",
        ),
    ],
)
def test_port_log_parser_rejects_incomplete_or_inconsistent_evidence(
    old: str, new: str, message: str
) -> None:
    with pytest.raises(BackendError, match=message):
        parse_port_eigenmode_log(_log().replace(old, new), selected_mode_index=0)


def test_port_log_parser_rejects_selection_and_absent_spectrum() -> None:
    with pytest.raises(BackendError, match="cannot be negative"):
        parse_port_eigenmode_log(_log(), selected_mode_index=-1)
    with pytest.raises(BackendError, match="absent"):
        parse_port_eigenmode_log(_log(), selected_mode_index=2)
    without_spectrum = "\n".join(
        line
        for line in _log().splitlines()
        if not line.startswith("EigenSolveComplex: 1 (")
        and not line.startswith("EigenSolveComplex: 2 (")
    )
    with pytest.raises(BackendError, match="no computed complex eigenvalues"):
        parse_port_eigenmode_log(without_spectrum, selected_mode_index=0)


_COMPONENTS = (
    "ef2d re 1",
    "ef2d re 2",
    "ef2d re 3",
    "ef2d im 1",
    "ef2d im 2",
    "ef2d im 3",
)


def _record(save_count: int, *, timestep: int = 1, offset: float = 0.0) -> str:
    lines = [f"Time: {save_count} {timestep} 1.00000000E+000"]
    for component_index, name in enumerate(_COMPONENTS):
        lines.append(name)
        if component_index == 0:
            lines.extend(("Perm: 5 2", "1 2", "2 1"))
        else:
            lines.append("Perm: use previous")
        start = 2 * component_index + 1
        lines.extend((str(start + offset), str(start + 1 + offset)))
    return "\n".join(lines)


def _result_text(*, second_offset: float = 0.0, second_save: int = 2) -> str:
    declarations = [
        "ef2d[ef2d re:3 ef2d im:3] : 12 5 6 : port mode_post",
        *(f"{name} : 2 5 1 : port mode_post" for name in _COMPONENTS),
    ]
    return "\n".join(
        (
            "ASCII 3",
            "!dynamic timestamp",
            "Degrees of freedom:",
            *declarations,
            "Total DOFs: 6",
            "Number Of Nodes: 2",
            _record(1),
            _record(second_save, offset=second_offset),
            "",
        )
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _replace_nth(text: str, old: str, new: str, occurrence: int) -> str:
    start = -1
    for _ in range(occurrence):
        start = text.find(old, start + 1)
        assert start >= 0
    return text[:start] + new + text[start + len(old) :]


def test_port_field_parser_reconstructs_complex_nodes_and_identical_repeated_saves(
    tmp_path: Path,
) -> None:
    parsed = read_port_electric_field_result(
        _write(tmp_path / "femx.result", _result_text()),
        expected_node_count=2,
    )

    expected = np.asarray(
        [[1.0 + 7.0j, 3.0 + 9.0j, 5.0 + 11.0j], [2.0 + 8.0j, 4.0 + 10.0j, 6.0 + 12.0j]]
    )
    np.testing.assert_array_equal(parsed.values, expected)
    assert parsed.record_count == 2
    assert parsed.final_save_count == 2
    assert parsed.final_timestep == 1
    assert parsed.permutation_size == 5


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("ASCII 3", "BINARY 3", "ASCII 3"),
        ("Total DOFs: 6", "", "exactly one total-DOF"),
        ("Total DOFs: 6", "Total DOFs: 5", "exactly six"),
        ("Number Of Nodes: 2", "Number Of Nodes: 3", "node header"),
        (
            "ef2d[ef2d re:3 ef2d im:3] : 12 5 6",
            "ef2d[ef2d re:3 ef2d im:3] : 11 5 6",
            "EF2D declaration",
        ),
        ("ef2d re 1 : 2 5 1", "ef2d re 1 : 1 5 1", "component declaration"),
        ("ef2d re 2 : 2 5 1", "other : 2 5 1", "invalid component"),
        ("ef2d re 2 : 2 5 1", "ef2d re 3 : 2 5 1", "canonical order"),
        ("Time: 1 1 1.00000000E+000", "", "begin immediately"),
        ("Time: 1 1 1.00000000E+000", "Time: 1 1 NaN", "begin immediately"),
        ("ef2d re 1\nPerm: 5 2", "wrong\nPerm: 5 2", "canonical EF2D"),
        ("Perm: 5 2", "Perm: use previous", "full first-component"),
        ("Perm: 5 2", "Perm: 4 2", "header is inconsistent"),
        ("Perm: 5 2", "Perm: 5 1", "header is inconsistent"),
        ("1 2\n2 1", "1\n2 1", "malformed permutation"),
        ("1 2\n2 1", "1 x\n2 1", "non-integer permutation"),
        ("1 2\n2 1", "1 1\n2 1", "nodal bijection"),
        ("1 2\n2 1", "2 2\n1 1", "nodal bijection"),
        ("ef2d re 2\nPerm: use previous", "ef2d re 2\nPerm: NULL", "reuse"),
        ("1.0\n2.0", "1.0 2.0\n2.0", "one scalar"),
        ("1.0\n2.0", "bad\n2.0", "invalid ef2d re 1"),
    ],
)
def test_port_field_parser_rejects_closed_contract_drift(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    text = _result_text().replace(old, new, 1)
    with pytest.raises(BackendError, match=message):
        read_port_electric_field_result(
            _write(tmp_path / "bad.result", text),
            expected_node_count=2,
        )


def test_port_field_parser_rejects_invalid_counts_repeated_drift_and_trailing_text(
    tmp_path: Path,
) -> None:
    with pytest.raises(BackendError, match="positive"):
        read_port_electric_field_result(tmp_path / "missing", expected_node_count=0)
    with pytest.raises(BackendError, match="does not exist"):
        read_port_electric_field_result(tmp_path / "missing", expected_node_count=2)
    with pytest.raises(BackendError, match="save counts must be contiguous"):
        read_port_electric_field_result(
            _write(tmp_path / "skipped.result", _result_text(second_save=3)),
            expected_node_count=2,
        )
    with pytest.raises(BackendError, match="repeated EF2D records disagree"):
        read_port_electric_field_result(
            _write(tmp_path / "different.result", _result_text(second_offset=0.25)),
            expected_node_count=2,
        )
    trailing = _result_text() + "unexpected\n"
    with pytest.raises(BackendError, match="outside a save record"):
        read_port_electric_field_result(
            _write(tmp_path / "trailing.result", trailing),
            expected_node_count=2,
        )


def test_port_field_parser_rejects_missing_or_noncanonical_declaration_blocks(
    tmp_path: Path,
) -> None:
    missing_component = _result_text().replace(
        "ef2d im 3 : 2 5 1 : port mode_post\n",
        "",
        1,
    )
    with pytest.raises(BackendError, match="one vector and six"):
        read_port_electric_field_result(
            _write(tmp_path / "missing-declaration.result", missing_component),
            expected_node_count=2,
        )

    wrong_vector = _result_text().replace(
        "ef2d[ef2d re:3 ef2d im:3] : 12 5 6",
        "other : 12 5 6",
        1,
    )
    with pytest.raises(BackendError, match="vector declaration"):
        read_port_electric_field_result(
            _write(tmp_path / "wrong-vector.result", wrong_vector),
            expected_node_count=2,
        )

    without_times = "\n".join(
        line for line in _result_text().splitlines() if not line.startswith("Time:")
    )
    with pytest.raises(BackendError, match="no save records"):
        read_port_electric_field_result(
            _write(tmp_path / "without-times.result", without_times),
            expected_node_count=2,
        )


def test_port_field_parser_rejects_component_without_a_permutation(tmp_path: Path) -> None:
    header = _result_text().split("Time:", 1)[0]
    truncated = header + "Time: 1 1 1.0\nef2d re 1\n"
    with pytest.raises(BackendError, match="missing an EF2D permutation"):
        read_port_electric_field_result(
            _write(tmp_path / "missing-permutation.result", truncated),
            expected_node_count=2,
        )


def test_port_field_parser_rejects_truncated_value_block(tmp_path: Path) -> None:
    truncated = _result_text().rstrip().rsplit("\n", 1)[0] + "\n"
    with pytest.raises(BackendError, match="missing EF2D values"):
        read_port_electric_field_result(
            _write(tmp_path / "truncated.result", truncated),
            expected_node_count=2,
        )


def _raw_record(
    save_count: int,
    *,
    timestep: int,
    raw_offset: float = 0.0,
    projected_offset: float = 0.0,
) -> str:
    lines = [f"Time: {save_count} {timestep} 1.00000000E+000", "eport re"]
    lines.extend(("Perm: 5 5", "1 5", "2 4", "3 3", "4 2", "5 1"))
    if save_count <= 2:
        raw_real = np.asarray((0.0, 0.0, 0.0, save_count + raw_offset, 0.0))
        raw_imaginary = np.asarray((0.0, 0.0, 0.0, -save_count, 0.0))
    else:
        raw_real = np.zeros(5)
        raw_imaginary = np.zeros(5)
    lines.extend(str(value) for value in raw_real)
    lines.extend(("eport im", "Perm: use previous"))
    lines.extend(str(value) for value in raw_imaginary)
    for component_index, name in enumerate(_COMPONENTS):
        lines.append(name)
        if component_index == 0:
            lines.extend(("Perm: 5 2", "1 2", "2 1"))
        else:
            lines.append("Perm: use previous")
        start = 2 * component_index + 1
        lines.extend((str(start + projected_offset), str(start + 1 + projected_offset)))
    return "\n".join(lines)


def _raw_result_text(
    *,
    second_raw_offset: float = 0.0,
    third_projected_offset: float = 0.0,
) -> str:
    declarations = (
        "eport[eport re:1 eport im:1] : 10 5 2 : port mode",
        "ef2d[ef2d re:3 ef2d im:3] : 12 5 6 : port mode_post",
        "eport re : 5 5 1 : port mode",
        "eport im : 5 5 1 : port mode",
        *(f"{name} : 2 5 1 : port mode_post" for name in _COMPONENTS),
    )
    return "\n".join(
        (
            "ASCII 3",
            "!dynamic timestamp",
            "Degrees of freedom:",
            *declarations,
            "Total DOFs: 8",
            "Number Of Nodes: 2",
            _raw_record(1, timestep=1),
            _raw_record(2, timestep=2, raw_offset=second_raw_offset),
            _raw_record(3, timestep=1, projected_offset=third_projected_offset),
            "",
        )
    )


def test_raw_port_parser_retains_every_mode_and_verifies_elmer_final_save(
    tmp_path: Path,
) -> None:
    parsed = read_port_eigenmode_result(
        _write(tmp_path / "raw.result", _raw_result_text()),
        expected_node_count=2,
        expected_edge_count=3,
        expected_mode_count=2,
    )

    np.testing.assert_array_equal(parsed.mixed.nodal_coefficients, np.zeros((2, 2)))
    np.testing.assert_array_equal(
        parsed.mixed.edge_coefficients,
        np.asarray(((0.0, 0.0), (1.0 - 1.0j, 2.0 - 2.0j), (0.0, 0.0))),
    )
    np.testing.assert_array_equal(parsed.mixed.target_permutation, (5, 4, 3, 2, 1))
    assert parsed.mixed.record_count == 3
    assert parsed.mixed.final_zero_record_verified
    np.testing.assert_array_equal(
        parsed.projected.values,
        np.asarray(
            [
                [1.0 + 7.0j, 3.0 + 9.0j, 5.0 + 11.0j],
                [2.0 + 8.0j, 4.0 + 10.0j, 6.0 + 12.0j],
            ]
        ),
    )
    assert parsed.projected.final_save_count == 3
    assert parsed.projected.final_timestep == 1


def test_raw_port_parser_rejects_save_field_and_permutation_drift(tmp_path: Path) -> None:
    bad_cases = (
        (
            _raw_result_text().replace("Time: 2 2", "Time: 2 1", 1),
            "save/timestep sequence",
        ),
        (
            _raw_result_text(third_projected_offset=0.25),
            "repeated EF2D records disagree",
        ),
        (
            _raw_result_text()
            .replace("Time: 3 1", "Time: 3 1", 1)
            .replace(
                "0.0\neport im\nPerm: use previous\n0.0\n0.0\n0.0\n0.0\n0.0\nef2d re 1",
                "1.0\neport im\nPerm: use previous\n0.0\n0.0\n0.0\n0.0\n0.0\nef2d re 1",
                1,
            ),
            "final save record",
        ),
        (
            _raw_result_text().replace("1 5\n2 4\n3 3\n4 2\n5 1", "1 5\n2 4\n3 3\n4 1\n5 1", 1),
            "not a bijection",
        ),
        (
            _replace_nth(
                _raw_result_text(),
                "1 5\n2 4\n3 3\n4 2\n5 1",
                "1 4\n2 5\n3 3\n4 2\n5 1",
                2,
            ),
            "mixed permutation changed",
        ),
        (
            _replace_nth(
                _raw_result_text(),
                "Perm: 5 2\n1 2\n2 1",
                "Perm: 5 2\n1 1\n2 2",
                2,
            ),
            "projected permutation changed",
        ),
    )
    for index, (text, message) in enumerate(bad_cases):
        with pytest.raises(BackendError, match=message):
            read_port_eigenmode_result(
                _write(tmp_path / f"bad-raw-{index}.result", text),
                expected_node_count=2,
                expected_edge_count=3,
                expected_mode_count=2,
            )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("Total DOFs: 8", "", "exactly one total-DOF"),
        ("eport im : 5 5 1 : port mode\n", "", "two vectors and eight"),
        (
            "eport[eport re:1 eport im:1] : 10 5 2",
            "other : 10 5 2",
            "vector declarations",
        ),
        (
            "eport[eport re:1 eport im:1] : 10 5 2",
            "eport[eport re:1 eport im:1] : 9 5 2",
            "mixed-vector declaration",
        ),
        (
            "ef2d[ef2d re:3 ef2d im:3] : 12 5 6",
            "ef2d[ef2d re:3 ef2d im:3] : 11 5 6",
            "projected-vector declaration",
        ),
        ("eport re : 5 5 1", "other : 5 5 1", "invalid mixed component"),
        ("eport re : 5 5 1", "eport re : 4 5 1", "mixed component"),
        ("eport im : 5 5 1", "eport re : 5 5 1", "mixed components"),
        ("ef2d re 1 : 2 5 1", "other : 2 5 1", "invalid projected component"),
        ("ef2d re 1 : 2 5 1", "ef2d re 1 : 1 5 1", "projected component"),
        ("ef2d re 2 : 2 5 1", "ef2d re 3 : 2 5 1", "projected components"),
        ("Number Of Nodes: 2", "Number Of Nodes: 3", "node header"),
        ("Total DOFs: 8", "Total DOFs: 7", "exactly eight"),
        ("Number Of Nodes: 2\nTime:", "Number Of Nodes: 2\nunexpected\nTime:", "begin after"),
        ("Perm: 5 5", "Perm: use previous", "full mixed permutation"),
        ("Perm: 5 5", "Perm: 5 4", "header is inconsistent"),
        ("1 5\n2 4", "1\n2 4", "permutation is malformed"),
        ("1 5\n2 4", "1 x\n2 4", "permutation is non-integer"),
        ("eport im\nPerm: use previous", "eport im\nPerm: NULL", "reuse the mixed"),
        ("\neport im\n", "\nwrong\n", "missing canonical eport im"),
        ("0.0\n0.0\n0.0\n1.0\n0.0", "0.0 1.0\n0.0\n0.0\n1.0\n0.0", "one scalar"),
    ),
)
def test_raw_port_parser_rejects_declaration_and_record_contract_drift(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        read_port_eigenmode_result(
            _write(tmp_path / "raw-contract-drift.result", _raw_result_text().replace(old, new, 1)),
            expected_node_count=2,
            expected_edge_count=3,
            expected_mode_count=2,
        )


def test_raw_port_parser_rejects_missing_records_truncation_and_trailing_text(
    tmp_path: Path,
) -> None:
    without_times = "\n".join(
        line for line in _raw_result_text().splitlines() if not line.startswith("Time:")
    )
    header = _raw_result_text().split("Time:", 1)[0]
    cases = (
        (without_times, "no save records"),
        (header + "Time: 1 1 1.0\neport re\n", "missing the mixed permutation"),
        (
            header + "Time: 1 1 1.0\neport re\nPerm: 5 5\n1 1\n",
            "missing the mixed permutation values",
        ),
        (
            header + "Time: 1 1 1.0\neport re\nPerm: 5 5\n1 1\n2 2\n3 3\n4 4\n5 5\n0.0\n",
            "missing eport re values",
        ),
        (_raw_result_text() + "unexpected\n", "outside a save record"),
    )
    for index, (text, message) in enumerate(cases):
        with pytest.raises(BackendError, match=message):
            read_port_eigenmode_result(
                _write(tmp_path / f"raw-truncated-{index}.result", text),
                expected_node_count=2,
                expected_edge_count=3,
                expected_mode_count=2,
            )


def test_raw_port_parser_rejects_nonpositive_counts_and_zero_requested_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(BackendError, match="all be positive"):
        read_port_eigenmode_result(
            tmp_path / "missing",
            expected_node_count=0,
            expected_edge_count=3,
            expected_mode_count=2,
        )
    zero_mode = _raw_result_text().replace(
        "0.0\n0.0\n0.0\n1.0\n0.0\neport im\nPerm: use previous\n0.0\n0.0\n0.0\n-1.0\n0.0",
        "0.0\n0.0\n0.0\n0.0\n0.0\neport im\nPerm: use previous\n0.0\n0.0\n0.0\n0.0\n0.0",
        1,
    )
    with pytest.raises(BackendError, match="zero requested eigenvector"):
        read_port_eigenmode_result(
            _write(tmp_path / "zero-mode.result", zero_mode),
            expected_node_count=2,
            expected_edge_count=3,
            expected_mode_count=2,
        )


def test_elmer_edge_coefficients_reorder_by_exact_topological_pair() -> None:
    coefficients = np.asarray(
        ((1.0 + 1.0j,), (2.0 + 2.0j,), (3.0 + 3.0j,)),
        dtype=np.complex128,
    )
    reordered = reorder_elmer_edge_coefficients(
        coefficients,
        elmer_edge_nodes=((1, 2), (0, 2), (0, 1)),
        canonical_edge_nodes=np.asarray(((0, 1), (0, 2), (1, 2))),
    )
    np.testing.assert_array_equal(reordered[:, 0], (3.0 + 3.0j, 2.0 + 2.0j, 1.0 + 1.0j))


@pytest.mark.parametrize(
    ("coefficients", "source", "target", "message"),
    (
        (np.ones(3), ((0, 1), (0, 2), (1, 2)), np.ones((3, 2), int), "rank-two"),
        (
            np.ones((3, 1), complex),
            ((0, 1), (0, 2), (1, 2)),
            np.ones((3, 2), float),
            "integer",
        ),
        (
            np.ones((2, 1), complex),
            ((0, 1), (0, 2), (1, 2)),
            np.asarray(((0, 1), (0, 2), (1, 2))),
            "counts",
        ),
        (
            np.ones((3, 1), complex),
            ((0, 1), (0, 1), (1, 2)),
            np.asarray(((0, 1), (0, 2), (1, 2))),
            "unique",
        ),
        (
            np.ones((3, 1), complex),
            ((0, 1), (0, 3), (1, 2)),
            np.asarray(((0, 1), (0, 2), (1, 2))),
            "sets",
        ),
    ),
)
def test_elmer_edge_reordering_fails_closed(
    coefficients: np.ndarray,
    source: tuple[tuple[int, int], ...],
    target: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        reorder_elmer_edge_coefficients(
            coefficients,
            elmer_edge_nodes=source,
            canonical_edge_nodes=target,
        )
