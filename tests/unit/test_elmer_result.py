from pathlib import Path

import numpy as np
import pytest

from femx.backends.elmer.result import (
    read_potential_result,
    read_scalar_fields_result,
    read_scalar_result,
    read_temperature_result,
)
from femx.core.errors import BackendError

pytestmark = pytest.mark.unit


def _result_text(
    *,
    header: str = "ASCII 3",
    declaration: str = "temperature : 3 3 1 : heat equation",
    total_dofs: str = "Total DOFs: 1",
    nodes: str = "Number Of Nodes: 3",
    time: str = "Time: 1 1 1.00000000D+000",
    field_name: str = "temperature",
    permutation: str = "Perm: 3 3\n1 3\n2 2\n3 1",
    values: str = "5.0000000000000000D-001\n0.0\n0.0",
    trailing: str = "",
) -> str:
    return (
        f"{header}\n"
        "!File started at: ignored\n"
        "Degrees of freedom:\n"
        f"{declaration}\n"
        f"{total_dofs}\n"
        f"{nodes}\n"
        f"{time}\n"
        f"{field_name}\n"
        f"{permutation}\n"
        f"{values}\n"
        f"{trailing}"
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _multi_result_text(
    *,
    declarations: str = ("Temperature : 3 3 1 : heat equation\nPotential : 3 3 1 : static current"),
    total_dofs: str = "Total DOFs: 2",
    nodes: str = "Number Of Nodes: 3",
    time: str = "Time: 1 7 0.0",
    sections: str = (
        "Temperature\n"
        "Perm: 3 3\n"
        "1 3\n2 2\n3 1\n"
        "300.0\n301.0\n302.0\n"
        "Potential\n"
        "Perm: use previous\n"
        "0.0\n0.1\n0.2"
    ),
    trailing: str = "",
) -> str:
    return (
        "ASCII 3\n"
        "!File started at: ignored\n"
        "Degrees of freedom:\n"
        f"{declarations}\n"
        f"{total_dofs}\n"
        f"{nodes}\n"
        f"{time}\n"
        f"{sections}\n"
        f"{trailing}"
    )


def test_ascii_result_parser_uses_source_node_order_not_solver_permutation(tmp_path) -> None:
    parsed = read_temperature_result(
        _write(tmp_path / "femx.result", _result_text()),
        expected_node_count=3,
    )

    np.testing.assert_array_equal(parsed.values, np.asarray((0.5, 0.0, 0.0)))
    assert parsed.save_count == 1
    assert parsed.timestep == 1


def test_ascii_result_parser_supports_the_typed_potential_contract(tmp_path) -> None:
    text = _result_text(
        declaration="Potential : 3 3 1 : static current",
        field_name="Potential",
    )

    parsed = read_potential_result(
        _write(tmp_path / "potential.result", text),
        expected_node_count=3,
    )

    np.testing.assert_array_equal(parsed.values, np.asarray((0.5, 0.0, 0.0)))
    assert parsed.save_count == 1
    assert parsed.timestep == 1


def test_multi_scalar_parser_supports_elmer_reused_permutations(tmp_path) -> None:
    parsed = read_scalar_fields_result(
        _write(tmp_path / "coupled.result", _multi_result_text()),
        expected_node_count=3,
        field_names=("potential", "temperature"),
    )

    assert tuple(parsed.values) == ("temperature", "potential")
    np.testing.assert_array_equal(parsed.values["temperature"], (300.0, 301.0, 302.0))
    np.testing.assert_array_equal(parsed.values["potential"], (0.0, 0.1, 0.2))
    assert parsed.save_count == 1
    assert parsed.timestep == 7
    with pytest.raises(TypeError):
        parsed.values["extra"] = np.zeros((3,))  # type: ignore[index]


def test_multi_scalar_parser_rejects_eof_immediately_after_time(tmp_path) -> None:
    text = _multi_result_text(sections="").rstrip()
    with pytest.raises(BackendError, match="missing a requested scalar field"):
        read_scalar_fields_result(
            _write(tmp_path / "truncated.result", text),
            expected_node_count=3,
            field_names=("potential", "temperature"),
        )


@pytest.mark.parametrize(
    ("kwargs", "field_names", "node_count", "message"),
    [
        ({}, (), 3, "non-empty and unique"),
        ({}, ("potential", "Potential"), 3, "non-empty and unique"),
        ({}, ("bad field",), 3, "ASCII identifier"),
        ({}, ("potential", "temperature"), 0, "positive"),
        (
            {"declarations": "Temperature : 3 3 1 : heat equation"},
            ("potential", "temperature"),
            3,
            "closed field set",
        ),
        (
            {
                "declarations": (
                    "Temperature : 3 3 1 : heat equation\nTemperature : 3 3 1 : duplicate"
                )
            },
            ("temperature",),
            3,
            "duplicate scalar field",
        ),
        (
            {
                "declarations": (
                    "Temperature : 2 3 1 : heat equation\nPotential : 3 3 1 : static current"
                )
            },
            ("potential", "temperature"),
            3,
            "field header",
        ),
        (
            {"total_dofs": "Total DOFs: 1"},
            ("potential", "temperature"),
            3,
            "total scalar DOFs",
        ),
        (
            {"nodes": "Number Of Nodes: 2"},
            ("potential", "temperature"),
            3,
            "node header",
        ),
        (
            {"time": "broken"},
            ("potential", "temperature"),
            3,
            "exactly one time",
        ),
        (
            {"time": "Time: 1 7 1e"},
            ("potential", "temperature"),
            3,
            "invalid simulation time",
        ),
        (
            {"sections": ""},
            ("potential", "temperature"),
            3,
            "missing a requested",
        ),
        (
            {"sections": "Other"},
            ("potential", "temperature"),
            3,
            "unexpected scalar field",
        ),
        (
            {"sections": "Temperature"},
            ("potential", "temperature"),
            3,
            "missing the temperature permutation",
        ),
        (
            {"sections": "Temperature\nPerm: 3 2"},
            ("potential", "temperature"),
            3,
            "full nodal map",
        ),
        (
            {"sections": "Temperature\nPerm: use previous"},
            ("potential", "temperature"),
            3,
            "missing previous permutation",
        ),
        (
            {"sections": "Temperature\nPerm: NULL"},
            ("potential", "temperature"),
            3,
            "full or reused",
        ),
        (
            {"sections": "Temperature\nPerm: 3 3\n1 1\n2\n3 3"},
            ("potential", "temperature"),
            3,
            "malformed permutation",
        ),
        (
            {"sections": "Temperature\nPerm: 3 3\n1 1\n2 x\n3 3"},
            ("potential", "temperature"),
            3,
            "non-integer permutation",
        ),
        (
            {"sections": "Temperature\nPerm: 3 3\n1 1\n2 1\n3 3"},
            ("potential", "temperature"),
            3,
            "nodal bijection",
        ),
        (
            {"sections": "Temperature\nPerm: 3 3\n1 1\n2 2\n3 3\n300.0"},
            ("potential", "temperature"),
            3,
            "missing temperature scalar records",
        ),
        (
            {"sections": ("Temperature\nPerm: 3 3\n1 1\n2 2\n3 3\n300.0 301.0\n301.0\n302.0")},
            ("potential", "temperature"),
            3,
            "one scalar",
        ),
        (
            {"trailing": "unexpected\n"},
            ("potential", "temperature"),
            3,
            "unexpected trailing",
        ),
    ],
)
def test_multi_scalar_parser_rejects_closed_contract_drift(
    tmp_path,
    kwargs: dict[str, str],
    field_names: tuple[str, ...],
    node_count: int,
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        read_scalar_fields_result(
            _write(tmp_path / "coupled.result", _multi_result_text(**kwargs)),
            expected_node_count=node_count,
            field_names=field_names,
        )


@pytest.mark.parametrize("field_name", ["", " potential", "potential field", "poténtial", "x-1"])
def test_generic_scalar_result_parser_rejects_unsafe_field_names(tmp_path, field_name: str) -> None:
    with pytest.raises(BackendError, match="ASCII identifier"):
        read_scalar_result(
            tmp_path / "unused.result",
            expected_node_count=3,
            field_name=field_name,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"header": "BINARY 3"}, "ASCII 3"),
        ({"declaration": "temperature : 2 3 1 : heat equation"}, "field header"),
        ({"declaration": "temperature : 3 3 2 : heat equation"}, "field header"),
        ({"total_dofs": "Total DOFs: 2"}, "one scalar degree"),
        ({"nodes": "Number Of Nodes: 2"}, "node header"),
        ({"time": "Time: 1 1 broken"}, "exactly one time"),
        ({"time": "Time: 1 1 NaN"}, "exactly one time"),
        ({"field_name": "other"}, "temperature section"),
        ({"permutation": "Perm: NULL"}, "explicit full"),
        ({"permutation": "Perm: 3 2\n1 1\n2 2\n3 3"}, "full nodal map"),
        ({"permutation": "Perm: 3 3\n1 1\n2 1\n3 3"}, "nodal bijection"),
        ({"permutation": "Perm: 3 3\n1 1\n3 2\n2 3"}, "nodal bijection"),
        ({"permutation": "Perm: 3 3\n1 1\n2 x\n3 3"}, "non-integer"),
        ({"permutation": "Perm: 3 3\n1 1\n2\n3 3"}, "malformed"),
        ({"values": "0.5 0.0\n0.0\n0.0"}, "one scalar"),
        ({"values": "bad\n0.0\n0.0"}, "invalid temperature"),
        ({"values": "NaN\n0.0\n0.0"}, "non-finite temperature"),
        ({"trailing": "unexpected\n"}, "unexpected trailing"),
    ],
)
def test_ascii_result_parser_rejects_contract_drift(
    tmp_path, overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(BackendError, match=message):
        read_temperature_result(
            _write(tmp_path / "femx.result", _result_text(**overrides)),
            expected_node_count=3,
        )


def test_ascii_result_parser_rejects_missing_duplicate_and_invalid_inputs(tmp_path) -> None:
    with pytest.raises(BackendError, match="positive"):
        read_temperature_result(tmp_path / "missing", expected_node_count=0)
    with pytest.raises(BackendError, match="does not exist"):
        read_temperature_result(tmp_path / "missing", expected_node_count=3)

    duplicate_header = _result_text(trailing="temperature : 3 3 1 : duplicate\n")
    with pytest.raises(BackendError, match="exactly one scalar temperature"):
        read_temperature_result(
            _write(tmp_path / "duplicate-header.result", duplicate_header),
            expected_node_count=3,
        )

    duplicate_time = _result_text(trailing="Time: 2 2 2.0\n")
    with pytest.raises(BackendError, match="exactly one time"):
        read_temperature_result(
            _write(tmp_path / "duplicate-time.result", duplicate_time),
            expected_node_count=3,
        )

    separated = _result_text().replace(
        "Time: 1 1 1.00000000D+000\ntemperature\n",
        "Time: 1 1 1.00000000D+000\n!unexpected gap\ntemperature\n",
    )
    with pytest.raises(BackendError, match="not adjacent"):
        read_temperature_result(
            _write(tmp_path / "separated.result", separated), expected_node_count=3
        )

    missing_permutation = _result_text().split("Perm:", maxsplit=1)[0]
    with pytest.raises(BackendError, match="missing the temperature permutation"):
        read_temperature_result(
            _write(tmp_path / "missing-perm.result", missing_permutation),
            expected_node_count=3,
        )

    invalid_time = _result_text(time="Time: 1 1 1e")
    with pytest.raises(BackendError, match="invalid simulation time"):
        read_temperature_result(
            _write(tmp_path / "invalid-time.result", invalid_time), expected_node_count=3
        )

    binary = tmp_path / "binary.result"
    binary.write_bytes(b"ASCII 3\n\xff")
    with pytest.raises(BackendError, match="not an ASCII"):
        read_temperature_result(binary, expected_node_count=3)
