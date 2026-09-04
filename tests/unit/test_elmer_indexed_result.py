from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from femx.backends.elmer.result import read_indexed_scalar_fields_result
from femx.core.errors import BackendError

pytestmark = pytest.mark.unit


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _mixed_result(
    *,
    declarations: str = ("Potential : 3 5 1 : static current\nTemperature : 5 5 1 : heat equation"),
    total_dofs: str = "Total DOFs: 2",
    nodes: str = "Number Of Nodes: 5",
    time: str = "Time: 1 2 0.0",
    sections: str = (
        "Potential\n"
        "Perm: 5 3\n"
        "1 3\n3 1\n5 2\n"
        "0.0\n0.5\n1.0\n"
        "Temperature\n"
        "Perm: 5 5\n"
        "1 5\n2 4\n3 3\n4 2\n5 1\n"
        "300.0\n301.0\n302.0\n303.0\n304.0"
    ),
    trailing: str = "",
) -> str:
    return (
        "ASCII 3\n"
        "!dynamic timestamp\n"
        "Degrees of freedom:\n"
        f"{declarations}\n"
        f"{total_dofs}\n"
        f"{nodes}\n"
        f"{time}\n"
        f"{sections}\n"
        f"{trailing}"
    )


def test_indexed_scalar_parser_preserves_partial_source_node_order(tmp_path: Path) -> None:
    parsed = read_indexed_scalar_fields_result(
        _write(tmp_path / "mixed.result", _mixed_result()),
        expected_node_count=5,
        field_node_ids={"potential": (0, 2, 4), "temperature": tuple(range(5))},
    )

    assert parsed.save_count == 1
    assert parsed.timestep == 2
    np.testing.assert_array_equal(parsed.fields["potential"].source_node_ids, (0, 2, 4))
    np.testing.assert_allclose(parsed.fields["potential"].values, (0.0, 0.5, 1.0))
    np.testing.assert_array_equal(parsed.fields["temperature"].source_node_ids, range(5))
    np.testing.assert_allclose(
        parsed.fields["temperature"].values,
        (300.0, 301.0, 302.0, 303.0, 304.0),
    )
    with pytest.raises(TypeError):
        parsed.fields["other"] = parsed.fields["potential"]  # type: ignore[index]


def test_indexed_scalar_parser_accepts_only_identical_reused_node_maps(tmp_path: Path) -> None:
    text = _mixed_result(
        declarations="First : 3 5 1 : first\nSecond : 3 5 1 : second",
        sections=(
            "First\n"
            "Perm: 5 3\n"
            "1 3\n3 1\n5 2\n"
            "1.0\n2.0\n3.0\n"
            "Second\n"
            "Perm: use previous\n"
            "4.0\n5.0\n6.0"
        ),
    )
    parsed = read_indexed_scalar_fields_result(
        _write(tmp_path / "reused.result", text),
        expected_node_count=5,
        field_node_ids={"first": (0, 2, 4), "second": (0, 2, 4)},
    )

    np.testing.assert_allclose(parsed.fields["second"].values, (4.0, 5.0, 6.0))


@pytest.mark.parametrize(
    ("field_node_ids", "node_count", "message"),
    (
        ({"potential": (0,)}, 0, "positive"),
        ({}, 5, "must not be empty"),
        ({"bad field": (0,)}, 5, "ASCII identifier"),
        ({"potential": (0,), "Potential": (0,)}, 5, "unique"),
        ({"potential": ()}, 5, "non-empty integer vector"),
        ({"potential": ((0,),)}, 5, "non-empty integer vector"),
        ({"potential": (0.0,)}, 5, "non-empty integer vector"),
        ({"potential": (True,)}, 5, "non-empty integer vector"),
        ({"potential": (2, 1)}, 5, "sorted, unique, and in range"),
        ({"potential": (1, 1)}, 5, "sorted, unique, and in range"),
        ({"potential": (-1,)}, 5, "sorted, unique, and in range"),
        ({"potential": (5,)}, 5, "sorted, unique, and in range"),
    ),
)
def test_indexed_scalar_parser_rejects_invalid_expected_maps(
    tmp_path: Path,
    field_node_ids: dict[str, tuple[object, ...]],
    node_count: int,
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        read_indexed_scalar_fields_result(
            tmp_path / "unused.result",
            expected_node_count=node_count,
            field_node_ids=field_node_ids,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"declarations": "Potential : 3 5 1 : static current"}, "indexed field set"),
        (
            {
                "declarations": (
                    "Potential : 3 5 1 : static current\n"
                    "Potential : 3 5 1 : duplicate\n"
                    "Temperature : 5 5 1 : heat equation"
                )
            },
            "duplicate scalar field",
        ),
        (
            {"declarations": "Potential : 2 5 1 : current\nTemperature : 5 5 1 : heat"},
            "field header",
        ),
        ({"total_dofs": "Total DOFs: 1"}, "total scalar DOFs"),
        ({"nodes": "Number Of Nodes: 4"}, "node header"),
        ({"time": "broken"}, "exactly one time"),
        ({"time": "Time: 1 2 1e"}, "invalid simulation time"),
        ({"sections": ""}, "missing a requested"),
        ({"sections": "Other"}, "unexpected indexed scalar"),
        ({"sections": "Potential"}, "missing the potential permutation"),
        (
            {"sections": "Potential\nPerm: 5 2"},
            "size or positive count differs",
        ),
        (
            {"sections": "Potential\nPerm: 5 3"},
            "missing the potential permutation pairs",
        ),
        (
            {"sections": "Potential\nPerm: 5 3\n1 1\n3\n5 2"},
            "malformed permutation",
        ),
        (
            {"sections": "Potential\nPerm: 5 3\n1 1\n3 x\n5 2"},
            "non-integer permutation",
        ),
        (
            {"sections": "Potential\nPerm: 5 3\n1 1\n2 3\n5 2"},
            "source-node map differs",
        ),
        (
            {"sections": "Potential\nPerm: 5 3\n1 1\n3 1\n5 2"},
            "target-DOF map is not a bijection",
        ),
        (
            {"sections": "Potential\nPerm: use previous"},
            "missing previous permutation",
        ),
        (
            {"sections": "Potential\nPerm: NULL"},
            "explicit or reused",
        ),
        (
            {"sections": "Potential\nPerm: 5 3\n1 1\n3 2\n5 3\n0.0"},
            "missing potential scalar records",
        ),
        (
            {"sections": ("Potential\nPerm: 5 3\n1 1\n3 2\n5 3\n0.0 1.0\n0.5\n1.0")},
            "one scalar",
        ),
        (
            {"sections": ("Potential\nPerm: 5 3\n1 1\n3 2\n5 3\n0.0\nNaN\n1.0")},
            "non-finite potential",
        ),
        ({"trailing": "unexpected\n"}, "unexpected trailing"),
    ),
)
def test_indexed_scalar_parser_rejects_result_contract_drift(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(BackendError, match=message):
        read_indexed_scalar_fields_result(
            _write(tmp_path / "mixed.result", _mixed_result(**overrides)),
            expected_node_count=5,
            field_node_ids={"potential": (0, 2, 4), "temperature": tuple(range(5))},
        )


def test_indexed_scalar_parser_rejects_reused_map_for_different_subset(tmp_path: Path) -> None:
    text = _mixed_result(
        sections=(
            "Potential\n"
            "Perm: 5 3\n"
            "1 3\n3 1\n5 2\n"
            "0.0\n0.5\n1.0\n"
            "Temperature\n"
            "Perm: use previous\n"
            "300.0\n301.0\n302.0\n303.0\n304.0"
        )
    )
    with pytest.raises(BackendError, match="reused permutation differs"):
        read_indexed_scalar_fields_result(
            _write(tmp_path / "reused-wrong.result", text),
            expected_node_count=5,
            field_node_ids={"potential": (0, 2, 4), "temperature": tuple(range(5))},
        )
