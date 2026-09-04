from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from femx.backends.jax.partition import balanced_lexicographic_cell_owners
from femx.core.errors import ContractError

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_balanced_lexicographic_owners_are_deterministic_and_complete() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (3.0, 0.0),
            (3.0, 1.0),
        ),
        dtype=np.float32,
    )
    cells = np.asarray(
        (
            (0, 1, 2),
            (1, 3, 2),
            (1, 4, 3),
            (4, 5, 3),
            (4, 6, 5),
            (6, 7, 5),
        ),
        dtype=np.int32,
    )
    owners = balanced_lexicographic_cell_owners(coordinates, cells, partition_count=4)
    np.testing.assert_array_equal(owners, (0, 0, 1, 2, 2, 3))
    np.testing.assert_array_equal(np.bincount(owners, minlength=4), (2, 1, 2, 1))
    assert owners.dtype == np.int64
    assert not owners.flags.writeable


@pytest.mark.parametrize(
    ("coordinates", "cells", "partition_count", "message"),
    (
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 1, "real coordinates"),
        (np.ones((1, 4)), [[0, 0, 0, 0, 0]], 1, "2D or 3D"),
        (np.asarray([[0.0, float("nan")], [1.0, 0.0], [0.0, 1.0]]), [[0, 1, 2]], 1, "finite"),
        (np.eye(3, 2), np.asarray([[0.0, 1.0, 2.0]]), 1, "integer cell"),
        (np.eye(3, 2), [[0, 1]], 1, "3 nodes"),
        (np.eye(3, 2), [[0, 1, 3]], 1, "out-of-range"),
        (np.eye(3, 2), [[0, 1, 1]], 1, "repeat"),
        (np.eye(3, 2), [[0, 1, 2]], 0, "partition count"),
        (np.eye(3, 2), [[0, 1, 2]], 2, "partition count"),
    ),
)
def test_balanced_lexicographic_owners_reject_invalid_inputs(
    coordinates: object,
    cells: object,
    partition_count: int,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        balanced_lexicographic_cell_owners(
            coordinates,
            cells,
            partition_count=partition_count,
        )
