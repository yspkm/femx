"""Deterministic host-side cell ownership for bounded JAX collective runs."""

from __future__ import annotations

import numpy as np

from femx.core.errors import ContractError


def balanced_lexicographic_cell_owners(
    coordinates: object,
    cells: object,
    *,
    partition_count: int,
) -> np.ndarray:
    """Assign consecutive centroid order to equal-count partitions.

    The first coordinate is the primary spatial key and later coordinates break ties in order.
    This is a deterministic evidence partitioner, not a graph partitioner or scaling claim.  It
    guarantees complete once-only cell ownership and a cell-count imbalance of at most one.
    """

    raw_coordinates = np.asarray(coordinates)
    if raw_coordinates.dtype.kind != "f" or raw_coordinates.ndim != 2:
        raise ContractError("balanced cell ownership requires rank-2 real coordinates")
    canonical_coordinates = np.ascontiguousarray(raw_coordinates, dtype=np.float64)
    if canonical_coordinates.shape[0] == 0 or canonical_coordinates.shape[1] not in (2, 3):
        raise ContractError("balanced cell ownership requires nonempty 2D or 3D coordinates")
    if not np.all(np.isfinite(canonical_coordinates)):
        raise ContractError("balanced cell ownership coordinates must be finite")

    raw_cells = np.asarray(cells)
    if raw_cells.dtype.kind not in "iu" or raw_cells.ndim != 2:
        raise ContractError("balanced cell ownership requires a rank-2 integer cell array")
    canonical_cells = np.ascontiguousarray(raw_cells, dtype=np.int64)
    expected_width = canonical_coordinates.shape[1] + 1
    if canonical_cells.shape[0] == 0 or canonical_cells.shape[1] != expected_width:
        raise ContractError(
            f"balanced cell ownership requires nonempty simplex cells with {expected_width} nodes"
        )
    if np.any(canonical_cells < 0) or np.any(canonical_cells >= canonical_coordinates.shape[0]):
        raise ContractError("balanced cell ownership contains an out-of-range node")
    if any(np.unique(cell).shape[0] != expected_width for cell in canonical_cells):
        raise ContractError("balanced cell ownership cells cannot repeat a node")
    if (
        isinstance(partition_count, bool)
        or not isinstance(partition_count, int)
        or partition_count <= 0
        or partition_count > canonical_cells.shape[0]
    ):
        raise ContractError(
            "balanced cell ownership partition count must be positive and no larger than cells"
        )

    centroids = np.mean(canonical_coordinates[canonical_cells], axis=1)
    keys = tuple(centroids[:, axis] for axis in range(centroids.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    owners = np.empty((canonical_cells.shape[0],), dtype=np.int64)
    owners[order] = (
        np.arange(canonical_cells.shape[0], dtype=np.int64) * partition_count
    ) // canonical_cells.shape[0]
    owners.setflags(write=False)
    return owners
