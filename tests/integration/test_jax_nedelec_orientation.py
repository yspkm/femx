import itertools

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends._hcurl import canonical_triangle_edge_map  # noqa: E402
from femx.backends.jax.elements.triangle_nedelec import (  # noqa: E402
    triangle_nedelec1_local_gram,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_jax]


def _assembled_gram(
    coordinates: np.ndarray,
    cell: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray((cell,), dtype=np.int32)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)
    local = triangle_nedelec1_local_gram(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(signs),
    )
    mass = np.zeros((edge_map.dof_count, edge_map.dof_count))
    curl_curl = np.zeros_like(mass)
    dofs = edge_map.cell_edge_dofs[0]
    mass[np.ix_(dofs, dofs)] += np.asarray(local.mass[0])
    curl_curl[np.ix_(dofs, dofs)] += np.asarray(local.curl_curl[0])
    return mass, curl_curl


def test_global_hcurl_gram_is_invariant_to_all_triangle_node_permutations() -> None:
    coordinates = np.asarray(((0.1, -0.2), (1.3, 0.1), (0.2, 1.1)), dtype=np.float64)
    reference_mass, reference_curl_curl = _assembled_gram(coordinates, (0, 1, 2))

    for permutation in itertools.permutations(range(3)):
        mass, curl_curl = _assembled_gram(coordinates, permutation)
        np.testing.assert_allclose(mass, reference_mass, rtol=2.0e-15, atol=2.0e-15)
        np.testing.assert_allclose(
            curl_curl,
            reference_curl_curl,
            rtol=2.0e-15,
            atol=2.0e-15,
        )
