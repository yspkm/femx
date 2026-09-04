# Native JAX mixed port operator

## Scope

This document fixes M4b: the dense native JAX assembly of the lossless planar port pencil used by
the locked Elmer `EMPortSolver`. It consumes the M4a canonical triangle edge map and lowest-order
first-family Nédélec element. It assembles and reduces matrices only; it does not solve or rank
modes.

The implemented subset is the same no-potential branch selected by the femx Elmer v1 case:

- affine first-order triangles;
- one scalar P1 nodal field plus one first-family edge field;
- piecewise-constant real `epsilon_r > 0` and `mu_r > 0`;
- positive finite frequency;
- homogeneous PEC on the complete external boundary;
- SI coordinates and physical vacuum constants.

## Local generalized pencil

For scalar bases `phi_i` and oriented edge bases `W_e`, let

$$
M^0_{ij}=\int_K\phi_i\phi_j,\qquad
G_{ie}=\int_K W_e\cdot\nabla\phi_i,
$$

$$
M^1_{ef}=\int_K W_e\cdot W_f,\qquad
C_{ef}=\int_K (\nabla_t\times W_e)(\nabla_t\times W_f).
$$

With $\epsilon=\epsilon_0\epsilon_r$,
$\nu=(\mu_0\mu_r)^{-1}$, and $\omega=2\pi f$, the cell matrices are

$$
A_K=
\begin{bmatrix}
-\epsilon M^0 & \epsilon G\\
\nu G^T & \nu C-\omega^2\epsilon M^1
\end{bmatrix},\qquad
B_K=
\begin{bmatrix}
0&0\\
0&\nu M^1
\end{bmatrix}.
$$

This is a direct algebraic match to the locked Elmer no-potential `LocalMatrix`; no Elmer code is
copied. The scalar mass is the exact P1 triangle mass. Because `grad(phi_i)` is constant and `W_e`
is affine, evaluating the coupling at the centroid times cell area is exact. The existing
degree-two edge Gram integration is exact for `M1` and `C`.

Elmer solves

$$
A x=\lambda Bx,\qquad \beta=\sqrt{-\lambda}.
$$

`B` has a zero scalar block and is intentionally singular. M4b therefore makes no assumption that
the next solver can use a positive-definite generalized Hermitian routine.

## Global ordering and PEC

Cell-local order is `[three scalar nodes, three canonical edges]`. Global order is

```text
[all scalar node DOFs, all lexicographically ordered edge DOFs].
```

`Port Ground` in locked Elmer constrains both `Eport {n}` and `Eport {e}`. femx mirrors this by
requiring the supplied boundary facets to equal the mesh's complete topological boundary, then
constraining:

- every nodal scalar DOF incident to a boundary facet;
- every edge DOF owned by a boundary facet.

The reduced problem is the free-DOF principal subpencil. Identity rows are not inserted because
they would add artificial eigenvalues. The full-to-reduced DOF indices are retained explicitly.

## Verification

Portable evidence includes:

- physical `epsilon` and `nu` scaling against the locked Elmer constants;
- every 6-by-6 local entry against an independent quadrature transcription of the four Elmer
  blocks;
- zero scalar rows and columns in the generalized mass and a positive edge mass block;
- global invariance under all 36 independent local-node permutation pairs on adjacent triangles;
- cell-order reversal with material values carried with their physical cells;
- exact rejection of missing, duplicated, non-mesh, and interior PEC facets;
- JIT assembly and reduction;
- reverse-mode geometry and relative-permittivity directional derivatives against central
  differences.

Near-cancelled dimensional matrix entries are not assessed with a misleading elementwise absolute
tolerance. The local blocks retain elementwise independent checks; global permutation evidence uses
a Frobenius relative error below `2e-15`.

## Next solver layer

M4c now preserves the pencil's finite spectrum by exact scalar-constraint Schur elimination and
solves the resulting dense edge problem with JAX. Its ordering, coefficient normalization,
blockwise residual, analytic PEC refinement, and degeneracy-aware subspace contracts are in the
[native port eigensolver document](PORT_EIGENSOLVER.md). M4b remains the authority for matrix
assembly and PEC reduction.

## Open gates

- implicit eigen-adjoint and finite-difference evidence;
- custom FDTDX source injection;
- sparse, matrix-free, distributed, and actual TPU execution.

[ADR 0020](../adr/0020-elmer-compatible-jax-port-pencil.md) records the binding decision.
