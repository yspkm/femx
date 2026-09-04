# Native JAX affine Tet4 H1 element

## Scope

This is the first local finite-element boundary for the M5 three-dimensional heater. It covers:

- affine four-node tetrahedra;
- scalar first-order conforming H1 basis functions;
- signed geometry and positive integration volume;
- isotropic or full $3\times3$ cell diffusion tensors;
- cell-local P1 material interpolation;
- the exact consistent P1 mass and nodal-source load; and
- JAX JIT and reverse-mode differentiation.

The element API itself does not assemble or solve a global system, ingest a Gmsh volume mesh,
register a 3D backend, run Elmer, execute on an accelerator, or establish a ring-heater result.
Later M5 layers now use it for portable global scalar and one-way electrothermal solves without
changing that local contract.

## Reference and physical maps

The reference tetrahedron has vertices

$$
\hat{x}_0=(0,0,0),\quad
\hat{x}_1=(1,0,0),\quad
\hat{x}_2=(0,1,0),\quad
\hat{x}_3=(0,0,1),
$$

with nodal basis

$$
N_0=1-r-s-t,\qquad N_1=r,\qquad N_2=s,\qquad N_3=t.
$$

For physical vertices $x_i$, the affine map is

$$
x=x_0+J\hat{x},\qquad
J=[x_1-x_0,\ x_2-x_0,\ x_3-x_0].
$$

The physical gradients and volume are

$$
\nabla_x N_i=J^{-T}\nabla_{\hat{x}}N_i,
\qquad
V_e=\frac{|\det J|}{6}.
$$

The implementation retains the signed determinant as data while using its absolute value for
integration. A node permutation therefore remains observable, but it cannot change the globally
scattered scalar operator. Degenerate-cell rejection belongs to mesh preparation; the raw JAX
kernel deliberately exposes a zero determinant and non-finite inverse instead of repairing the
cell.

## Weak operator and exact loads

For

$$
-\nabla\cdot(C\nabla u)=q,
$$

the affine cell matrix is

$$
K^e_{ij}=V_e(\nabla N_i)^T C_e\nabla N_j.
$$

The kernel accepts either an isotropic scalar or a complete physical tensor $C_e$. It does not
silently symmetrize or project the tensor. Symmetry and positive definiteness are properties of the
prepared material law and solver admission.

If a coefficient is represented by four cell-local nodal values, its P1 interpolation is

$$
C_h=\sum_{a=0}^{3}N_a C_a.
$$

Because the Tet4 gradients are constant and $\int_e N_a\,dV=V_e/4$, exact integration uses the
cell mean $\bar{C}_e=(C_0+C_1+C_2+C_3)/4$. The values remain cell-local so a material interface
does not accidentally identify coefficients at a shared geometric node.

The consistent P1 mass matrix is

$$
M^e_{ij}=\frac{V_e}{20}(1+\delta_{ij}),
$$

and a nodal source $q_h=\sum_jN_jq_j$ gives $f^e=M^eq$. Consequently,

$$
\sum_i f_i^e=\int_e q_h\,dV=\frac{V_e}{4}\sum_jq_j.
$$

This identity is the local source-conservation boundary for later Joule-to-heat coupling.

## Elmer alignment

The locked reference is Elmer commit
[`4f2d7e4`](https://github.com/ElmerCSC/elmerfem/commit/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594).

- [`PElementBase.F90`](https://github.com/ElmerCSC/elmerfem/blob/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594/fem/src/PElementBase.F90)
  defines the linear tetrahedral basis and reference gradients used above.
- [`ElemInfo.F90`](https://github.com/ElmerCSC/elmerfem/blob/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594/fem/src/ElemInfo.F90)
  maps those derivatives to global coordinates and returns the metric measure.
- [`StatCurrentSolve.F90`](https://github.com/ElmerCSC/elmerfem/blob/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594/fem/src/modules/StatCurrentSolve.F90)
  integrates $C_{ij}\,\partial_iN_q\,\partial_jN_p$ and the interpolated source.
- [`DiffuseConvectiveAnisotropic.F90`](https://github.com/ElmerCSC/elmerfem/blob/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594/fem/src/DiffuseConvectiveAnisotropic.F90)
  supplies the corresponding Cartesian heat-diffusion term consumed by `HeatSolve.F90`.

The femx implementation is an independent closed-form realization of that shared weak equation,
not copied Fortran. The source gate pins all four files, and the clean reviewed checkout has source
digest `59634b49c59da9c7d069d3ccb5de206ba66b055837c51ab2d7ca3f88ef1cd49d`.

## Current verification boundary

Portable tests establish:

1. exact reference gradients, volume, stiffness, and consistent mass;
2. symmetry, positive-semidefinite rank, and partition of unity;
3. isotropic and independent anisotropic weak-form agreement;
4. exact cell-mean reduction for scalar and tensor P1 material data;
5. exact source-integral conservation;
6. an affine scalar-field patch test on a skew tetrahedron;
7. identical globally scattered stiffness under all 24 node permutations; and
8. JIT plus geometry/material reverse mode against a directional central difference.

M5a.2 supplies the canonical nondegenerate volume-mesh boundary, and M5b.1 connects these local
matrices to the arbitrary-width owned/ghost action, fixed-capacity collective transport,
strong-boundary RHS, and residual-defined CG. The Tet4 transport uses a distinct v2 identity while
the existing triangle v1 evidence remains unchanged. M5b.2 adds exact triangular flux and Robin
terms, a compact electrical submesh, parent-cell Joule transfer, a manufactured 3D refinement
study, conservation audits, and a four-forced-CPU-device voltage VJP. Public-ring material and
boundary binding, same-mesh Elmer parity, physical TPU execution, and device-scale evidence remain
later and independent gates. [ADR 0056](../adr/0056-distinct-space-tet4-electrothermal.md) defines
the new boundary.
