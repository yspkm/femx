# Native JAX lowest-order Nédélec triangle

## Scope and terminology

This is the local `H(curl)` foundation for the native JAX Maxwell backend. The implemented subset
is deliberately narrow:

- affine two-dimensional triangles;
- Nédélec first family, one tangential moment per edge;
- real float64 geometry and basis kernels;
- explicit canonical edge signs;
- local unit `L2` and curl-curl Gram matrices;
- JAX JIT and reverse-mode differentiation.

`triangle_nedelec1` means the same lowest-order basis that the locked Elmer configuration calls
`EdgeBasisDegree = 1`. Some libraries label its polynomial subdegree zero. The DOF count and edge
moment definition, rather than an isolated order number, are authoritative.

## Reference element

For reference coordinates `(r, s)`, use

```text
lambda_0 = 1 - r - s
lambda_1 = r
lambda_2 = s
```

and directed edges `e_0=(0,1)`, `e_1=(1,2)`, `e_2=(2,0)`. The Whitney--Nédélec basis is

```text
N_0 = (1-s, r)
N_1 = (-s,  r)
N_2 = (-s,  r-1).
```

Every reference scalar curl is `2`. For the directed tangent of edge `e_j`,

```text
integral_ej N_i dot t_j dl = delta_ij.
```

If a P1 scalar has vertex values `p_i`, its exact edge coefficient is `p_j-p_i` on edge `(i,j)`.
Those coefficients reconstruct its constant gradient exactly and produce zero curl. This property
is a local discrete de Rham/exact-sequence check, not merely a sampled field comparison.

## Physical map and orientation

Let

```text
J = [x_1-x_0, x_2-x_0].
```

The physical `H(curl)` basis uses the covariant Piola map `J^{-T}` and its scalar curl is divided by
the signed determinant. The integration measure uses the absolute determinant. This distinction
allows the local kernel to remain mathematically correct under any node permutation, while normal
canonical meshes still preserve their recorded orientation.

The canonical global direction of edge `(a,b)` is `min(a,b) -> max(a,b)`. Preparation validates
the supplied sign for every local edge and assigns a deterministic lexicographic global edge DOF.
A coefficient stored once on a shared edge therefore gives a single-valued tangential trace from
both adjacent cells.

## Exact local matrices

On the unit reference triangle before canonical signs, the unit mass matrix is

```text
[[ 1/3, 0,   -1/6],
 [ 0,   1/6,  0   ],
 [-1/6, 0,    1/3 ]].
```

The unit curl-curl matrix has every entry equal to `2`. Canonical cell signs act on both sides of
each matrix. The three-point degree-two rule integrates these affine products exactly; no adaptive
or backend-selected quadrature is hidden in this slice.

## Elmer alignment

The locked source is Elmer commit
[`4f2d7e4`](https://github.com/ElmerCSC/elmerfem/commit/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594).
The relevant implementation is the triangle case of
[`EdgeElementInfo`](https://github.com/ElmerCSC/elmerfem/blob/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594/fem/src/ElemInfo.F90)
and the `LocalMatrix` use of Piola edge bases in
[`EMPort.F90`](https://github.com/ElmerCSC/elmerfem/blob/4f2d7e4b99f8f0dcf2f7ac579e056969373bf594/fem/src/modules/EMPort.F90).
The reviewed local hashes are `ElemInfo.F90=1b30819a...`, `ElementBasis.F90=b7386f1a...`, and
`PElementBase.F90=62b44b5f...`; the full values are retained in
[the source baseline](../SOURCE_BASELINE.md#reviewed-elmer-edge-element-source-witness).

Elmer's p-reference triangle has vertices `(-1,0)`, `(1,0)`, `(0,sqrt(3))`. Mapping femx's
reference basis to that triangle reproduces Elmer's three closed forms and curls exactly to the
float64 test tolerance, including the larger-global-node orientation rule. This is algebraic local
element alignment. It is not yet an independently executed global JAX/Elmer eigenmode comparison.

## Verification boundary

Portable tests establish:

1. Kronecker edge moments and constant curl;
2. exact P1-gradient reconstruction and zero reconstructed curl;
3. independent physical barycentric agreement on a skew triangle;
4. Elmer p-reference closed-form agreement;
5. one shared tangential trace across adjacent cells;
6. identical assembled Gram matrices under all six node permutations;
7. exact reference matrices, symmetry, rank, and positive mass;
8. JIT execution and reverse-mode agreement with central differences.

The [mixed `H(curl)`--`H1` port pencil](PORT_OPERATOR.md) and exact PEC reduction are now complete
as M4b. The generalized eigensolver remains a separate gate because it introduces singular-pencil,
ordering, residual, and degeneracy failure modes. Only after analytic refinement and same-mesh
Elmer beta/mode-subspace tests will femx advertise a native JAX port-eigenmode backend.

## References

- J.-C. Nédélec, *Mixed finite elements in R3*, Numerische Mathematik 35 (1980), 315--341,
  [doi:10.1007/BF01396415](https://doi.org/10.1007/BF01396415).
- D. N. Arnold, R. S. Falk, and R. Winther, *Finite element exterior calculus, homological
  techniques, and applications*, Acta Numerica 15 (2006), 1--155,
  [doi:10.1017/S0962492906210018](https://doi.org/10.1017/S0962492906210018).
- M. E. Rognes, R. C. Kirby, and A. Logg, *Efficient Assembly of H(div) and H(curl) Conforming
  Finite Elements*, SIAM Journal on Scientific Computing 31 (2009),
  [doi:10.1137/08073901X](https://doi.org/10.1137/08073901X).
