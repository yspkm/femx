# Native JAX finite port eigensolver

## Scope

M4c adds a dense serial JAX reference solver for the finite eigenpairs of the M4b mixed
`H(curl)`--`H1` port pencil. It is deliberately narrower than a public Maxwell backend: it accepts
an already validated and PEC-reduced pencil, computes coefficient-space modes, and returns
dimensionless residual evidence. M4d adds a bounds-checked reduced-to-full coefficient expansion
and the Cartesian P1 E projection used for same-mesh Elmer comparison. M4e wraps these operations in
a guarded public backend, reconstructs physical H, and normalizes E/H from signed native forward
power. M4f consumes its solver-neutral mixed fields in a separate exact-Yee interop layer. M4g adds
a residual-defined adjoint for one separated real mode and differentiates that mode through the Yee
sampler. M4g also adds basis-invariant observables for an isolated mode cluster. The JAX backend
still advertises no accelerator, sparse, matrix-free-eigensolver, open-boundary, or FDTDX-runtime
capability. M4h.1 supplies one unregistered matrix-free shifted-linear-action kernel only.

JAX float64 is required for comparisons with Elmer. The kernel does not select a device, enable
x64, or silently fall back to CPU. Its portable evidence currently runs on a declared serial CPU
runtime.

## Why the full mass matrix is not inverted

The locked Elmer no-potential formulation has

$$
\begin{bmatrix}A_{00}&A_{01}\\A_{10}&A_{11}\end{bmatrix}
\begin{bmatrix}s\\e\end{bmatrix}
=\lambda
\begin{bmatrix}0&0\\0&B_{11}\end{bmatrix}
\begin{bmatrix}s\\e\end{bmatrix}.
$$

The scalar rows are an algebraic constraint and the complete generalized mass is singular. Elmer
handles this with shifted solves involving `A - sigma B`; it does not compute `B^-1 A`. In the
dense reference, finite eigenpairs are preserved exactly by

$$
s=-A_{00}^{-1}A_{01}e,
\qquad
\left(A_{11}-A_{10}A_{00}^{-1}A_{01}\right)e=\lambda B_{11}e.
$$

The implementation uses `solve`, not an explicit inverse. If no scalar DOF survives complete PEC,
the condensed edge stiffness is simply `A11`.

## Scaling, ordering, and normalization

The standard dense problem is divided by an explicit propagation scale squared before
`jax.numpy.linalg.eig`. The intended scale is Elmer's automatic bound

$$
\beta_{\mathrm{limit}}=\omega\sqrt{\max(\epsilon)\max(\mu)}.
$$

The dimensional values are restored after the solve. Modes follow the solver-neutral convention

$$
\beta=\sqrt{-\lambda}
$$

and are ordered by decreasing real beta. Increasing imaginary beta is only a deterministic
secondary key. A repeated eigenvalue still defines a subspace, not authoritative individual
vectors.

Every edge coefficient column is normalized in the positive edge-mass metric,

$$
e^H B_{11}e=1.
$$

Its phase is fixed at the largest-magnitude edge coefficient. This is internal coefficient
normalization, not optical forward-power normalization. Scalar recovery inherits the same phase.
Raw scalar and edge magnitudes are not compared because they are different unknowns.

## Residual evidence

The solver reports separate normwise backward errors for the scalar constraint, the full edge
equation, and the condensed Schur equation. The authoritative mixed value is the maximum of the
first two blockwise errors. This avoids adding residual components with incompatible physical
scales and avoids a dimensional absolute floor.

For degenerate clusters, `compare_port_mode_subspaces` mass-orthonormalizes both bases and reports
principal angles plus the largest projector distance. Column phase, ordering, and nonsingular
mixing inside the cluster therefore do not create a false mismatch.

## Portable scientific witness

The analytic witness is a vacuum rectangular PEC waveguide:

```text
width:      2.0 um
height:     1.0 um
frequency:  100 THz
boundary:   complete PEC
```

This frequency lies above TE10 cutoff and below TE20/TE01 cutoff, so exactly one propagating mode
is expected. The exact value

$$
\beta_{10}=\sqrt{k_0^2-(\pi/a)^2}=1.3875032453\times10^6\ \mathrm{rad/m}
$$

is approached monotonically on the committed 4, 6, and 8 interval meshes. Relative errors are
`1.09077e-3`, `5.56035e-4`, and `3.26239e-4`; successive observed orders are `1.66` and `1.85`.
No additional propagating mode appears in the first six returned modes, and mixed backward errors
remain below `2e-14`.

These results validate the implemented PEC benchmark, not all waveguide geometries or absence of
all possible spurious modes.

## Same-mesh Elmer witness

M4d uses one canonical 268-node, 500-triangle Gmsh mesh for both the locked Elmer `EMPort` process
and the native JAX operator. Elmer's first-encounter edge numbering is mapped to femx's
lexicographic edge order by exact node pair, and all PEC coefficients are checked as exact zeros.
The raw Elmer save contract is also verified as eight eigenvectors followed by one zero final
record, with one unchanged selected `EF2D` projection.

The maximum relative beta error over eight modes is about `2.12e-9`. The full eight-dimensional
edge space has mass-weighted projector distance about `2.85e-6`; the selected Cartesian E field has
mass-weighted aligned relative error about `1.93e-14`. The maximum JAX mixed backward error is about
`1.98e-16`. Committed bounds are `5e-9`, `1e-5`, `1e-11`, and `1e-12`, respectively.

These are regression bounds for the locked source, build, mesh, and float64 CPU execution. They do
not imply open-boundary, lossy, higher-order, all-geometry, or fabricated-device agreement.

## Physical field and power boundary

For positive-z propagation the scalar variable gives $E_z=P_z/(i\beta)$. The backend reconstructs
physical H from Maxwell's curl equation with cellwise physical reluctivity. It integrates

$$
P=\frac12\mathrm{Re}\int(E\times H^*)\cdot\hat z\,dA
$$

on the native mixed field without taking an absolute value. A forward selected mode must therefore
have positive real beta and positive finite power. E and H receive the same amplitude and phase;
the returned nodal P1 projections are not used to redefine power.

Assembly, Schur reduction, dense eigensolve, selected-mode reconstruction, and integration execute
as one JAX transform. A single-triangle analytic witness checks all E/H components and signed power.
On the locked same mesh, reconstruction from Elmer's raw eigenvector matches Elmer's printed power
to about `3.3e-13` relative error, and separately normalized public-JAX and Elmer E fields agree
directly to about `1.63e-13` in the nodal-mass norm.

## Simple-mode residual adjoint

The condensed stiffness is generally nonsymmetric, so M4g does not apply a Hermitian eigenvalue
formula and does not differentiate the `jax.numpy.linalg.eig` trace. For one real simple mode it
augments $Se-\lambda Be=0$ with $e^TBe=1$ and solves the transpose of the bordered residual
Jacobian in reverse mode. JAX then carries the resulting $S$ and $B$ cotangents through scalar
Schur elimination, mixed assembly, material binding, full-DOF expansion, and exact Yee sampling.

One edge coefficient chosen at the validated baseline remains the sign anchor. Binding fails when
the selected eigenvalue is insufficiently separated, complex, poorly converged, or has an unstable
anchor. Evaluations outside declared parameter bounds, positive lossless material space, or the
simple-mode policy return non-finite values. Repeated modes require an invariant-subspace objective;
an individual vector is never differentiated through a crossing.

The portable witnesses cover a general nonsymmetric pencil, an independent bordered forward
sensitivity, central differences, JIT, exact degeneracy rejection, and the analytic material
derivative of the rectangular TE10 mode. On the locked 268-node, 500-triangle Si/SiO2 mesh, the
JAX beta derivative with respect to core relative permittivity is `608577.9366 rad/m`. It differs
from both the native-JAX and independent Elmer central differences by about `5.5e-9` relative. The
same parameter derivative through exact Yee evaluation and power correction differs from central
difference by about `8.7e-9` relative.

These are discrete same-mesh derivative checks, not a material-calibrated sensitivity or a
fabricated-device prediction. The opt-in dynamic FDTDX source continues this simple-mode path
through checkpointed time advance; HDF5 and the default imported source remain setup boundaries.

## Invariant cluster adjoint

An individual vector remains undefined at an eigenvalue crossing. For two or more consecutive
baseline modes, femx instead fixes an isolated dimensionless Riesz contour and solves

$$
(zB-\hat S)X(z)=BC
$$

at 32 trapezoidal quadrature points by default. Constant, $z$, and $\sqrt{-z}$ contour moments
produce the enclosed eigenvalue sum, propagation-constant sum, and a right invariant subspace. The
subspace is returned only through its $B$-orthogonal projector; changing phase, order, or basis
inside the cluster cannot change this output. Reverse mode differentiates the shifted residual
solves and never traces the dense eigensolver.

The contour is fixed at binding. Every evaluation checks its eigenvalue count and clearance,
quadrature refinement, shifted residuals, real-lossless condition, probe conditioning, mass
symmetry and positivity, and projector identities. A failed check produces non-finite values and
gradients instead of selecting a different cluster.

The portable exact-degeneracy witness uses a nonsymmetric pencil and verifies invariance under
independent probe-basis mixing. On the locked 232-node, 434-triangle square-silicon-core mesh, the
first two guided polarization modes have mean effective index `2.91318722205196` and relative
splitting about `2.23e-4`. Their propagation-sum derivative is `1281642.964934225 rad/m` with
respect to core relative permittivity. Relative differences from JAX and Elmer central differences
are `1.51e-9` and `1.50e-9`. The real waveguide pair is close but not exactly repeated; the
synthetic witness owns the exact-multiplicity claim.

## Full mixed matrix-free shifted action

Elmer's ARPACK boundary applies

$$
(A-\sigma B)^{-1}Bx
$$

to the full mixed pencil. M4h.1 follows that boundary directly from the cell-local M4b blocks. A
host-validated map sends constrained PEC coefficients to one discarded sentinel, while active
coefficients gather and scatter only in the reduced free space. No global square matrix or nested
$A_{00}$ solve is created by the operator.

Two-sided absolute-sum equilibration is stopped in reverse mode and leaves the exact residual
solution unchanged. An outer `jax.lax.custom_linear_solve` calls fresh GMRES solves for the primal
and transpose right-hand sides, then an independently recomputed equilibrated residual admits or
rejects the result. Failure returns non-finite coefficients rather than a dense fallback.

On 2, 4, and 8 interval rectangular PEC meshes, matrix-free actions agree with dense assembly
below `5.1e-16` blockwise relative error. The largest shifted-solution difference from an
independently solved dense equilibrated system is about `3.1e-11`, and the material derivative
agrees with central difference to about `2.5e-11`. The 8-interval explicit representation is about
10.14 times smaller than the pair of dense matrices by analytical byte count. These are serial CPU
linear-action results, not an iterative spectrum, runtime memory benchmark, or accelerator result.

## Generalized matrix-free Arnoldi

The finite-range Krylov operator is

$$
T=|\sigma|(A-\sigma B)^{-1}B.
$$

The start is projected through $T$ before normalization because $B$ is singular on the scalar
constraint. Two modified Gram--Schmidt passes use $x^HBy$ rather than an unscaled mixed Euclidean
inner product. A lower block-triangular diagonal preconditioner first approximates the scalar
constraint solve, then corrects the edge block. It is reused for every shift-invert application;
the complete equilibrated residual remains the solve authority.

For each Ritz value $\theta$,

$$
\lambda=\sigma+\frac{|\sigma|}{\theta},
\qquad
\beta=\sqrt{-\lambda}.
$$

Final residuals are normalized by element-assembled row magnitude bounds
$\sum_j(|A_{ij}|+|\lambda||B_{ij}|)|x_j|$. This makes the scalar constraint measurable even though
its correctly assembled result should cancel to zero.

Portable 4, 6, and 8 interval meshes verify six-mode dense parity, mass normalization,
$B$-orthogonality, and final scalar/edge equations. On the locked silicon-waveguide mesh, 967 free
mixed DOFs are solved with a 241-vector unrestarted space. All eight betas match dense JAX to about
`2.4e-12` relative, the largest componentwise residual is about `2.8e-11`, and the mode space passes
the existing locked-Elmer comparison. This is a forward-only serial reference. The projected
eigendecomposition and returned spectrum are stop-gradient.

## Open gates

- automatic cluster tracking across meshes and parameters, plus cluster-field reconstruction;
- cluster-mode injection into an FDTDX time advance and detector objective;
- open/PML boundaries, complex dispersive materials, higher-order elements;
- implicit or thick restart, sparse/coarse preconditioning, and public matrix-free backend wiring;
- owned/ghost multi-device, multi-host, checkpoint/restart, and actual TPU evidence.

[ADR 0021](../adr/0021-finite-port-spectrum-schur-solve.md) binds the finite solve, and
[ADR 0022](../adr/0022-same-mesh-elmer-jax-port-parity.md) binds the cross-backend comparison.
[ADR 0024](../adr/0024-exact-yee-port-mode-transfer.md) binds the exact Yee transfer and detector
boundary. [ADR 0027](../adr/0027-simple-port-eigen-adjoint.md) binds the simple-mode derivative and
degeneracy policy. [ADR 0029](../adr/0029-invariant-port-cluster-adjoint.md) binds the invariant
cluster derivative. [ADR 0030](../adr/0030-full-mixed-matrix-free-shift-solve.md) binds the first
matrix-free shifted action. [ADR 0031](../adr/0031-generalized-matrix-free-port-arnoldi.md) binds
the generalized Arnoldi spectrum.
