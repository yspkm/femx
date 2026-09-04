# Port eigenmode contract

## Scope

`PortEigenmode` is the first optical FEM problem in femx. A locked, separately installed Elmer
`EMPortSolver` is the reference oracle. A guarded native JAX backend now implements the same
lossless first-order serial CPU subset and returns a physical complex E/H pair.

The v1 domain is a two-dimensional Cartesian `x-y` waveguide cross section in SI metres with
positive-`z` propagation. It accepts lossless, isotropic, piecewise-constant real
`epsilon_r > 0` and `mu_r > 0`, and requires homogeneous PEC on every external boundary segment.
The caller provides these values explicitly, either as literals or dimensionless scalar parameter
references resolved at solve time. No material-catalog entry is silently selected.

## Formulation

The locked Elmer source implements the mixed port system

$$
\nabla_t\times(\nu_r\nabla_t\times E_t)
-\omega^2\epsilon_0\mu_0\epsilon_r E_t
+\nu_r\nabla_t P_z
=\lambda E_t,
$$

$$
-\nabla_t\cdot(\epsilon_r E_t)-\epsilon_r P_z=0,
\qquad P_z=\sqrt{\lambda}\,E_z,
$$

with first-family first-order edge elements for the transverse field, a first-order nodal
constraint, Piola transformation, and

$$
\beta=\sqrt{-\lambda},\qquad n_{eff}=\beta/k_0,qquad
k_0=2\pi f/c_0.
$$

The solver-neutral ordering is decreasing `Re(beta)`. For the current lossless subset this maps to
Elmer's smallest-real-`lambda` ordering. The selected mode index is zero-based in femx and
translated to Elmer's one-based `Eigenfunction Index`.

## Mesh and orientation

Preparation accepts only the concrete canonical `Mesh` with:

- exact float64 two-dimensional coordinates in metres;
- first-order triangle cells and explicit segment boundary facets;
- complete, non-overlapping region and PEC tag partitions;
- explicit integer edge signs for local edges `(0,1)`, `(1,2)`, `(2,0)`;
- each sign equal to the direction induced by increasing canonical global node id.

Gmsh is upstream of this boundary. Elmer and the later JAX backend consume the same canonical
coordinates, connectivity, tags, and orientations rather than independently remeshing.

## Execution and outputs

Preparation lowers the native Elmer mesh entirely in memory. Literal-material problems also render
their typed SIF at preparation; parameterized problems defer only SIF rendering until the exact
solve-time values pass schema, bounds, finiteness, and positivity checks. Solve requires a fresh
absolute run directory plus `execution_authorized=True` and `allow_external_process=True`. The
backend binds absolute paths and locked hashes for `ElmerSolver`, `EMPort.so`,
`ResultOutputSolve.so`, and `SaveData.so`.

The returned field is complex Cartesian `electric_field[node, xyz]` in `V/m`, represented as an
`H1` order-one vector because it is Elmer's nodal projection of the solved edge field. Raw edge
coefficients are retained in the raw ASCII result. After M4d validates Elmer's row ordering against
canonical mesh edges, the selected normalized mixed vector is also exposed as two explicitly named
solver-neutral coefficient fields: nodal longitudinal potential in `V/m^2` and transverse H(curl)
edge moments in `V`. Neither is mislabeled as a nodal Cartesian field. The run retains:

- the complete complex eigenvalue spectrum and printed absolute residuals;
- selected `beta`, `n_eff`, EMPort edge-field power, and impedance;
- every raw mixed eigenvector, the double-precision ASCII projected field, and raw binary VTU;
- every native-mesh/SIF/log/spectrum hash and locked source/runtime identity.

Elmer's unity-normalized eigenvector has arbitrary amplitude and phase. femx applies

$$
s=\sqrt{P_{target}/P_{Elmer}}
$$

and chooses the largest-magnitude projected component as a deterministic phase anchor. After
rotation and scaling that component is positive real. The positive Elmer power is computed from
the underlying edge field, not reconstructed from the nodal projection.

The native JAX backend consumes the same canonical mesh and mixed ordering. It assembles the
Elmer-compatible pencil, preserves the singular scalar constraint through Schur elimination, and
reconstructs

$$
E_z=\frac{P_z}{i\beta},\qquad H=\frac{1}{i\omega\mu}\nabla\times E.
$$

Its signed authority is $\frac12\mathrm{Re}\int(E\times H^*)\cdot\hat z\,dA$ on native
quadrature. Positive real beta and positive finite power are required. The same phase and amplitude
factor is applied to both fields before their nodal P1 L2 projections are returned. Those
projections are not treated as Yee samples or used as the power authority.

## Convergence and evidence

Process completion, Ritz convergence, and scientific validation are separate. A solution is
numerically converged only when Elmer reports every requested Ritz value converged. Elmer's printed
absolute `CheckResidualsComplex` values depend on equation and coordinate scaling, so femx records
them but does not compare them directly with the Ritz tolerance.

The locked local evidence on 2026-08-30 passed two optical gates:

1. Elmer's registered upstream `EM_port_eigen` case was regenerated with the matching ElmerGrid and
   reproduced `TEST.PASSED`, 30 Ritz values, `beta = 3.82726670`, and first eigenvalue
   `-14.64797039137`.
2. A 500 nm by 220 nm representative silicon core in a 4 micrometre by 3 micrometre silica box at
   1.55 micrometres was meshed at three exact size halvings. With explicit `n_core=3.48` and
   `n_clad=1.444`, `Re(n_eff)` changed
   `2.4167066863 -> 2.4410152886 -> 2.4473143953`; the successive-change ratio was about 0.259,
   corresponding to observed order about 1.95.

The second case establishes numerical mesh convergence for that fixed representative model. The
indices are explicit test inputs, not a calibrated foundry material claim, and the fixed PEC box
does not establish open-boundary convergence.

M4d then sent one 268-node, 500-triangle Gmsh mesh to both the locked Elmer process and the native
JAX mixed solver. Across eight modes, the maximum relative beta error was about `2.12e-9`; the
edge-mass projector distance for the full mode space was about `2.85e-6`. The selected Cartesian E
projection differed by about `1.93e-14` after one mass-weighted complex alignment, while the JAX
mixed backward error stayed below `2e-16`. This is same-discretization parity for the stated
lossless PEC model, not a universal accuracy bound.

M4e reconstructed the selected Elmer raw mixed eigenvector through the physical JAX E/H kernel.
The resulting native power was `1.045781910871658e-29 W`, versus Elmer's
`1.045781910872e-29 W`, for about `3.3e-13` relative difference. After independent one-watt
normalization and deterministic phase fixing, the public JAX and Elmer E projections differed by
about `1.63e-13` directly, without a fitted complex alignment. The committed bounds are `2e-12`
and `1e-10`, respectively.

M4f evaluates those normalized mixed fields directly at the six component-specific FDTDX Yee
positions. The first locked plan uses a 33-by-25-by-1 target plane and records every selected source
triangle, barycentric weight, canonical edge sign, coordinate axis, and hash. Elmer-derived and
JAX-derived transferred E and `eta0_H` agree to about `1.93e-14` and `3.42e-14`. Their raw
target-grid powers are about `1.09856 W` for a one-watt source mode, so the common correction scale
and the roughly 9.86 percent pre-correction error are retained in each `ModeBundle`. The corrected
arrays are consumed unchanged by the locked FDTDX custom overlap detector. On five exact-offset
planes from 16-by-12 through 256-by-192, absolute raw-power error decreases monotonically from
about 50.1 percent to 0.451 percent; all four successive observed orders exceed one. Both locked
backend bundles are then written and reconstructed through `femx.mode.hdf5/v1` before the final
cross-backend field comparison.

M4g binds core relative permittivity as the same dimensionless parameter in the native JAX and
Elmer backends. The JAX backend differentiates a separated real mode through the residual and
edge-mass normalization, not through the dense eigensolver trace. On the same 268-node mesh, the
beta derivative is `608577.9366 rad/m`; it differs from the native-JAX and independent Elmer
central differences by about `5.47e-9` and `5.50e-9` relative. The selected mode has relative gap
about `0.2269`. The derivative through exact Yee evaluation and signed-power correction differs
from central difference by about `8.63e-9`.

The sign anchor is fixed at the baseline. Insufficient gap, complex or poorly converged modes,
weak anchors, invalid parameter bounds, and nonpositive materials fail closed. Repeated-mode
derivatives require an invariant-subspace contract and are not represented by an arbitrary
individual vector.

Run the opted-in evidence with:

```bash
FEMX_RUN_ELMER_TESTS=1 \
FEMX_ELMER_EXECUTABLE=/absolute/path/to/ElmerSolver \
FEMX_RUN_GMSH_TESTS=1 \
FEMX_GMSH_EXECUTABLE=/absolute/path/to/gmsh \
pytest tests/integration/test_elmer_port_eigenmode_execution.py \
       tests/scientific/test_elmer_emport_oracle.py \
       tests/scientific/test_elmer_jax_port_eigenmode.py
```

## Open gates

- automatic mode and cluster tracking across mesh changes;
- cluster-field reconstruction and differentiable FDTDX cluster-source injection;
- dispersive/lossy materials, open boundaries, higher order, sparse execution, and TPU evidence.

[ADR 0018](../adr/0018-elmer-port-eigenmode-oracle.md) records the Elmer binding decision, and
[ADR 0022](../adr/0022-same-mesh-elmer-jax-port-parity.md) records the same-mesh parity decision.
[ADR 0023](../adr/0023-physical-port-fields-and-jax-backend.md) records the physical-field and
public-JAX boundary. [ADR 0024](../adr/0024-exact-yee-port-mode-transfer.md) records the exact Yee
transfer and FDTDX detector boundary. [ADR 0025](../adr/0025-durable-mode-bundle-hdf5.md) records
the durable mode artifact. [ADR 0027](../adr/0027-simple-port-eigen-adjoint.md) records the
simple-mode adjoint and degeneracy policy. [ADR 0029](../adr/0029-invariant-port-cluster-adjoint.md)
records the invariant cluster derivative.
