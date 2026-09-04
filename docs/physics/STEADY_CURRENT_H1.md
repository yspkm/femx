# Steady H1 electric-current contract

## Purpose

This M2a precursor establishes the electrical half of a Silicon Photonics electrothermal workflow
before coupling it to temperature. It is intentionally the same small scalar-H1 class in native JAX
and locked Elmer, so sign, material-interface, current-continuity, and Joule-energy errors are
observable before adding nonlinear materials or mesh transfer.

The implemented problem is two-dimensional Cartesian, steady, real-valued current conduction per
unit out-of-plane depth:

$$
-\nabla\cdot(\sigma\nabla\phi)=s.
$$

For test functions that vanish on strong-potential boundaries, femx uses

$$
\int_\Omega \nabla v^\mathsf{T}\sigma\nabla\phi\,d\Omega
=\int_\Omega vs\,d\Omega+\int_{\Gamma_g}vg\,d\Gamma.
$$

The natural coefficient `g` is defined by this positive right-hand-side sign. With
`E=-grad(phi)` and physical current `J=sigma*E`, it follows that `J dot n=-g`. A positive `g` is not
silently renamed or sign-reversed at either backend boundary.

The Joule heat density is

$$
q_J=J\cdot E=\nabla\phi^\mathsf{T}\sigma\nabla\phi,
$$

in `W/m^3`. Its two-dimensional integral is power per unit out-of-plane depth in `W/m`.

## Elmer derivation

The locked source is Elmer commit `4f2d7e4b99f8f0dcf2f7ac579e056969373bf594`. In
`fem/src/modules/StatCurrentSolve.F90`:

1. `Electric Conductivity` is read from each material and a scalar is expanded on the tensor
   diagonal.
2. `Current Source` is interpolated from the body force.
3. `StatCurrentCompose` adds `grad(N_p) dot sigma grad(N_q)` to the element matrix and
   `s N_p` to its force vector.
4. A boundary carrying `Current Density BC=True` reads `Current Density`; `StatCurrentBoundary`
   integrates it as a positive force term.
5. The postprocessor explicitly defines `E=-grad(phi)`, subtracts `sigma grad(phi)` when forming
   physical current, and accumulates `grad(phi) dot sigma grad(phi)` as Joule heating.

The generated SIF binds the verified `StatCurrentSolve.so` by absolute path and asks Elmer to emit
only the nodal `Potential` in restricted ASCII-v3 output. femx does not trust a second backend's
averaging convention for derived fields: it reconstructs cellwise `E`, `J`, and `q_J` independently
from the ingested potential and the original femx mesh.

## Implemented numerical scope

- H1-conforming first-order triangles for potential;
- explicit SI-metre coordinates and boundary segments;
- positive isotropic piecewise-constant conductivity in `S/m`;
- piecewise-constant volumetric current source in `A/m^3`;
- strong potential in `V` and positive-variational natural load in `A/m^2`;
- cellwise discontinuous L2-P0 electric field, current density, and Joule heat density;
- native JAX float64 dense serial CPU assembly and direct solve;
- a shared three-node matrix-free stiffness action with portable owned/ghost algebra;
- canonical `DESIGN`/`CONTROL` binding, potential state VJP, and total Joule-density VJP;
- separately installed serial Elmer `StatCurrentSolve` with locked executable/module identity.

Conductive regions partition every cell exactly once. Potential and current-load facets cannot
overlap. Conflicting corner potentials, overlapping region or boundary ownership, degenerate mesh
cells, wrong parameter units, nonpositive conductivity, non-CPU JAX selection, or disabled JAX
float64 fail explicitly.

## Evidence and non-claims

Portable evidence includes exact linear fields, natural-load sign, source and parameter contracts,
charge-equation backward error, Joule/variational energy balance, and measured second-order L2
potential convergence for `phi=x(1-x)`. A micrometre-scale two-region synthetic heater checks
piecewise conductivity, current continuity, analytic voltage drop, piecewise Joule density, and
integrated power. The conductivities are representative test constants, not a calibrated foundry
model.

The locked Elmer build has executed the identical femx meshes for a nonzero natural load, the
manufactured source, and the two-region heater. Potential fields agree at the committed absolute
tolerances; derived-field parity uses a normalized L2 metric so roundoff in analytically zero
transverse components is not divided by zero. Each solver separately satisfies the analytic field
and energy checks.

For active parameters `p`, the potential state map uses `R(phi,p)=A(p)phi-b(p)=0`. Its explicit
reverse rule solves `A.T lambda=phi_bar` and evaluates `-lambda.T R_p`. The Joule map additionally
computes the local pullback of `q_J(phi,p)`, so the reported total parameter gradient is the sum of
the direct material term and the indirect potential-adjoint term. Tests compare this split VJP with
native JAX reverse-mode and JAX central differences. Fresh locked-Elmer process runs provide an
independent central-difference check of integrated Joule power for conductivity, source, and
natural-load parameters.

M2e reuses the exact cell stiffness already assembled above in a free-node matrix-free action.
Representative heater/doped-region coefficients pass dense principal-operator, cell-energy,
one-to-four-partition, owner-policy, cell-order, JIT, and VJP checks. The internal collective path
also assembles source, natural-load, and nonzero-potential boundary terms and admits a global CG
solution by its recomputed residual on up to four forced CPU devices. Joule postprocessing and the
coupled electrothermal residual/VJP now reuse the same owner/ghost algebra in M2e.6a. That coupled
witness is still single-process forced-CPU portability, not a public distributed backend or a
physical accelerator result.

M2c now consumes this cellwise Joule map through the separately documented
[`SameMeshJouleHeating`](ELECTROTHERMAL_COUPLING.md) contract. The current solver itself still does
**not** claim:

- temperature-dependent conductivity or electrothermal feedback;
- a different-mesh transfer operator;
- calibrated implant, contact, metal, or foundry material data;
- FDTDX or thermo-optic field transfer;
- a production sparse or preconditioned backend, public parallel capability, GPU, TPU, or
  multi-host execution.

Those are later M2/M3/M4 gates and must not be inferred from the serial differentiable reference.
