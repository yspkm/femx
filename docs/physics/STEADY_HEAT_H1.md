# Steady H1 heat-conduction contract

## Purpose

This is femx's first executable scientific slice. It is deliberately small enough to compare
element by element with Elmer before adding electrothermal coupling, thermo-optic transfer, sparse
operators, or TPU distribution.

The implemented problem is two-dimensional Cartesian, steady, real-valued conduction per unit
out-of-plane depth:

$$
-\nabla\cdot(k\nabla T)=Q.
$$

For test functions that vanish on the strong-temperature boundary, femx assembles

$$
\int_\Omega \nabla v^\mathsf{T}k\nabla T\,d\Omega
=\int_\Omega vQ\,d\Omega+\int_{\Gamma_g}vg\,d\Gamma.
$$

The boundary coefficient `g` is defined by its variational sign: it is added to the right-hand
side. With outward normal `n` and Fourier heat flux `q = -k grad(T)`, this means
`q dot n = -g`; positive `g` is heat entering the domain. The name alone must never be used to infer
the sign.

## Elmer derivation

The locked reference source is Elmer commit
`4f2d7e4b99f8f0dcf2f7ac579e056969373bf594`. The relevant implementation chain is:

1. `fem/src/modules/HeatSolve.F90` reads `Heat Conductivity`. A scalar is expanded onto the
   conductivity-tensor diagonal.
2. The same module reads `Volumetric Heat Source`; only when that key is absent does the older
   `Heat Source` path multiply by density.
3. `fem/src/DiffuseConvectiveAnisotropic.F90::DiffuseConvectiveCompose` adds
   `k * grad(N_q) dot grad(N_p)` to the local stiffness and `Q * N_p` to the local force.
4. `HeatSolve.F90::AddHeatFluxBC` adds the SIF `Heat Flux` value to the boundary load, and
   `DiffuseConvectiveBoundary` integrates that load as a positive right-hand-side term.

An adjacent Elmer source comment describes the flux with a different-looking sign. femx therefore
locks the algebraic convention above and validates the SIF mapping by execution, not by the comment.
On the locked build, the canonical one-triangle case gives `T=[0.5, 0, 0]` for `Heat Flux=+1`, and
the generated square case with `k=2`, `g=+2`, and `T(left)=0` gives `T=x`. No sign reversal occurs
between the femx variational coefficient and Elmer SIF.

## M1 numerical scope

- H1-conforming, first-order triangular elements;
- explicit SI-metre coordinates and explicit segment boundary facets;
- positive, isotropic, piecewise-constant conductivity in `W/(m*K)`;
- piecewise-constant volumetric heat source in `W/m^3`;
- strong temperature constraints in `K`;
- piecewise-constant variational boundary heat load in `W/m^2`;
- symmetric Dirichlet elimination;
- native JAX float64 assembly and dense serial CPU direct solve;
- canonical `DESIGN`/`CONTROL` vector binding and a residual-defined adjoint state VJP.

Thermal regions must partition every cell exactly once. Explicit boundary facets must equal the
topological boundary of the triangle mesh. Conflicting boundary values, overlapping heat-load
tags, degenerate cells, wrong units, and a non-CPU or non-float64 local runtime fail explicitly.

The dense matrix is a correctness reference, not the intended large-scale implementation. M2e
reuses the same three-by-three cell matrices in a free-node owned/ghost action. Its internal JAX
collective path now assembles volumetric, facet, and nonzero-Dirichlet RHS terms and solves the
reduced system with residual-admitted CG on up to four forced CPU devices. It is not yet an
advertised heat backend or a physical accelerator result.

## Current evidence and non-claims

Portable tests establish the exact P1 reference-triangle matrix and load, symmetry-preserving
Dirichlet elimination, a linear solution driven by a nonzero boundary load, global reaction/load
balance, and second-order L2 convergence for the manufactured solution `T=x(1-x)`. An aligned
micrometre-scale Si/SiO2 interface benchmark additionally matches the piecewise analytic
temperature and preserves the normal `k grad(T)` load across the material interface using
representative constant conductivities; it is not a calibrated material model.

The locked Elmer `4f2d7e4b9` executable has now run the identical explicit femx meshes for the
nonzero-load, manufactured-source, and Si/SiO2 interface cases. Full nodal fields agree with the
JAX implementation at the committed tolerances. The adapter writes native Elmer mesh files and a
fixed typed SIF; it does not accept arbitrary SIF text. Numeric values are read from the
full-precision ASCII-v3 result with its explicit permutation, while binary VTU and raw logs remain
durable provenance.

For active parameters `p`, the JAX state map uses `R(T,p)=A(p)T-b(p)=0`. Its reverse rule solves
`A.T lambda = T_bar` and evaluates `p_bar = -lambda.T R_p`. The public differentiable boundary is
the bound temperature state map, not the Python `Solution` reporting envelope. A case with
parameterized conductivity, volumetric source, variational boundary load, and strong temperature
checks this VJP against native JAX reverse-mode, JAX central differences, and fresh locked-Elmer
central-difference runs. Parameter names and units remain aligned with schema order.

M2c extends the bound state map with an additive same-mesh `L2/P0` source in `W/m^3`. Its explicit
VJP returns separate cotangents for the thermal parameters and every source cell, allowing the
Joule/current adjoint to compose without changing the base `SteadyHeat` problem schema. See the
[electrothermal contract](ELECTROTHERMAL_COUPLING.md) for the transfer and cross-backend evidence.

This milestone does **not** yet claim:

- Silicon/SiO2/heater material calibration;
- thermo-optic transfer into FDTDX;
- forward-mode, nonlinear, eigenmode, shape, or mesh-coordinate derivatives;
- a production sparse or preconditioned backend, public parallel capability, multi-host, GPU, or
  TPU execution.

Those claims require the later evidence gates in `docs/VERIFICATION.md`.
