# Same-mesh electrothermal contracts

## Purpose and physical chain

M2c connects the validated steady-current and steady-heat reference equations for a
Silicon-Photonics microheater precursor:

```text
electrical parameters
  -> potential phi
  -> E = -grad(phi)
  -> J = sigma E
  -> q_J = J dot E
  -> additive volumetric heat source
  -> temperature T
```

The current solve and heat solve remain independent solver-neutral `Problem` objects. A frozen
`SameMeshJouleHeating` contract joins them only when both reference the exact same mesh object and
therefore the exact same cell order. `q_J` is an `L2` order-zero cell field in `W/m^3`; the heat
operator adds it to the region-declared source before assembling the P1 load vector. No averaging,
interpolation, coordinate conversion, or empirical efficiency factor is present.

For a two-dimensional model, integrating `q_J` over cell area gives `W/m`, power per unit
out-of-plane depth. The composed JAX result integrates the electrical and thermal sides with their
respective geometry arrays and reports their relative difference. It separately forms the
unconstrained heat residual and checks that total variational load plus Dirichlet reaction is zero.

## Chained differentiation

With electrical parameters `p_e`, thermal parameters `p_t`, and an objective cotangent `T_bar`, the
explicit pullback is evaluated from the end of the chain:

1. solve the heat adjoint `A_T.T lambda_T = T_bar`;
2. pull back to `p_t` and to one cotangent per additive source cell;
3. use that cell cotangent in the total Joule VJP;
4. evaluate the direct material derivative of `q_J` and the implicit potential adjoint;
5. return separate, schema-aligned gradients for `p_e` and `p_t`.

The same composed temperature function is a native JAX `jit`/`grad` boundary. Tests require the
native reverse derivative, the explicit two-adjoint result, and central finite differences to
agree. Invalid shapes, float32 inputs, parameter bounds, non-finite sources, and nonpositive
conductivities fail or produce explicit non-finite traced results according to the existing state
map contract; values are never clipped.

## Locked-Elmer realization

Elmer remains two external reference solves rather than a linked multiphysics library. The
validated micrometre-scale case has aligned doped-heater and contact regions. Its one-dimensional
potential makes Joule density constant within each region. The comparison first verifies this
property, then uses an area-weighted region value to create `Volumetric Heat Source` entries for a
fresh locked `HeatSolve` attempt. The area-weighted materialization must preserve the integrated
cellwise power before Elmer is run.

The committed scientific test compares:

- JAX and Elmer cellwise Joule density;
- JAX and Elmer full nodal temperature;
- electrical-to-thermal integrated power and final thermal reaction balance;
- the JAX two-adjoint gradients with fresh Elmer central differences for applied voltage, heater
  conductivity, and thermal conductivity.

Each Elmer attempt uses the locked executable and `StatCurrentSolve.so`/`HeatSolve.so` identities,
fresh directories, explicit execution authorization, and the existing source provenance contract.

### Three-dimensional distinct-space oracle

M5b.4 extends the one-way comparison to first-order tetrahedra without changing the two-dimensional
contract above. A deterministic native Elmer deck represents every thermal Tet4 body, but Elmer's
`Potential` variable exists only on the TiN-plus-aluminum conductor bodies. The result reader uses
the preserved global node identity to align that partial field with the compact JAX electrical
space; `Temperature` remains a full nodal field.

The public coarse ring comparison uses the exact imported mesh and the same source-pinned constant
coefficients, target-current voltage, bottom temperature, top convection, and adiabatic lateral
boundary. Complete potential and temperature vectors are compared. No region averaging or
coordinate interpolation enters the parity metric. The Elmer run is a serial direct reference
oracle; it is not the distributed execution path and does not change the JAX solver's ownership of
adjoints or accelerator execution.

## M2d self-consistent feedback

M2d adds the local law

$$
\sigma(T)=\sigma_{ref}/(1+\alpha(T-T_{ref})).
$$

Conductivity is evaluated at the three temperature DOFs of each triangle. It is deliberately
cell-local: a material interface may hold distinct values at the same geometric vertex. Because a
P1 potential has a constant gradient within a triangle, the electrical stiffness uses the mean of
the three nodal conductivity values. This is the exact integral for the linearly interpolated
coefficient. The three corresponding Joule values are integrated into the heat load with the
consistent P1 mass matrix. The operator identity is
`femx.transfer.same_mesh_cell_local_l2_p1_identity/v1`, distinct from M2c's P0 transfer.

The JAX forward reference performs current then heat solves with explicit block relaxation. It
requires both a scaled state update and offset-shifted free residuals to converge; the 300 K
reference level therefore cannot hide a source/reaction defect. The differentiable map defines the
concatenated residual `R((phi,T),p)=0` and solves one transposed coupled Jacobian for reverse mode.
Gradients are returned separately for electrical, thermal, and feedback schemas.

The locked Elmer realization is one external coupled run, not two materialized one-way runs. Its
typed SIF uses the standard nodal `Variable Temperature` conductivity path and `Joule Heat`, with
absolute verified procedure paths for both solver modules. A closed two-field ASCII result parser
reads potential and temperature, and independent NumPy reconstruction audits the cell-local
conductivity/Joule fields, residuals, power transfer, electrical energy, and thermal reaction.
Fresh Elmer central differences cover voltage, reference electrical conductivity, thermal
conductivity, and the temperature coefficient.

## M2e.6a distributed coupled residual

The first distributed extension keeps the exact M2d same-space discretization. It requires current
and heat to share coordinates, cell order, free-node identity, and Dirichlet-node identity. Cell
temperature reaches each owned cell through pairwise owner/ghost exchange; conductivity, Joule
values, and the consistent heat load remain cell-local. Global current and heat solves use masked
owner reductions, while the coupled reverse rule solves the transpose of the converged residual.

One-, two-, and four-partition forced-CPU tests compare complete potential and temperature fields
and all three parameter-gradient namespaces with the dense M2d authority. Native JAX reverse mode,
independent central differences, recomputed residuals, Joule-power transfer, and StableHLO
collective inspection are separate checks. No new Elmer process is needed for this algebraic
distribution gate because the same discrete M2d operator already has locked Elmer field and
finite-difference evidence.

## Scope limits

M2c proves the one-way P0 chain; M2d additionally proves the self-consistent cell-local P1 feedback
and coupled adjoint for the validated synthetic microheater. Neither claims:

- a transfer between different meshes or element families;
- calibrated silicon, implant, metal, contact, or foundry data;
- optical index perturbation, Maxwell FEM, or FDTDX execution;
- a public production sparse backend, physical GPU/TPU coupled run, or multi-host coupled run.

M2e.6a is only single-process forced-CPU multi-device portability. Physical process-set admission,
different-mesh coupling, and FDTDX composition remain subsequent evidence gates.
