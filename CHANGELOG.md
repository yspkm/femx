# Changelog

All notable changes to femx will be documented here. The project follows semantic versioning once
the first public API is released.

## Unreleased

### Added

- M0 architecture, backend, artifact, validation, and interoperability harness.
- Solver-neutral steady H1 heat-conduction physics and explicit boundary-facet contracts.
- Native float64 JAX P1-triangle assembly and dense serial CPU forward solve.
- Manufactured-solution convergence, Si/SiO2 interface continuity, boundary-load balance, and
  invalid-mesh regression tests.
- Steady current, Joule heating, same-mesh electrothermal coupling, temperature-dependent
  conductivity, and residual-defined implicit adjoints with locked Elmer finite differences.
- Provenance-first Silicon Photonics material records with explicit executable, reference-only,
  and calibration-required status.
- Guarded Gmsh MSH 4.1 ingestion with physical tags, orientation, source permutation, and hashes.
- Strict Tet4 volume ingestion with positive cell and outward boundary normalization, complete
  external-face coverage, physical-volume tags, v2 permutation provenance, and explicit 3D Gmsh
  execution without changing planar v1 record digests.
- A source-pinned public 3D SOI ring/heater OpenCASCADE recipe with explicitly separate electrical
  contacts, semantic volume and boundary partitions, three HXT refinement levels, analytic region
  volumes, fail-closed Tet4 and conformal electrical-interface reporting, and a deterministic
  real-Gmsh witness.
- Solver-neutral lossless port-eigenmode contract, locked Elmer EMPort oracle, complex projected-E
  ingestion, deterministic phase/power normalization, upstream regression, and waveguide mesh
  refinement evidence.
- Native JAX lowest-order Nedelec triangles, the Elmer-compatible mixed port pencil, finite-spectrum
  Schur solve, physical E/H reconstruction, and same-mesh Elmer parity.
- Exact FEM-to-Yee sampling, `eta0_H`, signed-power correction, checksummed ModeBundle HDF5, and
  locked FDTDX detector and custom-source boundaries.
- A residual-defined simple-mode eigen-adjoint with fixed-anchor and eigenvalue-gap safeguards,
  JAX and Elmer central-difference evidence, and differentiable exact-Yee sampling.
- An opt-in dynamic FDTDX mode-source contract with checkpointed source-profile differentiation
  from the native JAX eigen-adjoint through an actual FDTD phasor objective.
- A fixed-contour Riesz cluster adjoint for repeated or close lossless port modes, with invariant
  propagation aggregates, a mass-orthogonal projector, synthetic exact-degeneracy checks, and
  same-mesh JAX/Elmer finite-difference evidence.
- A full-mixed element-local port shift-invert action with stopped two-sided equilibration,
  independently admitted primal/transpose GMRES solves, dense parity, and material-gradient
  evidence.
- An unrestarted generalized matrix-free Arnoldi spectrum with a reusable block-triangular
  preconditioner, componentwise mixed residual admission, portable dense parity, and locked-Elmer
  silicon-waveguide evidence; accelerator execution remains a separate gate.
- A deterministic owned/ghost port-operator plan with canonical global DOFs, explicit value halos,
  owner-row contribution reduction, partition-invariant JIT actions, and reverse-mode parity; real
  collectives and accelerator execution remain separate gates.
- A shared arbitrary-width owned/ghost substrate and three-node H1/P1 heat/current matrix-free
  action with dense, energy, partition, JIT, and VJP evidence.
- An affine four-node tetrahedral scalar H1 element with signed geometry, scalar and tensor
  diffusion, exact consistent mass and P1 source integration, permutation checks, and JAX
  reverse-mode evidence against closed forms and central differences.
- A width-consistent Tet4 scalar owned/ghost, collective RHS, and residual-defined CG path with a
  separate v2 transport identity, element-incidence-linear host owner preparation, and four-device
  forced-CPU dense/VJP parity without all-gather; TPU, Elmer, and ring-physics gates remain open.
- A distinct-space Tet4 current/Joule/heat forward path with exact conductor-to-thermal parent-cell
  transfer, constant volume/flux/convection loads, shifted-temperature solves, independent balance
  admission, manufactured second-order convergence, float32 coverage, and four-forced-CPU-device
  partition/VJP evidence without all-gather; public-ring, Elmer, and TPU gates remain open.
- A source-pinned public 3D ring-heater application binding with explicit TiN-plus-aluminum current
  conduction, target-current rescaling, bottom-temperature and complete-top convection boundaries,
  exact Joule parent-cell transfer, and an admitted coarse/medium/fine JAX float64 forward witness
  whose four selected observable increments contract at the second refinement; a formal
  convergence order, fine-mesh Elmer parity, TPU execution, FDTDX response, and calibration remain
  open.
- Separate content-addressed 15 mA source-reproduction and 5 mA low-temperature ring-heater
  operating-point roles, with an explicit linear current-projection API and documented K/mW,
  thermal-domain, boundary-condition, literature-context, and device-calibration limits.
- A separately retained direct 5 mA coarse-ring JAX-Elmer rerun with complete open fields,
  process/numerical/scientific states, and explicit proof that no parity field was algebraically
  rescaled from the 15 mA artifact; panel a now renders the exact modeled substrate/BOX/device/
  heater order at one-to-one micrometre aspect with two faint 2D far-boundary cladding backdrops,
  complete lateral/z dimension labels, a versioned categorical material palette, a full-range
  perceptually uniform `inferno` temperature map, a symmetric `RdBu_r` difference map, restrained
  dual-contrast device outlines, an orthographic view, a versioned v1.1 light-canvas
  fill/frame palette, and a continuous Si substrate whose approximately 725 um nominal handle is
  depth-truncated below the explicitly marked 0.5 um solve boundary.
- A bounded one-device CPU float64 ring-heater thermal-envelope study crossing three domain widths
  and three modeled silicon depths, plus an ideal-isothermal sidewall bound, with open values and
  an exact-label visualization; this is sensitivity evidence rather than domain convergence or
  package calibration.
- A locked external Elmer Tet4 current/Joule/heat oracle with deterministic native 504/303 mesh
  lowering, partial-potential/full-temperature result ingestion, dual execution authorization,
  installed-module and raw-artifact provenance, a synthetic distinct-space field comparison, and
  complete same-mesh potential/temperature parity on the public coarse 3D ring heater.
- Owner-authoritative scalar H1 loads and unpreconditioned global CG with recomputed-residual
  admission, residual-defined reverse mode, one-to-four forced-CPU-device parity, and one retained
  four-process, 16-device physical TPU process set.
- An explicit hash-bound nested P1 hierarchy and symmetric additive Galerkin preconditioner for
  scalar H1 PCG, with four-device CPU refinement and coefficient-contrast evidence plus a retained
  four-process, 16-device physical TPU setup/forward/VJP process set with no all-gather.
- A distributed same-space electrothermal residual with cell-local temperature-dependent
  conductivity and consistent Joule loading, plus a coupled implicit VJP validated on one to four
  forced CPU devices against the dense M2d authority, native reverse mode, and finite differences.
- A process-complete eight-process, 32-device physical TPU v4 electrothermal witness with strict
  array-layout, dense-authority field/objective/gradient, right-preconditioned adjoint, StableHLO,
  compiler-estimate, and public-safe provenance admission.
- A hash-bound distributed P1 thermo-optic transfer that routes source-owned cell temperatures to
  destination-owned FDTDX x shards with `all_to_all`, plus an actual checkpointed FDTDX phasor
  objective and reverse-mode/finite-difference gate on four forced CPU devices.
- An immutable nested controller artifact for the physical distributed FDTDX gate, with
  independently reconstructed P1/routing operators and explicit ULP plus cell-fraction admission
  between float64 point-location coordinates and the x64-disabled FDTDX runtime grid.
- A fail-closed eight-process, 32-device TPU runner and process-set admission contract for the
  distributed electrothermal-to-FDTDX objective, including exact array reports, bounded
  replication, dual finite differences, compiler-HBM guards, and no-all-gather/no-f64 StableHLO;
  one retained physical TPU v4 process set now passes the digest-pinned admission.
- A fixed-capacity JAX `Mesh` transport lowering with pairwise owner/ghost `ppermute`, canonical
  pack/unpack validation, explicit partition/device/process mapping, real/complex VJP parity, and
  inspected no-all-gather StableHLO on four forced CPU devices; accelerator and multi-host evidence
  remain separate gates.

### Fixed

- Process zero now publishes controller-visible StableHLO copies so the generic TPU synchronizer
  can checksum and finalize every declared required artifact.
- The physical coupled-process admission now requires the sampled-cell pullback's direct potential
  cotangent to be zero, while retaining positive thermal cotangents and nonzero voltage gradients
  through the electrothermal residual adjoint.
- TPU coordinate admission now measures float32 ULP distance across negative and positive axes,
  matching the controller-side signed ordering used by the FDTDX physical gate.
- The coupled TPU runner now adopts FDTDX's topology-aware material Mesh order for every FEM and
  optical array and loads noncontiguous process partitions by exact shard index, preventing
  incompatible nested-JIT device assignments without a global gather.
- The complete coupled graph now passes all electrothermal, transfer, and FDTDX numerical state as
  explicit outer-JIT arguments, so multi-controller tracing never captures an array that spans
  non-addressable devices.
- Float32 distributed electrothermal scalar solves now admit by an explicit backward-error policy,
  retain fresh residual diagnostics, and use positive-diagonal Jacobi PCG without weakening the
  final coupled-residual gate.
- A fixed coupled-adjoint block inverse is now applied on the right of GMRES, so JAX restart
  convergence and femx admission both evaluate the original transpose residual despite the large
  electrical/thermal scale difference.
