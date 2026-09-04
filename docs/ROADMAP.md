# Roadmap

## M0 — architecture harness

- solver-neutral contracts and import boundaries;
- explicit backend capability negotiation and execution policy;
- mesh/function-space/DOF/partition schemas;
- run manifest, validation report, and FDTDX mode handoff;
- repository test, type, lint, architecture, and build gates.

M0 makes no production-solver claim.

## M1 — steady heat vertical slice

- frozen `SteadyHeat` physics specification;
- deterministic triangular H1-P1 mesh fixture;
- typed, fixed-scope Elmer native-mesh/SIF lowering and result parser;
- native JAX assembled reference implementation;
- manufactured solution, refinement order, flux balance, cross-backend comparison;
- conductivity gradient against central finite differences.

The forward JAX operator, deterministic Elmer lowering, same-mesh comparisons, and residual-defined
adjoint/finite-difference gradient gate are complete for the current serial H1 subset.

## M2 — electrothermal workflow

- static current and Joule-heating formulations (M2a forward JAX/Elmer reference complete);
- residual-defined potential and total Joule adjoints (M2b complete);
- typed same-mesh `L2/P0` Joule-to-heat transfer and chained adjoint (M2c complete);
- transfer-power, final heat-reaction, JAX finite-difference, and locked-Elmer chain evidence
  (M2c complete);
- temperature-dependent material coefficients (M2d complete for the resistivity law);
- explicit feedback iteration and nonlinear convergence evidence (M2d complete);
- concatenated current/heat implicit adjoint and locked-Elmer finite-difference evidence
  (M2d complete);
- different-mesh transfer and transfer-error evidence.

M2d establishes the differentiable self-consistent current/Joule/temperature loop on one exact
mesh. Different-mesh projection and calibrated material evidence remain open M2 work.

### M2e — distributed electrothermal execution

The next SiPh thermal scaling milestone proceeds in this order:

1. canonical owned/ghost partitioning for H1/P1 heat and current (portable algebra complete);
2. matrix-free diffusion and conduction actions with explicit halo exchange (in-process action,
   dense parity, energy, JIT, and VJP gate complete);
3. owner-authoritative load assembly and an unpreconditioned CG solve with residual,
   partition-invariance, and implicit-VJP evidence (one to four forced CPU devices complete);
4. physical accelerator and multi-process admission with process-local inputs and retained
   evidence (four-process, 16-device physical TPU process set complete);
5. an explicit nested multilevel preconditioner with iteration, refinement, coefficient-contrast,
   and implicit-VJP evidence (four forced CPU devices and one four-process, 16-device physical TPU
   process set complete);
6. the coupled electrothermal residual-defined implicit VJP
   (M2e.6a forced-CPU portability and M2e.6b physical process-set evidence complete); and
7. shard-preserving composition with the existing FDTDX thermo-optic objective
   (M2e.7a four-forced-CPU checkpointed-objective gate and M2e.7b immutable controller-input
   boundary plus physical eight-process, 32-device TPU v4 process set complete).

Steps 1-3 now establish the scalar operator, load, collective CG, and residual-defined reverse rule
on a single process with up to four forced CPU devices. This is portable multi-device evidence,
not accelerator or multi-host evidence. Existing TPU results for the port Maxwell operator cannot
be cited as scalar heat/current support. Step 4 is now closed by a retained physical float32 TPU
run with four processes, 16 devices, and 16 exactly-once-addressable FEM partitions. Both bounded
heat and current systems passed the independent NumPy float64 solution/adjoint authority and the
process-complete admission policy. Step 5 now adds a caller-supplied, hash-bound P1 hierarchy,
Galerkin coarse operators, and a stopped symmetric additive inverse. Four-device CPU evidence
shows bounded refinement growth, strong coefficient-contrast improvement, and a residual-defined
VJP. A retained four-process, 16-device TPU v5e process set now also admits explicit partitioned
fine transfer, bounded replicated coarse transfer, setup, forward, and residual-defined VJP for
the same heat/current systems. This physical result is a bounded correctness witness, not a
preconditioner scaling or live-HBM result. Step 6a now preserves the full M2d current, cell-local
Joule, and heat residual on one to four partitions and differentiates the converged coupled
residual with one transpose solve. Dense fields, all three parameter-gradient namespaces,
independent finite differences, and no-all-gather StableHLO pass on four forced CPU devices. This
is portable single-process evidence only. Step 6b is now closed by a retained eight-process,
32-device TPU v4 process set on a 289-node, 512-triangle problem. Every partition was addressable
exactly once; forward, explicit residual VJP, and native reverse passed the immutable dense float64
authority and no-all-gather StableHLO admission. The earlier scalar and left-preconditioned-adjoint
failures remain retained diagnostics rather than successful evidence. This is bounded 2D
same-discretization correctness, not 3D production, scaling, live HBM, fresh Elmer, FDTDX, or
recovery evidence. Step 7a now keeps cell-local P1 temperatures partitioned, routes barycentric
samples to FDTDX x owners with one `all_to_all`, preserves the concrete material-array sharding
through public `apply_params`, and differentiates a checkpointed `run_fdtd` phasor objective on
four forced CPU devices. Material and dense-parameter relative differences are below
$10^{-15}$, the applied-voltage gradient agrees with central difference to $1.42\times10^{-7}$,
and the combined StableHLO contains no all-gather. Step 7b is now closed by a retained
eight-process, 32-device TPU v4 process set using the immutable controller input. Every partition
was addressable exactly once; potential, temperature, material, and transfer differences were at
most $2.23\times10^{-6}$, native and explicit gradients agreed to $1.01\times10^{-7}$, and both
applied-voltage finite differences passed. All four compiled paths contain the required
`all_to_all`, `collective_permute`, and `all_reduce`, with no all-gather or f64. This remains a
bounded 2D correctness witness, not 3D FEM, ring-heater convergence, S-parameters, scaling,
live-HBM evidence, or inverse design.

## M3 — optical interoperability

- provenance-first Si/SiO2/Ge/Al/Cu/Ti/TiN/doped-Si catalog and guarded ElmerGUI importer
  (catalog infrastructure complete; most process models intentionally non-executable);
- hashed linear thermo-optic law and P1-to-FDTDX cell-center sampling (M3a complete);
- locked FDTDX `apply_params` field parity and composed electrothermal gradient evidence
  (M3a complete);
- actual `run_fdtd` source/phasor-detector precursor and end-to-end gradient evidence
  (M3b complete for checkpointed and reversible derivatives);
- shard-preserving distributed electrothermal-to-FDTDX material routing and checkpointed optical
  objective (M2e.7a portable gate and M2e.7b physical eight-process, 32-device TPU v4 process set
  complete);
- deterministic rectangular waveguide recipe, guarded Gmsh CLI execution, strict MSH 4.1
  ingestion, exact orientation/permutation provenance, and same-canonical-mesh JAX/Elmer handoff
  (M3c meshing foundation complete; no optical-solution claim);
- typed lossless `PortEigenmode`, pinned Elmer `EMPortSolver`/module identities, complex nodal-field
  ingestion, deterministic phase and Elmer-edge-power normalization, official upstream reference,
  and three-level effective-index refinement (M3d Elmer optical oracle complete);
- converged mode-normalized waveguide transmission and S-parameter evidence;
- dispersive/lossy and calibrated material models;
- Elmer EMWave scattering adapter and complex E/H export;
- native edge-element orientation, Piola, exact-sequence, and local Gram property tests
  (M4a local-element foundation complete; no global Maxwell solve);
- Elmer-compatible mixed nodal/edge pencil, dense global scatter, and exact PEC principal reduction
  (M4b operator foundation complete; no eigenpair claim);
- selected-mode FEM-to-Cartesian E projection (M4d complete);
- guarded native JAX port backend, physical E/H reconstruction, and signed native-power
  normalization (M4e complete);
- exact component-staggered FEM-to-Yee evaluation, `eta0_H`, power correction, transfer hashing,
  and `femx.mode/v1` construction (M4f implementation complete);
- locked FDTDX custom-overlap consumption (M4f detector boundary complete);
- five-level raw-power Yee-refinement evidence (M4f refinement boundary complete);
- checksummed `femx.mode.hdf5/v1` round trip (M4f durable-artifact boundary complete);
- public custom FDTDX source, exact mode/grid/medium binding, analytic one-watt time advance, and
  checkpointed/reversible downstream gradients (M4f source boundary complete);
- locked Elmer/JAX waveguide-mode HDF5 reload, source propagation, and downstream complex-field
  comparison (M4f waveguide-source boundary complete);
- cross-representation phase and forward-power conservation evidence.

M3a establishes the lossless scalar material boundary. M3b establishes Maxwell time integration
and an optical detector gradient on a deliberately small heated-core precursor. The M3c meshing
foundation establishes reproducible canonical geometry and identical backend input. M3d establishes
a locked Elmer mixed `H(curl)`/`H1` port reference, projected complex E, upstream regression, and
fixed-model mesh convergence. M4d adds one same-discretization native JAX/Elmer port agreement. M4e
adds the guarded JAX backend and a physical E/H pair normalized by signed native power. M4f now has
an exact-offset, hashed and differentiable transfer implementation plus real custom-detector
consumption. Its per-grid pre-correction power error is retained, and the five-level fixed-mode
study now establishes raw-power convergence. Checksummed HDF5 round trips also preserve both
backend bundles through the real detector boundary. The locked user FDTDX fork now also advances an
analytic imported mode and preserves downstream checkpointed and reversible gradients. The locked
Elmer/JAX waveguide bundles now separately drive one identical FDTDX scene and agree at the
downstream six-component phasor to about `1.83e-14`. Converged device performance, calibrated
silicon, and accelerator execution of those physical waveguide bundles remained open at that CPU
gate. A retained four-process, 16-device Spot TPU v5e witness now closes one bounded physical
source/downstream parity gate after explicit float32/complex64 lowering; convergence, absolute
transmission, S-parameters, scaling, adjoint, and Spot recovery remain open.

## M4 — native JAX Maxwell FEM

- lowest-order first-family Nédélec triangle, canonical edge map, covariant Piola, and local Gram
  operators (M4a complete);
- locked-Elmer no-potential mixed local/global pencil and exact PEC reduction (M4b complete);
- dense finite generalized eigensolver, analytic PEC refinement, residuals, and subspace metric
  (M4c complete);
- raw Elmer mixed-mode ingestion, bounds-safe JAX reconstruction, nodal E projection, and same-mesh
  eight-mode beta/subspace/field parity (M4d complete);
- guarded public JAX port backend and physical E/H power normalization (M4e complete);
- exact Yee transfer, `ModeBundle`, locked FDTDX detector consumption, durable artifact, and
  analytic custom-source gradient plus locked Elmer/JAX waveguide-source parity witness
  (M4f complete);
- explicit canonical-mode to TPU float32/complex64 source lowering with bounded coordinate,
  scalar, field, material, and signed-power error (portable precursor complete; no physical FDTDX
  TPU or multi-host claim);
- explicit FDTDX source E/H/time-offset placement on the source-material `NamedSharding`, with
  addressable-shard validation and outer-jitted time advance (portable precursor complete on four
  forced CPU devices; no physical TPU or multi-host claim);
- process-complete execution of that locked custom source on four JAX processes and 16 Spot TPU
  v5e devices, with exact source-shard coverage, 66 finite nonzero float32 time steps, and no
  all-gather in the four retained StableHLO records (one bounded homogeneous-source infrastructure
  witness complete; Elmer/JAX waveguide parity, convergence, S-parameters, scaling, adjoint, and
  Spot recovery remain open);
- process-complete execution of independently generated Elmer and JAX waveguide bundles through
  one shared 64-by-52-by-36 Si/SiO2 FDTDX scene on four JAX processes and 16 Spot TPU v5e devices,
  with exact source-shard coverage, 316 finite steps, no StableHLO all-gather, and downstream
  complex-field parity below `1e-7` (one bounded physical source/downstream parity witness complete;
  convergence, absolute transmission, S-parameters, scaling, adjoint, and Spot recovery remain
  open);
- residual-defined simple-mode adjoint, fixed-anchor/gap admission policy, JAX and Elmer central
  differences, differentiable exact-Yee sampling, and an opt-in checkpointed dynamic FDTDX source
  profile; fixed-contour invariant-cluster propagation aggregates and a mass projector with exact
  repeated-mode and same-mesh Elmer evidence (M4g complete for this dense lossless subset;
  automatic cluster tracking, cluster-field injection, reversible source cotangents, and total
  changing-scene derivatives remain open);
- full-mixed element-local application of Elmer's shifted port system, two-sided equilibration,
  independently admitted GMRES, and a residual-defined transpose solve (M4h.1 complete for the
  serial CPU linear action; this is not yet an iterative eigensolver);
- full-mixed generalized matrix-free Arnoldi with a reusable block preconditioner, portable dense
  parity, and locked-Elmer silicon-waveguide spectrum/subspace evidence (M4h.2 complete for the
  unrestarted serial CPU forward kernel; public backend registration, eigen-adjoint, implicit or
  thick restart, measured memory/speed, and accelerator projected eigensolving remain open);
- deterministic global cell/DOF ownership, variable-length owned/ghost vectors, explicit value
  halo and owner-row reduction, with serial/JIT/VJP partition-invariance evidence (M4h.3a complete
  for an in-process algebraic reference; no accelerator claim);
- fixed-capacity transport lowering on an explicit one-dimensional JAX `Mesh`, pairwise value and
  contribution `ppermute`, serial/JIT/VJP parity, StableHLO no-all-gather inspection, and explicit
  partition/device/process/addressability reporting (portable part of M4h.3b complete on four
  forced CPU devices; not accelerator or multi-host evidence);
- real collective execution on a declared accelerator and multi-host topology, with
  global/addressable layout, operator/VJP parity, compilation, and bounded runtime evidence;
  admission requires one consistent record per process, exact partition coverage, and
  process-complete numerical/timing/memory aggregation (M4h.3b complete for one retained
  eight-process, 32-device Spot TPU v4-64 action/VJP witness; eigensolve scaling, live HBM,
  and TPU execution of the matrix-free eigensolver remain open);
- topology-bound process-local checkpoints with atomic publication, exact same-topology restore,
  and restored-state action/VJP runner integration (portable implementation and one complete
  eight-fragment physical fresh-run round trip are complete; same-resource restart, durable
  off-node transfer, and controlled preemption recovery evidence remain required before any
  recovery capability or cross-topology claim). Until durable off-node checkpoint publication is
  implemented, the operational Spot Plan B is rapid replacement-resource provisioning and a
  deterministic retry from retained local inputs, not survival of a remote-only checkpoint.

## M5 — public 3D ring-resonator heater

M5 is the next release-defining milestone. It deliberately precedes inverse design.

- define a public, non-confidential SOI ring, oxide, substrate, contacts, and heater geometry with
  semantic Gmsh physical volumes and surfaces;
- ingest one deterministic first-order tetrahedral mesh without losing material, boundary, or
  orientation identity;
- implement matrix-free 3D H1/P1 Tet4 current conduction, cell-local Joule heating, and steady
  heat conduction in JAX without forming a dense global matrix;
- lower the same geometry, mesh, material law, sources, and boundary conditions to separately
  installed Elmer current and heat solvers;
- establish element algebra, manufactured solutions, mesh refinement, electrical/thermal energy
  balance, and same-mesh full-field JAX/Elmer parity before interpreting device behavior;
- run the admitted float32 forward problem on the declared eight-process, 32-device TPU topology
  with exact partition coverage, process-complete evidence, and no silent precision or backend
  fallback; and
- sample the verified 3D temperature field onto the matching FDTDX ring scene and demonstrate a
  forward thermally shifted optical response with grid/time/PML convergence appropriate to the
  reported observable.

M5 is complete only when the 3D electrical potential, Joule source, temperature field, balances,
Elmer parity, distributed-JAX execution, and FDTDX forward handoff all have retained reproducible
evidence. A rendered hot-ring image, solver exit code, same-discretization field match without
convergence, or uncalibrated resonance shift alone does not close the milestone. Foundry-calibrated
device prediction remains a stronger later evidence tier.

M5a.1 is complete: the independent JAX affine Tet4 scalar H1 kernel now fixes signed geometry,
physical basis gradients, scalar/tensor diffusion, exact consistent mass and P1 load, cell-local
material interpolation, permutation invariance, and reverse-mode behavior against closed-form and
finite-difference references. This is the first local-algebra part of the third item above. Mesh
ingestion, global matrix-free solve, coupled physics, Elmer parity, TPU execution, and the FDTDX
ring experiment remain open.

M5a.2 is complete for the ingestion foundation: the guarded Gmsh CLI now supports an explicit 3D
request, and a versioned strict importer preserves physical volumes, the complete external surface,
positive cell orientation, outward boundary orientation, edge/face signs, and exact source
permutations. A repeated real-Gmsh box run establishes format compatibility and determinism. The
actual public ring recipe and its mesh-quality/refinement evidence remain a separate gate.

M5a.3 is complete for geometry and meshing. A source-pinned independent reconstruction of the
public 3D SOI ring, two buses, TiN heater, BOX, wafer, and cladding adds two explicitly
femx-originated top contacts for the future current solve. Eight material volumes and six boundary
groups partition the imported mesh without overlap or omission, and both contacts share exact
triangular faces with the TiN heater at every refinement level. Gmsh 4.12.1 HXT refinement grows
from 71,808 to 435,574 to 3,179,879 Tet4 cells; minimum mean-ratio quality rises from 0.137 to
0.250 to 0.323, and the maximum analytic region-volume error falls from
$5.96\times10^{-6}$ to $5.92\times10^{-7}$. A repeated coarse run is byte-identical. This closes
the first two M5 bullets as geometry/mesh contracts, not as a device simulation. The matrix-free
3D current/Joule/heat path is next.

M5b.1 is complete for the portable scalar solve substrate. The scalar owned/ghost topology,
strong-boundary RHS, fixed-capacity collective transport, and residual-defined CG now derive their
local width from either triangle P1 or Tet4 P1 cells. The existing 2D v1 layout identity is
unchanged; Tet4 uses a separate v2 identity. A 45-node, 96-Tet4 problem agrees with a dense
authority across one, two, and four forced CPU-device partitions, and its material-scale VJP agrees
with central differences. Host owner preparation now uses element-incidence passes rather than an
$O(N_{dof}N_{cell})$ scan. This is not a ring solve, Elmer comparison, or TPU result. M5b.2 must
bind the 3D current, Joule, thermal boundary, balance, manufactured-solution, and refinement
contracts before the public ring forward solve is attempted.

M5b.2 is complete for the portable one-way electrothermal physics contract. Electrical potential
is solved on a compact conductor submesh while temperature is solved on the complete Tet4 mesh;
exact parent-cell and owner identities transfer cell-local Joule density without interpolation or
all-gather. Constant volume and face loads, convection, nonzero Dirichlet data, charge balance,
electrical variational energy, Joule-power conservation, and thermal balance are independently
checked. A manufactured 3D heat problem has nodal RMS orders 2.07 and 2.14. On four forced CPU
devices, a 125-node/384-cell thermal box with a 75-node/192-cell conductor agrees across one, two,
and four partitions; the voltage VJP agrees with central difference to $1.17\times10^{-11}$
relative error and the lowered program contains no all-gather. This is not the public-ring solve,
Elmer parity, TPU execution, calibrated material, or device evidence. M5b.3 must bind the public
ring mesh, material records, and boundary values and establish a mesh-refined JAX forward result.

M5b.3a is complete for the source-pinned public-device binding and its first JAX forward witness.
The application contract admits the public Gmsh mesh, keeps electrical potential on TiN plus the
two declared aluminum contacts, transfers Joule density by exact parent-cell identity, fixes the
substrate bottom at 300 K, and applies the published 10 W m$^{-2}$ K$^{-1}$ convection coefficient
to the complete top plane. A unit-voltage solve is rescaled and then re-solved at the voltage that
produces the published 15 mA target. The coarse and medium meshes both satisfy every algebraic and
conservation gate; conductance changes by 0.192 percent, while the peak, ring-mean, and heater-mean
temperature rises change by 2.10, 1.76, and 2.36 percent.

This is a two-level single-device CPU float64 mesh-sensitivity witness for an uncalibrated public
benchmark. It is not a formal convergence order, continuum solution, Elmer comparison, TPU run,
FDTDX response, or fabricated-device prediction.

M5b.3b is complete for the fine JAX CPU float64 solve and the three-level refinement
interpretation. The 3,179,879-Tet4 level independently passes every algebraic and conservation
gate. From medium to fine, conductance changes by 0.146 percent; peak, ring-mean, and heater-mean
temperature rises change by 1.035, 0.806, and 1.186 percent. The corresponding absolute increments
are 0.761, 0.499, 0.461, and 0.509 times their coarse-to-medium increments, with the same signed
direction at both steps.

This is a three-level mesh-sensitivity witness, not a fitted convergence order or
continuum-extrapolated solution. It remains single-device CPU float64 evidence for constant,
uncalibrated properties. Fine-mesh Elmer parity, physical distributed execution, FDTDX response,
foundry accuracy, and fabricated-device validation do not follow.

M5b.4 is complete for a locked same-mesh 3D Elmer oracle and the public coarse-ring comparison.
The adapter emits the exact imported Tet4 mesh as native Elmer 504/303 elements, keeps potential
on the TiN-plus-aluminum conductor nodes, and reads temperature on every node. At the same 15 mA
target-current voltage, the 12,761-node, 71,808-Tet4 comparison has maximum potential and
temperature differences of $2.31\times10^{-10}$ V and $1.90\times10^{-7}$ K. The relative L2
temperature-rise difference is $2.30\times10^{-10}$. A separate small distinct-space case also
matches both fields and the JAX power/balance reconstruction. The retained
[figure, open fields, and provenance](assets/readme/3d_ring_heater_reference/README.md) present the
public witness without replacing its numerical gates.

This establishes algebraic parity for one coarse, constant-property, uncalibrated 3D benchmark.
It is not fine-mesh Elmer parity, continuum convergence, physical distributed execution, FDTDX
ring response, foundry accuracy, or measurement agreement. Physical distributed JAX execution
and the FDTDX response remain later independent M5 gates.

M5b.5 remains open for thermal applicability. The immutable 15 mA source-reproduction case remains
the source benchmark. A separately retained 5 mA bundle now reruns native JAX and locked external
Elmer at 0.229573 V and establishes direct same-mesh parity at that lower-temperature point; the
current itself is selected by the documented linear projection. An initial one-device CPU float64
envelope now crosses three lateral extents
and three modeled substrate depths and adds an ideal-isothermal sidewall bound. It shows that width
and depth interact and does not support assigning the source-envelope temperature to the 0.5 um
substrate alone. It is not domain convergence or device calibration. A device-representative model
must still bind a target wafer/die/package configuration; test a justified backside impedance;
introduce process-appropriate resistance and temperature-dependent properties; and compare heater
and waveguide K/mW with measurements. Neither the direct 5 mA rerun nor its agreement with the
linear projection closes those applicability gates. See the
[thermal-scope and operating-point note](physics/PUBLIC_RING_HEATER_THERMAL_SCOPE.md).

## M6 — adjoint inverse design and optimization

M6 begins only after the M5 forward pipeline is stable. It adds the residual-defined 3D
electrothermal adjoint, the complete temperature-to-FDTDX objective pullback, finite-difference
spot checks, a fabrication-aware parameterization, durable optimization checkpoints, and
independent final-design forward validation. Gradient correctness, optimization progress, and
robust device performance remain separate claims. No inverse-design result is required to close
M5 or make the first 3D heater release.
