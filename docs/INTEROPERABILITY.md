# FDTDX interoperability

Compatibility is defined at three levels:

1. semantic: SI units, axes, tensor ordering, scalar type, frequency, and direction;
2. workflow: static problem, parameters, preparation, solve, observation, and gradient capability;
3. artifact: fields, modes, material perturbations, observables, hashes, and provenance.

It does not mean sharing FDTDX's Yee-grid state containers or presenting FEM coefficients as dense
FDTD arrays.

## Mode boundary

`femx.mode/v1` requires:

- positive frequency and complex effective index/propagation constant;
- complex Cartesian E and H at the exact component-specific Yee positions;
- three physical Yee edge-coordinate axes and one cell on the propagation axis;
- propagation axis and sign;
- deterministic phase reference;
- forward-power normalization and target power;
- FDTDX-compatible `eta0_H` at the mode boundary;
- solver name/version, source revision, config hash, and mesh hash;
- a transfer hash, pre/post-correction signed power, correction scale, and target FDTDX identity.

FEM H(curl) coefficients cannot be copied into a mode bundle as Cartesian samples. A transfer
operator must evaluate/orient the edge-element field on the target grid and report power
conservation. Conversion between physical H and `eta0_H` is explicit.

## Differentiation boundary

Mode data imported through an external eigensolver is not assumed differentiable. The adapter must
mark or stop that path explicitly. The native JAX port solver now exposes a guarded simple-mode
adjoint after finite-difference checks, fixed-anchor phase handling, and fail-closed eigenvalue-gap
checks. It also exposes basis-invariant propagation aggregates and an edge-mass projector for an
isolated repeated or close cluster through a fixed Riesz contour. Individual cluster vectors and
cluster-to-FDTDX field injection remain unsupported.

M3d first provided the Elmer port-mode precursor: complete complex eigenspectrum, projected nodal E,
Elmer edge power, and deterministic phase. M4d later validated and canonicalized Elmer's raw mixed
coefficients; M4e reconstructed physical H and applied one phase/power scale to the full mixed
field. The original projected field remains inspection data and is never relabeled as a Yee field.
See [the port-eigenmode contract](physics/PORT_EIGENMODE.md).

M4f adds the first exact mixed-FEM-to-Yee operator for the lossless positive-z 2D subset. Static
preparation locates each of Ex/Ey/Ez/Hx/Hy/Hz at the corresponding offsets from FDTDX's locked
`calculate_spatial_offsets_yee`; it rejects extrapolation and ambiguous triangle ownership unless
the caller explicitly selects a deterministic tie policy. Source mesh, canonical edge topology,
target grid, selected cells, barycentric weights, and policy are hashed.

The JAX evaluation reconstructs H from the Maxwell curl equation, checks signed physical power on
the target grid, applies one positive scale to both fields, and converts H to `eta0_H`. The bundle
retains the uncorrected power and scale: exact post-correction one-watt power is not presented as an
interpolation-convergence result. The locked same-mesh witness produced a roughly 9.86 percent
pre-correction error on a 33-by-25-by-1 plane, while Elmer-derived and JAX-derived transferred E and
H agreed to about `1.93e-14` and `3.42e-14`. The actual locked FDTDX
`CustomModeOverlapDetector` consumed the bundle unchanged on CPU. A five-level exact-offset study
then reduced the absolute raw-power error monotonically from about 50.1 percent at 16-by-12 to
0.451 percent at 256-by-192, with a minimum observed order of about 1.24.

Both backend bundles are now written and reconstructed through the checksummed
`femx.mode.hdf5/v1` boundary before their final comparison, and the locked FDTDX integration
reloads the same artifact type before detector placement. The optional codec preserves exact array
dtype, canonical metadata, an independent logical content digest, and the complete-file
`ArtifactRef`; serialization remains outside the differentiation path. FDTDX currently has no
equivalent public custom-mode source callback in its upstream baseline.

The locked user FDTDX fork now provides `CustomModePlaneSource`. The femx
`femx.fdtdx.mode_source/v1` adapter supplies the bundle only through that public API and binds the
complete bundle, exact source-plane grid and medium, dtype, source policy, and FDTDX source
fingerprint. The first subset is positive-z, lossless, isotropic, nonmagnetic, and
host-addressable. It preserves the supplied E and `eta0_H` without another normalization or silent
cast, and it fails closed if the placed FDTDX source no longer matches the contract.

The default imported mode fields are a stop-gradient setup boundary. On the locked CPU runtime,
both FDTDX checkpointed reverse mode and its reversible custom VJP agree with central differences
for a material segment downstream of the custom source. This verifies downstream-material
gradient continuity, not an eigensolve or transfer derivative.

TPU execution uses a separate explicit scalar boundary. The canonical HDF5 mode remains
float64/complex128 when produced at that precision; `lower_mode_source_inputs_for_tpu()` derives a
read-only float32/complex64 runtime copy, reproduces FDTDX's cast-then-invert material arithmetic,
recomputes signed power on the cast Yee grid, and records coordinate, optical-scalar, field,
material, and power errors in a hash-bound report. Coordinate collapse or an error outside the
admitted bounds fails before FDTDX placement. The nonmagnetic permeability remains FDTDX's host
scalar-one sentinel, not a fictitious float32 material array. This portable
lowering is not itself accelerator or multi-host evidence. [ADR 0036](adr/0036-explicit-tpu-mode-precision.md)
binds the boundary.

FDTDX's distributed runtime uses a second explicit boundary. The imported source requires the same
global one-axis `NamedSharding` as its source-plane material. Each controller validates only its
addressable material shards against the complete hash-bound host snapshot, then JAX materializes E
and `eta0_H` directly on the global sharding. After source application, both Yee time-offset arrays
are rebound to that layout. The FDTD loop must be outer-jitted with arrays, objects, and config as
explicit arguments; a bound source container is not captured as a process-local constant. A forced
four-device CPU run passes placement, binding, compilation, and finite nonzero time advance, but it
is not TPU or multi-host evidence. [ADR 0037](adr/0037-distributed-fdtdx-mode-source.md) records the
contract and remaining physical gate.

One retained physical witness closes that infrastructure gate for a bounded homogeneous source.
At femx revision `6c21321006302a81972efc29c7d3128672cf460e`, four JAX processes used 16 Spot
TPU v5e devices with JAX/JAXLIB 0.10.1 and the locked FDTDX 0.6.2 source revision
`81a58da9cde4a4ff822f835b63597c0d0d8ba978`. Sixteen addressable x ranges cover the complex64
source shape 3-by-32-by-8-by-1 exactly once. The 32-by-8-by-20 float32 scene completed all 66
steps with finite nonzero E/H and nonzero downstream E on every process; all four retained
StableHLO records report zero all-gather occurrences. The process-set admission and raw-record
hashes are retained in the [public aggregate](assets/readme/fdtdx_tpu_source/evidence.json).

This result verifies source sharding, public-source injection, and time advance on the recorded
physical topology. It does not verify an Elmer-derived or JAX-derived silicon mode on TPU,
transmission convergence, S-parameters, scaling, live HBM, source-profile adjoints, or Spot
recovery. [ADR 0038](adr/0038-process-complete-tpu-fdtdx-source-evidence.md) binds that limit.

A second retained process set closes the next bounded physical boundary. Independently generated
Elmer and JAX bundles from the same 268-node/500-triangle lossless PEC waveguide were lowered to
float32/complex64 and passed through one shared 64-by-52-by-36 Si/SiO2 FDTDX scene on four JAX
processes and 16 Spot TPU v5e devices. Both sources completed 316 steps; their 16 shards covered
the complete 64-cell source axis exactly once, and all four StableHLO records contain no
all-gather. Source E and `eta0_H` relative L2 differences are `3.04e-13` and `2.39e-11`; the
downstream six-component complex phasor difference is `9.91e-8`, below the admitted `2e-5` bound.

The [public evidence projection](assets/readme/fdtdx_tpu_waveguide_source/evidence.json) binds the
exact process-set digest while removing private infrastructure identifiers. This verifies one
fixed Elmer/JAX HDF5-to-sharded-source-to-downstream-field path. It is not spatial or temporal
convergence, absolute transmission, S-parameters, eigen-adjoint validation, performance scaling,
fabricated-device agreement, live HBM measurement, or Spot recovery.
[ADR 0039](adr/0039-process-complete-tpu-waveguide-source-parity.md) binds this limit.

Separately, the native JAX simple-mode state now composes directly with
`sample_port_mode_to_yee` before serialization. On the locked 268-node Si/SiO2 mesh, the beta
adjoint agrees with an Elmer material central difference to about `5.5e-9` relative, and a scalar
objective on the exact Yee E samples agrees with its JAX central difference to about `8.7e-9`.
This closes the differentiable FEM-to-Yee boundary in memory. It does not cross the current
host-addressed `ModeBundle` or HDF5 boundary; those remain deliberately static.

The opt-in `femx.fdtdx.dynamic_mode_source/v1` path crosses the remaining in-memory boundary. It
binds the baseline static-source contract, ordered FEM parameters and units, exact transfer hash,
target power, fixed source-plane medium, and the locked FDTDX identity. A parameterized JAX mode is
sampled at the exact Yee offsets and supplied through FDTDX's public `with_mode_profile` method.
Profile and source-time-offset cotangents are preserved, while the source-plane material multiplier
is stopped. Only checkpointed reverse mode is supported; reversible execution fails closed because
its custom VJP does not carry a source-object cotangent.

On the locked 268-node Si/SiO2 mesh and a 14-by-10-by-18 CPU scene, the complete source-profile
gradient through the six-component Yee fields, checkpointed time advance, and downstream phasor
objective agrees with a central difference to about `1.10e-7` relative error. JAX/JAXLIB 0.10.1,
Gmsh 4.12.1, float64/complex128, and FDTDX revision `81a58da9cde4a4ff822f835b63597c0d0d8ba978`
are part of the witness.

The three-dimensional FDTD material arrays are fixed at their baseline in this test. It therefore
isolates the source-profile derivative and does not claim the total derivative of a propagation
scene that changes with the same material parameter.

The locked same-mesh Elmer and JAX waveguide modes are also each reconstructed from independent
HDF5 artifacts and injected into an identical 70-by-52-by-36 rectilinear CPU scene. The source
plane is an explicit 10-by-4-cell silicon core in silica, with transverse PEC and longitudinal PML.
At a six-component complex phasor plane 720 nm downstream, the two runs differ by about
`1.83e-14` in relative L2 norm. Their source E and `eta0_H` arrays differ by about `1.99e-14` and
`2.83e-14`. The exact source-plane medium is snapshotted before source construction and checked
again after placement.

The source grid's pre-correction power error is about 5.57 percent. Final one-watt normalization
does not turn this into transmission convergence. Absolute transmission, S-parameters, time/grid
and PML convergence, total changing-scene derivatives, reversible source-profile derivatives, and
production-scale physical multi-host execution remain open. The retained TPU witness above closes
only fixed-scene source/downstream parity.

[ADR 0024](adr/0024-exact-yee-port-mode-transfer.md) binds the transfer boundary,
[ADR 0025](adr/0025-durable-mode-bundle-hdf5.md) binds the durable artifact, and
[ADR 0026](adr/0026-locked-fdtdx-custom-mode-source.md) binds source injection.
[ADR 0027](adr/0027-simple-port-eigen-adjoint.md) binds the simple-mode derivative, and
[ADR 0028](adr/0028-dynamic-fdtdx-mode-source.md) binds the guarded checkpointed FDTD path.
[ADR 0029](adr/0029-invariant-port-cluster-adjoint.md) binds the invariant cluster derivative.

The validated thermal precursor uses a different, narrower boundary: a canonical active-parameter
vector maps to the nodal FEM temperature field and supplies an explicit state VJP. Thermo-optic and
FDTDX objectives may compose around that map, but no field-transfer derivative is implied until the
transfer operator has its own gradient and conservation evidence.

The validated electrothermal precursor has two levels. M2c maps separate electrical and thermal
active vectors through an exact same-mesh `L2/P0` transfer and chained adjoints. M2d adds a feedback
parameter vector, cell-local P1 temperature-dependent conductivity/Joule integration, and one
implicit VJP of the coupled current/heat residual. Both preserve explicit `W/m^3`, cell ordering,
and integrated-power evidence. Neither implies a thermo-optic material map or FDTDX derivative.
Those require a versioned temperature-to-index law and FEM/FDTD sampling transfer with their own
conservation and finite-difference evidence.

## Thermo-optic material boundary

`femx.fdtdx.thermo_optic_parameter/v1` is the first implemented temperature-to-FDTDX boundary. It
uses a wavelength-specific, real isotropic law

$$
n(T)=n_{ref}+(dn/dT)(T-T_{ref}),\qquad \epsilon_r(T)=n(T)^2.
$$

The law is hashed and does not imply dispersion or calibrated silicon data. A versioned dense
reference operator evaluates the 2D P1 FEM field at the actual FDTDX cell centers, with explicit
FEM-to-FDTDX plane axes and explicit extrusion on the remaining axis. All target points must be
covered; there is no nearest-cell fallback or extrapolation. Source mesh, target axes, selected
triangles, barycentric weights, and operator identity are hashed.

FDTDX receives

$$
s=(\epsilon_r-\epsilon_{low})/(\epsilon_{high}-\epsilon_{low})
$$

through its public `apply_params` path. The v1 adapter requires a raw continuous, non-etching device
with one design voxel per FDTD cell and exactly two lossless isotropic bracket materials. The
bracket and float32/float64 cast are explicit. Out-of-bracket values become invalid NaNs rather than
being clipped. Package version, caller-attested source revision/digest, physical-law hash,
coordinate hash, and transfer hash must all match.

The v1 differentiable runtime also rejects any electromagnetic source whose placed grid slice
overlaps the active thermo-optic `Device`. FDTDX samples source-local material state during
`apply_params`, while the validated Maxwell derivative boundary is the material array. A source
inside the changing region therefore adds a parameter-dependent source-amplitude path that is not
part of the demonstrated array VJP. Static silicon access waveguides place the source outside the
heated segment instead of silently accepting that missing derivative.

Portable evidence checks affine sampling, the analytic sampling transpose, JAX reverse mode, and
central differences. The locked FDTDX test additionally places the real device and composes
`self-consistent electrothermal temperature -> sampling -> thermo-optic law -> apply_params ->
inverse permittivity`; its gradient reaches the electrical design parameters through the coupled
adjoint. This is a material-boundary result, not yet `run_fdtd`, a mode/source/detector objective,
or an end-to-end photonic optimization result. [ADR 0014](adr/0014-fdtdx-thermo-optic-material-boundary.md)
records the exact decision and scope.

M3b adds one real `run_fdtd` precursor on the locked source. A point source drives a short silicon
core in silica, an interior raw `Device` represents the heated segment, and a reduced complex
phasor detector defines a scalar optical objective at 1.55 micrometres. The electrical design
gradient crosses the M2d coupled residual adjoint, P1 sampling, thermo-optic law, public
`apply_params`, Maxwell time integration, and detector reduction. Both FDTDX checkpointed reverse
mode and reversible custom-VJP agree with independent central differences without relaxing the
threshold. The grid and periodic-boundary scene are deliberately small integration evidence; they
do not establish mesh/time convergence, port normalization, transmission, S-parameters, or a
publishable phase-shifter result. [ADR 0015](adr/0015-fdtdx-optical-objective-precursor.md) records
the evidence boundary and the source-overlap rule.

M2e.7a composes the distributed M2e.6 state with the same optical boundary without reconstructing
the global FEM temperature. The host-owned plan binds every canonical P1 target sample to its
source-cell owner and its destination-local FDTDX x index. Each source shard evaluates its local
barycentric samples, one `all_to_all` exchanges them, and each destination shard scatters exactly
one value per local voxel before applying the thermo-optic law. Source mesh, distributed FEM
layout, canonical sampler, target coordinates, and the distributed routing operator all have
separate hashes.

The locked FDTDX revision `0c05c4784b2be83b42d9b46ab089265981ba157f` captures concrete material
array shardings before JAX tracing and accepts them explicitly in `apply_params`. This keeps
repeated active-device updates on the original x-sharded material layout during reverse mode. The
four-forced-CPU executable witness then runs the distributed electrothermal implicit VJP,
destination-local thermo-optic update, checkpointed Maxwell time advance, and phasor objective in
one differentiated graph. It is a portable layout and derivative gate, not physical TPU,
multi-host, convergence, port normalization, S-parameter, 3D FEM, calibrated-material, or device
evidence. [ADR 0049](adr/0049-shard-preserving-distributed-thermo-optic-objective.md) records the
contract and the physical gate that remained at that stage.

M2e.7b has a separate physical runner and process-set schema. It reconstructs the ADR 0050
controller artifact on an exact eight-process, 32-device TPU v4 topology and records both sides of
the distributed boundary: source-owned FEM/transfer arrays and the actual destination-sharded
FDTDX thermo-optic parameter and inverse permittivity. The reference phasor, forward objective,
explicit residual VJP, and native reverse mode must all retain the required collectives without an
all-gather or f64 operation. Material parity is reduced from process-local inverse-permittivity
shards; the complete material array is never gathered for comparison.

A retained eight-process, 32-device TPU v4 aggregate now passes and is pinned at logical SHA-256
`1dd42ac8f51bff53a17814e7f923581f08fd2dd2f1aec11604223b33354e8654`. This closes the bounded
2D physical M2e.7b execution gate. It still cannot be cited as 3D ring-heater convergence,
S-parameter, scaling, live-HBM, optimization, measured-device, foundry, or preemption-recovery
evidence. See the [public aggregate](assets/readme/distributed_fdtdx_thermo_optic_tpu/evidence.json),
its [scope note](assets/readme/distributed_fdtdx_thermo_optic_tpu/README.md), and [ADR
0051](adr/0051-process-complete-distributed-fdtdx-tpu-evidence.md).

## Version policy

Compatibility is tested against exact FDTDX revisions and recorded in release notes. DeepWiki and
README feature descriptions are navigation aids, not executable compatibility evidence.
