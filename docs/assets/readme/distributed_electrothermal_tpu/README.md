# Physical TPU distributed-electrothermal witness

This bundle records one process-complete physical witness for femx's bounded 2D coupled
current-to-Joule-to-heat JAX path. It used eight JAX processes and 32 TPU v4 devices at femx
revision `6c344613f1bfacaaf39ebfa0751e0b1e85581b5e`.

The worker runtime used JAX/JAXLIB 0.11.0 with x64 disabled. The repository's portable dependency
baseline at that revision is JAX 0.10.1, so this evidence is not labelled as a locked-0.10.1 TPU
reproduction; the exact physical runtime identity remains in the aggregate.

The [aggregate](evidence.json) was admitted from all eight raw process records. Its canonical
logical SHA-256 is
`ba48ad3d6d6334ecae01db1effa63989d96118f6102729c3a91e10a4ae424b7f`. It retains source,
configuration, plan, runtime, array-layout, numerical, StableHLO, timing, compiler-estimate, and
raw-record hashes without publishing the private profile, run ID, worker mapping, machine address,
hostname, cloud project, zone, or TPU resource name. Raw logs, input arrays, per-process StableHLO,
and executable artifacts remain outside Git.

All 32 FEM partitions were addressable exactly once. The 289-node, 512-triangle problem completed
14 self-consistent iterations with finite forward, explicit-adjoint, and native-reverse results.
The coupled-adjoint backward error was `2.12e-4`; the largest of the current, thermal, and feedback
gradient differences from the dense float64 authority was `2.20e-3`, below the committed `5e-3`
bound. All three retained executable identities contain no all-gather.

This is a same-discretization float32 correctness witness for one bounded 2D problem. It is not a
fresh Elmer execution, 3D production FEM, a scaling result, live HBM measurement, FDTDX
composition, calibrated foundry prediction, measured-device validation, or preemption-recovery
test. The executable admission rules are in
`src/femx/validation/tpu_distributed_electrothermal_evidence.py`; see
[ADR 0048](../../../adr/0048-process-complete-tpu-distributed-electrothermal-evidence.md) for the
claim boundary.
