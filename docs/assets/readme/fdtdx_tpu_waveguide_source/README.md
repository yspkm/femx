# Physical TPU Elmer/JAX waveguide-source witness

This bundle records one process-complete physical witness for femx's Elmer/JAX-to-FDTDX
waveguide path. It used four JAX processes and 16 Spot TPU v5e devices at femx revision
`a89849d7ece46927456ac13fb81936c2e7cd8e05` with the locked FDTDX revision
`81a58da9cde4a4ff822f835b63597c0d0d8ba978`.

The exact process set was admitted from all four records. Its canonical logical SHA-256 is
`e909db1632769775ee15c3927e48e636568746ef8f878140456a78a188e1cf56`. The tracked
[public projection](evidence.json) retains source, artifact, runtime, shard, numerical, timing,
compiler-estimate, and raw-record hashes while removing the private profile, run ID, machine
addresses, cloud project, zone, resource name, and hostnames. Raw logs, per-process StableHLO,
input HDF5 files, and executable artifacts remain outside Git.

Elmer and JAX independently solved the same 268-node, 500-triangle lossless PEC waveguide problem.
Their HDF5 bundles were lowered explicitly to float32/complex64 and run through one shared
64-by-52-by-36 Si/SiO2 FDTDX scene. All 316 steps completed. Source E and `eta0_H` differed by
`3.04e-13` and `2.39e-11` relative L2; the downstream six-component complex phasor differed by
`9.91e-8`. Sixteen shards covered the complete 64-cell source axis exactly once for both sources,
and every retained StableHLO record reported zero all-gather occurrences.

This is one bounded same-mesh, same-scene parity witness. It is not spatial or temporal
convergence, absolute transmission, an S-parameter, eigen-adjoint validation, performance scaling,
fabricated-device agreement, live HBM measurement, or observed Spot-preemption recovery. The
executable admission rules are in
`src/femx/validation/tpu_fdtdx_waveguide_source_evidence.py`; see
[ADR 0039](../../../adr/0039-process-complete-tpu-waveguide-source-parity.md) for the claim
boundary.
