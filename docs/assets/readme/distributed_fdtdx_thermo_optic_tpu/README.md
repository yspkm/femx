# Physical TPU distributed electrothermal-to-FDTDX witness

This bundle records one process-complete physical witness for femx's bounded 2D distributed
current/Joule/heat-to-FDTDX derivative graph. It used eight JAX processes and 32 TPU v4 devices at
femx revision `e5860a99e4bb3c49f6665f8c0f8e38d4d533fd3c`, with FDTDX 0.6.2 at revision
`0c05c4784b2be83b42d9b46ab089265981ba157f`.

The [aggregate](evidence.json) was admitted from all eight original process-local records. Its
canonical logical SHA-256 is
`1dd42ac8f51bff53a17814e7f923581f08fd2dd2f1aec11604223b33354e8654`. It retains source,
configuration, input, layout, numerical, StableHLO, timing, compiler-estimate, and raw-record
hashes without publishing the private profile, run ID, worker mapping, hostnames, cloud project,
zone, or resource name. Raw logs, inputs, and per-process StableHLO remain outside Git.

All 32 partitions were addressable exactly once. The 289-node, 512-triangle electrothermal problem
converged in 14 coupled iterations. Relative differences from the immutable controller authority
were `2.23e-6` for potential and `6.37e-9` for temperature; material and transfer differences were
`4.67e-8` and `4.82e-8`. The coupled-adjoint backward error was `1.70e-4`. Native and explicit
gradients agreed to at most `1.01e-7`, and both applied-voltage central differences agreed to
`1.83e-3`, below the committed `2e-2` bound. The sampled-cell pullback has a zero direct potential
cotangent by construction; its nonzero voltage gradient passes through Joule coupling in the
residual adjoint.

All four executable identities contain `all_to_all`, `collective_permute`, and `all_reduce`, with
no all-gather and no f64. The largest JAX compiler estimate was 513,943,040 bytes per device, or
1.50% of the declared HBM capacity. This is a compiler estimate, not a live-HBM measurement.

The executed revision retained HLO authorities under process-local output paths while the generic
synchronizer expected controller-visible process-zero copies. After the remote-index checksums had
verified all original files, exact process-zero copies and a recovery receipt were created locally;
the second full synchronization verified 66 remote files totaling 718,494,561 bytes and completed
with no required artifact missing. The aggregate above was built from the original process-local
records. The runner now publishes those controller-visible copies directly for future runs.

This closes one bounded M2e.7b physical execution gate. It is not 3D FEM, ring-heater convergence,
an S-parameter, a scaling result, live-HBM evidence, a measured-device or foundry claim,
preemption-recovery evidence, or an inverse-design result. The executable admission rules are in
`src/femx/validation/tpu_distributed_fdtdx_thermo_optic_evidence.py`; see
[ADR 0051](../../../adr/0051-process-complete-distributed-fdtdx-tpu-evidence.md) for the claim
boundary.
