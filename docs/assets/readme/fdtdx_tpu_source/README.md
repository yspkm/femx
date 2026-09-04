# Physical TPU FDTDX source witness

This bundle records one process-complete infrastructure witness for femx's locked FDTDX custom
mode source. It used four JAX processes and 16 Spot TPU v5e devices at femx revision
`6c21321006302a81972efc29c7d3128672cf460e`.

The [aggregate](evidence.json) was admitted from all four process records. Its canonical logical
SHA-256 is `4bdd3e2642b8e0fb86340a0b2f9f87df3f156912698853d4e131b2abf432c189`.
It retains source, configuration, runtime, FDTDX, shard, numerical, timing, memory-estimate, and
raw-record hashes without publishing machine addresses, a cloud project, a zone, or a TPU resource
name. Raw logs, per-process StableHLO, and executable artifacts are retained outside Git.

The source is an analytic one-watt homogeneous positive-z port, not an Elmer or JAX waveguide
mode. The result verifies exact process-set admission, source-plane sharding, public FDTDX source
injection, and a finite nonzero time advance on the recorded physical topology. It does not verify
Elmer parity, silicon-waveguide accuracy, convergence, S-parameters, scaling, live HBM, an adjoint,
or Spot-preemption recovery.

The executable admission rules are in
`src/femx/validation/tpu_fdtdx_mode_source_evidence.py`; the runner and aggregator are
`scripts/run_tpu_fdtdx_mode_source_evidence.py` and
`scripts/aggregate_tpu_fdtdx_mode_source_evidence.py`. See
[ADR 0038](../../../adr/0038-process-complete-tpu-fdtdx-source-evidence.md) for the claim boundary.
