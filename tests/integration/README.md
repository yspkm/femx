# Integration tests

Integration tests cross an implementation boundary but do not by themselves establish scientific
accuracy. External Elmer tests must use `requires_elmer`; optional JAX tests use `requires_jax`.
Tests skip when their declared prerequisite is absent and must not silently select another backend.
Real Elmer execution additionally requires `FEMX_RUN_ELMER_TESTS=1`; use
`FEMX_ELMER_EXECUTABLE` for an absolute executable override. This opt-in does not bypass the runtime
`ExecutionPolicy` gate.

Real Gmsh execution similarly requires `FEMX_RUN_GMSH_TESTS=1`; use `FEMX_GMSH_EXECUTABLE` for an
absolute executable override. The test repeats generation under one recorded executable identity,
imports both outputs, and checks the canonical mesh passed to JAX and Elmer lowering. It is process
and mesh-handoff evidence, not Maxwell or optical-solution agreement.

The ElmerGUI material-library integration test is read-only and does not run ElmerSolver. It checks
the optional locked sibling checkout and skips when that source tree is absent.
