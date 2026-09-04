# Scientific tests

This directory is the evidence boundary for numerical claims. A test belongs here when it checks
an analytic/manufactured solution, convergence order, a conservation law, a cross-solver physical
quantity, or an adjoint gradient against finite differences.

Every test must use `pytest.mark.scientific`. A solver exit code, output-file existence, or embedded
reference norm belongs in integration/regression testing and is not sufficient scientific evidence.

Accelerator and multi-host claims must run on the stated physical topology and add
`requires_accelerator` or `multihost` respectively.
