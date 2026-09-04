# femx

[![CI](https://github.com/yspkm/femx/actions/workflows/ci.yml/badge.svg)](https://github.com/yspkm/femx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11--3.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/yspkm/femx/blob/main/LICENSE)

> **Verification-first, differentiable electrothermal FEM for silicon photonics.**

`femx` connects electrical conduction, Joule heating, thermal transport, thermo-optic material
updates, waveguide modes, and FDTDX optical objectives through explicit numerical contracts.

- **JAX** provides native differentiable finite-element operators and implicit adjoints.
- **Elmer** provides an independently executed open-source FEM reference.
- **FDTDX** consumes explicit material, mode, source, detector, and result contracts.

> **Research preview:** APIs, evidence schemas, and experimental distributed interfaces may change
> before the first stable release.

## 3D silicon-photonics ring heater

![Physical-scale 3D ring-heater material stack and direct 5 mA JAX-Elmer thermal fields](https://raw.githubusercontent.com/yspkm/femx/main/docs/assets/readme/3d_ring_heater_5ma_reference/figure.png)

Native JAX and locked external Elmer independently solve the same coarse Gmsh Tet4 problem at
5 mA. No displayed temperature or parity field is rescaled from the retained 15 mA result.

| Direct 5 mA reference | Value |
|---|---:|
| Mesh | 12,761 nodes; 71,808 Tet4 cells |
| Voltage | 0.229573 V |
| Joule power | 1.14786 mW |
| Peak temperature rise | 18.2608 K |
| Maximum JAX-Elmer temperature difference | $2.12\times10^{-8}\ \mathrm{K}$ |
| Relative $L_2$ temperature-rise difference | $2.31\times10^{-10}$ |
| Maximum conductor-potential difference | $7.72\times10^{-11}\ \mathrm{V}$ |

This 3D same-discretization parity result is solver evidence for an uncalibrated,
constant-property benchmark. The
modeled thermal domain is 20 um by 20 um, contains only 0.5 um of silicon below the 2 um BOX, fixes
the bottom at 300 K, uses adiabatic sides, and applies top convection to 300 K with
$h=10\ \mathrm{W\,m^{-2}\,K^{-1}}$. In the retained coarse result, 99.997 percent of the heat
exits through the fixed bottom. The result is not a fabricated-device prediction.

The original 15 mA source-reproduction reaches a 164.348 K peak rise and is retained for source
traceability, not as a recommended operating point. See the
[direct 5 mA method and open fields](https://github.com/yspkm/femx/blob/main/docs/assets/readme/3d_ring_heater_5ma_reference/README.md),
[thermal-scope note](https://github.com/yspkm/femx/blob/main/docs/physics/PUBLIC_RING_HEATER_THERMAL_SCOPE.md),
[original 15 mA bundle](https://github.com/yspkm/femx/blob/main/docs/assets/readme/3d_ring_heater_reference/README.md),
and [earlier 2D adjoint reference](https://github.com/yspkm/femx/blob/main/docs/assets/readme/siph_thermal_reference/README.md).

### Thermal-envelope sensitivity

![Ring-heater domain-width and modeled-substrate-depth sensitivity](https://raw.githubusercontent.com/yspkm/femx/main/docs/assets/readme/ring_heater_thermal_sensitivity/figure.png)

A separate CPU float64 study crosses 20, 40, and 80 um square domains with 0.5, 5, and 50 um
modeled silicon depths while holding material values and the mesh-size policy fixed. The widest,
deepest case differs from the source envelope by -1.41 percent in peak K/mW and -3.23 percent in
ring-mean K/mW. Width and depth interact; this bounded study is not formal domain convergence or
device calibration. The [complete case evidence](https://github.com/yspkm/femx/blob/main/docs/assets/readme/ring_heater_thermal_sensitivity/README.md)
is public.

## What femx provides

- Solver-neutral problem, mesh, material, artifact, and validation contracts.
- JAX-native steady heat, current, Joule-heating, and electrothermal paths.
- Independent same-mesh checks against separately installed Elmer.
- Residual-defined differentiation with finite-difference or independent-adjoint checks.
- Explicit FEM-to-Yee mode transfer and FDTDX interoperability.
- Experimental distributed JAX paths with
  [bounded physical TPU evidence](https://github.com/yspkm/femx/blob/main/docs/assets/readme/distributed_fdtdx_thermo_optic_tpu/evidence.json).

Process success, numerical convergence, and scientific validation are reported separately. A test
or completed executable does not by itself establish a physical claim.

## Current public scope

| Capability | Status |
|---|---|
| 2D steady heat and current FEM | Available and cross-validated |
| 2D electrothermal coupling and implicit adjoint | Available for the documented subset |
| 3D Tet4 current, Joule heating, and steady heat | Available for the constant-property forward subset |
| 3D ring-heater JAX-Elmer field parity | Available on the public coarse mesh |
| Three-level 3D ring-heater mesh sensitivity | Available through 3,179,879 Tet4 cells |
| 2D waveguide port modes and documented eigen-adjoints | Available for the lossless PEC subset |
| FEM-to-Yee transfer and FDTDX consumers | Available for the documented contracts |
| Distributed scalar and electrothermal JAX paths | Experimental; bounded TPU evidence retained |
| 3D ring-heater FDTDX resonance response | In development |

Transient thermal analysis, foundry-calibrated prediction, and a continuum-converged ring-heater
value are not currently claimed.

## Installation

`femx` supports Python 3.11 to 3.14. Install the published development release from PyPI with:

```bash
pip install "femx==0.1.0.dev0"
femx doctor
```

For an editable source checkout, install [uv](https://docs.astral.sh/uv/) and run:

```bash
git clone https://github.com/yspkm/femx.git
cd femx
uv sync --locked --extra jax --extra artifacts --extra meshing
uv run femx doctor
```

The optional `jax`, `artifacts`, and `meshing` extras may be selected for workflows that need them.
Gmsh, Elmer, and FDTDX remain separately installed external tools.

Run the portable public CI selection with:

```bash
uv run pytest -m "unit or architecture"
```

Elmer comparison requires JAX float64. Set `JAX_ENABLE_X64=1` before importing JAX.

## Runtime model

```python
prepared = femx.prepare(problem, backend, request=prepare_request)
solution = femx.solve(prepared, backend, request=solve_request)
```

`Problem` contains solver-neutral physics. `PreparedProblem` binds one exact backend payload, and
`Solution` records fields, observables, convergence, and validation state. femx does not silently
change the backend, precision, mesh, element family, or scalar type.

## Documentation

- [Roadmap](https://github.com/yspkm/femx/blob/main/docs/ROADMAP.md)
- [Materials and provenance](https://github.com/yspkm/femx/blob/main/docs/MATERIALS.md)
- [FDTDX interoperability](https://github.com/yspkm/femx/blob/main/docs/INTEROPERABILITY.md)
- [Tet4 scalar H1 formulation](https://github.com/yspkm/femx/blob/main/docs/physics/TETRAHEDRON_H1.md)
- [Steady heat formulation](https://github.com/yspkm/femx/blob/main/docs/physics/STEADY_HEAT_H1.md)
- [Steady current formulation](https://github.com/yspkm/femx/blob/main/docs/physics/STEADY_CURRENT_H1.md)
- [Electrothermal coupling](https://github.com/yspkm/femx/blob/main/docs/physics/ELECTROTHERMAL_COUPLING.md)
- [Port eigenmode formulation](https://github.com/yspkm/femx/blob/main/docs/physics/PORT_EIGENMODE.md)
- [Contributing](https://github.com/yspkm/femx/blob/main/CONTRIBUTING.md)

For reproducible work, record the exact femx version or source commit together with external solver
revisions, mesh and material identities, numerical policy, and the relevant evidence artifact.

## License

`femx` is distributed under the [MIT License](https://github.com/yspkm/femx/blob/main/LICENSE).
Elmer, FDTDX, JAX, and Gmsh are separate projects with their own licenses; femx does not vendor
their implementations.

## Acknowledgements

We gratefully acknowledge Google's TPU Research Cloud (TRC) program for providing access to Cloud
TPU resources that supported the development and distributed validation of `femx`.
