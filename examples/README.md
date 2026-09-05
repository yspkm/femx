# femx examples

Each example is a narrow, reproducible experiment. Passing its numerical checks does not by
itself establish full-device accuracy, Elmer parity, or agreement with measurements.

## README 3D ring-heater reference

[`readme_3d_ring_heater_reference.py`](readme_3d_ring_heater_reference.py) generates the public
bundle at
[`docs/assets/readme/3d_ring_heater_reference`](../docs/assets/readme/3d_ring_heater_reference/README.md).
It builds the admitted coarse 3D ring, solves current, Joule heating, and heat flow with native JAX
and locked external Elmer on the exact same Tet4 mesh, checks complete potential and temperature
fields, and writes open CSV data with a final-size SVG and 300 dpi PNG.

The script retains separate 15 mA source-reproduction and 5 mA direct-solve bundles. The latter is
generated with `--operating-point low_temperature_projection`: that role selects the current, then
native JAX and external Elmer each solve again at the resulting target voltage. The retained CSV
fields can be re-rendered without starting Gmsh, JAX, or Elmer. The 3D panel preserves the
source-pinned layer elevations and uses an explicit bottom-to-top material order: modeled silicon
substrate, full SiO2 BOX with two far-side 2D cladding-extent backdrops, silicon ring and buses,
TiN heater, aluminum contacts, then three isosurfaces from the retained direct JAX field. The
upper-cladding volume is not rendered; its two faint backdrop planes share the BOX layer and stay
behind the device solids.
Panel a uses an orthographic axonometric view and a one-to-one scale for solved geometry. It labels
the +/-10 um adiabatic lateral extent and all vertical thicknesses, including both 0.44 um features
that share the top z span. The 0.5 um solved substrate continues visually as the same Si solid into
a depth-truncated, not-to-scale depiction of an approximately 725 um nominal handle wafer; a dashed
line marks the solve boundary. Temperature panels use the linear full-range `inferno` map; signed
differences use symmetric zero-centered `RdBu_r`. Material solids use the versioned light-canvas
categorical material palette v1.1 fill/frame pairs, with identical SiO2 colour for BOX and cladding
backdrop and opacity as the role cue. The horizontal and vertical temperature panels overlay the
matching source-CAD geometry as thin muted-neutral dual-contrast outlines; panel-c line styles
distinguish material roles without encoding another scalar.

The script requires explicit Gmsh and Elmer paths plus `--allow-external`. It also requires a clean
locked Elmer source checkout. It does not download or install software. The committed result is a
coarse same-discretization parity witness, not mesh convergence, calibrated device prediction,
physical TPU execution, or an FDTDX resonance result.

## README ring-heater thermal sensitivity

[`readme_ring_heater_thermal_sensitivity.py`](readme_ring_heater_thermal_sensitivity.py) renders
the bounded 3 by 3 domain-width and modeled-substrate-depth study at
[`docs/assets/readme/ring_heater_thermal_sensitivity`](../docs/assets/readme/ring_heater_thermal_sensitivity/README.md).
It consumes retained one-device CPU float64 evidence, writes an open summary CSV, and labels every
heatmap cell and relative-change point directly. The result isolates a computational-envelope
sensitivity; it is not formal domain convergence, Elmer parity, package calibration, or physical
TPU evidence.

## README thermal reference

[`readme_siph_thermal_reference.py`](readme_siph_thermal_reference.py) generates the public figure
bundle at [`docs/assets/readme/siph_thermal_reference`](../docs/assets/readme/siph_thermal_reference/README.md).
It meshes the published ring-resonator cross-section once, solves the exact same P1 system with
the native float64 JAX and external Elmer backends, and compares a JAX implicit adjoint with an
Elmer central difference. External execution is disabled unless `--allow-external` is present.

Use this script for the earlier reduced 2D parity and adjoint evidence. Use the notebook below for
an interactive JAX walkthrough; its different boundary model is intentionally not labelled as an
Elmer comparison.

## Thermally tuned ring resonator

[`thermally_tuned_ring_fem.ipynb`](thermally_tuned_ring_fem.ipynb) is an English, hands-on finite
element tutorial based on the physical dimensions published in Flexcompute's
[thermally tuned ring-resonator example](https://www.flexcompute.com/tidy3d/examples/notebooks/ThermallyTunedRingResonator/).
It independently reconstructs the silicon ring, two bus waveguides, 100 nm coupling gaps, and the
notched TiN heater. It then solves the exact $x=0$ vertical section with femx's native float64 JAX
P1 heat operator.

The notebook shows the tagged triangular mesh, a local element matrix, the temperature field, and
normalized heat-flow direction. It records residual, energy-balance, boundary, and temperature
ordering checks, then compares an implicit reverse-mode gradient with central finite differences.

The solve is a **reduced 2D tutorial**, not a full 3D Tidy3D or Elmer parity result. The notebook
ends with the explicit verification gates needed before a 3D thermal field can be handed to FDTDX.

### Run it

The portable default is CPU:

```bash
FEMX_NOTEBOOK_PLATFORM=cpu uv run --with jupyter --with nbformat \
  jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=900 \
  examples/thermally_tuned_ring_fem.ipynb
```

If the active environment already provides a CUDA-enabled JAX installation, request CUDA
explicitly before the kernel starts:

```bash
FEMX_NOTEBOOK_PLATFORM=cuda jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.timeout=900 \
  examples/thermally_tuned_ring_fem.ipynb
```

Set `FEMX_MRR_MESH_PROFILE=fine` to double the interval count in each direction. The committed
notebook uses the portable `standard` mesh.

The notebook is generated deterministically. Rebuilding clears its outputs, so execute it again
afterward:

```bash
uv run --with nbformat python examples/build_thermally_tuned_ring_fem.py
```
