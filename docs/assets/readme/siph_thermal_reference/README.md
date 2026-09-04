# Silicon Photonics thermal FEM reference

This bundle backs the thermal figure near the top of the femx README. It independently solves one
canonical first-order triangular mesh with native JAX float64 and a separately installed Elmer
`HeatSolve` module. The geometry is the $x=0$ vertical solid section reconstructed from
Flexcompute's public [thermally tuned ring-resonator example](https://www.flexcompute.com/tidy3d/examples/notebooks/ThermallyTunedRingResonator/).

The model uses a 300 K fixed substrate boundary and adiabatic remaining exterior boundaries. Its
Si, SiO2, and TiN values are public tutorial parameters. It is a 2D, per-unit-depth numerical
reference—not a 3D thermal prediction, foundry calibration, or measurement comparison. The very
small JAX–Elmer difference establishes parity for this shared discrete model only.

## Bundle contents

- [`figure.png`](figure.png): 300 dpi README preview.
- [`figure.svg`](figure.svg): publication-scale hybrid vector/raster figure.
- [`nodes.csv`](nodes.csv): coordinates, JAX and Elmer nodal temperatures, and signed difference.
- [`cells.csv`](cells.csv): mesh connectivity, material tags, centroids, and JAX heat flux.
- [`evidence.json`](evidence.json): fixed thresholds, metrics, versions, hashes, conventions, and
  run identities.

The two field panels share a full, unclipped temperature scale. The difference panel is signed,
unclipped, and symmetric about zero. Neither temperature panel overlays vector glyphs, so JAX and
Elmer use the same visual encoding without marks that could resemble mesh gaps. The JAX cell heat
flux remains available in `cells.csv`. The vertical plotting scale is enlarged for legibility and
labelled in the figure.

## Reproduce

Install the locked femx development environment, Gmsh, and Elmer separately. Build Elmer from the
source identity recorded in `evidence.json` or provide an installation with the same recorded
solver identity. Then run:

```bash
uv sync --locked --group dev
uv run --with matplotlib==3.11.0 python \
  examples/readme_siph_thermal_reference.py \
  --allow-external \
  --elmer-executable /absolute/path/to/ElmerSolver
```

The generator runs `make source-check`, refuses a non-CPU or non-float64 JAX runtime, and requires
explicit permission before starting Gmsh or Elmer. Raw meshes, SIF files, VTU results, and process
logs remain under ignored `.femx/readme-thermal-reference/` run directories; the compact public
bundle retains their identities and SHA-256 digests.
