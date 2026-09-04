# Ring-heater thermal-envelope sensitivity

![Bounded domain-width and modeled-substrate-depth sensitivity](figure.png)

`ring-heater-thermal-sensitivity-cb9cafed8f1e2328` summarizes ten retained 5 mA, one-device CPU float64 solves: a 3 by 3 factorial
over 20/40/80 um square domains and 0.5/5/50 um modeled silicon depths, plus one ideal-isothermal
sidewall bound. Every case uses the same constant material properties and mesh-size policy.

The source envelope gives 15.909 K/mW peak and
5.727 K/mW ring mean. The widest/deepest tested envelope
gives 15.684 and
5.542 K/mW, respectively. The results do not support
attributing the source-envelope thermal resistance to the 0.5 um modeled substrate alone: at fixed
20 um width, increasing the modeled depth raises both observables, while increasing width partly
offsets that change.

## Files

- `summary.csv`: compact values plotted in the figure;
- `evidence.json`: complete case, mesh, boundary, numerical, runtime, source-hash, and figure data;
- `figure.svg` and `figure.png`: publication-scale vector and 300 dpi raster forms.

## Reproduce the presentation

```bash
uv run python examples/readme_ring_heater_thermal_sensitivity.py \
  --evidence docs/assets/readme/ring_heater_thermal_sensitivity/evidence.json \
  --output /temporary/new/rendered-bundle
```

## Claim boundary

This is a bounded computational sensitivity check, not formal domain convergence, a full wafer or
package model, temperature-dependent material calibration, Elmer parity, physical TPU evidence,
or fabricated-device agreement. The 5 mA point is a low-temperature operating illustration under
the same linear material assumptions.
