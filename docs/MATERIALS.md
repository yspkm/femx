# Material data and provenance

## Scope

femx treats a material value as a specimen- and condition-dependent scientific record, not as a
global scalar keyed only by a chemical name. Every curated property carries a stable material and
property id, SI unit, usage status, validity conditions, exact citations, uncertainty text when
available, and notes about form or process. The complete canonical catalog is hashed.

The v1 statuses are deliberately distinct:

- `executable`: femx has transcribed and tested the numerical representation and rejects missing
  variables or extrapolation;
- `reference_only`: an authoritative source is identified, but its tables or coefficients are not
  yet an executable femx model;
- `requires_calibration`: the literature source is useful, but a deposited film, doped region, or
  foundry process must be calibrated before a numerical model may be selected.

`executable` means that a transcription and its software contract are executable. It does not mean
that the record is automatically valid for a particular fabricated device.

## Built-in catalog

The packaged `femx.material_catalog/v1` currently contains 11 records and 18 citations. Its
canonical digest is
`d9b6fcc9cca67910e73f1f7ac9482d2332cbf646733d60ce70d8515dc99a18ff`.

| Record | Implemented use | Deliberately guarded use |
|---|---|---|
| `si_crystalline_intrinsic` | source routing for optical, thermo-optic, thermal conductivity, and heat capacity | no executable telecom index or universal `dn/dT` |
| `sio2_fused` | executable Malitson refractive index | not an automatic thermal-oxide model |
| `sio2_thermal` | Herzinger et al. source routing | no fused-silica fallback |
| `ge_crystalline` | transparent and complex-optics source routing | Li model starts at 1.9 um and cannot represent lossless Ge at 1.55 um |
| `al_reference`, `cu_reference`, `ti_reference` | optical, thermal, and electrical source routing | no universal bulk/thin-film scalar |
| `tin_reference` | specimen-specific NIST density and 300-1800 K heat-capacity table | optical permittivity requires film/process calibration |
| `si_n_phosphorus`, `si_n_arsenic`, `si_p_boron` | dopant-specific mobility and free-carrier source routing | no default concentration, activation, conductivity, index, or absorption |

The n-type records are split by phosphorus and arsenic, and the p-type record is explicitly boron.
There is intentionally no ambiguous `n-Si` or `p-Si` scalar record.

### Executable fused-silica model

The Malitson model is evaluated with vacuum wavelength in micrometres:

$$
n^2-1 =
\frac{0.6961663\lambda^2}{\lambda^2-0.0684043^2}+
\frac{0.4079426\lambda^2}{\lambda^2-0.1162414^2}+
\frac{0.8974794\lambda^2}{\lambda^2-9.896161^2}.
$$

The implementation requires exactly 293.15 K and a wavelength within 0.21-3.71 um. At 1.55 um it
returns `1.4440236217032607`. Missing temperature, range violations, and resonance singularities
fail; femx never silently extrapolates.

### Executable TiN specimen records

NIST SCD record Z00752 describes a 99.6 percent TiN polycrystalline sample made by reacting powdered
titanium in purified nitrogen and hydrogen. The measured density is retained as `5240 kg m^-3` for
that sample only. The qualified NIST specific-heat table supplies 16 values from 300 to 1800 K,
including `601.71 J kg^-1 K^-1` at 300 K and `913.11 J kg^-1 K^-1` at 1800 K. femx uses an explicit
linear interpolation of those rounded SI table values and rejects extrapolation. The source states
a 0.5 percent relative uncertainty for its fitted original molar heat-capacity equation.

This does not make the same density or heat capacity authoritative for an arbitrary sputtered
TiN$_x$ heater. Taylor and Morreale's high-temperature thermal-conductivity measurement is
retained only as a `reference_only` record, while Kearney et al.'s sputtered-film resistivity and
Reddy et al.'s optical record remain `requires_calibration`. Stoichiometry, deposition, substrate,
anneal, thickness, texture, contacts, and temperature materially affect TiN.

## Selection and hashing

```python
from femx.materials import builtin_catalog

catalog = builtin_catalog()
n_silica = catalog.evaluate(
    "sio2_fused",
    "optical.refractive_index.malitson1965",
    temperature_k=293.15,
    vacuum_wavelength_m=1.55e-6,
)
catalog_sha256 = catalog.digest()
```

Publication/run provenance must retain at least the catalog digest, catalog version, material id,
property id, independent variables, citation ids, and any process calibration identity. A future
physics adapter will bind a selected property to Elmer/JAX/FDTDX coefficients; v1 does not silently
replace the explicit coefficients already used by those backends.

## ElmerGUI compatibility library

The locked Elmer source includes
`ElmerGUI/Application/edf/egmaterials.xml`. It is a useful compatibility reference, but it has no
property-level citations, uncertainty, specimen description, or optical/low-frequency distinction.
It also belongs to the GPL ElmerGUI area. femx therefore does not copy or redistribute it.

`load_elmer_material_library()` reads an explicitly supplied external path, preserves the
whitespace-normalized XML character value, parses only whole finite scalar values, records the
absolute source path, source revision, and exact file SHA-256, and labels every result
`legacy_unverified`. MATC/table strings remain raw strings and are never executed by the importer.
There is no implicit sibling-directory discovery.

```python
from pathlib import Path

from femx.materials import load_elmer_material_library

legacy = load_elmer_material_library(
    Path("/path/to/elmerfem/ElmerGUI/Application/edf/egmaterials.xml"),
    source_revision="4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
    selected_names=("Silicon (solid)", "Fused Silica (25 C)"),
)
```

The locked-source witness on 2026-08-30 reported XML SHA-256
`50793332435448c2a7249e03ab32d1f4ae28c2413b6ffd927a10c4aae8662d29`. Selecting Al, Cu, solid Si,
and fused silica produced path-independent selection digest
`ab666739fce9320d0d3e043bd1c33092b99551264da9f2a910dea326106e5696`. The exact source values were:

| ElmerGUI name | Density | Thermal conductivity | Heat capacity | Electrical/relative-permittivity field |
|---|---:|---:|---:|---:|
| `Aluminium (generic)` | 2700 | 237 | 897 | electrical conductivity `37.73e6` |
| `Copper (generic)` | 8960 | 401 | 385 | electrical conductivity `59.59e6` |
| `Silicon (solid)` | 2330 | temperature table beginning at 156 | 555.8 | electrical conductivity `1.0e-3` |
| `Fused Silica (25 C)` | 2200 | 1.46 | 670 | relative permittivity `3.75` |

These values are compatibility observations, not publication defaults. In particular, Elmer's
uncited fused-silica relative permittivity must not be substituted for an optical
refractive-index-squared model. The XML contains no Ti, TiN, Ge, or dopant-specific silicon record.

## Public ring-heater benchmark values

The M5 public ring-heater application uses a closed, benchmark-only parameter record rather than
silently selecting entries from the curated catalog. Ambient temperature, target current,
convection, Si/SiO2/TiN thermal conductivity, and TiN electrical conductivity are pinned to the
reviewed public Tidy3D tutorial. The two femx-only aluminum contacts use the locked ElmerGUI values
above and retain their `legacy_unverified` status. Every value and source identity contributes to
the application digest.

This record exists to make JAX and future Elmer executions compare the same stated model. It is
not a universal TiN or aluminum model, a temperature-dependent property law, or a process
calibration. The original tutorial inferred a uniform TiN heat source; femx instead solves the
declared TiN-plus-aluminum electrical conductor and rescales its linear unit-voltage solution to
15 mA. [ADR 0057](adr/0057-public-ring-heater-forward-binding.md) records that deliberate model
extension.

The retained 15 mA coarse result reaches 464.348 K at its hottest node, but the silicon, silica,
TiN, and aluminum values remain fixed at their source-pinned constants. A 5 mA projection lowers
the modeled peak rise to 18.2608 K by exact linear rescaling; it does not make those properties
process-calibrated or repair the thermal-domain boundary assumptions. The
[public ring-heater thermal-scope note](physics/PUBLIC_RING_HEATER_THERMAL_SCOPE.md) records the
normalized K/mW values, literature context, and the separate gate for temperature-dependent and
measured-device inputs.

## Primary and official sources

- I. H. Malitson, “Interspecimen Comparison of the Refractive Index of Fused Silica,” *JOSA* 55,
  1205 (1965), [doi:10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).
- H. H. Li, “Refractive index of silicon and germanium and its wavelength and temperature
  derivatives,” *J. Phys. Chem. Ref. Data* 9, 561-658 (1980),
  [doi:10.1063/1.555624](https://doi.org/10.1063/1.555624).
- Martin A. Green, “Self-consistent optical parameters of intrinsic silicon at 300 K including
  temperature coefficients,” *Solar Energy Materials and Solar Cells* 92, 1305-1310 (2008),
  [doi:10.1016/j.solmat.2008.06.009](https://doi.org/10.1016/j.solmat.2008.06.009).
- G. Cocorullo, F. G. Della Corte, and I. Rendina, “Temperature dependence of the thermo-optic
  coefficient in crystalline silicon between room temperature and 550 K at the wavelength of
  1523 nm,” *Applied Physics Letters* 74, 3338-3340 (1999),
  [doi:10.1063/1.123337](https://doi.org/10.1063/1.123337).
- C. J. Glassbrenner and Glen A. Slack, “Thermal Conductivity of Silicon and Germanium from 3 K to
  the Melting Point,” *Physical Review* 134, A1058-A1069 (1964),
  [doi:10.1103/PhysRev.134.A1058](https://doi.org/10.1103/PhysRev.134.A1058).
- G. Masetti, M. Severi, and S. Solmi, “Modeling of carrier mobility against carrier concentration
  in arsenic-, phosphorus-, and boron-doped silicon,” *IEEE Transactions on Electron Devices* 30,
  764-769 (1983),
  [doi:10.1109/T-ED.1983.21207](https://doi.org/10.1109/T-ED.1983.21207).
- R. Soref and B. Bennett, “Electrooptical effects in silicon,” *IEEE Journal of Quantum
  Electronics* 23, 123-129 (1987),
  [doi:10.1109/JQE.1987.1073206](https://doi.org/10.1109/JQE.1987.1073206).
- Aleksandar D. Rakic et al., “Optical properties of metallic films for vertical-cavity
  optoelectronic devices,” *Applied Optics* 37, 5271 (1998),
  [doi:10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271).
- J. G. Hust and A. B. Lankford, *Thermal conductivity of aluminum, copper, iron, and tungsten for
  temperatures from 1 K to the melting point*, NBSIR 84-3007 (1984),
  [doi:10.6028/NBS.IR.84-3007](https://doi.org/10.6028/NBS.IR.84-3007).
- NIST, “NIST alloy data,” NIST Public Data Repository mds2-2153 (modified 2019),
  [doi:10.18434/M32153](https://doi.org/10.18434/M32153),
  [official metadata](https://data.nist.gov/od/id/mds2-2153).
- R. W. Powell and R. P. Tye, “The thermal and electrical conductivity of titanium and its
  alloys,” *Journal of the Less Common Metals* 3, 226-233 (1961),
  [doi:10.1016/0022-5088(61)90064-9](https://doi.org/10.1016/0022-5088(61)90064-9).
- R. E. Taylor and J. Morreale, “Thermal Conductivity of Titanium Carbide, Zirconium Carbide, and
  Titanium Nitride at High Temperatures,” *Journal of the American Ceramic Society* 47, 69-73
  (1964),
  [doi:10.1111/j.1151-2916.1964.tb15657.x](https://doi.org/10.1111/j.1151-2916.1964.tb15657.x).
- B. T. Kearney et al., “Substrate and annealing temperature dependent electrical resistivity of
  sputtered titanium nitride thin films,” *Thin Solid Films* 661, 78-83 (2018),
  [doi:10.1016/j.tsf.2018.07.001](https://doi.org/10.1016/j.tsf.2018.07.001).
- Harsha Reddy et al., “Temperature-Dependent Optical Properties of Plasmonic Titanium Nitride
  Thin Films,” *ACS Photonics* 4, 1413-1420 (2017),
  [doi:10.1021/acsphotonics.7b00127](https://doi.org/10.1021/acsphotonics.7b00127).
- NIST Structural Ceramics Database, SRD 30, TiN record Z00752,
  [doi:10.18434/T4F30D](https://doi.org/10.18434/T4F30D),
  [official record](https://srdata.nist.gov/CeramicDataPortal/Scd/Z00752).
- Timothy Nathan Nunley et al., “Optical constants of germanium and thermally grown germanium
  dioxide from 0.5 to 6.6 eV via a multisample ellipsometry investigation,” *JVST B* 34, 061205
  (2016), [doi:10.1116/1.4963075](https://doi.org/10.1116/1.4963075).
- C. M. Herzinger et al., “Ellipsometric determination of optical constants for silicon and
  thermally grown silicon dioxide via a multi-sample, multi-wavelength, multi-angle investigation,”
  *Journal of Applied Physics* 83, 3323-3336 (1998),
  [doi:10.1063/1.367101](https://doi.org/10.1063/1.367101).
- NIST, [NIST-JANAF Thermochemical Tables, Fourth Edition: Silicon](https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Silicon.pdf)
  (1998).

## Remaining gates

The catalog is provenance infrastructure, not calibrated Silicon Photonics evidence. Before a
property becomes an Elmer/JAX/FDTDX publication input, its exact numerical transcription, unit
conversion, interpolation, complex-number convention, validity range, finite-difference sensitivity,
and cross-backend behavior require tests. Foundry holdout data remain a separate validation gate.
