#!/usr/bin/env python3
"""Render the bounded ring-heater thermal-envelope sensitivity bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "femx-matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from femx.artifacts import sha256_file

RUN_SCHEMA = "femx.ring-heater-thermal-sensitivity-evidence/v1"
BUNDLE_SCHEMA = "femx.figure.ring-heater-thermal-sensitivity/v1"
FIGURE_PREFIX = "ring-heater-thermal-sensitivity"
WIDTHS_UM = (20.0, 40.0, 80.0)
DEPTHS_UM = (0.5, 5.0, 50.0)
METRICS = (
    ("peak_K_per_mW", "Peak", "#0072B2", "o"),
    ("ring_mean_K_per_mW", "Ring mean", "#D55E00", "s"),
    ("heater_mean_K_per_mW", "Heater mean", "#009E73", "^"),
)


def _read_source_evidence(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema == BUNDLE_SCHEMA:
        source = payload.get("source_evidence")
        if not isinstance(source, dict):
            raise RuntimeError("figure bundle does not contain source_evidence")
        return source, str(payload["source_evidence_sha256"])
    if schema != RUN_SCHEMA:
        raise RuntimeError(f"unsupported thermal-sensitivity evidence schema: {schema!r}")
    return payload, sha256_file(path)


def _records_by_name(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if source.get("status") != "passed":
        raise RuntimeError("thermal-sensitivity source evidence is not passed")
    runtime = source.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("thermal-sensitivity evidence has no runtime record")
    if runtime.get("backend") != "cpu" or runtime.get("x64_enabled") is not True:
        raise RuntimeError("thermal-sensitivity figure requires retained CPU float64 evidence")
    records = source.get("cases")
    if not isinstance(records, list):
        raise RuntimeError("thermal-sensitivity evidence has no case list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "passed":
            raise RuntimeError("thermal-sensitivity evidence contains an invalid case")
        case = record.get("case")
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise RuntimeError("thermal-sensitivity case has no name")
        result[str(case["name"])] = record
    required = {
        "source_envelope",
        "substrate_5um",
        "substrate_50um",
        "domain_40um",
        "domain_80um",
        "domain_40um_substrate_5um",
        "domain_40um_substrate_50um",
        "domain_80um_substrate_5um",
        "domain_80um_substrate_50um",
        "ideal_isothermal_sidewall_bound",
    }
    if set(result) != required:
        raise RuntimeError("thermal-sensitivity evidence does not contain the canonical ten cases")
    return result


def _geometry_um(record: dict[str, Any]) -> tuple[float, float]:
    geometry = record["case"]["geometry_m"]
    return float(geometry["domain_x"]) * 1.0e6, float(geometry["substrate_thickness"]) * 1.0e6


def _factorial_record(
    records: dict[str, dict[str, Any]], width_um: float, depth_um: float
) -> dict[str, Any]:
    matches = []
    for name, record in records.items():
        if name == "ideal_isothermal_sidewall_bound":
            continue
        width, depth = _geometry_um(record)
        if np.isclose(width, width_um) and np.isclose(depth, depth_um):
            matches.append(record)
    if len(matches) != 1:
        raise RuntimeError(f"expected one factorial case at width={width_um}, depth={depth_um}")
    return matches[0]


def _matrix(records: dict[str, dict[str, Any]], metric: str) -> np.ndarray:
    return np.asarray(
        [
            [
                float(_factorial_record(records, width, depth)["temperature"][metric])
                for width in WIDTHS_UM
            ]
            for depth in DEPTHS_UM
        ],
        dtype=np.float64,
    )


def _relative_change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def _comparison_values(records: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    source = records["source_envelope"]["temperature"]
    deep_narrow = records["substrate_50um"]["temperature"]
    wide_deep = records["domain_80um_substrate_50um"]["temperature"]
    sidewall_reference = records["domain_40um_substrate_5um"]["temperature"]
    sidewall_bound = records["ideal_isothermal_sidewall_bound"]["temperature"]
    return {
        "Deepen only\nW20: D0.5→50": [
            _relative_change(float(deep_narrow[key]), float(source[key])) for key, *_rest in METRICS
        ],
        "Widen + deepen\nW80, D50": [
            _relative_change(float(wide_deep[key]), float(source[key])) for key, *_rest in METRICS
        ],
        "Ideal side sink\nvs W40, D5": [
            _relative_change(float(sidewall_bound[key]), float(sidewall_reference[key]))
            for key, *_rest in METRICS
        ],
    }


def _annotated_heatmap(
    axis: mpl.axes.Axes,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
) -> mpl.image.AxesImage:
    image = axis.imshow(values, cmap="inferno", aspect="equal", origin="lower")
    midpoint = 0.5 * (float(np.min(values)) + float(np.max(values)))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < midpoint else "#111827",
                fontsize=6.4,
                fontweight="bold",
            )
    axis.add_patch(
        mpl.patches.Rectangle(
            (-0.48, -0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor="#00E5FF",
            linewidth=1.5,
        )
    )
    axis.set(
        xticks=range(len(WIDTHS_UM)),
        xticklabels=[f"{value:g}" for value in WIDTHS_UM],
        yticks=range(len(DEPTHS_UM)),
        yticklabels=[f"{value:g}" for value in DEPTHS_UM],
        xlabel="Domain width (µm)",
        ylabel="Modeled Si depth (µm)",
        title=title,
    )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label(colorbar_label)
    return image


def _render(
    output_dir: Path,
    records: dict[str, dict[str, Any]],
    *,
    figure_id: str,
) -> dict[str, Any]:
    peak = _matrix(records, "peak_K_per_mW")
    ring = _matrix(records, "ring_mean_K_per_mW")
    comparisons = _comparison_values(records)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "svg.hashsalt": figure_id,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.28)},
    )
    _annotated_heatmap(
        axes[0],
        peak,
        title="a  Peak thermal resistance",
        colorbar_label="Peak rise / power (K/mW)",
    )
    _annotated_heatmap(
        axes[1],
        ring,
        title="b  Ring-mean thermal resistance",
        colorbar_label="Ring-mean rise / power (K/mW)",
    )

    comparison_names = tuple(comparisons)
    y_centers = np.arange(len(comparison_names), dtype=np.float64)
    offsets = (-0.18, 0.0, 0.18)
    for metric_index, ((_key, label, color, marker), offset) in enumerate(
        zip(METRICS, offsets, strict=True)
    ):
        values = np.asarray(
            [comparisons[name][metric_index] for name in comparison_names], dtype=np.float64
        )
        axes[2].scatter(
            values,
            y_centers + offset,
            label=label,
            color=color,
            marker=marker,
            s=26.0,
            zorder=3,
        )
        for x_value, y_value in zip(values, y_centers + offset, strict=True):
            axes[2].annotate(
                f"{x_value:+.2f}%",
                (float(x_value), float(y_value)),
                xytext=(4 if x_value >= 0.0 else -4, 0),
                textcoords="offset points",
                ha="left" if x_value >= 0.0 else "right",
                va="center",
                fontsize=5.5,
                color=color,
            )
    axes[2].axvline(0.0, color="#6B7280", linewidth=0.8, linestyle="--", zorder=1)
    axes[2].grid(axis="x", color="#D1D5DB", linewidth=0.45, alpha=0.8)
    axes[2].set(
        xlim=(-5.0, 17.0),
        ylim=(-0.55, len(comparison_names) - 0.45),
        yticks=y_centers,
        yticklabels=comparison_names,
        xlabel="Relative change (%)",
        title="c  Controlled changes",
    )
    axes[2].invert_yaxis()
    axes[2].legend(loc="lower right", frameon=False)

    figure.suptitle(
        "Bounded ring-heater thermal-envelope sensitivity | 5 mA CPU solves",
        fontsize=8.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.005,
        (
            "constant properties • common mesh-size policy • bottom fixed at 300 K • "
            "cyan cell = source envelope • not device calibration"
        ),
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#4B5563",
    )
    png_path = output_dir / "figure.png"
    svg_path = output_dir / "figure.svg"
    figure.savefig(png_path, dpi=300, metadata={"Software": "femx", "Description": figure_id})
    figure.savefig(svg_path, metadata={"Date": None, "Description": figure_id})
    plt.close(figure)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "question": (
            "How do bounded lateral extent, modeled silicon depth, and an ideal side sink change "
            "the coarse model's thermal resistance?"
        ),
        "takeaway": (
            "Within this 3 by 3 envelope, deeper silicon does not monotonically reduce thermal "
            "resistance; width and depth interact, and the ideal side-sink bound is negligible "
            "at W40 um and D5 um."
        ),
        "physical_size_inches": [7.2, 3.0],
        "png_pixels": [2160, 900],
        "png_dpi": 300,
        "heatmap_colormap": "inferno",
        "non_color_encoding": "every heatmap cell and comparison point has an exact value label",
        "source_envelope_outline": "cyan border at W20 um and D0.5 um",
        "widths_um": list(WIDTHS_UM),
        "substrate_depths_um": list(DEPTHS_UM),
        "peak_K_per_mW": peak.tolist(),
        "ring_mean_K_per_mW": ring.tolist(),
        "relative_comparisons_percent": comparisons,
        "clipping": "none; plotted limits include every retained value",
        "figure_id": figure_id,
    }


def _write_summary_csv(path: Path, records: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "case",
                "domain_width_um",
                "substrate_depth_um",
                "lateral_condition",
                "node_count",
                "tetrahedron_count",
                "power_mW",
                "peak_rise_K",
                "peak_K_per_mW",
                "ring_mean_K_per_mW",
                "heater_mean_K_per_mW",
            )
        )
        for name, record in records.items():
            width_um, depth_um = _geometry_um(record)
            temperature = record["temperature"]
            boundary = record["boundary"]
            writer.writerow(
                (
                    name,
                    f"{width_um:.12g}",
                    f"{depth_um:.12g}",
                    boundary["lateral"]["condition"],
                    record["mesh"]["node_count"],
                    record["mesh"]["tetrahedron_count"],
                    f"{float(record['numerics']['electrical_joule_power_W']) * 1.0e3:.12g}",
                    f"{float(temperature['peak_rise_K']):.12g}",
                    f"{float(temperature['peak_K_per_mW']):.12g}",
                    f"{float(temperature['ring_mean_K_per_mW']):.12g}",
                    f"{float(temperature['heater_mean_K_per_mW']):.12g}",
                )
            )


def _write_readme(path: Path, figure_id: str, records: dict[str, dict[str, Any]]) -> None:
    baseline = records["source_envelope"]["temperature"]
    wide_deep = records["domain_80um_substrate_50um"]["temperature"]
    content = f"""# Ring-heater thermal-envelope sensitivity

![Bounded domain-width and modeled-substrate-depth sensitivity](figure.png)

`{figure_id}` summarizes ten retained 5 mA, one-device CPU float64 solves: a 3 by 3 factorial
over 20/40/80 um square domains and 0.5/5/50 um modeled silicon depths, plus one ideal-isothermal
sidewall bound. Every case uses the same constant material properties and mesh-size policy.

The source envelope gives {float(baseline["peak_K_per_mW"]):.3f} K/mW peak and
{float(baseline["ring_mean_K_per_mW"]):.3f} K/mW ring mean. The widest/deepest tested envelope
gives {float(wide_deep["peak_K_per_mW"]):.3f} and
{float(wide_deep["ring_mean_K_per_mW"]):.3f} K/mW, respectively. The results do not support
attributing the source-envelope thermal resistance to the 0.5 um modeled substrate alone: at fixed
20 um width, increasing the modeled depth raises both observables, while increasing width partly
offsets that change.

## Files

- `summary.csv`: compact values plotted in the figure;
- `evidence.json`: complete case, mesh, boundary, numerical, runtime, source-hash, and figure data;
- `figure.svg` and `figure.png`: publication-scale vector and 300 dpi raster forms.

## Reproduce the presentation

```bash
uv run python examples/readme_ring_heater_thermal_sensitivity.py \\
  --evidence docs/assets/readme/ring_heater_thermal_sensitivity/evidence.json \\
  --output /temporary/new/rendered-bundle
```

## Claim boundary

This is a bounded computational sensitivity check, not formal domain convergence, a full wafer or
package model, temperature-dependent material calibration, Elmer parity, physical TPU evidence,
or fabricated-device agreement. The 5 mA point is a low-temperature operating illustration under
the same linear material assumptions.
"""
    path.write_text(content, encoding="utf-8", newline="\n")


def build_bundle(evidence_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build one deterministic public figure bundle from retained sensitivity evidence."""

    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("output directory must not already exist")
    source, source_sha256 = _read_source_evidence(evidence_path)
    records = _records_by_name(source)
    generator_sha256 = sha256_file(Path(__file__).resolve())
    identity = {
        "schema_version": BUNDLE_SCHEMA,
        "source_evidence_sha256": source_sha256,
        "generator_sha256": generator_sha256,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    figure_id = f"{FIGURE_PREFIX}-{hashlib.sha256(encoded).hexdigest()[:16]}"
    output_dir.mkdir(parents=True)
    _write_summary_csv(output_dir / "summary.csv", records)
    figure = _render(output_dir, records, figure_id=figure_id)
    _write_readme(output_dir / "README.md", figure_id, records)
    artifact_names = ("README.md", "summary.csv", "figure.png", "figure.svg")
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "figure_id": figure_id,
        "source_evidence_sha256": source_sha256,
        "generator_sha256": generator_sha256,
        "source_evidence": source,
        "figure": figure,
        "artifacts": {name: sha256_file(output_dir / name) for name in artifact_names},
        "claim_scope": (
            "bounded constant-property CPU sensitivity; not domain convergence, package "
            "calibration, Elmer parity, TPU evidence, or fabricated-device agreement"
        ),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = build_bundle(arguments.evidence.resolve(), arguments.output.resolve())
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
