from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from scripts._tpu_public_ring_heater_plan import (  # noqa: E402
    ARRAY_FIELDS,
    ARTIFACT_SCHEMA,
    PublicRingHeaterTPUAuthority,
    read_public_ring_heater_cpu_authority,
    read_public_ring_heater_tpu_artifact,
    write_public_ring_heater_tpu_artifact,
)

from femx.applications.ring_heater import (  # noqa: E402
    PublicRingHeaterForwardPlan,
    PublicRingHeaterMeshAdmissionPolicy,
    PublicRingHeaterReferenceParameters,
)
from femx.backends.jax.partition import balanced_lexicographic_cell_owners  # noqa: E402
from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    prepare_tet4_electrothermal_plan,
)
from femx.meshing.gmsh import PublicRingHeaterMeshReport  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(
        (
            (0, 1, 3, 7),
            (0, 3, 2, 7),
            (0, 2, 6, 7),
            (0, 6, 4, 7),
            (0, 4, 5, 7),
            (0, 5, 1, 7),
        ),
        dtype=np.int64,
    )
    return coordinates, cells


def _forward_plan() -> PublicRingHeaterForwardPlan:
    coordinates, cells = _mesh()
    owners = balanced_lexicographic_cell_owners(coordinates, cells, partition_count=2)
    current_boundary = np.asarray((0, 7), dtype=np.int64)
    thermal_boundary = np.flatnonzero(np.isclose(coordinates[:, 2], 0.0))
    empty_facets = np.empty((0, 3), dtype=np.int64)
    empty_values = np.empty((0,), dtype=np.float64)
    tet4 = prepare_tet4_electrothermal_plan(
        coordinates,
        cells,
        owners,
        np.arange(cells.shape[0], dtype=np.int64),
        current_conductivity=np.full((cells.shape[0],), 2.0),
        current_cell_source=np.zeros((cells.shape[0],)),
        current_flux_facets=empty_facets,
        current_facet_flux=empty_values,
        current_dirichlet_nodes=current_boundary,
        current_dirichlet_base=np.zeros(current_boundary.shape),
        current_dirichlet_voltage_scale=np.asarray((0.0, 1.0)),
        thermal_conductivity=np.full((cells.shape[0],), 4.0),
        thermal_cell_source=np.zeros((cells.shape[0],)),
        thermal_flux_facets=empty_facets,
        thermal_facet_flux=empty_values,
        thermal_robin_facets=empty_facets,
        thermal_robin_transfer=empty_values,
        thermal_robin_ambient=empty_values,
        thermal_dirichlet_nodes=thermal_boundary,
        thermal_dirichlet_values=np.full(thermal_boundary.shape, 300.0),
        thermal_reference=300.0,
        partition_count=2,
    )
    reference = PublicRingHeaterReferenceParameters()
    report = PublicRingHeaterMeshReport(
        recipe_sha256="1" * 64,
        import_record_sha256="2" * 64,
        node_count=coordinates.shape[0],
        tetrahedron_count=cells.shape[0],
        boundary_triangle_count=12,
        total_volume_m3=1.0,
        full_domain_relative_volume_error=0.0,
        minimum_cell_volume_m3=1.0 / 6.0,
        maximum_cell_volume_m3=1.0 / 6.0,
        minimum_edge_length_m=1.0,
        maximum_edge_length_m=3.0**0.5,
        minimum_mean_ratio=0.5,
        percentile_1_mean_ratio=0.5,
        median_mean_ratio=0.5,
        maximum_region_volume_relative_error=0.0,
        region_cell_counts=(("synthetic", cells.shape[0]),),
        region_volumes_m3=(("synthetic", 1.0),),
        region_volume_relative_errors=(("synthetic", 0.0),),
        electrical_interface_triangle_counts=(("synthetic", 1),),
        surface_triangle_counts=(("synthetic", 12),),
    )
    return PublicRingHeaterForwardPlan(
        reference=reference,
        mesh_report=report,
        mesh_admission=PublicRingHeaterMeshAdmissionPolicy(),
        tet4=tet4,
        current_region_cell_counts=(
            ("tin_heater", 2),
            ("al_contact_negative", 2),
            ("al_contact_positive", 2),
        ),
        terminal_node_counts=(("terminal_negative", 1), ("terminal_positive", 1)),
    )


def _authority(plan: PublicRingHeaterForwardPlan) -> PublicRingHeaterTPUAuthority:
    coordinates, _cells = _mesh()
    potential = coordinates[:, 0].copy()
    temperature = 300.0 + coordinates[:, 2]
    record = {
        "schema_version": "femx.public-ring-heater-forward.cpu-witness/v1",
        "status": "passed",
        "profile": "fine",
        "excitation": {
            "target_voltage_V": 0.5,
            "target_current_A": 0.015,
            "predicted_joule_power_W": 0.0075,
        },
        "numerics": {
            "potential_sha256_float64": hashlib.sha256(
                np.ascontiguousarray(potential, dtype="<f8").tobytes()
            ).hexdigest(),
            "temperature_sha256_float64": hashlib.sha256(
                np.ascontiguousarray(temperature, dtype="<f8").tobytes()
            ).hexdigest(),
        },
    }
    authority = PublicRingHeaterTPUAuthority(potential, temperature, record)
    authority.validate(plan)
    return authority


def test_ring_heater_tpu_artifact_round_trips_as_memory_mapped_arrays(tmp_path: Path) -> None:
    plan = _forward_plan()
    authority = _authority(plan)
    root = tmp_path / "input"
    manifest = write_public_ring_heater_tpu_artifact(
        root,
        plan,
        authority,
        source_commit="a" * 40,
        source_msh_sha256="b" * 64,
        partition_owner_sha256="c" * 64,
        silicon_ring_cell_ids=np.asarray((0,), dtype=np.int64),
        tin_heater_cell_ids=np.asarray((1,), dtype=np.int64),
    )
    loaded = read_public_ring_heater_tpu_artifact(root)

    assert manifest["schema_version"] == ARTIFACT_SCHEMA
    assert loaded.logical_sha256 == manifest["logical_sha256"]
    assert loaded.runtime_plan.source_plan_sha256 == plan.tet4.digest()
    assert set(loaded.arrays) == set(ARRAY_FIELDS)
    assert all(isinstance(value, np.memmap) for value in loaded.arrays.values())
    assert loaded.arrays["authority_potential"].dtype == np.float32
    assert np.all(np.isfinite(loaded.arrays["authority_potential"]))


def test_ring_heater_tpu_artifact_rejects_a_changed_array(tmp_path: Path) -> None:
    plan = _forward_plan()
    root = tmp_path / "input"
    write_public_ring_heater_tpu_artifact(
        root,
        plan,
        _authority(plan),
        source_commit="a" * 40,
        source_msh_sha256="b" * 64,
        partition_owner_sha256="c" * 64,
        silicon_ring_cell_ids=(0,),
        tin_heater_cell_ids=(1,),
    )
    path = root / "arrays" / "thermal_cell_volumes.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes((original[0] ^ 1,)))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_public_ring_heater_tpu_artifact(root)


def test_cpu_authority_pair_reader_is_strict(tmp_path: Path) -> None:
    plan = _forward_plan()
    authority = _authority(plan)
    record_path = tmp_path / "record.json"
    state_path = tmp_path / "state.npz"
    record_path.write_text(json.dumps(authority.record), encoding="utf-8")
    np.savez(state_path, potential=authority.potential, temperature=authority.temperature)

    loaded = read_public_ring_heater_cpu_authority(record_path, state_path)
    loaded.validate(plan)
    np.testing.assert_array_equal(loaded.temperature, authority.temperature)

    wrong_path = tmp_path / "wrong.npz"
    np.savez(wrong_path, potential=authority.potential)
    with pytest.raises(ValueError, match="exactly potential and temperature"):
        read_public_ring_heater_cpu_authority(record_path, wrong_path)
