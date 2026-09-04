from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from scripts import (  # noqa: E402
    build_tpu_distributed_electrothermal_inputs as electrothermal_builder,
)
from scripts._distributed_electrothermal_case import (  # noqa: E402
    bind_jax_self_consistent_microheater,
    distributed_electrothermal_iteration_policy,
)
from scripts._distributed_fdtdx_thermo_optic_case import (  # noqa: E402
    GRID_CENTER_M,
    GRID_SHAPE,
    GRID_SPACING_M,
    MESH_AXIS_NAME,
    RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR,
    RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR,
    build_scene,
    device_contract,
    scene_metadata,
    thermo_optic_law,
)
from scripts._tpu_distributed_electrothermal_plan import (  # noqa: E402
    read_distributed_electrothermal_artifact,
)
from scripts._tpu_distributed_fdtdx_thermo_optic_plan import (  # noqa: E402
    ARRAYS_FILENAME,
    ARTIFACT_SCHEMA,
    ELECTROTHERMAL_DIRECTORY,
    MANIFEST_FILENAME,
    read_distributed_fdtdx_thermo_optic_artifact,
    write_distributed_fdtdx_thermo_optic_artifact,
)

from femx.interop.fdtdx import (  # noqa: E402
    build_triangle_p1_sampling_plan,
    prepare_distributed_triangle_p1_sampling_plan,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _axis_centers(axis: int, start: int, stop: int) -> np.ndarray:
    indices = np.arange(start, stop, dtype=np.float64)
    return GRID_CENTER_M[axis] + (indices - 0.5 * GRID_SHAPE[axis] + 0.5) * GRID_SPACING_M


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_artifact(tmp_path: Path) -> Path:
    nested_root = tmp_path / "electrothermal-source"
    electrothermal_builder.build_inputs(nested_root, intervals=2, partition_count=4)
    electrothermal = read_distributed_electrothermal_artifact(nested_root)
    system = bind_jax_self_consistent_microheater(
        intervals=2,
        iteration=distributed_electrothermal_iteration_policy(),
    )
    payload = system.current._engine.payload
    sampling = build_triangle_p1_sampling_plan(
        np.asarray(payload.coordinates, dtype=np.float64),
        np.asarray(payload.cells, dtype=np.int64),
        (
            _axis_centers(0, 32, 64),
            _axis_centers(1, 1, 3),
            _axis_centers(2, 2, 6),
        ),
        plane_axes=(0, 2),
    )
    transfer = prepare_distributed_triangle_p1_sampling_plan(
        sampling,
        electrothermal.plan.layout.transport.cell_ids,
        source_layout_sha256=electrothermal.plan.layout.digest(),
        mesh_axis_name=MESH_AXIS_NAME,
    )
    root = tmp_path / "coupled-input"
    write_distributed_fdtdx_thermo_optic_artifact(
        root,
        electrothermal_root=nested_root,
        source_commit=str(electrothermal.manifest["source_commit"]),
        sampling=sampling,
        transfer=transfer,
        law=thermo_optic_law(),
        contract=device_contract(sampling, parameter_dtype="float32"),
        scene=scene_metadata(time_steps=302),
    )
    return root


def test_coupled_input_roundtrips_nested_authority_and_both_transfer_operators(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)

    loaded = read_distributed_fdtdx_thermo_optic_artifact(root)

    assert loaded.manifest["schema_version"] == ARTIFACT_SCHEMA
    assert loaded.electrothermal.plan.layout.partition_count == 4
    assert loaded.electrothermal.plan.layout.topology.node_count == 9
    assert loaded.sampling.target_shape == (32, 2, 4)
    assert loaded.transfer.target_shard_shape == (8, 2, 4)
    assert loaded.transfer.source_layout_sha256 == loaded.electrothermal.plan.layout.digest()
    assert loaded.transfer.sampling_operator_sha256 == loaded.sampling.operator_sha256
    assert loaded.contract.parameter_dtype == "float32"
    assert loaded.contract.target_coordinate_sha256 == loaded.sampling.target_coordinate_sha256
    assert loaded.law.sha256 == thermo_optic_law().sha256
    assert loaded.scene == scene_metadata(time_steps=302)
    assert loaded.scene["runtime_target_coordinate_tolerance"] == {
        "comparison": "float64_controller_vs_float32_fdtdx_cell_centers",
        "max_ulp_error": RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR,
        "max_grid_spacing_fraction_error": RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR,
    }
    assert len(loaded.arrays_sha256) == 64
    assert {path.name for path in root.iterdir()} == {
        ARRAYS_FILENAME,
        ELECTROTHERMAL_DIRECTORY,
        MANIFEST_FILENAME,
    }


def test_coupled_input_writer_refuses_replacement_and_contract_drift(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    loaded = read_distributed_fdtdx_thermo_optic_artifact(root)
    nested_root = root / ELECTROTHERMAL_DIRECTORY

    with pytest.raises(ValueError, match="new absolute path"):
        write_distributed_fdtdx_thermo_optic_artifact(
            root,
            electrothermal_root=nested_root,
            source_commit=str(loaded.manifest["source_commit"]),
            sampling=loaded.sampling,
            transfer=loaded.transfer,
            law=loaded.law,
            contract=loaded.contract,
            scene=loaded.scene,
        )
    with pytest.raises(ValueError, match="source commits must match"):
        write_distributed_fdtdx_thermo_optic_artifact(
            tmp_path / "wrong-commit",
            electrothermal_root=nested_root,
            source_commit="a" * 40,
            sampling=loaded.sampling,
            transfer=loaded.transfer,
            law=loaded.law,
            contract=loaded.contract,
            scene=loaded.scene,
        )
    with pytest.raises(ValueError, match="canonical float32"):
        write_distributed_fdtdx_thermo_optic_artifact(
            tmp_path / "wrong-contract",
            electrothermal_root=nested_root,
            source_commit=str(loaded.manifest["source_commit"]),
            sampling=loaded.sampling,
            transfer=loaded.transfer,
            law=loaded.law,
            contract=replace(loaded.contract, parameter_dtype="float64"),
            scene=loaded.scene,
        )
    with pytest.raises(ValueError, match="time-step count differs"):
        write_distributed_fdtdx_thermo_optic_artifact(
            tmp_path / "wrong-scene",
            electrothermal_root=nested_root,
            source_commit=str(loaded.manifest["source_commit"]),
            sampling=loaded.sampling,
            transfer=loaded.transfer,
            law=loaded.law,
            contract=loaded.contract,
            scene={**loaded.scene, "time_steps": 301},
        )


def test_coupled_input_reader_rejects_outer_and_nested_byte_tampering(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    arrays_path = root / ARRAYS_FILENAME
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"byte count|SHA-256"):
        read_distributed_fdtdx_thermo_optic_artifact(root)

    second = _build_artifact(tmp_path / "second")
    nested_arrays = second / ELECTROTHERMAL_DIRECTORY / "arrays.npz"
    nested_arrays.write_bytes(nested_arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"byte count|SHA-256"):
        read_distributed_fdtdx_thermo_optic_artifact(second)


def test_coupled_input_reader_reconstructs_instead_of_trusting_rehashed_arrays(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    arrays_path = root / ARRAYS_FILENAME
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    tampered = np.array(arrays["sampling_target_cell_indices"], copy=True)
    tampered.flat[0] = (int(tampered.flat[0]) + 1) % 8
    arrays["sampling_target_cell_indices"] = tampered
    np.savez(arrays_path, **arrays)  # type: ignore[arg-type]
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"]["sha256"] = _sha256(arrays_path)
    manifest["arrays"]["byte_count"] = arrays_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="stored sampling cell indices"):
        read_distributed_fdtdx_thermo_optic_artifact(root)


def test_coupled_input_reader_rejects_manifest_and_array_dtype_drift(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scene"]["grid_shape_xyz"] = [64, 4, 8]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical physical-gate contract"):
        read_distributed_fdtdx_thermo_optic_artifact(root)

    second = _build_artifact(tmp_path / "second")
    arrays_path = second / ARRAYS_FILENAME
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["source_cells"] = arrays["source_cells"].astype(np.float64)
    np.savez(arrays_path, **arrays)  # type: ignore[arg-type]
    second_manifest_path = second / MANIFEST_FILENAME
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    second_manifest["arrays"]["sha256"] = _sha256(arrays_path)
    second_manifest["arrays"]["byte_count"] = arrays_path.stat().st_size
    second_manifest_path.write_text(json.dumps(second_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"source_cells.*int64"):
        read_distributed_fdtdx_thermo_optic_artifact(second)


def test_scene_helper_rejects_an_unknown_backend_without_importing_runtime() -> None:
    with pytest.raises(ValueError, match="must be 'cpu' or 'tpu'"):
        build_scene(None, None, None, backend="gpu")
