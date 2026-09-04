from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.port_collective import (  # noqa: E402
    PortCollectiveLayout,
    prepare_collective_port_layout,
)
from femx.backends.jax.port_collective_checkpoint import (  # noqa: E402
    PORT_COLLECTIVE_CHECKPOINT_COMPLETE_SCHEMA,
    PORT_COLLECTIVE_CHECKPOINT_REPORT_SCHEMA,
    PortCollectiveCheckpointFragment,
    _canonical_json,
    _canonical_local_array,
    _checkpoint_container,
    _exact_keys,
    _fragment_report,
    _require_component,
    _require_nonnegative_integer,
    _require_sha256,
    _strict_json,
    port_collective_checkpoint_fragment_path,
    restore_port_collective_checkpoint_fragment,
    write_port_collective_checkpoint_fragment,
)
from femx.backends.jax.port_collective_runtime import (  # noqa: E402
    collective_port_named_sharding,
    make_collective_port_array_from_process_local_data,
)
from femx.backends.jax.port_owned_ghost import (  # noqa: E402
    prepare_owned_ghost_port_topology,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

SOURCE_SHA = "a" * 64
CONFIG_SHA = "b" * 64
CHECKPOINT_ID = "port-step-000007"


def _layout() -> PortCollectiveLayout:
    topology = prepare_owned_ghost_port_topology(
        np.arange(6, dtype=np.int64).reshape(1, 6),
        np.asarray((0,), dtype=np.int64),
        free_dof_count=6,
        partition_count=1,
    )
    return prepare_collective_port_layout(topology)


def _mesh(*, axis_name: str = "partition") -> Mesh:
    return Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), (axis_name,))


def _arrays(mesh: Mesh) -> dict[str, Any]:
    values: Any = {
        "cell-map": np.arange(6, dtype=np.int64).reshape(1, 1, 6),
        "complex-state": np.asarray(((1.0 + 2.0j, -3.0 + 0.5j),), dtype=np.complex128),
        "real-state": np.asarray(((0.25, -0.75),), dtype=np.float64),
    }
    return {
        name: make_collective_port_array_from_process_local_data(name, value, mesh)[0]
        for name, value in values.items()
    }


def _write(tmp_path: Path, **changes: Any) -> PortCollectiveCheckpointFragment:
    layout = changes.pop("layout", _layout())
    mesh = changes.pop("mesh", _mesh())
    arrays = changes.pop("arrays", _arrays(mesh))
    values: Any = {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 7,
        "source_sha256": SOURCE_SHA,
        "config_sha256": CONFIG_SHA,
        "layout": layout,
        "mesh": mesh,
        "arrays": arrays,
    }
    values.update(changes)
    return write_port_collective_checkpoint_fragment(tmp_path / "checkpoints", **values)


def _restore(
    fragment: PortCollectiveCheckpointFragment,
    *,
    layout: Any | None = None,
    mesh: Mesh | None = None,
    templates: dict[str, Any] | None = None,
    **changes: Any,
) -> tuple[dict[str, Any], PortCollectiveCheckpointFragment]:
    actual_layout = _layout() if layout is None else layout
    actual_mesh = _mesh() if mesh is None else mesh
    actual_templates = _arrays(actual_mesh) if templates is None else templates
    values: Any = {
        "expected_checkpoint_id": CHECKPOINT_ID,
        "expected_step": 7,
        "expected_source_sha256": SOURCE_SHA,
        "expected_config_sha256": CONFIG_SHA,
        "layout": actual_layout,
        "mesh": actual_mesh,
        "templates": actual_templates,
    }
    values.update(changes)
    return restore_port_collective_checkpoint_fragment(fragment.path, **values)


def _resign_manifest(fragment: PortCollectiveCheckpointFragment, mutate: Any) -> None:
    manifest_path = fragment.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    encoded = _canonical_json(manifest)
    manifest_path.write_bytes(encoded)
    marker = {
        "schema_version": PORT_COLLECTIVE_CHECKPOINT_COMPLETE_SCHEMA,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    (fragment.path / "COMPLETE").write_bytes(_canonical_json(marker))


def _resign_shard_file(fragment: PortCollectiveCheckpointFragment, shard_path: Path) -> None:
    def update(manifest: dict[str, Any]) -> None:
        for array in manifest["arrays"]:
            for shard in array["shards"]:
                if shard["relative_path"] == shard_path.relative_to(fragment.path).as_posix():
                    shard["file_size_bytes"] = shard_path.stat().st_size
                    shard["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
                    return
        raise AssertionError("test shard was absent from its manifest")

    _resign_manifest(fragment, update)


def test_checkpoint_round_trip_is_process_local_atomic_and_path_independent(tmp_path: Path) -> None:
    mesh = _mesh()
    source = _arrays(mesh)
    fragment = _write(tmp_path, mesh=mesh, arrays=source)

    assert fragment.path == (tmp_path / "checkpoints" / CHECKPOINT_ID / "process-00000")
    assert fragment.path.is_dir()
    assert not fragment.path.with_name("process-00000.incomplete").exists()
    assert fragment.array_names == ("cell-map", "complex-state", "real-state")
    assert fragment.step == 7
    assert fragment.process_index == 0
    assert fragment.process_count == 1
    assert fragment.canonical_data() == {
        "schema_version": PORT_COLLECTIVE_CHECKPOINT_REPORT_SCHEMA,
        "checkpoint_id": CHECKPOINT_ID,
        "step": 7,
        "source_sha256": SOURCE_SHA,
        "config_sha256": CONFIG_SHA,
        "layout_sha256": _layout().digest(),
        "process_index": 0,
        "process_count": 1,
        "manifest_sha256": fragment.manifest_sha256,
        "array_names": ["cell-map", "complex-state", "real-state"],
        "completion_scope": "one process-local fragment",
        "restore_policy": "exact same topology only; no resharding",
    }

    restored, observed = _restore(fragment, mesh=mesh, templates=source)
    assert observed == fragment
    for name, expected in source.items():
        np.testing.assert_array_equal(np.asarray(jax.device_get(restored[name])), expected)
        assert restored[name].sharding.is_equivalent_to(expected.sharding, expected.ndim)

    second = _write(tmp_path / "second", mesh=mesh, arrays=source)
    assert second.manifest_sha256 == fragment.manifest_sha256
    for first_path in sorted((fragment.path / "arrays").glob("*.npy")):
        second_path = second.path / "arrays" / first_path.name
        assert (
            hashlib.sha256(first_path.read_bytes()).digest()
            == hashlib.sha256(second_path.read_bytes()).digest()
        )


def test_checkpoint_refuses_overwrite_and_stale_incomplete_publication(tmp_path: Path) -> None:
    fragment = _write(tmp_path)
    with pytest.raises(ContractError, match="overwrite is forbidden"):
        _write(tmp_path)

    second_root = tmp_path / "second" / "checkpoints"
    incomplete = port_collective_checkpoint_fragment_path(second_root, CHECKPOINT_ID, 0).with_name(
        "process-00000.incomplete"
    )
    incomplete.parent.mkdir(parents=True)
    incomplete.mkdir()
    with pytest.raises(ContractError, match="stale incomplete"):
        _write(tmp_path / "second")
    assert fragment.path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_id", "../escape", "path component"),
        ("step", True, "nonnegative integer"),
        ("step", -1, "nonnegative integer"),
        ("source_sha256", "A" * 64, "canonical SHA-256"),
        ("config_sha256", "short", "canonical SHA-256"),
        ("arrays", {}, "at least one"),
        ("arrays", {"bad/name": jnp.ones((1, 1))}, "path component"),
        ("arrays", {1: jnp.ones((1, 1)), "state": jnp.ones((1, 1))}, "path component"),
        ("arrays", {"state": np.ones((1, 1))}, "JAX arrays"),
    ],
)
def test_checkpoint_writer_rejects_invalid_identity_or_array_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _write(tmp_path, **{field: value})


def test_checkpoint_rejects_relative_or_non_directory_roots(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="root must be absolute"):
        _checkpoint_container(Path("relative"), CHECKPOINT_ID)
    with pytest.raises(ContractError, match="root must be absolute"):
        port_collective_checkpoint_fragment_path(Path("relative"), CHECKPOINT_ID, 0)
    root_file = tmp_path / "file"
    root_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ContractError, match="real directory"):
        _checkpoint_container(root_file, CHECKPOINT_ID)
    container_file = tmp_path / "root" / CHECKPOINT_ID
    container_file.parent.mkdir()
    container_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ContractError, match="container"):
        _checkpoint_container(container_file.parent, CHECKPOINT_ID)
    with pytest.raises(ContractError, match="process index"):
        port_collective_checkpoint_fragment_path(tmp_path, CHECKPOINT_ID, True)


def test_checkpoint_rejects_nonfinite_and_unsupported_shard_values(tmp_path: Path) -> None:
    mesh = _mesh()
    nonfinite = _arrays(mesh)
    nonfinite["real-state"] = jax.device_put(
        jnp.asarray(((jnp.nan, 1.0),)), collective_port_named_sharding(mesh, 2)
    )
    with pytest.raises(ContractError, match="only finite"):
        _write(tmp_path / "nonfinite", mesh=mesh, arrays=nonfinite)

    unsupported = _arrays(mesh)
    unsupported["bool-state"] = jax.device_put(
        jnp.asarray(((True, False),)), collective_port_named_sharding(mesh, 2)
    )
    with pytest.raises(ContractError, match="unsupported checkpoint dtype"):
        _write(tmp_path / "unsupported", mesh=mesh, arrays=unsupported)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"path": Path("relative")}, "path must be absolute"),
        ({"checkpoint_id": ".."}, "path component"),
        ({"step": False}, "nonnegative integer"),
        ({"manifest_sha256": "A" * 64}, "canonical SHA-256"),
        ({"process_count": 0}, "outside the process count"),
        ({"process_index": 1}, "outside the process count"),
        ({"array_names": ()}, "nonempty, unique, and sorted"),
        ({"array_names": ("z", "a")}, "nonempty, unique, and sorted"),
    ],
)
def test_checkpoint_fragment_report_is_strict(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    report = PortCollectiveCheckpointFragment(
        path=tmp_path,
        checkpoint_id=CHECKPOINT_ID,
        step=7,
        source_sha256=SOURCE_SHA,
        config_sha256=CONFIG_SHA,
        layout_sha256="c" * 64,
        process_index=0,
        process_count=1,
        manifest_sha256="d" * 64,
        array_names=("state",),
    )
    with pytest.raises(ContractError, match=message):
        replace(report, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_checkpoint_id", "wrong", "checkpoint id does not match"),
        ("expected_step", 8, "checkpoint step does not match"),
        ("expected_source_sha256", "c" * 64, "source sha256 does not match"),
        ("expected_config_sha256", "d" * 64, "config sha256 does not match"),
        ("templates", {}, "requires named"),
        ("templates", {"state": np.ones((1, 1))}, "must be JAX arrays"),
        ("templates", {"bad/name": jnp.ones((1, 1))}, "path component"),
        ("templates", {1: jnp.ones((1, 1)), "state": jnp.ones((1, 1))}, "path component"),
    ],
)
def test_checkpoint_restore_rejects_identity_or_template_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fragment = _write(tmp_path)
    with pytest.raises(ContractError, match=message):
        _restore(fragment, **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "v2", "manifest schema"),
        ("layout_sha256", "c" * 64, "layout sha256"),
        ("process_index", 1, "process index"),
        ("process_count", 2, "process count"),
        ("completion_scope", "global", "completion scope"),
        ("restore_policy", "reshard", "restore policy"),
    ],
)
def test_checkpoint_restore_rejects_resigned_manifest_identity_tampering(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    fragment = _write(tmp_path)
    _resign_manifest(fragment, lambda manifest: manifest.__setitem__(field, value))
    with pytest.raises(ContractError, match=message):
        _restore(fragment)


def test_checkpoint_restore_rejects_manifest_hash_schema_and_directory_identity(
    tmp_path: Path,
) -> None:
    fragment = _write(tmp_path / "hash")
    (fragment.path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="does not match COMPLETE"):
        _restore(fragment)

    fragment = _write(tmp_path / "marker")
    marker_path = fragment.path / "COMPLETE"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["schema_version"] = "v2"
    marker_path.write_bytes(_canonical_json(marker))
    with pytest.raises(ContractError, match="marker schema"):
        _restore(fragment)

    fragment = _write(tmp_path / "directory")
    moved = fragment.path.with_name("process-00001")
    fragment.path.rename(moved)
    changed = replace(fragment, path=moved)
    with pytest.raises(ContractError, match="directory disagrees"):
        _restore(changed)


def test_checkpoint_restore_rejects_array_and_shard_metadata_tampering(tmp_path: Path) -> None:
    cases = (
        (
            lambda manifest: manifest["arrays"][0].__setitem__("name", "unexpected"),
            "unexpected array",
        ),
        (
            lambda manifest: manifest["arrays"][0]["array_report"].__setitem__("dtype", "float32"),
            "array report",
        ),
        (
            lambda manifest: manifest["arrays"][0].__setitem__("shards", []),
            "shard set",
        ),
        (
            lambda manifest: manifest["arrays"][0]["shards"][0].__setitem__("device_id", 99),
            "shard metadata",
        ),
        (
            lambda manifest: manifest["arrays"][0]["shards"][0].__setitem__(
                "relative_path", "arrays/wrong.npy"
            ),
            "path is not canonical",
        ),
    )
    for index, (mutate, message) in enumerate(cases):
        fragment = _write(tmp_path / str(index))
        _resign_manifest(fragment, mutate)
        with pytest.raises(ContractError, match=message):
            _restore(fragment)


def test_checkpoint_restore_rejects_shard_file_and_directory_tampering(tmp_path: Path) -> None:
    fragment = _write(tmp_path / "digest")
    shard = next((fragment.path / "arrays").glob("*.npy"))
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="file size"):
        _restore(fragment)

    fragment = _write(tmp_path / "unexpected")
    (fragment.path / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ContractError, match="unexpected file"):
        _restore(fragment)

    fragment = _write(tmp_path / "directory")
    (fragment.path / "unexpected").mkdir()
    with pytest.raises(ContractError, match="unexpected directory"):
        _restore(fragment)


def test_checkpoint_restore_rejects_same_size_digest_and_invalid_npy_payloads(
    tmp_path: Path,
) -> None:
    fragment = _write(tmp_path / "digest")
    shard = next((fragment.path / "arrays").glob("*.npy"))
    changed = bytearray(shard.read_bytes())
    changed[-1] ^= 1
    shard.write_bytes(changed)
    with pytest.raises(ContractError, match="shard SHA-256"):
        _restore(fragment)

    fragment = _write(tmp_path / "invalid")
    shard = next((fragment.path / "arrays").glob("*.npy"))
    shard.write_bytes(b"not-an-npy")
    _resign_shard_file(fragment, shard)
    with pytest.raises(ContractError, match="not a valid non-pickle NPY"):
        _restore(fragment)

    fragment = _write(tmp_path / "oversized")
    shard = next((fragment.path / "arrays").glob("*.npy"))
    shard.write_bytes(shard.read_bytes() + b"x" * ((1 << 16) + 1))
    _resign_shard_file(fragment, shard)
    with pytest.raises(ContractError, match="bounded logical size"):
        _restore(fragment)


def test_checkpoint_restore_rejects_wrong_payload_shape_and_noncanonical_array_order(
    tmp_path: Path,
) -> None:
    fragment = _write(tmp_path / "shape")
    shard = fragment.path / "arrays" / "cell-map--partition-00000.npy"
    with shard.open("wb") as stream:
        np.save(stream, np.arange(6, dtype=np.int64).reshape(1, 6), allow_pickle=False)
    _resign_shard_file(fragment, shard)
    with pytest.raises(ContractError, match="payload shape"):
        _restore(fragment)

    fragment = _write(tmp_path / "order")
    _resign_manifest(fragment, lambda manifest: manifest["arrays"].reverse())
    with pytest.raises(ContractError, match="not canonical"):
        _restore(fragment)


def test_checkpoint_restore_rejects_incomplete_array_set_missing_shard_and_symlink(
    tmp_path: Path,
) -> None:
    fragment = _write(tmp_path / "set")
    templates = _arrays(_mesh())
    templates.pop("real-state")
    with pytest.raises(ContractError, match="array set"):
        _restore(fragment, templates=templates)

    fragment = _write(tmp_path / "missing")
    shard = next((fragment.path / "arrays").glob("*.npy"))
    shard.unlink()
    with pytest.raises(ContractError, match="not a regular file"):
        _restore(fragment)

    fragment = _write(tmp_path / "symlink")
    (fragment.path / "extra-link").symlink_to(fragment.path / "manifest.json")
    with pytest.raises(ContractError, match="symbolic links"):
        _restore(fragment)


def test_checkpoint_restore_requires_an_absolute_complete_directory(tmp_path: Path) -> None:
    layout = _layout()
    mesh = _mesh()
    kwargs: Any = {
        "expected_checkpoint_id": CHECKPOINT_ID,
        "expected_step": 7,
        "expected_source_sha256": SOURCE_SHA,
        "expected_config_sha256": CONFIG_SHA,
        "layout": layout,
        "mesh": mesh,
        "templates": _arrays(mesh),
    }
    with pytest.raises(ContractError, match="path must be absolute"):
        restore_port_collective_checkpoint_fragment(Path("relative"), **kwargs)
    with pytest.raises(ContractError, match="complete real directory"):
        restore_port_collective_checkpoint_fragment(tmp_path / "missing", **kwargs)


def test_writer_detects_internally_inconsistent_runtime_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import femx.backends.jax.port_collective_checkpoint as checkpoint

    mesh = _mesh()
    arrays = _arrays(mesh)
    original_describe: Any = checkpoint.describe_collective_port_array  # type: ignore[attr-defined]
    first = original_describe("real-state", arrays["real-state"], mesh)

    monkeypatch.setattr(
        checkpoint,
        "describe_collective_port_array",
        lambda name, array, mesh, axis_name="partition": replace(
            original_describe(name, array, mesh, axis_name=axis_name),
            process_index=first.process_index + (name == "real-state"),
            process_count=2 if name == "real-state" else 1,
            addressable_shards=(
                replace(
                    original_describe(name, array, mesh, axis_name=axis_name).addressable_shards[0],
                    process_index=first.process_index + (name == "real-state"),
                ),
            ),
        ),
    )
    with pytest.raises(ContractError, match="disagree on JAX process identity"):
        _write(tmp_path / "process", mesh=mesh, arrays=arrays)

    monkeypatch.setattr(checkpoint, "describe_collective_port_array", original_describe)
    monkeypatch.setattr(
        checkpoint,
        "describe_collective_port_mesh",
        lambda layout, mesh, axis_name="partition": SimpleNamespace(process_count=2),
    )
    with pytest.raises(ContractError, match="array and Mesh process counts disagree"):
        _write(tmp_path / "mesh", mesh=mesh, arrays=arrays)


def test_writer_detects_shard_report_and_local_payload_inconsistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import femx.backends.jax.port_collective_checkpoint as checkpoint

    mesh = _mesh()
    arrays = {"real-state": _arrays(mesh)["real-state"]}
    original_describe: Any = checkpoint.describe_collective_port_array  # type: ignore[attr-defined]
    report = original_describe("real-state", arrays["real-state"], mesh)
    monkeypatch.setattr(
        checkpoint,
        "describe_collective_port_array",
        lambda name, array, mesh, axis_name="partition": SimpleNamespace(
            process_index=report.process_index,
            process_count=report.process_count,
            addressable_shards=(),
        ),
    )
    with pytest.raises(ContractError, match="shard data disagrees"):
        _write(tmp_path / "report", mesh=mesh, arrays=arrays)

    monkeypatch.setattr(checkpoint, "describe_collective_port_array", original_describe)
    monkeypatch.setattr(
        checkpoint,
        "_canonical_local_array",
        lambda value, expected_dtype, label: np.zeros((2, 1), dtype=np.float64),
    )
    with pytest.raises(ContractError, match="shape or byte count disagrees"):
        _write(tmp_path / "payload", mesh=mesh, arrays=arrays)


def test_strict_json_and_internal_contract_helpers_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key":1,"key":2}', encoding="utf-8")
    with pytest.raises(ContractError, match="repeats JSON key"):
        _strict_json(duplicate)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ContractError, match="not valid JSON"):
        _strict_json(invalid)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * ((1 << 20) + 1))
    with pytest.raises(ContractError, match="exceeds"):
        _strict_json(oversized)
    with pytest.raises(ContractError, match="not a regular file"):
        _strict_json(tmp_path / "missing.json")

    with pytest.raises(ContractError, match="versioned schema fields"):
        _exact_keys({"extra": 1}, frozenset(), label="record")
    with pytest.raises(ContractError, match="canonical finite JSON"):
        _canonical_json({"nan": float("nan")})
    with pytest.raises(ContractError, match="path component"):
        _require_component(1, label="name")
    with pytest.raises(ContractError, match="canonical SHA-256"):
        _require_sha256(None, label="digest")
    with pytest.raises(ContractError, match="nonnegative integer"):
        _require_nonnegative_integer(1.5, label="step")
    with pytest.raises(ContractError, match="unsupported checkpoint dtype"):
        _canonical_local_array(np.ones(1), expected_dtype="uint8", label="state")
    with pytest.raises(ContractError, match="dtype disagrees"):
        _canonical_local_array(np.ones(1), expected_dtype="float32", label="state")


def test_fragment_report_requires_a_json_array_list(tmp_path: Path) -> None:
    manifest: dict[str, object] = {
        "checkpoint_id": CHECKPOINT_ID,
        "step": 7,
        "source_sha256": SOURCE_SHA,
        "config_sha256": CONFIG_SHA,
        "layout_sha256": "c" * 64,
        "process_index": 0,
        "process_count": 1,
        "arrays": {},
    }
    with pytest.raises(ContractError, match="JSON list"):
        _fragment_report(tmp_path, manifest, "d" * 64)
