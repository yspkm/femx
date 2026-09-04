from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

import femx.interop.fdtdx.mode_hdf5 as mode_hdf5
from femx.artifacts import ArtifactRef, ArtifactRole, sha256_file
from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.errors import ArtifactError
from femx.interop.fdtdx import (
    MODE_HDF5_MEDIA_TYPE,
    FDTDXFingerprint,
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeBundleHDF5Artifact,
    ModeNormalization,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeGrid,
    YeeVectorField,
    build_yee_grid,
    read_mode_bundle_hdf5,
    write_mode_bundle_hdf5,
)

pytestmark = pytest.mark.unit

SHA = "a" * 64
COMPLEX128 = np.dtype(np.complex128)


def _bundle(*, dtype: np.dtype = COMPLEX128) -> ModeBundle:
    grid = build_yee_grid(
        (
            np.asarray((-0.8e-6, -0.1e-6, 0.9e-6)),
            np.asarray((-0.7e-6, -0.2e-6, 0.4e-6, 0.8e-6)),
            np.asarray((0.0, 20.0e-9)),
        )
    )
    shape = (3, *grid.shape)
    base = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
    electric = np.asarray((base + 1.0) + 1j * (0.25 * base - 0.5), dtype=dtype)
    magnetic = np.asarray((0.5 * base + 2.0) + 1j * (base + 0.75), dtype=dtype)
    target_power = 1.0
    pre_correction_power = 1.21
    fdtdx = FDTDXFingerprint("0.6.2", "b" * 40, "c" * 64)
    return ModeBundle(
        frequency_hz=193.414e12,
        effective_index=2.419 + 2.0e-6j,
        beta_per_m=9.806e6 + 8.1j,
        electric=YeeVectorField(electric, grid, YeeFieldKind.ELECTRIC, "V/m"),
        magnetic=YeeVectorField(magnetic, grid, YeeFieldKind.MAGNETIC, "V/m"),
        propagation=AxisDirection(Axis.Z, Direction.POSITIVE),
        magnetic_convention=MagneticFieldConvention.ETA0_H,
        normalization=ModeNormalization(target_power_watts=target_power),
        solver=SolverFingerprint("jax-port-eigenmode", "1", SHA, "d" * 64, "e" * 40),
        transfer=TransferReport(
            source_representation=FieldRepresentation.FEM_DOFS,
            target_representation=FieldRepresentation.CARTESIAN_YEE_SAMPLES,
            operator_sha256="f" * 64,
            relative_power_error=0.0,
            relative_interpolation_error=0.0045,
            source_power_watts=target_power,
            pre_correction_power_watts=pre_correction_power,
            relative_pre_correction_power_error=0.21,
            transferred_power_watts=target_power,
            power_correction_scale=math.sqrt(target_power / pre_correction_power),
            target_runtime_name="fdtdx",
            target_runtime_version=fdtdx.package_version,
            target_source_revision=fdtdx.source_revision,
            target_source_digest=fdtdx.source_digest,
        ),
    )


def _refreshed(reference: ArtifactRef, run_root: Path) -> ArtifactRef:
    path = run_root / reference.path
    return replace(reference, sha256=sha256_file(path), size_bytes=path.stat().st_size)


def _rewrite_metadata(path: Path, transform) -> None:
    with h5py.File(path, "r+") as handle:
        raw = bytes(np.asarray(handle["metadata_json"][...], dtype=np.uint8).tobytes())
        data = json.loads(raw.decode("utf-8"))
        transformed = transform(data)
        encoded = json.dumps(
            transformed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        del handle["metadata_json"]
        handle.create_dataset("metadata_json", data=np.frombuffer(encoded, dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.dtype(np.complex64), np.dtype(np.complex128)])
def test_mode_hdf5_round_trip_preserves_every_convention_and_dtype(
    tmp_path: Path,
    dtype: np.dtype[np.complexfloating],
) -> None:
    bundle = _bundle(dtype=dtype)
    if dtype == np.dtype(np.complex64):
        bundle = replace(
            bundle,
            solver=replace(bundle.solver, source_revision=None),
            transfer=replace(bundle.transfer, relative_interpolation_error=None),
        )
    run_root = tmp_path / "run"
    run_root.mkdir()

    written = write_mode_bundle_hdf5(run_root, "modes/port.h5", bundle)
    loaded = read_mode_bundle_hdf5(run_root, written.reference)

    assert written.reference.role is ArtifactRole.CANONICAL_NUMERICAL
    assert written.reference.media_type == MODE_HDF5_MEDIA_TYPE
    assert written.reference.sha256 == sha256_file(run_root / written.reference.path)
    assert np.asarray(bundle.electric.values).flags.writeable
    assert np.asarray(bundle.magnetic.values).flags.writeable
    assert written.bundle.electric.values.flags.writeable is False
    assert written.bundle.magnetic.values.flags.writeable is False
    assert written.bundle.electric.values is not bundle.electric.values
    np.testing.assert_array_equal(written.bundle.electric.values, bundle.electric.values)
    np.testing.assert_array_equal(written.bundle.magnetic.values, bundle.magnetic.values)
    assert loaded.content_sha256 == written.content_sha256
    assert loaded.logical_data_bytes == written.logical_data_bytes
    assert loaded.bundle.frequency_hz == bundle.frequency_hz
    assert loaded.bundle.effective_index == bundle.effective_index
    assert loaded.bundle.beta_per_m == bundle.beta_per_m
    assert loaded.bundle.propagation == bundle.propagation
    assert loaded.bundle.magnetic_convention is MagneticFieldConvention.ETA0_H
    assert loaded.bundle.normalization == bundle.normalization
    assert loaded.bundle.solver == bundle.solver
    assert loaded.bundle.transfer == bundle.transfer
    for actual, expected in zip(
        loaded.bundle.electric.grid.edge_coordinates,
        bundle.electric.grid.edge_coordinates,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable is False
    np.testing.assert_array_equal(loaded.bundle.electric.values, bundle.electric.values)
    np.testing.assert_array_equal(loaded.bundle.magnetic.values, bundle.magnetic.values)
    assert loaded.bundle.electric.values.dtype == dtype
    assert loaded.bundle.magnetic.values.dtype == dtype
    assert loaded.bundle.electric.values.flags.writeable is False

    with h5py.File(run_root / written.reference.path, "r") as handle:
        assert handle.attrs["container_schema"] == "femx.mode.hdf5/v1"
        assert handle["fields/electric"].compression == "gzip"
        assert handle["fields/electric"].shuffle
        assert handle["fields/electric"].fletcher32
        assert handle["grid/x_edges_m"].fletcher32

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = write_mode_bundle_hdf5(second_root, "same.hdf5", bundle)
    assert second.content_sha256 == written.content_sha256


def test_mode_hdf5_writer_is_fail_closed_and_never_overwrites(tmp_path: Path) -> None:
    bundle = _bundle()
    run_root = tmp_path / "run"
    run_root.mkdir()
    written = write_mode_bundle_hdf5(run_root, "mode.h5", bundle)
    original = (run_root / written.reference.path).read_bytes()

    with pytest.raises(ArtifactError, match="already exists"):
        write_mode_bundle_hdf5(run_root, "mode.h5", bundle)
    assert (run_root / written.reference.path).read_bytes() == original

    for relative_path in (
        "",
        " ../mode.h5",
        "../mode.h5",
        "/mode.h5",
        "a\\mode.h5",
        "./mode.h5",
        "nested//mode.h5",
        "mode.bin",
    ):
        with pytest.raises(ArtifactError, match=r"path|end in"):
            write_mode_bundle_hdf5(run_root, relative_path, bundle)
    with pytest.raises(ArtifactError, match="run root"):
        write_mode_bundle_hdf5(tmp_path / "missing", "mode.h5", bundle)
    with pytest.raises(ArtifactError, match="requires a ModeBundle"):
        write_mode_bundle_hdf5(run_root, "bad.h5", object())  # type: ignore[arg-type]

    outside = tmp_path / "outside"
    outside.mkdir()
    (run_root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError, match="escapes"):
        write_mode_bundle_hdf5(run_root, "escape/mode.h5", bundle)


def test_mode_hdf5_writer_revalidates_grid_and_field_data(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle = _bundle()
    bad_digest_grid = replace(bundle.electric.grid, coordinate_sha256="0" * 64)
    bad_digest_bundle = replace(
        bundle,
        electric=replace(bundle.electric, grid=bad_digest_grid),
        magnetic=replace(bundle.magnetic, grid=bad_digest_grid),
    )
    with pytest.raises(ArtifactError, match="coordinate digest"):
        write_mode_bundle_hdf5(run_root, "bad-digest.h5", bad_digest_bundle)

    changed_axes = list(bundle.magnetic.grid.edge_coordinates)
    changed_axes[0] = np.asarray((-0.8e-6, 0.0, 0.9e-6))
    mismatched_grid = YeeGrid(
        tuple(changed_axes),  # type: ignore[arg-type]
        bundle.electric.grid.coordinate_sha256,
    )
    mismatched_bundle = replace(bundle, magnetic=replace(bundle.magnetic, grid=mismatched_grid))
    with pytest.raises(ArtifactError, match="different Yee coordinates"):
        write_mode_bundle_hdf5(run_root, "mismatch.h5", mismatched_bundle)

    nonfinite = np.asarray(bundle.electric.values).copy()
    nonfinite.flat[0] = complex(math.nan, 0.0)
    with pytest.raises(ArtifactError, match="finite values"):
        write_mode_bundle_hdf5(
            run_root,
            "nonfinite.h5",
            replace(bundle, electric=replace(bundle.electric, values=nonfinite)),
        )

    extended = np.asarray(bundle.electric.values, dtype=np.clongdouble)
    if extended.dtype.itemsize not in (8, 16):
        with pytest.raises(ArtifactError, match="complex64 or complex128"):
            write_mode_bundle_hdf5(
                run_root,
                "extended.h5",
                replace(
                    bundle,
                    electric=replace(bundle.electric, values=extended),
                    magnetic=replace(
                        bundle.magnetic,
                        values=np.asarray(bundle.magnetic.values, dtype=np.clongdouble),
                    ),
                ),
            )

    with pytest.raises(ArtifactError, match="one-dimensional"):
        mode_hdf5._canonical_axis(np, np.zeros((2, 2)))
    with pytest.raises(ArtifactError, match="strictly increasing"):
        mode_hdf5._canonical_axis(np, np.asarray((0.0, 0.0)))


def test_mode_hdf5_optional_dependency_and_write_failure_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    original_import = mode_hdf5.import_module

    def missing_h5py(name: str):
        if name == "h5py":
            raise ImportError("missing")
        return original_import(name)

    monkeypatch.setattr(mode_hdf5, "import_module", missing_h5py)
    with pytest.raises(ArtifactError, match=r"femx\[artifacts\]"):
        write_mode_bundle_hdf5(run_root, "mode.h5", _bundle())
    monkeypatch.setattr(mode_hdf5, "import_module", original_import)

    class BrokenH5py:
        def File(self, *_args, **_kwargs):
            raise OSError("synthetic write failure")

    monkeypatch.setattr(mode_hdf5, "_numeric_modules", lambda: (BrokenH5py(), np))
    with pytest.raises(ArtifactError, match="failed to write"):
        write_mode_bundle_hdf5(run_root, "broken.h5", _bundle())
    assert not (run_root / "broken.h5").exists()
    assert list(run_root.glob(".broken.h5.*.tmp")) == []

    target = run_root / "appeared.h5"

    class RacingFile:
        def __init__(self, *args, **kwargs) -> None:
            self._file = h5py.File(*args, **kwargs)

        def __enter__(self):
            return self._file.__enter__()

        def __exit__(self, *args) -> bool:
            result = self._file.__exit__(*args)
            target.write_bytes(b"concurrent writer")
            return result

    class RacingH5py:
        File = RacingFile

    monkeypatch.setattr(mode_hdf5, "_numeric_modules", lambda: (RacingH5py(), np))
    with pytest.raises(ArtifactError, match="appeared during write"):
        write_mode_bundle_hdf5(run_root, "appeared.h5", _bundle())
    assert target.read_bytes() == b"concurrent writer"
    assert list(run_root.glob(".appeared.h5.*.tmp")) == []
    target.unlink()

    monkeypatch.setattr(mode_hdf5, "_numeric_modules", lambda: (h5py, np))

    def occupied_link(_source: Path, _target: Path) -> None:
        raise FileExistsError("synthetic atomic race")

    monkeypatch.setattr(mode_hdf5.os, "link", occupied_link)
    with pytest.raises(ArtifactError, match="appeared during write"):
        write_mode_bundle_hdf5(run_root, "atomic-race.h5", _bundle())
    assert not (run_root / "atomic-race.h5").exists()
    assert list(run_root.glob(".atomic-race.h5.*.tmp")) == []


def test_mode_hdf5_reader_validates_external_reference_and_limits(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    written = write_mode_bundle_hdf5(run_root, "mode.h5", _bundle())
    reference = written.reference

    with pytest.raises(ArtifactError, match="canonical numerical"):
        read_mode_bundle_hdf5(run_root, replace(reference, role=ArtifactRole.RAW_PROVENANCE))
    with pytest.raises(ArtifactError, match="media type"):
        read_mode_bundle_hdf5(run_root, replace(reference, media_type="application/octet-stream"))
    with pytest.raises(ArtifactError, match="size or SHA"):
        read_mode_bundle_hdf5(run_root, replace(reference, sha256="0" * 64))
    with pytest.raises(ArtifactError, match="size or SHA"):
        read_mode_bundle_hdf5(run_root, replace(reference, size_bytes=reference.size_bytes + 1))
    with pytest.raises(ArtifactError, match="positive"):
        read_mode_bundle_hdf5(run_root, reference, maximum_data_bytes=0)
    with pytest.raises(ArtifactError, match="positive"):
        read_mode_bundle_hdf5(run_root, reference, maximum_data_bytes=True)
    with pytest.raises(ArtifactError, match="exceeds limit"):
        read_mode_bundle_hdf5(run_root, reference, maximum_data_bytes=1)

    missing = replace(reference, path="missing.h5")
    with pytest.raises(ArtifactError, match="non-symlink"):
        read_mode_bundle_hdf5(run_root, missing)
    symlink = run_root / "link.h5"
    symlink.symlink_to(run_root / reference.path)
    with pytest.raises(ArtifactError, match="non-symlink"):
        read_mode_bundle_hdf5(run_root, replace(reference, path="link.h5"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda h: h.attrs.__setitem__("container_schema", "femx.mode.hdf5/v2"), "schema"),
        (lambda h: h.attrs.__setitem__("content_sha256", "bad"), "content digest"),
        (lambda h: h.create_group("unexpected"), "members differ"),
        (lambda h: h["fields"].__delitem__("magnetic"), "members differ"),
        (
            lambda h: (
                h["grid"].__delitem__("x_edges_m"),
                h["grid"].__setitem__("x_edges_m", h5py.SoftLink("/grid/y_edges_m")),
            ),
            "internal hard link",
        ),
        (lambda h: h["fields/electric"].__setitem__((0, 0, 0, 0), 999 + 1j), "array metadata"),
        (
            lambda h: (
                h["fields"].__delitem__("electric"),
                h["fields"].create_group("electric"),
            ),
            "wrong object type",
        ),
    ],
)
def test_mode_hdf5_reader_rejects_tampered_hdf5_structure(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    written = write_mode_bundle_hdf5(run_root, "mode.h5", _bundle())
    with h5py.File(run_root / written.reference.path, "r+") as handle:
        mutation(handle)
    refreshed = _refreshed(written.reference, run_root)
    with pytest.raises(ArtifactError, match=message):
        read_mode_bundle_hdf5(run_root, refreshed)


def test_mode_hdf5_reader_rejects_metadata_tampering_and_invalid_files(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    written = write_mode_bundle_hdf5(run_root, "mode.h5", _bundle())
    path = run_root / written.reference.path

    _rewrite_metadata(path, lambda data: {**data, "frequency_hz": "193414000000000.0"})
    with pytest.raises(ArtifactError, match="reproducible"):
        read_mode_bundle_hdf5(run_root, _refreshed(written.reference, run_root))

    coordinate_root = tmp_path / "coordinate"
    coordinate_root.mkdir()
    coordinate = write_mode_bundle_hdf5(coordinate_root, "mode.h5", _bundle())
    _rewrite_metadata(
        coordinate_root / coordinate.reference.path,
        lambda data: {
            **data,
            "grid": {**data["grid"], "coordinate_sha256": "0" * 64},
        },
    )
    with pytest.raises(ArtifactError, match="coordinate digest"):
        read_mode_bundle_hdf5(coordinate_root, _refreshed(coordinate.reference, coordinate_root))

    axes_root = tmp_path / "axes"
    axes_root.mkdir()
    axes = write_mode_bundle_hdf5(axes_root, "mode.h5", _bundle())
    _rewrite_metadata(
        axes_root / axes.reference.path,
        lambda data: {
            **data,
            "grid": {**data["grid"], "axes": data["grid"]["axes"][:2]},
        },
    )
    with pytest.raises(ArtifactError, match="three grid axes"):
        read_mode_bundle_hdf5(axes_root, _refreshed(axes.reference, axes_root))

    mapping_root = tmp_path / "mapping"
    mapping_root.mkdir()
    mapping = write_mode_bundle_hdf5(mapping_root, "mode.h5", _bundle())
    _rewrite_metadata(mapping_root / mapping.reference.path, lambda _data: [])
    with pytest.raises(ArtifactError, match="root must be an object"):
        read_mode_bundle_hdf5(mapping_root, _refreshed(mapping.reference, mapping_root))

    other_root = tmp_path / "other"
    other_root.mkdir()
    other = write_mode_bundle_hdf5(other_root, "mode.h5", _bundle())
    other_path = other_root / other.reference.path
    with h5py.File(other_path, "r+") as handle:
        del handle["metadata_json"]
        handle.create_dataset("metadata_json", data=np.asarray((0xFF,), dtype=np.uint8))
    with pytest.raises(ArtifactError, match="UTF-8 JSON"):
        read_mode_bundle_hdf5(other_root, _refreshed(other.reference, other_root))

    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    corrupt_path = corrupt_root / "mode.h5"
    corrupt_path.write_bytes(b"not hdf5")
    corrupt_ref = ArtifactRef(
        "mode.h5",
        ArtifactRole.CANONICAL_NUMERICAL,
        MODE_HDF5_MEDIA_TYPE,
        sha256_file(corrupt_path),
        corrupt_path.stat().st_size,
    )
    with pytest.raises(ArtifactError, match="invalid mode HDF5"):
        read_mode_bundle_hdf5(corrupt_root, corrupt_ref)


def test_mode_hdf5_reader_rejects_layout_content_and_read_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout_root = tmp_path / "layout"
    layout_root.mkdir()
    layout = write_mode_bundle_hdf5(layout_root, "mode.h5", _bundle())
    with h5py.File(layout_root / layout.reference.path, "r+") as handle:
        del handle["metadata_json"]
        handle.create_dataset("metadata_json", data=np.asarray(1, dtype=np.uint8))
    with pytest.raises(ArtifactError, match="invalid layout"):
        read_mode_bundle_hdf5(layout_root, _refreshed(layout.reference, layout_root))

    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    canonical = write_mode_bundle_hdf5(canonical_root, "mode.h5", _bundle())
    with h5py.File(canonical_root / canonical.reference.path, "r+") as handle:
        raw = bytes(np.asarray(handle["metadata_json"][...], dtype=np.uint8).tobytes())
        decoded = json.loads(raw.decode("utf-8"))
        encoded = json.dumps(decoded, ensure_ascii=False).encode("utf-8")
        del handle["metadata_json"]
        handle.create_dataset("metadata_json", data=np.frombuffer(encoded, dtype=np.uint8))
    with pytest.raises(ArtifactError, match="not canonical"):
        read_mode_bundle_hdf5(
            canonical_root,
            _refreshed(canonical.reference, canonical_root),
        )

    digest_root = tmp_path / "digest"
    digest_root.mkdir()
    digest = write_mode_bundle_hdf5(digest_root, "mode.h5", _bundle())
    with h5py.File(digest_root / digest.reference.path, "r+") as handle:
        handle.attrs["content_sha256"] = "0" * 64
    with pytest.raises(ArtifactError, match="logical content digest"):
        read_mode_bundle_hdf5(digest_root, _refreshed(digest.reference, digest_root))

    external_root = tmp_path / "external"
    external_root.mkdir()
    external = write_mode_bundle_hdf5(external_root, "mode.h5", _bundle())
    with h5py.File(external_root / external.reference.path, "r+") as handle:
        values = handle["grid/x_edges_m"][...]
        del handle["grid/x_edges_m"]
        dataset = handle["grid"].create_dataset(
            "x_edges_m",
            shape=values.shape,
            dtype=values.dtype,
            external=[(str(external_root / "axis.raw"), 0, h5py.h5f.UNLIMITED)],
        )
        dataset[...] = values
    with pytest.raises(ArtifactError, match="external storage"):
        read_mode_bundle_hdf5(external_root, _refreshed(external.reference, external_root))

    virtual_root = tmp_path / "virtual"
    virtual_root.mkdir()
    virtual = write_mode_bundle_hdf5(virtual_root, "mode.h5", _bundle())
    source_path = virtual_root / "axis-source.h5"
    with h5py.File(source_path, "w") as source:
        source.create_dataset("axis", data=np.asarray((-0.8e-6, -0.1e-6, 0.9e-6)))
    with h5py.File(virtual_root / virtual.reference.path, "r+") as handle:
        values = handle["grid/x_edges_m"][...]
        del handle["grid/x_edges_m"]
        layout = h5py.VirtualLayout(shape=values.shape, dtype=values.dtype)
        layout[:] = h5py.VirtualSource(str(source_path), "axis", shape=values.shape)
        handle["grid"].create_virtual_dataset("x_edges_m", layout)
    with pytest.raises(ArtifactError, match="virtual storage"):
        read_mode_bundle_hdf5(virtual_root, _refreshed(virtual.reference, virtual_root))

    race_root = tmp_path / "race"
    race_root.mkdir()
    race = write_mode_bundle_hdf5(race_root, "mode.h5", _bundle())
    calls = 0
    original_sha256_file = mode_hdf5.sha256_file

    def changing_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_sha256_file(path, chunk_size=chunk_size)
        return "0" * 64

    monkeypatch.setattr(mode_hdf5, "sha256_file", changing_sha256)
    with pytest.raises(ArtifactError, match="changed while"):
        read_mode_bundle_hdf5(race_root, race.reference)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"bundle": object()}, "ModeBundle"),
        (
            {"reference": ArtifactRef("mode.h5", ArtifactRole.LOG, MODE_HDF5_MEDIA_TYPE, SHA, 1)},
            "canonical numerical",
        ),
        (
            {
                "reference": ArtifactRef(
                    "mode.h5", ArtifactRole.CANONICAL_NUMERICAL, "bad/type", SHA, 1
                )
            },
            "media type",
        ),
        ({"content_sha256": "bad"}, "content digest"),
        ({"logical_data_bytes": 0}, "data size"),
        ({"container_schema": "v2"}, "unsupported"),
    ],
)
def test_mode_hdf5_artifact_record_rejects_invalid_identity(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "bundle": _bundle(),
        "reference": ArtifactRef(
            "mode.h5", ArtifactRole.CANONICAL_NUMERICAL, MODE_HDF5_MEDIA_TYPE, SHA, 1
        ),
        "content_sha256": SHA,
        "logical_data_bytes": 1,
    }
    values.update(changes)
    with pytest.raises(ArtifactError, match=message):
        ModeBundleHDF5Artifact(**values)  # type: ignore[arg-type]
