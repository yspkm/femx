from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import build_tpu_fdtdx_waveguide_source_inputs as builder
from scripts import run_tpu_fdtdx_waveguide_source_evidence as runner

pytestmark = pytest.mark.unit


class _FakeSource:
    def __init__(self, **values: object) -> None:
        self.values = values
        for name, value in values.items():
            setattr(self, name, value)

    def aset(self, name: str, value: object, *, create_new_ok: bool) -> _FakeSource:
        assert create_new_ok
        replacement = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        return _FakeSource(**{**self.values, name: replacement})


class _FakeObjects:
    def __init__(self, source: _FakeSource, *, static_identity: str) -> None:
        self.source = source
        self.static_identity = static_identity

    def __getitem__(self, name: str) -> _FakeSource:
        assert name == "femx-waveguide-port"
        return self.source

    def index(self, name: str) -> int:
        assert name == "femx-waveguide-port"
        return 9

    def aset(self, path: str, source: _FakeSource) -> _FakeObjects:
        assert path == "object_list->[9]"
        return _FakeObjects(source, static_identity=self.static_identity)


def _fake_prepared(static_identity: str) -> tuple[object, ...]:
    fields = {
        name: np.asarray([index], dtype=np.float32)
        for index, name in enumerate(runner._RUNTIME_MODE_SOURCE_FIELDS)
    }
    return (
        object(),
        _FakeObjects(_FakeSource(**fields), static_identity=static_identity),
        object(),
        SimpleNamespace(bundle=object()),
        SimpleNamespace(source_name="femx-waveguide-port"),
        SimpleNamespace(sha256="0" * 64),
    )


def _manifest() -> dict[str, object]:
    return {
        "schema_version": runner.INPUT_MANIFEST_SCHEMA,
        "status": "passed",
        "geometry": {
            "grid_shape_xyz": [64, 52, 36],
            "source_z_index": 6,
            "detector_z_index": 24,
            "core_cells_xy": [8, 4],
            "core_width_m": 0.5e-6,
            "core_height_m": 0.22e-6,
            "core_refractive_index": 3.48,
            "cladding_refractive_index": 1.444,
        },
        "runtime": {
            "fdtdx_fingerprint": {
                "package_version": "0.6.2",
                "source_revision": "81a58da9cde4a4ff822f835b63597c0d0d8ba978",
                "source_digest": (
                    "c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c"
                ),
            }
        },
        "errors": {
            "canonical_source_electric_relative_l2": 1.0e-14,
            "canonical_source_magnetic_relative_l2": 2.0e-14,
        },
    }


def test_waveguide_input_manifest_is_bounded_and_hash_bound(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / runner.INPUT_MANIFEST_NAME
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, digest = runner._load_input_manifest(tmp_path)

    assert loaded == manifest
    assert len(digest) == 64
    runner._verify_manifest_contract(loaded)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["geometry"].__setitem__("grid_shape_xyz", [32, 52, 36]),
            "geometry",
        ),
        (
            lambda value: value["runtime"]["fdtdx_fingerprint"].__setitem__(
                "source_revision", "0" * 40
            ),
            "fingerprint",
        ),
        (
            lambda value: value["errors"].__setitem__(
                "canonical_source_electric_relative_l2", 2.0e-10
            ),
            "parity bound",
        ),
        (
            lambda value: value["errors"].__setitem__(
                "canonical_source_magnetic_relative_l2", "bad"
            ),
            "not finite",
        ),
    ],
)
def test_waveguide_input_manifest_fails_closed(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    manifest = _manifest()
    mutate(manifest)
    path = tmp_path / runner.INPUT_MANIFEST_NAME
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, _digest = runner._load_input_manifest(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        runner._verify_manifest_contract(loaded)


def test_waveguide_input_manifest_rejects_missing_oversized_or_nonfinite_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        runner._load_input_manifest(tmp_path)
    path = tmp_path / runner.INPUT_MANIFEST_NAME
    invalid = _manifest()
    invalid["schema_version"] = "v0"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported"):
        runner._load_input_manifest(tmp_path)
    path.write_text("NaN", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        runner._load_input_manifest(tmp_path)
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "MAXIMUM_MANIFEST_BYTES", 1)
    with pytest.raises(RuntimeError, match="size"):
        runner._load_input_manifest(tmp_path)


def test_exact_source_medium_and_pairwise_error_contract() -> None:
    inverse = runner._source_inverse_permittivity()
    epsilon = 1.0 / inverse[0, :, :, 0]

    assert inverse.shape == (1, 64, 52, 1)
    assert np.count_nonzero(epsilon == runner.CORE_INDEX**2) == 32
    assert np.count_nonzero(epsilon == runner.CLADDING_INDEX**2) == 64 * 52 - 32
    assert runner._relative_l2(np.array([1.0 + 0.0j]), np.array([1.0 + 0.0j])) == 0.0
    with pytest.raises(RuntimeError, match="zero norm"):
        runner._relative_l2(np.ones(1), np.zeros(1))


def test_candidate_source_reuses_one_compiled_scene_pytree() -> None:
    baseline = _fake_prepared("baseline")
    candidate = _fake_prepared("candidate")

    result = runner._reuse_scene_with_candidate_source(
        baseline,
        candidate,
        tree_structure=lambda value: (value[0], value[2], value[1].static_identity),
    )

    assert result[0] is baseline[0]
    assert result[2] is baseline[2]
    assert result[3:] == candidate[3:]
    rebound = result[1]["femx-waveguide-port"]
    candidate_source = candidate[1]["femx-waveguide-port"]
    for field in runner._RUNTIME_MODE_SOURCE_FIELDS:
        assert getattr(rebound, field) is not getattr(candidate_source, field)
        assert runner._same_addressable_runtime_leaf(
            getattr(rebound, field), getattr(candidate_source, field)
        )


def test_runtime_leaf_comparison_rejects_value_shape_and_dtype_changes() -> None:
    reference = np.asarray([1.0, 2.0], dtype=np.float32)

    assert runner._same_addressable_runtime_leaf(reference, reference.copy())
    assert not runner._same_addressable_runtime_leaf(reference, np.asarray([1.0, 3.0]))
    assert not runner._same_addressable_runtime_leaf(reference, np.asarray([[1.0, 2.0]]))
    assert not runner._same_addressable_runtime_leaf(reference, object())


def test_candidate_source_rebinding_fails_closed() -> None:
    baseline = _fake_prepared("baseline")
    candidate = list(_fake_prepared("candidate"))
    candidate[4] = SimpleNamespace(source_name="different")
    with pytest.raises(RuntimeError, match="canonical source name"):
        runner._reuse_scene_with_candidate_source(
            baseline,
            tuple(candidate),
            tree_structure=lambda value: value,
        )

    candidate = _fake_prepared("candidate")
    structures = iter(("baseline", "changed"))
    with pytest.raises(RuntimeError, match="pytree structure"):
        runner._reuse_scene_with_candidate_source(
            baseline,
            candidate,
            tree_structure=lambda _value: next(structures),
        )


def test_builder_makes_exact_64_by_52_waveguide_grid() -> None:
    validated = SimpleNamespace(coordinates=np.array([[-2.0e-6, -1.5e-6], [2.0e-6, 1.5e-6]]))
    recipe = builder.RectangularWaveguideCrossSection()

    x_edges, y_edges, z_edges = builder._fdtd_edges(validated, recipe)

    assert (x_edges.size - 1, y_edges.size - 1, z_edges.size - 1) == (64, 52, 36)
    assert x_edges[28] == pytest.approx(-0.25e-6, abs=2.0e-12)
    assert x_edges[36] == pytest.approx(0.25e-6, abs=3.0e-12)
    assert y_edges[24] == pytest.approx(-0.11e-6, abs=3.0e-12)
    assert y_edges[28] == pytest.approx(0.11e-6, abs=3.0e-12)


def test_builder_atomic_json_refuses_to_replace_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    builder._atomic_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        builder._atomic_json(path, {"value": 2})
