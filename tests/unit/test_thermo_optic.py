from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import femx.interop.fdtdx.thermo_optic as module  # noqa: E402
from femx.core.errors import ContractError  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXDeviceParameterContract,
    FDTDXFingerprint,
    ThermoOpticLaw,
    apply_thermo_optic_to_fdtdx,
    build_triangle_p1_sampling_plan,
    sample_triangle_p1,
    target_coordinate_digest,
    thermo_optic_parameter_state,
    with_fdtdx_device_parameter,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

REVISION = "eaab78a42cd1351b7f447f312fa50c9febfe4b99"
SOURCE_DIGEST = "cf7bf29a1aa2411f2ffc84dcf2c1806d43d1823e32a60f234134298171da7d08"


def _fingerprint() -> FDTDXFingerprint:
    return FDTDXFingerprint("0.6.2", REVISION, SOURCE_DIGEST)


def _law() -> ThermoOpticLaw:
    return ThermoOpticLaw(
        material_region="silicon",
        reference_temperature_k=300.0,
        reference_refractive_index=2.0,
        thermo_optic_coefficient_per_k=1.0e-2,
        vacuum_wavelength_m=1.55e-6,
    )


def _plan():
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
    target = (
        np.asarray((0.25, 0.75)),
        np.asarray((-0.5, 0.5)),
        np.asarray((0.25, 0.75)),
    )
    return build_triangle_p1_sampling_plan(
        coordinates,
        cells,
        target,
        plane_axes=(0, 2),
    )


def _contract(*, dtype: str = "float64") -> FDTDXDeviceParameterContract:
    plan = _plan()
    return FDTDXDeviceParameterContract(
        device_name="heated-silicon",
        target_shape=plan.target_shape,
        plane_axes=plan.plane_axes,
        lower_relative_permittivity=3.5,
        upper_relative_permittivity=5.0,
        parameter_dtype=dtype,
        thermo_optic_law_sha256=_law().sha256,
        target_coordinate_sha256=plan.target_coordinate_sha256,
        transfer_operator_sha256=plan.operator_sha256,
        fdtdx=_fingerprint(),
    )


def _isotropic_property(value: float) -> tuple[float, ...]:
    return (value, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, value)


class _Material:
    def __init__(
        self,
        permittivity: float,
        *,
        permeability: float = 1.0,
        electric_conductivity: float = 0.0,
        magnetic_conductivity: float = 0.0,
        dispersion: object | None = None,
    ) -> None:
        self.permittivity = _isotropic_property(permittivity)
        self.permeability = _isotropic_property(permeability)
        self.electric_conductivity = _isotropic_property(electric_conductivity)
        self.magnetic_conductivity = _isotropic_property(magnetic_conductivity)
        self.dispersion = dispersion


class _Grid:
    def __init__(self, axes) -> None:
        self._axes = axes

    def centers(self, axis: int):
        return self._axes[axis]


def _runtime(contract: FDTDXDeviceParameterContract):
    plan = _plan()
    device = SimpleNamespace(
        name=contract.device_name,
        matrix_voxel_grid_shape=contract.target_shape,
        single_voxel_grid_shape=(1, 1, 1),
        use_etching=False,
        param_transforms=(),
        materials={"lower": _Material(3.5), "upper": _Material(5.0)},
        grid_slice_tuple=((0, 2), (0, 2), (0, 2)),
    )
    arrays = SimpleNamespace(inv_permittivities=jnp.ones((1, 2, 2, 2)))
    objects = SimpleNamespace(devices=[device], sources=[])
    config = SimpleNamespace(resolved_grid=_Grid(plan.target_coordinates))
    parameters = {
        contract.device_name: jnp.zeros(contract.target_shape, dtype=contract.parameter_dtype)
    }
    return arrays, objects, config, parameters, device


def test_contracts_preserve_physical_and_runtime_identity() -> None:
    plan = _plan()
    contract = _contract()

    assert plan.target_shape == (2, 2, 2)
    assert plan.maximum_partition_error < 2.0e-16
    assert plan.minimum_barycentric_weight >= 0.0
    assert len(plan.source_mesh_sha256) == 64
    assert target_coordinate_digest(plan.target_coordinates) == plan.target_coordinate_sha256
    assert _law().canonical_data()["loss_model"] == "none"
    assert len(_law().sha256) == 64
    metadata = contract.canonical_data()
    assert metadata["parameter_semantics"] == "linear_relative_permittivity_fraction"
    assert metadata["out_of_range_policy"] == "nan_no_clipping"
    assert metadata["fdtdx_source_revision"] == REVISION


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("", REVISION, SOURCE_DIGEST), "package version"),
        (("0.6.2", "A" * 40, SOURCE_DIGEST), "source revision"),
        (("0.6.2", REVISION, "0" * 63), "source digest"),
    ),
)
def test_fdtdx_fingerprint_rejects_ambiguous_identity(arguments, message) -> None:
    with pytest.raises(ContractError, match=message):
        FDTDXFingerprint(*arguments)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"material_region": " silicon"}, "material region"),
        ({"reference_temperature_k": np.nan}, "reference temperature"),
        ({"reference_temperature_k": 0.0}, "reference temperature must be positive"),
        ({"reference_refractive_index": np.inf}, "reference refractive index"),
        ({"reference_refractive_index": 0.0}, "refractive index must be positive"),
        ({"thermo_optic_coefficient_per_k": np.nan}, "coefficient"),
        ({"vacuum_wavelength_m": np.inf}, "vacuum wavelength"),
        ({"vacuum_wavelength_m": 0.0}, "vacuum wavelength must be positive"),
        ({"schema_version": "future"}, "unsupported thermo-optic"),
    ),
)
def test_thermo_optic_law_rejects_invalid_metadata(changes, message) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_law(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"device_name": ""}, "device name"),
        ({"target_shape": (2, 0, 2)}, "target shape"),
        ({"plane_axes": (0, 0)}, "distinct axes"),
        ({"plane_axes": (0, 3)}, "x/y/z"),
        ({"lower_relative_permittivity": 0.0}, "lower relative"),
        ({"upper_relative_permittivity": np.nan}, "upper relative"),
        ({"lower_relative_permittivity": 5.0}, "strictly ordered"),
        ({"parameter_dtype": "bfloat16"}, "float32 or float64"),
        ({"thermo_optic_law_sha256": "0" * 63}, "physical-law"),
        ({"target_coordinate_sha256": "0" * 63}, "target-coordinate"),
        ({"transfer_operator_sha256": "0" * 63}, "transfer-operator"),
        ({"schema_version": "future"}, "unsupported FDTDX"),
    ),
)
def test_device_contract_rejects_silent_semantic_changes(changes, message) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_contract(), **changes)


def test_sampling_plan_is_deterministic_and_affine_exact() -> None:
    plan = _plan()
    second = _plan()
    coordinates = np.asarray(plan.source_coordinates)
    nodal = jnp.asarray(300.0 + 2.0 * coordinates[:, 0] + 3.0 * coordinates[:, 1])
    sampled = np.asarray(sample_triangle_p1(plan, nodal))
    x, _y, z = np.meshgrid(*plan.target_coordinates, indexing="ij")

    np.testing.assert_allclose(sampled, 300.0 + 2.0 * x + 3.0 * z, atol=2.0e-13)
    assert plan.source_mesh_sha256 == second.source_mesh_sha256
    assert plan.target_coordinate_sha256 == second.target_coordinate_sha256
    assert plan.operator_sha256 == second.operator_sha256


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"source_coordinates": np.zeros((4, 3))}, "source coordinates"),
        ({"source_cells": np.zeros((2, 4), dtype=int)}, "source cells"),
        ({"source_cells": np.empty((0, 3), dtype=int)}, "at least one"),
        ({"plane_axes": (0, 0)}, "distinct axes"),
        ({"plane_axes": (0, 3)}, "x/y/z"),
        (
            {"target_coordinates": (np.zeros((2, 1)), np.zeros((2,)), np.zeros((2,)))},
            "one-dimensional",
        ),
        (
            {"target_coordinates": (np.zeros((0,)), np.zeros((2,)), np.zeros((2,)))},
            "cannot be empty",
        ),
        ({"target_cell_indices": np.zeros((1, 2, 2), dtype=int)}, "cell-index"),
        ({"barycentric_weights": np.zeros((2, 2, 2, 2))}, "barycentric"),
        ({"source_cells": np.zeros((2, 3), dtype=float)}, "integer dtype"),
        ({"target_cell_indices": np.zeros((2, 2, 2), dtype=float)}, "cell indices"),
        ({"source_mesh_sha256": "0" * 63}, "source mesh digest"),
        ({"containment_tolerance": np.nan}, "must be finite"),
        ({"containment_tolerance": 0.0}, "must be positive"),
        ({"maximum_partition_error": -1.0}, "cannot be negative"),
        ({"minimum_barycentric_weight": -1.0}, "outside"),
        ({"schema_version": "future"}, "unsupported P1"),
    ),
)
def test_sampling_plan_rejects_corrupted_operator_metadata(changes, message) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_plan(), **changes)


def test_target_coordinate_digest_requires_xyz_axes() -> None:
    with pytest.raises(ContractError, match="three axes"):
        target_coordinate_digest((np.zeros((1,)), np.zeros((1,))))


@pytest.mark.parametrize(
    ("coordinates", "cells", "targets", "axes", "tolerance", "message"),
    (
        (
            np.zeros((2, 3)),
            np.asarray(((0, 1, 1),)),
            ((0.5,), (0.0,), (0.5,)),
            (0, 2),
            1e-12,
            "coordinates",
        ),
        (
            np.asarray(((0j, 0j), (1j, 0j), (0j, 1j))),
            np.asarray(((0, 1, 2),)),
            ((0.2,), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "finite real",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, np.nan))),
            np.asarray(((0, 1, 2),)),
            ((0.2,), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "finite real",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.empty((0, 3), dtype=int),
            ((0.2,), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "non-empty",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0.0, 1.0, 2.0),)),
            ((0.2,), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "integer dtype",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 3),)),
            ((0.2,), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "node range",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            ((0.2,), (0.0,), (0.2,)),
            (0, 0),
            1e-12,
            "plane_axes",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            ((0.2,), (0.0,), (0.2,)),
            (0, 2),
            0.0,
            "tolerance",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            ((0.2,), (0.0,)),
            (0, 2),
            1e-12,
            "three axes",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            (np.asarray(()), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "non-empty vector",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            (np.asarray((0.2j,)), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "finite real",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            ((0.2, 0.1), (0.0,), (0.2,)),
            (0, 2),
            1e-12,
            "strictly increasing",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            np.asarray(((0, 1, 2),)),
            ((1.2,), (0.0,), (1.2,)),
            (0, 2),
            1e-12,
            "outside",
        ),
        (
            np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            np.asarray(((0, 1, 2),)),
            ((0.2,), (0.0,), (0.0,)),
            (0, 2),
            1e-12,
            "degenerate",
        ),
    ),
)
def test_sampling_builder_rejects_invalid_or_uncovered_geometry(
    coordinates, cells, targets, axes, tolerance, message
) -> None:
    with pytest.raises(ContractError, match=message):
        build_triangle_p1_sampling_plan(
            coordinates,
            cells,
            targets,
            plane_axes=axes,
            containment_tolerance=tolerance,
        )


def test_sampling_and_state_reject_contract_mismatches() -> None:
    plan = _plan()
    nodal = jnp.full((4,), 300.0)
    contract = _contract()
    with pytest.raises(ContractError, match="shape"):
        sample_triangle_p1(plan, jnp.ones((3,)))
    with pytest.raises(ContractError, match="floating"):
        sample_triangle_p1(plan, jnp.ones((4,), dtype=jnp.int32))
    for changed, message in (
        (replace(contract, target_shape=(1, 2, 2)), "target shape"),
        (replace(contract, plane_axes=(1, 2)), "plane axes"),
        (replace(contract, target_coordinate_sha256="0" * 64), "target coordinates"),
        (replace(contract, transfer_operator_sha256="0" * 64), "transfer operator"),
        (replace(contract, thermo_optic_law_sha256="0" * 64), "physical law"),
    ):
        with pytest.raises(ContractError, match=message):
            thermo_optic_parameter_state(plan, nodal, _law(), changed)


def test_thermo_optic_state_is_explicitly_typed_and_never_clips() -> None:
    plan = _plan()
    valid = thermo_optic_parameter_state(plan, jnp.full((4,), 302.0), _law(), _contract())
    expected_epsilon = (2.0 + 0.01 * 2.0) ** 2
    np.testing.assert_allclose(valid.relative_permittivity, expected_epsilon)
    np.testing.assert_allclose(valid.parameter, (expected_epsilon - 3.5) / 1.5)
    assert valid.parameter.dtype == jnp.float64
    assert bool(valid.all_valid)

    invalid_contract = replace(
        _contract(dtype="float32"),
        lower_relative_permittivity=4.5,
        upper_relative_permittivity=5.0,
    )
    invalid = thermo_optic_parameter_state(
        plan,
        jnp.full((4,), 302.0),
        _law(),
        invalid_contract,
    )
    assert invalid.parameter.dtype == jnp.float32
    assert not bool(invalid.all_valid)
    assert np.isnan(np.asarray(invalid.parameter)).all()


def test_parameter_container_copy_is_closed_and_dtype_exact() -> None:
    state = thermo_optic_parameter_state(_plan(), jnp.full((4,), 302.0), _law(), _contract())
    parameters = {
        "heated-silicon": jnp.zeros((2, 2, 2), dtype=jnp.float64),
        "other": jnp.ones((1,)),
    }
    updated = with_fdtdx_device_parameter(parameters, state, _contract())

    assert updated is not parameters
    assert updated["other"] is parameters["other"]
    assert updated["heated-silicon"] is state.parameter
    for bad, message in (
        ({"other": jnp.zeros((1,))}, "no device"),
        ({"heated-silicon": {"params": jnp.zeros((2, 2, 2))}}, "not a mapping"),
        ({"heated-silicon": object()}, "not array-like"),
        ({"heated-silicon": jnp.zeros((1, 2, 2), dtype=jnp.float64)}, "shape"),
        ({"heated-silicon": jnp.zeros((2, 2, 2), dtype=jnp.float32)}, "dtype"),
    ):
        with pytest.raises(ContractError, match=message):
            with_fdtdx_device_parameter(bad, state, _contract())
    with pytest.raises(ContractError, match="parameter shape"):
        with_fdtdx_device_parameter(
            parameters,
            state._replace(parameter=jnp.zeros((1, 2, 2), dtype=jnp.float64)),
            _contract(),
        )
    with pytest.raises(ContractError, match="parameter dtype"):
        with_fdtdx_device_parameter(
            parameters,
            state._replace(parameter=jnp.zeros((2, 2, 2), dtype=jnp.float32)),
            _contract(),
        )


def test_runtime_adapter_calls_public_fdtdx_apply_params(monkeypatch) -> None:
    contract = _contract()
    state = thermo_optic_parameter_state(_plan(), jnp.full((4,), 302.0), _law(), contract)
    arrays, objects, config, parameters, _device = _runtime(contract)
    seen = {}

    def fake_apply_params(*, arrays, objects, params, key=None):
        seen.update(arrays=arrays, objects=objects, params=params, key=key)
        return arrays, objects, {"called": True}

    monkeypatch.setattr(module, "package_version", lambda name: "0.6.2")
    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: SimpleNamespace(apply_params=fake_apply_params),
    )
    result = apply_thermo_optic_to_fdtdx(
        arrays,
        objects,
        parameters,
        config,
        state,
        contract,
        verified_fingerprint=_fingerprint(),
        key="key",
    )

    assert result[2] == {"called": True}
    assert seen["params"][contract.device_name] is state.parameter
    assert seen["key"] == "key"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda a, o, c, p, d: setattr(o, "devices", (d,)), "device list"),
        (lambda a, o, c, p, d: setattr(o, "devices", []), "exactly one"),
        (lambda a, o, c, p, d: setattr(d, "matrix_voxel_grid_shape", (1, 2, 2)), "grid shape"),
        (
            lambda a, o, c, p, d: setattr(d, "single_voxel_grid_shape", (2, 1, 1)),
            "one design voxel",
        ),
        (lambda a, o, c, p, d: setattr(d, "use_etching", True), "etching"),
        (lambda a, o, c, p, d: setattr(d, "param_transforms", (object(),)), "raw continuous"),
        (lambda a, o, c, p, d: setattr(d, "materials", {"only": _Material(3.5)}), "exactly two"),
        (
            lambda a, o, c, p, d: setattr(
                d, "materials", {"a": _Material(3.5), "b": _Material(4.9)}
            ),
            "bracket",
        ),
        (
            lambda a, o, c, p, d: setattr(
                d.materials["lower"],
                "permittivity",
                (3.5, 0.1, 0.0, 0.1, 3.5, 0.0, 0.0, 0.0, 3.5),
            ),
            "must be isotropic",
        ),
        (
            lambda a, o, c, p, d: setattr(
                d, "materials", {"a": _Material(3.5, permeability=2.0), "b": _Material(5.0)}
            ),
            "non-magnetic",
        ),
        (
            lambda a, o, c, p, d: setattr(
                d,
                "materials",
                {"a": _Material(3.5, electric_conductivity=1.0), "b": _Material(5.0)},
            ),
            "cannot be conductive",
        ),
        (
            lambda a, o, c, p, d: setattr(
                d,
                "materials",
                {"a": _Material(3.5, magnetic_conductivity=1.0), "b": _Material(5.0)},
            ),
            "magnetically lossy",
        ),
        (
            lambda a, o, c, p, d: setattr(
                d, "materials", {"a": _Material(3.5, dispersion=object()), "b": _Material(5.0)}
            ),
            "cannot be dispersive",
        ),
        (lambda a, o, c, p, d: setattr(d, "grid_slice_tuple", None), "grid slice"),
        (lambda a, o, c, p, d: setattr(o, "sources", ()), "source list"),
        (
            lambda a, o, c, p, d: setattr(
                o,
                "sources",
                [SimpleNamespace(name="source", grid_slice_tuple=None)],
            ),
            "source has no resolved",
        ),
        (
            lambda a, o, c, p, d: setattr(
                o,
                "sources",
                [
                    SimpleNamespace(
                        name="source",
                        grid_slice_tuple=((1, 2), (0, 1), (0, 1)),
                    )
                ],
            ),
            "overlaps the active",
        ),
        (lambda a, o, c, p, d: setattr(c, "resolved_grid", None), "resolved grid"),
        (
            lambda a, o, c, p, d: setattr(d, "grid_slice_tuple", ((0,), (0, 2), (0, 2))),
            "grid bounds",
        ),
        (
            lambda a, o, c, p, d: setattr(
                c,
                "resolved_grid",
                _Grid((np.asarray((0.2, 0.8)), np.asarray((-0.5, 0.5)), np.asarray((0.25, 0.75)))),
            ),
            "coordinates",
        ),
        (lambda a, o, c, p, d: setattr(a, "inv_permittivities", object()), "inverse-permittivity"),
        (
            lambda a, o, c, p, d: setattr(a, "inv_permittivities", jnp.ones((3, 2, 2, 2))),
            "isotropic",
        ),
    ),
)
def test_runtime_adapter_rejects_fdtdx_semantic_drift(monkeypatch, mutation, message) -> None:
    contract = _contract()
    state = thermo_optic_parameter_state(_plan(), jnp.full((4,), 302.0), _law(), contract)
    arrays, objects, config, parameters, device = _runtime(contract)
    mutation(arrays, objects, config, parameters, device)
    monkeypatch.setattr(module, "package_version", lambda name: "0.6.2")
    monkeypatch.setattr(
        module, "import_module", lambda name: SimpleNamespace(apply_params=lambda **kwargs: None)
    )
    with pytest.raises(ContractError, match=message):
        apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=_fingerprint(),
        )


def test_runtime_adapter_requires_installed_and_attested_fdtdx(monkeypatch) -> None:
    contract = _contract()
    state = thermo_optic_parameter_state(_plan(), jnp.full((4,), 302.0), _law(), contract)
    arrays, objects, config, parameters, _device = _runtime(contract)

    with pytest.raises(ContractError, match="source identity"):
        apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=replace(_fingerprint(), source_digest="0" * 64),
        )
    monkeypatch.setattr(
        module,
        "package_version",
        lambda name: (_ for _ in ()).throw(module.PackageNotFoundError()),
    )
    with pytest.raises(ContractError, match="not installed"):
        apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=_fingerprint(),
        )
    monkeypatch.setattr(module, "package_version", lambda name: "0.6.3")
    monkeypatch.setattr(module, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ContractError, match="version mismatch"):
        apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=_fingerprint(),
        )
    monkeypatch.setattr(module, "package_version", lambda name: "0.6.2")
    with pytest.raises(ContractError, match="callable apply_params"):
        apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=_fingerprint(),
        )
