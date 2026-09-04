from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.errors import ContractError
from femx.interop.fdtdx import mode_source as mode_source_module
from femx.interop.fdtdx.dynamic_mode_source import (
    FDTDXDynamicModeSourceContract,
    build_fdtdx_dynamic_mode_source_contract,
    make_fdtdx_dynamic_mode_source,
    with_fdtdx_dynamic_mode_profile,
)
from femx.interop.fdtdx.mode_bundle import (
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeVectorField,
)
from femx.interop.fdtdx.mode_source import (
    build_fdtdx_mode_source_contract,
    make_fdtdx_mode_source,
    make_fdtdx_mode_source_function,
    validate_fdtdx_mode_source,
)
from femx.interop.fdtdx.mode_transfer import build_yee_grid
from femx.interop.fdtdx.thermo_optic import FDTDXFingerprint

pytestmark = pytest.mark.unit

SHA = "a" * 64
FDTDX = FDTDXFingerprint(
    package_version="0.6.2",
    source_revision="81a58da9cde4a4ff822f835b63597c0d0d8ba978",
    source_digest="c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c",
)


def _bundle() -> ModeBundle:
    grid = build_yee_grid(
        (
            np.asarray((0.0, 100e-9, 300e-9), dtype=np.float64),
            np.asarray((0.0, 120e-9, 350e-9), dtype=np.float64),
            np.asarray((80e-9, 120e-9), dtype=np.float64),
        )
    )
    shape = (3, *grid.shape)
    electric = np.zeros(shape, dtype=np.complex128)
    magnetic = np.zeros(shape, dtype=np.complex128)
    electric[0] = 1.0 + 0.25j
    magnetic[1] = 0.5 - 0.125j
    return ModeBundle(
        frequency_hz=193.41448903225806e12,
        effective_index=2.4 + 0.0j,
        beta_per_m=9.729e6 + 0.0j,
        electric=YeeVectorField(electric, grid, YeeFieldKind.ELECTRIC, "V/m"),
        magnetic=YeeVectorField(magnetic, grid, YeeFieldKind.MAGNETIC, "V/m"),
        propagation=AxisDirection(Axis.Z, Direction.POSITIVE),
        magnetic_convention=MagneticFieldConvention.ETA0_H,
        normalization=ModeNormalization(target_power_watts=1.0),
        solver=SolverFingerprint("test-port", "1", SHA, SHA, "revision"),
        transfer=TransferReport(
            source_representation=FieldRepresentation.FEM_DOFS,
            target_representation=FieldRepresentation.CARTESIAN_YEE_SAMPLES,
            operator_sha256="b" * 64,
            relative_power_error=0.0,
            source_power_watts=1.0,
            pre_correction_power_watts=1.0,
            relative_pre_correction_power_error=0.0,
            transferred_power_watts=1.0,
            power_correction_scale=1.0,
            target_runtime_name="fdtdx",
            target_runtime_version=FDTDX.package_version,
            target_source_revision=FDTDX.source_revision,
            target_source_digest=FDTDX.source_digest,
        ),
    )


def _medium(bundle: ModeBundle) -> tuple[np.ndarray, np.ndarray]:
    inverse_permittivity = np.full(
        (1, *bundle.electric.grid.shape),
        1.0 / 2.085136,
        dtype=np.float64,
    )
    return inverse_permittivity, np.asarray(1.0, dtype=np.float64)


def _contract(bundle: ModeBundle | None = None):
    bundle = _bundle() if bundle is None else bundle
    inverse_permittivity, inverse_permeability = _medium(bundle)
    return build_fdtdx_mode_source_contract(
        bundle,
        source_name="fem-port",
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=inverse_permeability,
        fdtdx=FDTDX,
    )


def _coordinates(bundle: ModeBundle):
    centers = [
        0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
        for edges in bundle.electric.grid.edge_coordinates
    ]
    return tuple(np.asarray(values) for values in np.meshgrid(*centers, indexing="ij"))


def test_mode_source_contract_snapshots_medium_and_hashes_identity() -> None:
    bundle = _bundle()
    inverse_permittivity, inverse_permeability = _medium(bundle)
    contract = _contract(bundle)

    assert contract.grid_shape == (2, 2, 1)
    assert contract.propagation_axis == 2
    assert contract.propagation_direction == "+"
    assert contract.field_dtype == "complex128"
    assert len(contract.mode_bundle_sha256) == 64
    assert len(contract.sha256) == 64
    assert contract.canonical_data()["source_mode_gradient_policy"] == "constant_stop_gradient"
    assert contract.canonical_data()["setup_addressability"] == "host_addressable"
    assert contract.expected_inverse_permittivity is not inverse_permittivity
    assert contract.expected_inverse_permeability is not inverse_permeability
    assert not np.asarray(contract.expected_inverse_permittivity).flags.writeable
    assert not np.asarray(contract.expected_inverse_permeability).flags.writeable
    rebuilt = _contract(bundle)
    assert contract.canonical_data() == rebuilt.canonical_data()
    assert contract.sha256 == rebuilt.sha256
    np.testing.assert_array_equal(
        contract.expected_inverse_permittivity,
        rebuilt.expected_inverse_permittivity,
    )


def test_mode_source_bundle_digest_covers_semantic_metadata() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    variants = (
        replace(bundle, beta_per_m=bundle.beta_per_m * 1.01),
        replace(
            bundle,
            electric=replace(bundle.electric, function_space="changed Yee space"),
        ),
        replace(
            bundle,
            normalization=replace(bundle.normalization, phase_reference="different phase"),
        ),
        replace(bundle, solver=replace(bundle.solver, source_revision="different")),
        replace(
            bundle,
            transfer=replace(bundle.transfer, relative_interpolation_error=1e-6),
        ),
    )

    for variant in variants:
        assert mode_source_module._mode_bundle_sha256(variant) != contract.mode_bundle_sha256
        with pytest.raises(ContractError, match="content differs"):
            make_fdtdx_mode_source_function(variant, contract)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_name": ""}, "name"),
        ({"grid_shape": (2, 0, 1)}, "three positive"),
        ({"grid_shape": (2, 2, 2)}, "one cell thick"),
        ({"propagation_axis": 3}, "axis must be"),
        ({"propagation_axis": 1, "grid_shape": (2, 1, 2)}, "positive-z"),
        ({"propagation_direction": "-"}, "positive-z"),
        ({"frequency_hz": 0.0}, "frequency"),
        ({"effective_index": -1.0 + 0.0j}, "effective index"),
        ({"effective_index": complex(np.nan, 0.0)}, "effective index"),
        ({"field_dtype": "float64"}, "field dtype"),
        ({"coordinate_sha256": "bad"}, "coordinate digest"),
        ({"mode_bundle_sha256": "bad"}, "bundle digest"),
        ({"schema_version": "future"}, "unsupported"),
    ],
)
def test_mode_source_contract_rejects_invalid_metadata(changes, message) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_contract(), **changes)


def test_mode_source_contract_rejects_invalid_medium_and_digests() -> None:
    contract = _contract()
    shape = contract.grid_shape

    with pytest.raises(ContractError, match="shape"):
        replace(
            contract,
            expected_inverse_permittivity=np.ones((1, 2, 1, 1), dtype=np.float64),
        )
    with pytest.raises(ContractError, match="precision"):
        replace(
            contract,
            expected_inverse_permittivity=np.ones((1, *shape), dtype=np.float32),
        )
    with pytest.raises(ContractError, match="finite and positive"):
        replace(
            contract,
            expected_inverse_permittivity=np.zeros((1, *shape), dtype=np.float64),
        )
    with pytest.raises(ContractError, match="permittivity digest"):
        replace(contract, inverse_permittivity_sha256="c" * 64)
    with pytest.raises(ContractError, match="scalar sentinel"):
        replace(
            contract,
            expected_inverse_permeability=np.ones((1,), dtype=np.float64),
        )
    with pytest.raises(ContractError, match="float64 scalar"):
        replace(
            contract,
            expected_inverse_permeability=np.asarray(1.0, dtype=np.float32),
        )
    with pytest.raises(ContractError, match="non-magnetic"):
        replace(
            contract,
            expected_inverse_permeability=np.asarray(0.5, dtype=np.float64),
        )
    with pytest.raises(ContractError, match="permeability digest"):
        replace(contract, inverse_permeability_sha256="d" * 64)


def test_mode_source_contract_rejects_unsupported_arrays_and_target_identity() -> None:
    bundle = _bundle()
    inverse_permittivity, inverse_permeability = _medium(bundle)
    with pytest.raises(ContractError, match="unsupported dtype"):
        build_fdtdx_mode_source_contract(
            bundle,
            source_name="fem-port",
            expected_inverse_permittivity=np.ones((1, *bundle.electric.grid.shape), dtype=np.int64),
            expected_inverse_permeability=inverse_permeability,
            fdtdx=FDTDX,
        )
    with pytest.raises(ContractError, match="target identity"):
        build_fdtdx_mode_source_contract(
            bundle,
            source_name="fem-port",
            expected_inverse_permittivity=inverse_permittivity,
            expected_inverse_permeability=inverse_permeability,
            fdtdx=replace(FDTDX, source_digest="f" * 64),
        )
    backward = replace(bundle, propagation=AxisDirection(Axis.Z, Direction.NEGATIVE))
    with pytest.raises(ContractError, match="positive-z"):
        build_fdtdx_mode_source_contract(
            backward,
            source_name="fem-port",
            expected_inverse_permittivity=inverse_permittivity,
            expected_inverse_permeability=inverse_permeability,
            fdtdx=FDTDX,
        )


def test_mode_source_callback_preserves_fields_and_validates_exact_medium() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    callback = make_fdtdx_mode_source_function(bundle, contract)
    electric, magnetic = callback(
        coordinates=_coordinates(bundle),
        frequency=bundle.frequency_hz,
        propagation_axis=2,
        inv_permittivity=contract.expected_inverse_permittivity,
        inv_permeability=contract.expected_inverse_permeability,
    )
    np.testing.assert_array_equal(electric, bundle.electric.values)
    np.testing.assert_array_equal(magnetic, bundle.magnetic.values)

    wrong_permittivity = np.asarray(contract.expected_inverse_permittivity).copy()
    wrong_permittivity[0, 0, 0, 0] *= 0.9
    with pytest.raises(ContractError, match="inverse permittivity differs"):
        callback(
            coordinates=_coordinates(bundle),
            frequency=bundle.frequency_hz,
            propagation_axis=2,
            inv_permittivity=wrong_permittivity,
            inv_permeability=contract.expected_inverse_permeability,
        )
    with pytest.raises(ContractError, match="inverse permeability differs"):
        callback(
            coordinates=_coordinates(bundle),
            frequency=bundle.frequency_hz,
            propagation_axis=2,
            inv_permittivity=contract.expected_inverse_permittivity,
            inv_permeability=np.asarray(0.9, dtype=np.float64),
        )


def test_mode_source_callback_reports_non_addressable_setup() -> None:
    bundle = _bundle()
    contract = _contract(bundle)

    class NonAddressable:
        shape = contract.grid_shape
        ndim = 3
        dtype = np.dtype(np.float64)

        def __array__(self):
            raise RuntimeError("not addressable")

    with pytest.raises(ContractError, match="host-addressable"):
        mode_source_module._runtime_array(NonAddressable(), label="probe")

    changed_electric = replace(
        bundle.electric,
        values=np.asarray(bundle.electric.values) * (1.0 + 1e-6),
    )
    with pytest.raises(ContractError, match="content differs"):
        make_fdtdx_mode_source_function(replace(bundle, electric=changed_electric), contract)


def test_bundle_contract_comparison_rejects_every_identity_dimension() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    changed_target = replace(FDTDX, source_digest="f" * 64)

    reshaped_eps, reshaped_eps_digest = mode_source_module._canonical_array(
        np.full((1, 1, 4, 1), 1.0 / 2.085136, dtype=np.float64),
        label="source_plane/inverse_permittivity",
    )
    float32_eps, float32_eps_digest = mode_source_module._canonical_array(
        np.asarray(contract.expected_inverse_permittivity, dtype=np.float32),
        label="source_plane/inverse_permittivity",
    )
    float64_mu, float64_mu_digest = mode_source_module._canonical_array(
        np.asarray(1.0, dtype=np.float64),
        label="source_plane/inverse_permeability",
    )
    variants = (
        (replace(contract, fdtdx=changed_target), "target identity"),
        (
            replace(
                contract,
                grid_shape=(1, 4, 1),
                expected_inverse_permittivity=reshaped_eps,
                inverse_permittivity_sha256=reshaped_eps_digest,
            ),
            "grid shape",
        ),
        (replace(contract, coordinate_sha256="c" * 64), "coordinate identity"),
        (
            replace(
                contract,
                field_dtype="complex64",
                expected_inverse_permittivity=float32_eps,
                expected_inverse_permeability=float64_mu,
                inverse_permittivity_sha256=float32_eps_digest,
                inverse_permeability_sha256=float64_mu_digest,
            ),
            "field precision",
        ),
        (replace(contract, frequency_hz=contract.frequency_hz * 0.5), "frequency"),
        (replace(contract, effective_index=2.3 + 0.0j), "effective index"),
    )
    for variant, message in variants:
        with pytest.raises(ContractError, match=message):
            make_fdtdx_mode_source_function(bundle, variant)


def test_bundle_target_fingerprint_rejects_incomplete_metadata() -> None:
    incomplete = SimpleNamespace(
        transfer=SimpleNamespace(
            target_runtime_name=None,
            target_runtime_version=None,
            target_source_revision=None,
            target_source_digest=None,
        )
    )
    with pytest.raises(ContractError, match="no complete FDTDX target"):
        mode_source_module._bundle_target_fingerprint(incomplete)


def test_mode_source_factory_uses_only_the_locked_public_source(monkeypatch) -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    captured: dict[str, object] = {}

    class FakeWaveCharacter:
        def __init__(self, *, frequency):
            self.frequency = frequency

    def source_constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    module = SimpleNamespace(
        CustomModePlaneSource=source_constructor,
        WaveCharacter=FakeWaveCharacter,
    )
    monkeypatch.setattr(mode_source_module, "package_version", lambda _name: "0.6.2")
    monkeypatch.setattr(mode_source_module, "import_module", lambda _name: module)
    temporal_profile = object()
    source = make_fdtdx_mode_source(
        bundle,
        contract,
        verified_fingerprint=FDTDX,
        temporal_profile=temporal_profile,
    )

    assert source.name == "fem-port"
    assert source.partial_grid_shape == contract.grid_shape
    assert source.wave_character.frequency == contract.frequency_hz
    assert source.direction == "+"
    assert source.effective_index == contract.effective_index
    assert source.normalize is False
    assert source.allow_device_overlap is False
    assert source.temporal_profile is temporal_profile
    assert callable(captured["mode_function"])


def test_mode_source_factory_fails_closed_on_runtime_identity(monkeypatch) -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    with pytest.raises(ContractError, match="verified FDTDX"):
        make_fdtdx_mode_source(
            bundle,
            contract,
            verified_fingerprint=replace(FDTDX, source_digest="f" * 64),
        )

    monkeypatch.setattr(mode_source_module, "package_version", lambda _name: "9.9.9")
    monkeypatch.setattr(mode_source_module, "import_module", lambda _name: SimpleNamespace())
    with pytest.raises(ContractError, match="package version mismatch"):
        make_fdtdx_mode_source(bundle, contract, verified_fingerprint=FDTDX)

    def missing_package(_name):
        raise mode_source_module.PackageNotFoundError("fdtdx")

    monkeypatch.setattr(mode_source_module, "package_version", missing_package)
    with pytest.raises(ContractError, match="not installed"):
        make_fdtdx_mode_source(bundle, contract, verified_fingerprint=FDTDX)


@pytest.mark.parametrize("missing", ["source", "wave"])
def test_mode_source_factory_requires_new_public_fdtdx_api(monkeypatch, missing) -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    module = SimpleNamespace(
        CustomModePlaneSource=None if missing == "source" else lambda **kwargs: kwargs,
        WaveCharacter=None if missing == "wave" else lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(mode_source_module, "package_version", lambda _name: "0.6.2")
    monkeypatch.setattr(mode_source_module, "import_module", lambda _name: module)
    expected = "CustomModePlaneSource" if missing == "source" else "WaveCharacter"
    with pytest.raises(ContractError, match=expected):
        make_fdtdx_mode_source(bundle, contract, verified_fingerprint=FDTDX)


class _Grid:
    def __init__(self, bundle: ModeBundle, *, shift_axis: int | None = None):
        self._edges = [np.asarray(values) for values in bundle.electric.grid.edge_coordinates]
        if shift_axis is not None:
            self._edges[shift_axis] = self._edges[shift_axis] + 1e-12

    def edges(self, axis: int):
        return self._edges[axis]


def _placed_source(bundle: ModeBundle, contract, *, shift_axis: int | None = None):
    return SimpleNamespace(
        name=contract.source_name,
        grid_shape=contract.grid_shape,
        propagation_axis=contract.propagation_axis,
        direction=contract.propagation_direction,
        normalize=False,
        allow_device_overlap=False,
        wave_character=SimpleNamespace(get_frequency=lambda: contract.frequency_hz),
        _E=np.asarray(bundle.electric.values),
        _H=np.asarray(bundle.magnetic.values),
        _inv_permittivity=np.asarray(contract.expected_inverse_permittivity),
        _inv_permeability=np.asarray(contract.expected_inverse_permeability),
        _config=SimpleNamespace(resolved_grid=_Grid(bundle, shift_axis=shift_axis)),
        grid_slice_tuple=((0, 2), (0, 2), (0, 1)),
    )


def test_validate_placed_fdtdx_mode_source_accepts_exact_binding() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    validate_fdtdx_mode_source(_placed_source(bundle, contract), bundle, contract)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("name", "other", "name"),
        ("grid_shape", (1, 2, 1), "shape"),
        ("propagation_axis", 0, "axis"),
        ("direction", "-", "direction"),
        ("normalize", True, "normalization"),
        ("allow_device_overlap", True, "Device overlap"),
        ("wave_character", None, "frequency"),
        ("_E", np.zeros((3, 2, 2, 1), dtype=np.complex128), "electric field"),
        ("_H", np.ones((3, 2, 2, 1), dtype=np.complex128), "magnetic field"),
        (
            "_inv_permittivity",
            np.ones((1, 2, 2, 1), dtype=np.float64),
            "inverse permittivity",
        ),
        ("_inv_permeability", np.asarray(0.5, dtype=np.float64), "inverse permeability"),
        ("_config", None, "resolved three-axis grid"),
    ],
)
def test_validate_placed_fdtdx_mode_source_rejects_mutation(attribute, value, message) -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    source = _placed_source(bundle, contract)
    setattr(source, attribute, value)
    with pytest.raises(ContractError, match=message):
        validate_fdtdx_mode_source(source, bundle, contract)


def test_validate_placed_fdtdx_mode_source_rejects_frequency_and_edge_changes() -> None:
    bundle = _bundle()
    contract = _contract(bundle)
    source = _placed_source(bundle, contract)
    source.wave_character = SimpleNamespace(get_frequency=lambda: contract.frequency_hz * 2.0)
    with pytest.raises(ContractError, match="frequency"):
        validate_fdtdx_mode_source(source, bundle, contract)

    with pytest.raises(ContractError, match="axis 1"):
        validate_fdtdx_mode_source(
            _placed_source(bundle, contract, shift_axis=1),
            bundle,
            contract,
        )


def _dynamic_contract(bundle: ModeBundle | None = None) -> FDTDXDynamicModeSourceContract:
    bundle = _bundle() if bundle is None else bundle
    return build_fdtdx_dynamic_mode_source_contract(
        bundle,
        _contract(bundle),
        parameter_names=("core_epsilon_r",),
        parameter_units=("1",),
    )


def test_dynamic_mode_source_contract_records_only_static_gradient_identity() -> None:
    contract = _dynamic_contract()

    assert contract.parameter_names == ("core_epsilon_r",)
    assert contract.transfer_operator_sha256 == "b" * 64
    assert contract.target_power_watts == pytest.approx(1.0)
    assert len(contract.sha256) == 64
    data = contract.canonical_data()
    assert data["source_mode_gradient_policy"] == "dynamic_profile_checkpointed_reverse"
    assert data["reversible_gradient_policy"] == "unsupported_source_object_cotangent"
    assert data["source_plane_medium_policy"] == "fixed_baseline_snapshot"
    assert contract.sha256 == _dynamic_contract().sha256


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"parameter_names": ()}, "at least one"),
        ({"parameter_units": ()}, "must align"),
        ({"parameter_names": ("p", "p"), "parameter_units": ("1", "1")}, "unique"),
        ({"parameter_names": (" p",)}, "non-empty and trimmed"),
        ({"parameter_units": ("",)}, "units must be"),
        ({"transfer_operator_sha256": "bad"}, "SHA-256"),
        ({"target_power_watts": 0.0}, "target power"),
        ({"schema_version": "future"}, "unsupported"),
    ],
)
def test_dynamic_mode_source_contract_rejects_invalid_metadata(changes, message) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_dynamic_contract(), **changes)


def _dynamic_placed_source(bundle: ModeBundle, contract: FDTDXDynamicModeSourceContract):
    source = _placed_source(bundle, contract.baseline)
    source.allow_profile_updates = True
    captured: dict[str, object] = {}

    def update(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(updated=True, **kwargs)

    source.with_mode_profile = update
    return source, captured


def test_dynamic_mode_source_binds_arrays_without_host_conversion() -> None:
    bundle = _bundle()
    contract = _dynamic_contract(bundle)
    source, captured = _dynamic_placed_source(bundle, contract)
    effective_index = np.asarray(bundle.effective_index, dtype=np.complex128)

    updated = with_fdtdx_dynamic_mode_profile(
        source,
        contract,
        electric_v_per_m=bundle.electric.values,
        magnetic_eta0_v_per_m=bundle.magnetic.values,
        effective_index=effective_index,
    )

    assert updated.updated is True
    assert captured["mode_E"] is bundle.electric.values
    assert captured["mode_H"] is bundle.magnetic.values
    assert captured["effective_index"] is effective_index


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: setattr(source, "name", "other"), "name"),
        (lambda source: setattr(source, "grid_shape", (1, 4, 1)), "shape"),
        (lambda source: setattr(source, "propagation_axis", 0), "axis"),
        (lambda source: setattr(source, "direction", "-"), "direction"),
        (lambda source: setattr(source, "normalize", True), "normalization"),
        (lambda source: setattr(source, "allow_device_overlap", True), "overlap"),
        (lambda source: setattr(source, "allow_profile_updates", False), "profile updates"),
        (lambda source: setattr(source, "with_mode_profile", None), "no public"),
    ],
)
def test_dynamic_mode_source_rejects_runtime_contract_mutation(mutation, message) -> None:
    bundle = _bundle()
    contract = _dynamic_contract(bundle)
    source, _captured = _dynamic_placed_source(bundle, contract)
    mutation(source)
    with pytest.raises(ContractError, match=message):
        with_fdtdx_dynamic_mode_profile(
            source,
            contract,
            electric_v_per_m=bundle.electric.values,
            magnetic_eta0_v_per_m=bundle.magnetic.values,
            effective_index=np.asarray(bundle.effective_index, dtype=np.complex128),
        )


def test_dynamic_mode_source_rejects_profile_shape_and_precision() -> None:
    bundle = _bundle()
    contract = _dynamic_contract(bundle)
    source, _captured = _dynamic_placed_source(bundle, contract)

    with pytest.raises(ContractError, match="electric profile must have shape"):
        with_fdtdx_dynamic_mode_profile(
            source,
            contract,
            electric_v_per_m=bundle.electric.values[:, :-1],
            magnetic_eta0_v_per_m=bundle.magnetic.values,
            effective_index=np.asarray(bundle.effective_index, dtype=np.complex128),
        )
    with pytest.raises(ContractError, match="magnetic profile precision"):
        with_fdtdx_dynamic_mode_profile(
            source,
            contract,
            electric_v_per_m=bundle.electric.values,
            magnetic_eta0_v_per_m=bundle.magnetic.values.astype(np.complex64),
            effective_index=np.asarray(bundle.effective_index, dtype=np.complex128),
        )
    with pytest.raises(ContractError, match="effective index must be a scalar"):
        with_fdtdx_dynamic_mode_profile(
            source,
            contract,
            electric_v_per_m=bundle.electric.values,
            magnetic_eta0_v_per_m=bundle.magnetic.values,
            effective_index=np.asarray((bundle.effective_index,), dtype=np.complex128),
        )
    with pytest.raises(ContractError, match="effective-index precision"):
        with_fdtdx_dynamic_mode_profile(
            source,
            contract,
            electric_v_per_m=bundle.electric.values,
            magnetic_eta0_v_per_m=bundle.magnetic.values,
            effective_index=np.asarray(bundle.effective_index, dtype=np.complex64),
        )


def test_dynamic_mode_source_factory_requires_public_profile_method(monkeypatch) -> None:
    bundle = _bundle()
    contract = _dynamic_contract(bundle)
    captured: dict[str, object] = {}

    class FakeWaveCharacter:
        def __init__(self, *, frequency):
            self.frequency = frequency

    def source_constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    module = SimpleNamespace(
        CustomModePlaneSource=source_constructor,
        WaveCharacter=FakeWaveCharacter,
    )
    monkeypatch.setattr(mode_source_module, "package_version", lambda _name: "0.6.2")
    monkeypatch.setattr(mode_source_module, "import_module", lambda _name: module)
    with pytest.raises(ContractError, match="no public dynamic-profile"):
        make_fdtdx_dynamic_mode_source(
            bundle,
            contract,
            verified_fingerprint=FDTDX,
        )
    assert captured["allow_profile_updates"] is True

    module.CustomModePlaneSource = lambda **kwargs: SimpleNamespace(
        **kwargs,
        with_mode_profile=lambda **values: values,
    )
    source = make_fdtdx_dynamic_mode_source(
        bundle,
        contract,
        verified_fingerprint=FDTDX,
    )
    assert source.allow_profile_updates is True
